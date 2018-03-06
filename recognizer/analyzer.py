from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.http import JsonResponse
from tempfile import mkstemp
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import os.path
import sys
import tarfile
import requests
from six.moves import urllib
import urllib2
from urllib2 import HTTPError
import tensorflow as tf
import numpy as np
import re

IMAGE_DIR = '/tmp/imagenet'

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

class AnalyzeError(Exception):
    def __init__(self, message):
        self.message = message

def analyze(request):
    temp_path = None
    classifications = None
    try:
        ensure_model_downloaded()
        url = username = request.GET.get('url', None)
        val = URLValidator()
        val(url)
        if not valid_url_extension(url):
            raise AnalyzeError("Must be valid image file type.")

        fd, temp_path = mkstemp()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1636.0 Safari/537.36',
        }
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            os.write(fd,r.content)
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                raise AnalyzeError("URL not found. (404)")
            else:
                raise AnalyzeError("Unable to open URL (" + str(err.code) + ")")
        finally:
            os.close(fd)

        classifcations = classify_image(temp_path)

        os.remove(temp_path)
    except (AnalyzeError, ValidationError) as err:
        return build_response(None, str(err.message))
    finally:
        if (temp_path and os.path.isfile(temp_path)):
            os.remove(temp_path)

    return build_response(classifcations, None)

def build_response(classifications, error_response):
    data = {}
    data['is_success'] = 'False' if (error_response) else 'True'
    if (error_response):
        data['error_response'] = error_response
    else:
        data['classifications'] = classifications

    return JsonResponse(data)

def _progress(count, block_size, total_size):
#    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

def ensure_model_downloaded():
    """Download and extract model tar file."""
    dest_directory = IMAGE_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filepath = local_filepath_for_url(DATA_URL, dest_directory)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filepath, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def local_filepath_for_url(url,dest_directory):
    filename = url.split('/')[-1]
    return os.path.join(dest_directory, filename)

VALID_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
]

def valid_url_extension(url):
    return any([url.split('?')[0].lower().endswith(e) for e in VALID_IMAGE_EXTENSIONS])

def classify_image(image):

  classifications = []

  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-5:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      # if string is multiple comma-separated classifications, just take the first
      human_string = human_string.split(",")[0]
      score = predictions[node_id]*100
      if score >= 1:
          score_str = str(int(score))
      else:
          score_str = "<1"
      classifications.append({'name': human_string, 'score': score_str})

  return classifications

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      IMAGE_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          IMAGE_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          IMAGE_DIR, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]
