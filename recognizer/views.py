from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
import requests
from . import analyzer

def index(request):
    context = {}
    return render(request, "index.html", context)

def analyze(request):
    return analyzer.analyze(request)
