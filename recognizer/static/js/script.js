
$(document).ready(function() {

	"use strict";
	$(window).load(function () {
		$(".loaded").fadeOut();
		$(".preloader").delay(1000).fadeOut("slow");
	});

	$( "#submit" ).click(function() {
		var url = $("#url").val();
		$("#response-text").html("");
		$("#show-image").html("");
			$(".preloader").fadeIn("fast");
		$(".loaded").fadeIn();
		$.ajax({
			url: '/ajax/analyze',
			data: {
				'url': url
			},
			dataType: 'json',
			success: function (data) {
				hideProgressSpinner();
				var message = "";
				if (data.is_success === 'True') {
					var html = '<img src="' + url + '" alt="" style="width:100%;max-width:640px"/>';
					$("#show-image").html(html);
					var classification = data.classifications[0];
					message = "<p>I think this is a:<br/><span style=\"font-size: 1.8em;padding-bottom:8px\"><b>"
						+ classification.name + "</b></span><br/>(" + classification.score + "% confidence)</p>";

					if (data.classifications.length > 1) {
						message += "<p>Other guesses:<p><p>";
						for (var i=1; i < data.classifications.length; i++) {
							classification = data.classifications[i];
							message +=  classification.name + " (" + classification.score + "% confidence)<br/>"
						}
						message += "</p>"
					}
				}
				else {
					message = data.error_response;
				}
				$("#response-text").html(message);
			},
			error: function (xhr, ajaxOptions, thrownError) {
				hideProgressSpinner();
				var message = "Sorry, there was an error.<br/>(" + xhr.status + ")";
				$("#response-text").html(message);
			}
		});
	});

	function hideProgressSpinner() {
		$(".loaded").fadeOut();
		$(".preloader").fadeOut("fast");
	}

	$('.owl-carousel').owlCarousel({
	    loop:true,
	    margin:10,
	    nav:true,
			autoplay: true,
			autoplayTimeout: 2000,
	    responsive:{
	        0:{
	            items:1
	        },
	        600:{
	            items:3
	        },
	        1000:{
	            items:5
	        }
	    }
	});

	$('.owl-carousel').on('click', '.item', function () {
			var relative_url = $(this).children("img").attr('src');
			var url = new URL(relative_url, window.location.href).href
			$('#url').val(url);
			$('#submit').click();
			$(window).scrollTop(0);
	});

});
