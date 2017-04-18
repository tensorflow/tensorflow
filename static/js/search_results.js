var reset_height = function() {
  var row = $('#image-grid');
  $.each(row, function() {
    $.each($(this).find('div[class^="col-"]'), function() {
      $(this).height('auto');
    })
  })
}

var normalize_grid = function() {
  var row = $('#image-grid');
  $.each(row, function() {
    var maxh = 0;
    $.each($(this).find('div[class^="col-"]'), function() {
      if ($(this).height() > maxh)
        maxh = $(this).height();
    });
    $.each($(this).find('div[class^="col-"]'), function() {
      $(this).height(maxh);
    });
  });
}

$(document).ready(function() {  // wait until document is ready
    normalize_grid();
});

$(window).resize(function() {
  reset_height();
  normalize_grid();
});
