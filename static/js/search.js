var requesting = false;

var submitForm = function() {
  var species = $("#species").val();
  var entries = 36;
  var page = 1;

  var url = encodeURI("/images/search?entries=" + entries + "&species=" + species + "&page=" + page);
  window.location.href = url;
  //var url = "/images/search";

  /*if (!requesting) {
    requesting = true;
    $.ajax({
      url:url,
      type:"get",
      data: {
        species:species,
        entries:36,
        page:1
      },
      success: function(response) {
        //document.html = response;
        requesting = false;
      },
      error: function(xhr) {

      }
    });
  }*/
}
