/// <reference path="../../typings/tsd.d.ts" />
/// <reference path="../../bower_components/plottable/plottable.d.ts" />

module TF {
  export module Urls {

    export var routes = ["runs", "scalars", "histograms",
                         "compressedHistograms", "images",
                         "individualImage", "graph"];

    function router(route: string): ((tag: string, run: string) => string) {
      return function(tag: string, run: string): string {
        return "/" + route + "?tag=" + encodeURIComponent(tag)
                           + "&run=" + encodeURIComponent(run);
      };
    }

    export function runsUrl() {
      return "/runs";
    }
    export var scalarsUrl = router("scalars");
    export var histogramsUrl = router("histograms");
    export var compressedHistogramsUrl = router("compressedHistograms");
    export var imagesUrl = router("images");
    export function individualImageUrl(query: string) {
      return "/individualImage?" + query;
    }
    export function graphUrl(run: string) {
      return "/graph?run=" + encodeURIComponent(run);
    }

  }
}
