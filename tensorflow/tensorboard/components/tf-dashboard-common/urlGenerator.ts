/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
