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
module TF.Backend {
  export type RunTagUrlFn = (tag: string, run: string) => string;

  export interface Router {
    runs: () => string;
    scalars: RunTagUrlFn;
    histograms: RunTagUrlFn;
    compressedHistograms: RunTagUrlFn;
    images: RunTagUrlFn;
    individualImage: (query: string) => string;
    graph: (run: string, limit_attr_size?: number, large_attrs_key?: string)
        => string;
    runMetadata: RunTagUrlFn;
  };

  /**
   * The standard router for communicating with the TensorBoard backend
   * @param dataDir {string} The base prefix for finding data on server.
   * @param demoMode {boolean} Whether to modify urls for filesystem demo usage.
   */
  export function router(dataDir = "/data", demoMode = false): Router {
    var clean = demoMode ? demoify : (x) => x;
    if (dataDir[dataDir.length - 1] === "/") {
      dataDir = dataDir.slice(0, dataDir.length - 1);
    }
    function standardRoute(route: string): ((tag: string, run: string) => string) {
      return function(tag: string, run: string): string {
        var url = dataDir + "/" + route + clean(queryEncoder({tag: tag, run: run}));
        if (demoMode) {
          url += ".json";
        }
        return url;
      };
    }
    function individualImageUrl(query: string) {
      var url = dataDir + "/" + clean("individualImage?" + query);
      if (demoMode) {
        url += ".png";
      }
      return url;
    }
    function graphUrl(run: string, limit_attr_size?: number,
        large_attrs_key?: string) {
      let query_params = [["run", clean(run)]];
      if (limit_attr_size != null && !demoMode) {
        query_params.push(["limit_attr_size", String(limit_attr_size)]);
      }
      if (large_attrs_key != null && !demoMode) {
        query_params.push(["large_attrs_key", large_attrs_key]);
      }
      let query = query_params.map(param => {
        return param[0] + "=" + encodeURIComponent(param[1]);
      }).join("&");
      var url = dataDir + "/graph" + clean("?" + query);
      if (demoMode) {
        url += ".pbtxt";
      }
      return url;
    }
    return {
      runs: () => dataDir + "/runs" + (demoMode ? ".json" : ""),
      individualImage: individualImageUrl,
      graph: graphUrl,
      scalars: standardRoute("scalars"),
      histograms: standardRoute("histograms"),
      compressedHistograms: standardRoute("compressedHistograms"),
      images: standardRoute("images"),
      runMetadata: standardRoute("run_metadata"),
    };
  };
}
