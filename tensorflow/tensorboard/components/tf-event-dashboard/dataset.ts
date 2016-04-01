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
module TF {
  /* An extension of Plottable.Dataset that knows how to load data from a backend.
   */
  export class Dataset extends Plottable.Dataset {
    public tag: string;
    public run: string;
    private lastLoadTime: number;
    private lastRequest;
    private urlGenerator: Function;

    constructor(tag: string, run: string, urlGenerator: (run: string, tag: string) => string) {
      super([], {tag: tag, run: run});
      this.tag = tag;
      this.run = run;
      this.urlGenerator = urlGenerator;
    }

    public load = _.debounce(this._load, 10);

    private _load() {
      var url = this.urlGenerator(this.tag, this.run);
      if (this.lastRequest != null) {
        this.lastRequest.abort();
      }
      this.lastRequest = d3.json(url, (error, json) => {
        this.lastRequest = null;
        if (error) {
          /* tslint:disable */
          console.log(error);
          /* tslint:enable */
          throw new Error("Failure loading JSON at url: \"" + url + "\"");
        } else {
          this.data(json);
        }
      });
    }
  }
}
