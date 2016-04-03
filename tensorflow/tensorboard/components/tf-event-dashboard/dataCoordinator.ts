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

  /* The DataCoordinator generates TF.Datasets for each run/tag combination,
   * and is responsible for communicating with the backend to load data into them.
   * A key fact about this design is that when Datasets modify their data, they
   * automatically notify all dependent Plottable charts.
   */
  export class DataCoordinator {
    private urlGenerator: (tag: string, run: string) => string;
    private datasets: {[key: string]: TF.Dataset};
    private runToTag: {[run: string]: string[]};

    constructor(urlGenerator: (tag: string, run: string) => string,
                runToTag: {[run: string]: string[]}) {
      this.datasets = {};
      this.urlGenerator = urlGenerator;
      this.runToTag = runToTag;
    }

    /* Create or return an array of Datasets for the given
     * tag and runs. It filters which runs it uses by checking
     * that data exists for each tag-run combination.
     * Calling this triggers a load on the dataset.
     */
    public getDatasets(tag: string, runs: string[]) {
      var usableRuns = runs.filter((r) => {
        var tags = this.runToTag[r];
        return tags.indexOf(tag) !== -1;
      });
      return usableRuns.map((r) => this.getDataset(tag, r));
    }

    /* Create or return a Dataset for given tag and run.
     * Calling this triggers a load on the dataset.
     */
    public getDataset(tag: string, run: string): TF.Dataset {
      var dataset = this._getDataset(tag, run);
      dataset.load();
      return dataset;
    }

    private _getDataset(tag: string, run: string): TF.Dataset {
      var key = [tag, run].toString();
      var dataset: TF.Dataset;
      if (this.datasets[key] != null) {
        dataset = this.datasets[key];
      } else {
        dataset = new TF.Dataset(tag, run, this.urlGenerator);
        this.datasets[key] = dataset;
      }
      return dataset;
    }
  }
}
