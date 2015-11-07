/// <reference path="../../typings/tsd.d.ts" />
/// <reference path="../../bower_components/plottable/plottable.d.ts" />

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
