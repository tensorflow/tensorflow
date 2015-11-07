/// <reference path="../../typings/tsd.d.ts" />
/// <reference path="../../bower_components/plottable/plottable.d.ts" />

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
