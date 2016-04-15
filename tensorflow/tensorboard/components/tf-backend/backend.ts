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
  // TODO(cassandrax): Remove this interface.
  export interface RunEnumeration {
    histograms: string[];
    compressedHistogramTuples: string[];
    scalars: string[];
    images: string[];
    graph: boolean;
    run_metadata: string[];
  }


  // TODO(cassandrax): Remove this interface.
  export interface RunsResponse { [runName: string]: RunEnumeration; }

  export type RunToTag = {[run: string]: string[]};

  export interface Datum {
    wall_time: Date;
    step: number;
  }

  export type ScalarDatum = Datum & Scalar;
  export interface Scalar { scalar: number; }

  export type HistogramDatum = Datum & Histogram;
  export interface Histogram {
    min: number;
    max: number;
    nItems?: number;
    sum?: number;
    sumSquares?: number;
    bucketRightEdges: number[];
    bucketCounts: number[];
  }

  export interface HistogramBin { x: number, dx: number, y: number }
  export type HistogramSeriesDatum = HistogramSeries & Datum;
  export interface HistogramSeries { bins: HistogramBin[] }

  export type ImageDatum = Datum & Image;
  export interface Image {
    width: number;
    height: number;
    url: string;
  }


  export var TYPES = [
    'scalar', 'histogram', 'compressedHistogram', 'graph', 'image',
    'runMetadata'
  ];
  /**
   * The Backend class provides a convenient and typed interface to the backend.
   *
   * It provides methods corresponding to the different data sources on the
   * TensorBoard backend. These methods return a promise containing the data
   * from the backend. This class does some post-processing on the data; for
   * example, converting data elements tuples into js objects so that they can
   * be accessed in a more convenient and clearly-documented fashion.
   */
  export class Backend {
    public router: Router;
    public requestManager: RequestManager;

    /**
     * Construct a Backend instance.
     * @param router the Router with info on what urls to get data from
     * @param requestManager The RequestManager, overwritable so you may
     * manually clear request queue, etc. Defaults to a new RequestManager.
     */
    constructor(r: Router, requestManager?: RequestManager) {
      this.router = r;
      this.requestManager = requestManager || new RequestManager();
    }

    /**
     * Returns a listing of all the available data in the TensorBoard backend.
     * Will be deprecated in the future, in favor of
     * per-data-type methods.
     */
    public runs(): Promise<RunsResponse> {
      return this.requestManager.request(this.router.runs());
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for scalar data.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public scalarRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'scalars'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for histogram data.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public histogramRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'histograms'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for image data.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public imageRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'images'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for compressedHistogram
     * data.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public compressedHistogramRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'compressedHistograms'));
    }

    /**
     * Return a promise showing list of runs that contain graphs.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public graphRuns(): Promise<string[]> {
      return this.runs().then(
          (x) => { return _.keys(x).filter((k) => x[k].graph); });
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for run_metadata objects.
     * TODO(cassandrax): Replace this with the direct route, when
     * available.
     */
    public runMetadataRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'run_metadata'));
    }

    /**
     * Return a promise of a graph string from the backend.
     */
    public graph(
        tag: string, limit_attr_size?: number,
        large_attrs_key?: string): Promise<string> {
      let url = this.router.graph(tag, limit_attr_size, large_attrs_key);
      return this.requestManager.request(url);
    }

    /**
     * Return a promise containing ScalarDatums for given run and tag.
     */
    public scalar(tag: string, run: string): Promise<Array<ScalarDatum>> {
      let p: Promise<TupleData<number>[]>;
      let url = this.router.scalars(tag, run);
      p = this.requestManager.request(url);
      return p.then(map(detupler(createScalar)));
    }

    /**
     * Return a promise containing HistogramDatums for given run and tag.
     */
    public histogram(tag: string, run: string):
        Promise<Array<HistogramSeriesDatum>> {
      let p: Promise<TupleData<HistogramTuple>[]>;
      let url = this.router.histograms(tag, run);
      p = this.requestManager.request(url);
      return p.then(map(detupler(createHistogram))).then(function(histos) {
        return histos.map(function(histo, i) {
          return {
            wall_time: histo.wall_time,
            step: histo.step,
            bins: convertBins(histo)
          };
        });
      });
    }

    /**
     * Return a promise containing ImageDatums for given run and tag.
     */
    public image(tag: string, run: string): Promise<Array<ImageDatum>> {
      let url = this.router.images(tag, run);
      let p: Promise<ImageMetadata[]>;
      p = this.requestManager.request(url);
      return p.then(map(this.createImage.bind(this)));
    }

    /**
     * Returns a promise to load the string RunMetadata for given run/tag.
     */
    public runMetadata(tag: string, run: string): Promise<string> {
      let url = this.router.runMetadata(tag, run);
      return this.requestManager.request(url);
    }

    /**
     * Get compressedHistogram data.
     * Unlike other methods, don't bother reprocessing this data into a nicer
     * format. This is because we will deprecate this route.
     */
    private compressedHistogram(tag: string, run: string):
        Promise<Array<Datum&CompressedHistogramTuple>> {
      let url = this.router.compressedHistograms(tag, run);
      let p: Promise<TupleData<CompressedHistogramTuple>[]>;
      p = this.requestManager.request(url);
      return p.then(map(detupler((x) => x)));
    }

    private createImage(x: ImageMetadata): Image&Datum {
      return {
        width: x.width,
        height: x.height,
        wall_time: timeToDate(x.wall_time),
        step: x.step,
        url: this.router.individualImage(x.query),
      };
    }
  }

  /** Given a RunToTag, return sorted array of all runs */
  export function getRuns(r: RunToTag): string[] { return _.keys(r).sort(); }

  /** Given a RunToTag, return array of all tags (sorted + dedup'd) */
  export function getTags(r: RunToTag): string[] {
    return _.union.apply(null, _.values(r)).sort();
  }

  /**
   * Given a RunToTag and an array of runs, return every tag that appears for
   * at least one run.
   * Sorted, deduplicated.
   */
  export function filterTags(r: RunToTag, runs: string[]): string[] {
    var result = [];
    runs.forEach((x) => result = result.concat(r[x]));
    return _.uniq(result).sort();
  }

  function timeToDate(x: number): Date { return new Date(x * 1000); };

  /**  Just a curryable map to make things cute and tidy. */
  function map<T, U>(f: (x: T) => U): (arr: T[]) => U[] {
    return function(arr: T[]): U[] { return arr.map(f); };
  };

  /**
   * This is a higher order function that takes a function that transforms a
   * T into a G, and returns a function that takes TupleData<T>s and converts
   * them into the intersection of a G and a Datum.
   */
  function detupler<T, G>(xform: (x: T) => G): (t: TupleData<T>) => Datum & G {
    return function(x: TupleData<T>): Datum & G {
      // Create a G, assert it has type <G & Datum>
      let obj = <G&Datum>xform(x[2]);
      // ... patch in the properties of datum
      obj.wall_time = timeToDate(x[0]);
      obj.step = x[1];
      return obj;
    };
  };

  function createScalar(x: number): Scalar { return {scalar: x}; };

  function createHistogram(x: HistogramTuple): Histogram {
    return {
      min: x[0],
      max: x[1],
      nItems: x[2],
      sum: x[3],
      sumSquares: x[4],
      bucketRightEdges: x[5],
      bucketCounts: x[6],
    };
  };

  /**
   * Takes histogram data as stored by tensorboard backend and converts it to
   * the standard d3 histogram data format to make it more compatible and easier
   * to visualize. When visualizing histograms, having the left edge and width
   * makes things quite a bit easier.
   *
   * @param {histogram} Histogram - A histogram from tensorboard backend.
   * @return {HistogramBin[]} - Each bin has an x (left edge), a dx (width), and a y (count).
   *
   * If given rightedges are inclusive, then these left edges (x) are exclusive.
   */
  export function convertBins(histogram: Histogram) {
    if (histogram.bucketRightEdges.length !== histogram.bucketCounts.length) {
      throw(new Error('Edges and counts are of different lengths.'))
    }

    var previousRightEdge = histogram.min;
    return histogram.bucketRightEdges.map(function(
        rightEdge: number, i: number) {

      // Use the previous bin's rightEdge as the new leftEdge
      var left = previousRightEdge;

      // We need to clip the rightEdge because right-most edge can be
      // infinite-sized
      var right = Math.min(histogram.max, rightEdge);

      // Store rightEdgeValue for next iteration
      previousRightEdge = rightEdge;

      return {x: left, dx: right - left, y: histogram.bucketCounts[i]};
    });
  }

  /**
    * The following interfaces (TupleData, HistogramTuple,
    * CompressedHistogramTuple, and ImageMetadata) describe how the data is sent
    * over from the backend; the numbers are wall_time then step
    */
  type TupleData<T> = [number, number, T];

  // Min, Max, nItems, Sum, Sum_Squares, right edges of buckets, nItems in
  // buckets
  type HistogramTuple =
      [number, number, number, number, number, number[], number[]];
  type CompressedHistogramTuple = [number, number][];  // percentile, value
  interface ImageMetadata {
    width: number;
    height: number;
    wall_time: number;
    step: number;
    query: string;
  }
}
