/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
  export interface RunEnumeration {
    histograms: string[];
    compressedHistogramTuples: string[];
    scalars: string[];
    images: string[];
    audio: string[];
    graph: boolean;
    run_metadata: string[];
  }

  export interface LogdirResponse { logdir: string; }

  export interface RunsResponse { [runName: string]: RunEnumeration; }

  export type RunToTag = {[run: string]: string[];};

  export interface Datum {
    wall_time: Date;
    step: number;
  }

  export type ScalarDatum = Datum & Scalar;
  export interface Scalar { scalar: number; }

  export interface Text { text: string; }
  export type TextDatum = Datum & Text;

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

  export interface HistogramBin {
    x: number;
    dx: number;
    y: number;
  }
  export type HistogramSeriesDatum = HistogramSeries & Datum;
  export interface HistogramSeries { bins: HistogramBin[]; }

  export type ImageDatum = Datum & Image;
  export interface Image {
    width: number;
    height: number;
    url: string;
  }

  export type AudioDatum = Datum & Audio;
  export interface Audio {
    content_type: string;
    url: string;
  }

  // A health pill encapsulates an overview of tensor element values. The value
  // field is a list of 12 numbers that shed light on the status of the tensor.
  export interface HealthPill {
    node_name: string;
    output_slot: number;
    value: number[];
  };
  // When updating this type, keep it consistent with the HealthPill interface
  // in tf_graph_common/lib/scene/scene.ts.
  export type HealthPillDatum = Datum & HealthPill;
  // A health pill response is a mapping from node name to a list of health pill
  // data entries.
  export interface HealthPillsResponse { [key: string]: HealthPillDatum[]; };

  export var TYPES = [
    'scalar', 'histogram', 'compressedHistogram', 'graph', 'image', 'audio',
    'runMetadata', 'text'
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
    constructor(router: Router, requestManager?: RequestManager) {
      this.router = router;
      this.requestManager = requestManager || new RequestManager();
    }

    /**
     * Returns a promise for requesting the logdir string.
     */
    public logdir(): Promise<LogdirResponse> {
      return this.requestManager.request(this.router.logdir());
    }

    /**
     * Returns a listing of all the available data in the TensorBoard backend.
     */
    public runs(): Promise<RunsResponse> {
      return this.requestManager.request(this.router.runs());
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for scalar data.
     */
    public scalarRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'scalars'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for histogram data.
     */
    public histogramRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'histograms'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for image data.
     */
    public imageRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'images'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for audio data.
     */
    public audioRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'audio'));
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for compressedHistogram
     * data.
     */
    public compressedHistogramRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'compressedHistograms'));
    }

    /**
     * Return a promise showing list of runs that contain graphs.
     */
    public graphRuns(): Promise<string[]> {
      return this.runs().then(
          (x) => { return _.keys(x).filter((k) => x[k].graph); });
    }

    /**
     * Return a promise showing the Run-to-Tag mapping for run_metadata objects.
     */
    public runMetadataRuns(): Promise<RunToTag> {
      return this.runs().then((x) => _.mapValues(x, 'run_metadata'));
    }


    /**
     * Returns a promise showing the Run-to-Tag mapping for text data.
     */
    public textRuns(): Promise<RunToTag> {
      return this.requestManager.request(this.router.textRuns());
    }


    /**
     * Returns a promise containing TextDatums for given run and tag.
     */
    public text(tag: string, run: string): Promise<TextDatum[]> {
      let url = this.router.text(tag, run);
      // tslint:disable-next-line:no-any it's convenient and harmless here
      return this.requestManager.request(url).then(map(function(x: any) {
        x.wall_time = timeToDate(x.wall_time);
        return x;
      }));
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
     * Returns a promise for requesting the health pills for a list of nodes.
     */
    public healthPills(nodeNames: string[], step?: number):
        Promise<HealthPillsResponse> {
      let postData = {'node_names': JSON.stringify(nodeNames)};
      if (step !== undefined) {
        // The user requested health pills for a specific step. This request
        // might be slow since the backend reads events sequentially from disk.
        postData['step'] = step;
      }
      return this.requestManager.request(this.router.healthPills(), postData);
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
        // Get the minimum and maximum values across all histograms so that the
        // visualization is aligned for all timesteps.
        let min = d3.min(histos, d => d.min);
        let max = d3.max(histos, d => d.max);

        return histos.map(function(histo, i) {
          return {
            wall_time: histo.wall_time,
            step: histo.step,
            bins: convertBins(histo, min, max)
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
     * Return a promise containing AudioDatums for given run and tag.
     */
    public audio(tag: string, run: string): Promise<Array<AudioDatum>> {
      let url = this.router.audio(tag, run);
      let p: Promise<AudioMetadata[]>;
      p = this.requestManager.request(url);
      return p.then(map(this.createAudio.bind(this)));
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
        url: this.router.individualImage(x.query, x.wall_time),
      };
    }

    private createAudio(x: AudioMetadata): Audio&Datum {
      return {
        content_type: x.content_type,
        wall_time: timeToDate(x.wall_time),
        step: x.step,
        url: this.router.individualAudio(x.query),
      };
    }
  }

  /** Given a RunToTag, return sorted array of all runs */
  export function getRuns(r: RunToTag): string[] {
    return _.keys(r).sort(VZ.Sorting.compareTagNames);
  }

  /** Given a RunToTag, return array of all tags (sorted + dedup'd) */
  export function getTags(r: RunToTag): string[] {
    return _.union.apply(null, _.values(r)).sort(VZ.Sorting.compareTagNames);
  }

  /**
   * Given a RunToTag and an array of runs, return every tag that appears for
   * at least one run.
   * Sorted, deduplicated.
   */
  export function filterTags(r: RunToTag, runs: string[]): string[] {
    var result = [];
    runs.forEach((x) => result = result.concat(r[x]));
    return _.uniq(result).sort(VZ.Sorting.compareTagNames);
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
   * makes things quite a bit easier. The bins are also converted to have an
   * uniform width, what makes the visualization easier to understand.
   *
   * @param histogram A histogram from tensorboard backend.
   * @param min The leftmost edge. The binning will start on it.
   * @param max The rightmost edge. The binning will end on it.
   * @param numBins The number of bins of the converted data. The default of 30
   * is a sensible default, using more starts to get artifacts because the event
   * data is stored in buckets, and you start being able to see the aliased
   * borders between each bucket.
   * @return A histogram bin. Each bin has an x (left edge), a dx (width),
   *     and a y (count).
   *
   * If given rightedges are inclusive, then these left edges (x) are exclusive.
   */
  export function convertBins(
      histogram: Histogram, min: number, max: number, numBins = 30) {
    if (histogram.bucketRightEdges.length !== histogram.bucketCounts.length) {
      throw(new Error('Edges and counts are of different lengths.'));
    }

    if (max === min) {
      // Create bins even if all the data has a single value.
      max = min * 1.1 + 1;
      min = min / 1.1 - 1;
    }
    let binWidth = (max - min) / numBins;
    let bucketLeft = min;  // Use the min as the starting point for the bins.
    let bucketPos = 0;
    return d3.range(min, max, binWidth).map(function(binLeft) {
      let binRight = binLeft + binWidth;

      // Take the count of each existing bucket, multiply it by the proportion
      // of overlap with the new bin, then sum and store as the count for the
      // new bin. If no overlap, will add to zero, if 100% overlap, will include
      // the full count into new bin.
      let binY = 0;
      while (bucketPos < histogram.bucketRightEdges.length) {
        // Clip the right edge because right-most edge can be infinite-sized.
        let bucketRight = Math.min(max, histogram.bucketRightEdges[bucketPos]);

        let intersect =
            Math.min(bucketRight, binRight) - Math.max(bucketLeft, binLeft);
        let count = (intersect / (bucketRight - bucketLeft)) *
            histogram.bucketCounts[bucketPos];

        binY += intersect > 0 ? count : 0;

        // If bucketRight is bigger than binRight, than this bin is finished and
        // there is data for the next bin, so don't increment bucketPos.
        if (bucketRight > binRight) {
          break;
        }
        bucketLeft = Math.max(min, bucketRight);
        bucketPos++;
      };

      return {x: binLeft, dx: binWidth, y: binY};
    });
  }

  /**
   * The following interfaces (TupleData, HistogramTuple,
   * CompressedHistogramTuple, ImageMetadata, and AudioMetadata) describe how
   * the data is sent over from the backend.
   */
  type TupleData<T> = [number, number, T];  // wall_time, step

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
  interface AudioMetadata {
    content_type: string;
    wall_time: number;
    step: number;
    query: string;
  }
}
