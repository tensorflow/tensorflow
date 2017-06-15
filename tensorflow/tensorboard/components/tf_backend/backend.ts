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

import {compareTagNames} from '../vz-sorting/sorting';
import {RequestManager} from './requestManager';
import {getRouter} from './router';
import {demoify, queryEncoder} from './urlPathHelpers';

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

export type RunToTag = {
  [run: string]: string[];
};

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
  device_name: string;
  node_name: string;
  output_slot: number;
  dtype: string;
  shape: number[];
  value: number[];
}

// When updating this type, keep it consistent with the HealthPill interface
// in tf_graph_common/lib/scene/scene.ts.
export type HealthPillDatum = Datum & HealthPill;
// A health pill response is a mapping from node name to a list of health pill
// data entries.
export interface HealthPillsResponse { [key: string]: HealthPillDatum[]; }

// An object that encapsulates an alert issued by the debugger. This alert is
// sent by debugging libraries after bad values (NaN, +/- Inf) are encountered.
export interface DebuggerNumericsAlertReport {
  device_name: string;
  tensor_name: string;
  first_timestamp: number;
  nan_event_count: number;
  neg_inf_event_count: number;
  pos_inf_event_count: number;
}
// A DebuggerNumericsAlertReportResponse contains alerts issued by the debugger
// in ascending order of timestamp. This helps the user identify for instance
// when bad values first appeared in the model.
export type DebuggerNumericsAlertReportResponse = DebuggerNumericsAlertReport[];

export const TYPES = [
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
  public requestManager: RequestManager;

  /**
   * Construct a Backend instance.
   * @param requestManager The RequestManager, overwritable so you may
   * manually clear request queue, etc. Defaults to a new RequestManager.
   */
  constructor(requestManager?: RequestManager) {
    this.requestManager = requestManager || new RequestManager();
  }

  /**
   * Returns a promise for requesting the logdir string.
   */
  public logdir(): Promise<LogdirResponse> {
    return this.requestManager.request(getRouter().logdir());
  }

  /**
   * Returns a listing of all the available data in the TensorBoard backend.
   */
  public runs(): Promise<RunsResponse> {
    return this.requestManager.request(getRouter().runs());
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for scalar data.
   */
  public scalarTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('scalars', '/tags'));
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for histogram data.
   */
  public histogramTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('histograms', '/tags'));
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for image data.
   */
  public imageTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('images', '/tags'));
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for audio data.
   */
  public audioTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('audio', '/tags'));
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for compressedHistogram
   * data.
   */
  public compressedHistogramTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('distributions', '/tags'));
  }

  /**
   * Returns a promise showing the Run-to-Tag mapping for profile data.
   */
  public profileTags(): Promise<RunToTag> {
    let url = getRouter().pluginRoute('profile', '/tags');
    if (getRouter().isDemoMode()) {
      url += '.json';
    }
    return this.requestManager.request(url);
  }

  /**
   * Return a promise showing list of runs that contain graphs.
   */
  public graphRuns(): Promise<string[]> {
    return this.requestManager.request(
        getRouter().pluginRoute('graphs', '/runs'));
  }

  /**
   * Return a promise showing the Run-to-Tag mapping for run_metadata objects.
   */
  public runMetadataTags(): Promise<RunToTag> {
    return this.requestManager.request(
        getRouter().pluginRoute('graphs', '/run_metadata_tags'));
  }


  /**
   * Returns a promise showing the Run-to-Tag mapping for text data.
   */
  public textRuns(): Promise<RunToTag> {
    return this.requestManager.request(getRouter().textRuns());
  }


  /**
   * Returns a promise containing TextDatums for given run and tag.
   */
  public text(tag: string, run: string): Promise<TextDatum[]> {
    const url = getRouter().text(tag, run);
    // tslint:disable-next-line:no-any it's convenient and harmless here
    return this.requestManager.request(url).then(map((x: any) => {
      x.wall_time = timeToDate(x.wall_time);
      return x;
    }));
  }

  /**
   * Return a URL to fetch a graph (cf. method 'graph').
   */
  public graphUrl(run: string, limitAttrSize?: number, largeAttrsKey?: string):
      string {
    const demoMode = getRouter().isDemoMode();
    const base = getRouter().pluginRoute('graphs', '/graph');
    const optional = (p) => (p != null && !demoMode || undefined) && p;
    const parameters = {
      'run': run,
      'limit_attr_size': optional(limitAttrSize),
      'large_attrs_key': optional(largeAttrsKey),
    };
    const extension = demoMode ? '.pbtxt' : '';
    return base + queryEncoder(parameters) + extension;
  }

  public graph(run: string, limitAttrSize?: number, largeAttrsKey?: string):
      Promise<string> {
    const url = this.graphUrl(run, limitAttrSize, largeAttrsKey);
    return this.requestManager.request(url);
  }

  /**
   * Return a promise containing ScalarDatums for given run and tag.
   */
  public scalar(tag: string, run: string): Promise<Array<ScalarDatum>> {
    let p: Promise<TupleData<number>[]>;
    const url = getRouter().pluginRunTagRoute('scalars', '/scalars')(tag, run);
    p = this.requestManager.request(url);
    return p.then(map(detupler(createScalar)));
  }

  /**
   * Returns a promise for requesting the health pills for a list of nodes. This
   * route is used by the debugger plugin.
   */
  public healthPills(nodeNames: string[], step?: number):
      Promise<HealthPillsResponse> {
    const postData = {
      'node_names': JSON.stringify(nodeNames),

      // Events files with debugger data fall under this special run.
      'run': '__debugger_data__',
    };
    if (step !== undefined) {
      // The user requested health pills for a specific step. This request
      // might be slow since the backend reads events sequentially from disk.
      postData['step'] = step;
    }
    return this.requestManager.request(getRouter().healthPills(), postData);
  }

  /**
   * Returns a promise for alerts for bad values (detected by the debugger).
   * This route is used by the debugger plugin.
   */
  public debuggerNumericsAlerts():
      Promise<DebuggerNumericsAlertReportResponse> {
    return this.requestManager.request(
        getRouter().pluginRoute('debugger', '/numerics_alert_report'));
  }

  /**
   * Return a promise containing HistogramDatums for given run and tag.
   */
  public histogram(tag: string, run: string):
      Promise<Array<HistogramSeriesDatum>> {
    let p: Promise<TupleData<HistogramTuple>[]>;
    const url =
        getRouter().pluginRunTagRoute('histograms', '/histograms')(tag, run);
    p = this.requestManager.request(url);
    return p.then(map(detupler(createHistogram))).then(function(histos) {
      // Get the minimum and maximum values across all histograms so that the
      // visualization is aligned for all timesteps.
      const min = d3.min(histos, d => d.min);
      const max = d3.max(histos, d => d.max);

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
    const url = (getRouter().pluginRunTagRoute('images', '/images')(tag, run));
    let p: Promise<ImageMetadata[]>;
    p = this.requestManager.request(url);
    return p.then(map(this.createImage.bind(this)));
  }

  /**
   * Return a promise containing AudioDatums for given run and tag.
   */
  public audio(tag: string, run: string): Promise<Array<AudioDatum>> {
    const url = (getRouter().pluginRunTagRoute('audio', '/audio')(tag, run));
    let p: Promise<AudioMetadata[]>;
    p = this.requestManager.request(url);
    return p.then(map(this.createAudio.bind(this)));
  }

  /**
   * Returns a promise containing profile data for given run and tag.
   */
  public profile(tag: string, run: string): Promise<string> {
    let url = (getRouter().pluginRunTagRoute('profile', '/data')(tag, run));
    if (getRouter().isDemoMode()) {
      url += '.json';
    }
    return this.requestManager.request(url);
  }

  /**
   * Returns the url for the RunMetadata for the given run/tag.
   */
  public runMetadataUrl(tag: string, run: string): string {
    return getRouter().pluginRunTagRoute('graphs', '/run_metadata')(tag, run);
  }

  /**
   * Returns a promise to load the string RunMetadata for given run/tag.
   */
  public runMetadata(tag: string, run: string): Promise<string> {
    const url = this.runMetadataUrl(tag, run);
    return this.requestManager.request(url);
  }

  /**
   * Get compressedHistogram data.
   * Unlike other methods, don't bother reprocessing this data into a nicer
   * format. This is because we will deprecate this route.
   */
  private compressedHistogram(tag: string, run: string):
      Promise<Array<Datum&CompressedHistogramTuple>> {
    const url = (getRouter().pluginRunTagRoute(
        'distributions', '/distributions')(tag, run));
    let p: Promise<TupleData<CompressedHistogramTuple>[]>;
    p = this.requestManager.request(url);
    return p.then(map(detupler((x) => x)));
  }

  private createImage(x: ImageMetadata): Image&Datum {
    const pluginRoute = getRouter().pluginRoute('images', '/individualImage');

    let query = x.query;
    if (pluginRoute.indexOf('?') > -1) {
      // The route already has GET parameters. Append our parameters to them.
      query = '&' + query;
    } else {
      // The route lacks GET parameters. We append them.
      query = '?' + query;
    }

    if (getRouter().isDemoMode()) {
      query = demoify(query);
    }

    let individualImageUrl = pluginRoute + query;
    // Include wall_time just to disambiguate the URL and force the browser
    // to reload the image when the URL changes. The backend doesn't care
    // about the value.
    individualImageUrl +=
        getRouter().isDemoMode() ? '.png' : '&ts=' + x.wall_time;

    return {
      width: x.width,
      height: x.height,
      wall_time: timeToDate(x.wall_time),
      step: x.step,
      url: individualImageUrl,
    };
  }

  private createAudio(x: AudioMetadata): Audio&Datum {
    const pluginRoute = getRouter().pluginRoute('audio', '/individualAudio');

    let query = x.query;
    if (pluginRoute.indexOf('?') > -1) {
      // The route already has GET parameters. Append our parameters to them.
      query = '&' + query;
    } else {
      // The route lacks GET parameters. We append them.
      query = '?' + query;
    }

    if (getRouter().isDemoMode()) {
      query = demoify(query);
    }

    let individualAudioUrl = pluginRoute + query;
    // Include wall_time just to disambiguate the URL and force the browser
    // to reload the audio when the URL changes. The backend doesn't care
    // about the value.
    individualAudioUrl +=
        getRouter().isDemoMode() ? '.wav' : '&ts=' + x.wall_time;

    return {
      content_type: x.content_type,
      wall_time: timeToDate(x.wall_time),
      step: x.step,
      url: individualAudioUrl,
    };
  }
}

/** Given a RunToTag, return sorted array of all runs */
export function getRuns(r: RunToTag): string[] {
  return _.keys(r).sort(compareTagNames);
}

/** Given a RunToTag, return array of all tags (sorted + dedup'd) */
export function getTags(r: RunToTag): string[] {
  return _.union.apply(null, _.values(r)).sort(compareTagNames);
}

/**
 * Given a RunToTag and an array of runs, return every tag that appears for
 * at least one run.
 * Sorted, deduplicated.
 */
export function filterTags(r: RunToTag, runs: string[]): string[] {
  let result = [];
  runs.forEach((x) => result = result.concat(r[x]));
  return _.uniq(result).sort(compareTagNames);
}

function timeToDate(x: number): Date {
  return new Date(x * 1000);
};

/**  Just a curryable map to make things cute and tidy. */
function map<T, U>(f: (x: T) => U): (arr: T[]) => U[] {
  return function(arr: T[]): U[] {
    return arr.map(f);
  };
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

function createScalar(x: number): Scalar {
  return {scalar: x};
}

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
}

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
  const binWidth = (max - min) / numBins;
  let bucketLeft = min;  // Use the min as the starting point for the bins.
  let bucketPos = 0;
  return d3.range(min, max, binWidth).map((binLeft) => {
    const binRight = binLeft + binWidth;

    // Take the count of each existing bucket, multiply it by the proportion
    // of overlap with the new bin, then sum and store as the count for the
    // new bin. If no overlap, will add to zero, if 100% overlap, will include
    // the full count into new bin.
    let binY = 0;
    while (bucketPos < histogram.bucketRightEdges.length) {
      // Clip the right edge because right-most edge can be infinite-sized.
      const bucketRight = Math.min(max, histogram.bucketRightEdges[bucketPos]);

      const intersect =
          Math.min(bucketRight, binRight) - Math.max(bucketLeft, binLeft);
      const count = (intersect / (bucketRight - bucketLeft)) *
          histogram.bucketCounts[bucketPos];

      binY += intersect > 0 ? count : 0;

      // If bucketRight is bigger than binRight, than this bin is finished and
      // there is data for the next bin, so don't increment bucketPos.
      if (bucketRight > binRight) {
        break;
      }
      bucketLeft = Math.max(min, bucketRight);
      bucketPos++;
    }

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
