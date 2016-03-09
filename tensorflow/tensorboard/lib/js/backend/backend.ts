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
  export interface RunEnumeration {
    histograms: string[];
    compressedHistogramTuples: string[];
    scalars: string[];
    images: string[];
    graph: boolean;
  }
  export interface RunsResponse {
    [runName: string]: RunEnumeration;
  }

  export interface Datum {
    wall_time: Date;
    step: number;
  }

  export type ScalarDatum = Datum & Scalar;
  export interface Scalar {
    scalar: number;
  }

  export type HistogramDatum = Datum & Histogram;
  export interface Histogram {
    min: number;
    max: number;
    nItems: number;
    sum: number;
    sumSquares: number;
    bucketRightEdges: number[];
    bucketCounts: number[];
  }

  export type ImageDatum = Datum & Image
  export interface Image {
    width: number;
    height: number;
    url: string;
  }

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
    public baseRoute: string;
    public requestManager: RequestManager;
    public demoMode = false;

    /**
     * Construct a Backend instance.
     * @param baseRoute the route prepended to each dispatched request
     * @param requestManager The RequestManager, overwritable so you may
     * manually clear request queue, etc.
     * @param demoMode Whether to launch regular http requests, or special ones
     * to read from filesystem serialized tensorboard (serialize_tensorboard.py)
     */
    constructor(baseRoute = "/data", requestManager?: RequestManager, demoMode?) {
      if (demoMode) {
        this.demoMode = demoMode;
      }
      if (baseRoute[baseRoute.length - 1] === "/") {
        baseRoute = baseRoute.substr(0, baseRoute.length - 1);
      }
      this.baseRoute = baseRoute;
      this.requestManager = requestManager || new RequestManager();
    }

    /**
     * Returns a listing of all the available data in the TensorBoard backend.
     * This API point may be deprecated in the future, in favor of
     * per-data-type methods.
     */
    public runs(): Promise<RunsResponse> {
      return this.req("runs");
    }

    public scalars(run: string, tag: string): Promise<Array<ScalarDatum>> {
      let p: Promise<TupleData<number>[]>;
      p = this.req("scalars", {run: run, tag: tag});
      return p.then(map(detupler(createScalar)));
    }

    public histograms(run: string, tag: string): Promise<Array<HistogramDatum>> {
      let p: Promise<TupleData<HistogramTuple>[]>;
      p = this.req("histograms", {run: run, tag: tag});
      return p.then(map(detupler(createHistogram)));
    }

    public images(run: string, tag: string): Promise<Array<ImageDatum>> {
      let p: Promise<ImageMetadata[]> = this.req("images", {run: run, tag: tag});
      return p.then(map(this.createImage.bind(this)));
    }

    /**
     * Get compressedHistogram data.
     * Unlike other methods, don't bother reprocessing this data into a nicer
     * format. This is because we will deprecate this route.
     */
    private compressedHistograms(run: string, tag: string): Promise<TupleData<CompressedHistogramTuple>[]> {
      return this.req( "compressedHistogramTuples", {run: run, tag: tag});
    }

    private req(route: string, params?: any): Promise<any> {
      let target = route + UrlPathHelpers.queryEncoder(params);
      if (this.demoMode) {
        target = UrlPathHelpers.clean(target);
      }
      let url = this.baseRoute + "/" + target;
      return this.requestManager.request(url);
    }

    private createImage(x: ImageMetadata): Image & Datum {
      let url = "individualImage?" + x.query;
      if (this.demoMode) {
        url = UrlPathHelpers.clean(url);
      }
      url = this.baseRoute + "/" + url;
      return {
        width: x.width,
        height: x.height,
        wall_time: timeToDate(x.wall_time),
        step: x.step,
        url: url,
      };
    }
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
      let obj = <G & Datum> xform(x[2]);
      // ... patch in the properties of datum
      obj.wall_time = timeToDate(x[0]);
      obj.step = x[1];
      return obj;
    };
  };

  function createScalar(x: number): Scalar {
    return {scalar: x};
  };

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

  // The following interfaces (TupleData, HistogramTuple, CompressedHistogramTuple,
  // and ImageMetadata) describe how the data is sent over from the backend, and thus
  // wall_time, step, value
  type TupleData<T> = [number, number, T];

  // Min, Max, nItems, Sum, Sum_Squares, right edges of buckets, nItems in buckets
  type HistogramTuple = [number, number, number, number, number, number[], number[]];
  type CompressedHistogramTuple = [number, number][];  // percentile, value
  interface ImageMetadata {
    width: number;
    height: number;
    wall_time: number;
    step: number;
    query: string;
  }
}
