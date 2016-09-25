/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

import {runAsyncTask} from './async';
import {TSNE} from './bh_tsne';
import * as knn from './knn';
import * as scatterPlot from './scatterPlot';
import {shuffle} from './util';
import * as vector from './vector';

/**
 * A DataSource is our ground truth data. The original parsed data should never
 * be modified, only copied out.
 */
export class DataSource {
  originalDataSet: DataSet;
  spriteImage: HTMLImageElement;
  metadata: DatasetMetadata;

  /** A shallow-copy constructor. */
  makeShallowCopy(): DataSource {
    let copy = new DataSource();
    copy.originalDataSet = this.originalDataSet;
    copy.spriteImage = this.spriteImage;
    copy.metadata = this.metadata;
    return copy;
  }

  /** Returns a new dataset. */
  getDataSet(subset?: number[]): DataSet {
    let pointsSubset = subset ?
        subset.map(i => this.originalDataSet.points[i]) :
        this.originalDataSet.points;
    return new DataSet(pointsSubset);
  }
}

export interface DataPoint extends scatterPlot.DataPoint {
  /** The point in the original space. */
  vector: number[];

  /*
   * Metadata for each point. Each metadata is a set of key/value pairs
   * where the value can be a string or a number.
   */
  metadata: {[key: string]: number | string};

  /** This is where the calculated projections space are cached */
  projections: {[key: string]: number};
}

/** Checks to see if the browser supports webgl. */
function hasWebGLSupport(): boolean {
  try {
    let c = document.createElement('canvas');
    let gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    return gl != null && typeof weblas !== 'undefined';
  } catch (e) {
    return false;
  }
}

const WEBGL_SUPPORT = hasWebGLSupport();
/** Sampling is used when computing expensive operations such as T-SNE. */
export const SAMPLE_SIZE = 10000;
/** Number of dimensions to sample when doing approximate PCA. */
export const PCA_SAMPLE_DIM = 200;
/** Number of pca components to compute. */
const NUM_PCA_COMPONENTS = 10;
/** Reserved metadata attribute used for trace information. */
const TRACE_METADATA_ATTR = '__next__';

/**
 * Dataset contains a DataPoints array that should be treated as immutable. This
 * acts as a working subset of the original data, with cached properties
 * from computationally expensive operations. Because creating a subset
 * requires normalizing and shifting the vector space, we make a copy of the
 * data so we can still always create new subsets based on the original data.
 */
export class DataSet implements scatterPlot.DataSet {
  points: DataPoint[];
  traces: scatterPlot.DataTrace[];

  sampledDataIndices: number[] = [];

  /**
   * This keeps a list of all current projections so you can easily test to see
   * if it's been calculated already.
   */
  projections = d3.set();
  nearest: knn.NearestEntry[][];
  nearestK: number;
  tSNEShouldStop = true;
  dim = [0, 0];
  hasTSNERun: boolean = false;

  private tsne: TSNE;

  /**
   * Creates a new Dataset by copying out data from an array of datapoints.
   * We make a copy because we have to modify the vectors by normalizing them.
   */
  constructor(points: DataPoint[]) {
    // Keep a list of indices seen so we don't compute traces for a given
    // point twice.
    let indicesSeen: boolean[] = [];

    this.points = [];
    points.forEach(dp => {
      this.points.push({
        metadata: dp.metadata,
        dataSourceIndex: dp.dataSourceIndex,
        vector: dp.vector.slice(),
        projectedPoint: [0, 0, 0],
        projections: {}
      });
      indicesSeen.push(false);
    });

    this.sampledDataIndices =
        shuffle(d3.range(this.points.length)).slice(0, SAMPLE_SIZE);
    this.traces = this.computeTraces(points, indicesSeen);

    this.normalize();
    this.dim = [this.points.length, this.points[0].vector.length];
  }

  private computeTraces(points: DataPoint[], indicesSeen: boolean[]) {
    // Compute traces.
    let indexToTrace: {[index: number]: scatterPlot.DataTrace} = {};
    let traces: scatterPlot.DataTrace[] = [];
    for (let i = 0; i < points.length; i++) {
      if (indicesSeen[i]) {
        continue;
      }
      indicesSeen[i] = true;

      // Ignore points without a trace attribute.
      let next = points[i].metadata[TRACE_METADATA_ATTR];
      if (next == null || next === '') {
        continue;
      }
      if (next in indexToTrace) {
        let existingTrace = indexToTrace[+next];
        // Pushing at the beginning of the array.
        existingTrace.pointIndices.unshift(i);
        indexToTrace[i] = existingTrace;
        continue;
      }
      // The current point is pointing to a new/unseen trace.
      let newTrace: scatterPlot.DataTrace = {pointIndices: []};
      indexToTrace[i] = newTrace;
      traces.push(newTrace);
      let currentIndex = i;
      while (points[currentIndex]) {
        newTrace.pointIndices.push(currentIndex);
        let next = points[currentIndex].metadata[TRACE_METADATA_ATTR];
        if (next != null && next !== '') {
          indicesSeen[+next] = true;
          currentIndex = +next;
        } else {
          currentIndex = -1;
        }
      }
    }
    return traces;
  }

  /**
   * Computes the centroid, shifts all points to that centroid,
   * then makes them all unit norm.
   */
  private normalize() {
    // Compute the centroid of all data points.
    let centroid =
        vector.centroid(this.points, () => true, a => a.vector).centroid;
    if (centroid == null) {
      throw Error('centroid should not be null');
    }
    // Shift all points by the centroid and make them unit norm.
    for (let id = 0; id < this.points.length; ++id) {
      let dataPoint = this.points[id];
      dataPoint.vector = vector.sub(dataPoint.vector, centroid);
      vector.unit(dataPoint.vector);
    }
  }

  /** Projects the dataset onto a given vector and caches the result. */
  projectLinear(dir: vector.Vector, label: string) {
    this.projections.add(label);
    this.points.forEach(dataPoint => {
      dataPoint.projections[label] = vector.dot(dataPoint.vector, dir);
    });
  }

  /** Projects the dataset along the top 10 principal components. */
  projectPCA(): Promise<void> {
    if (this.projections.has('pca-0')) {
      return Promise.resolve<void>(null);
    }
    return runAsyncTask('Computing PCA...', () => {
      // Approximate pca vectors by sampling the dimensions.
      let dim = this.points[0].vector.length;
      let vectors = this.points.map(d => d.vector);
      if (dim > PCA_SAMPLE_DIM) {
        vectors = vector.projectRandom(vectors, PCA_SAMPLE_DIM);
      }
      let sigma = numeric.div(
          numeric.dot(numeric.transpose(vectors), vectors), vectors.length);
      let U: any;
      U = numeric.svd(sigma).U;
      let pcaVectors = vectors.map(vector => {
        let newV: number[] = [];
        for (let d = 0; d < NUM_PCA_COMPONENTS; d++) {
          let dot = 0;
          for (let i = 0; i < vector.length; i++) {
            dot += vector[i] * U[i][d];
          }
          newV.push(dot);
        }
        return newV;
      });
      for (let j = 0; j < NUM_PCA_COMPONENTS; j++) {
        let label = 'pca-' + j;
        this.projections.add(label);
        this.points.forEach((d, i) => {
          d.projections[label] = pcaVectors[i][j];
        });
      }
    });
  }

  /** Runs tsne on the data. */
  projectTSNE(
      perplexity: number, learningRate: number, tsneDim: number,
      stepCallback: (iter: number) => void) {
    this.hasTSNERun = true;
    let k = Math.floor(3 * perplexity);
    let opt = {epsilon: learningRate, perplexity: perplexity, dim: tsneDim};
    this.tsne = new TSNE(opt);
    this.tSNEShouldStop = false;
    let iter = 0;

    let step = () => {
      if (this.tSNEShouldStop) {
        stepCallback(null);
        return;
      }
      this.tsne.step();
      let result = this.tsne.getSolution();
      this.sampledDataIndices.forEach((index, i) => {
        let dataPoint = this.points[index];

        dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
        dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
        if (tsneDim === 3) {
          dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
        }
      });
      iter++;
      stepCallback(iter);
      requestAnimationFrame(step);
    };

    // Nearest neighbors calculations.
    let knnComputation: Promise<knn.NearestEntry[][]>;

    if (this.nearest != null && k === this.nearestK) {
      // We found the nearest neighbors before and will reuse them.
      knnComputation = Promise.resolve(this.nearest);
    } else {
      let sampledData = this.sampledDataIndices.map(i => this.points[i]);
      this.nearestK = k;
      knnComputation = WEBGL_SUPPORT ?
          knn.findKNNGPUCosine(sampledData, k, (d => d.vector)) :
          knn.findKNN(
              sampledData, k, (d => d.vector),
              (a, b, limit) => vector.cosDistNorm(a, b));
    }
    knnComputation.then(nearest => {
      this.nearest = nearest;
      runAsyncTask('Initializing T-SNE...', () => {
        this.tsne.initDataDist(this.nearest);
      }).then(step);

    });
  }

  stopTSNE() { this.tSNEShouldStop = true; }
}

export interface DatasetMetadata {
  /**
   * Metadata for an associated image sprite. The sprite should be a matrix
   * of smaller images, filled in a row-by-row order.
   *
   * E.g. the image for the first data point should be in the upper-left
   * corner, and to the right of it should be the image of the second data
   * point.
   */
  image?: {
    /** The file path pointing to the sprite image. */
    sprite_fpath: string;
    /** The dimensions of the image for a single data point. */
    single_image_dim: [number, number];
  };
}
