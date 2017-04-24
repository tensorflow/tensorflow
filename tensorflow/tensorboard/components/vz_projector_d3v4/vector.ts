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

import * as d3 from 'd3';  // from //third_party/javascript/typings/d3_v4
import {assert} from './util';

/**
 * @fileoverview Useful vector utilities.
 */

export type Vector = Float32Array | number[];
export type Point2D = [number, number];
export type Point3D = [number, number, number];

/** Returns the dot product of two vectors. */
export function dot(a: Vector, b: Vector): number {
  assert(a.length === b.length, 'Vectors a and b must be of same length');
  let result = 0;
  for (let i = 0; i < a.length; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

/** Sums all the elements in the vector */
export function sum(a: Vector): number {
  let result = 0;
  for (let i = 0; i < a.length; ++i) {
    result += a[i];
  }
  return result;
}

/** Returns the sum of two vectors, i.e. a + b */
export function add(a: Vector, b: Vector): Float32Array {
  assert(a.length === b.length, 'Vectors a and b must be of same length');
  let result = new Float32Array(a.length);
  for (let i = 0; i < a.length; ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

/** Subtracts vector b from vector a, i.e. returns a - b */
export function sub(a: Vector, b: Vector): Float32Array {
  assert(a.length === b.length, 'Vectors a and b must be of same length');
  let result = new Float32Array(a.length);
  for (let i = 0; i < a.length; ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

/** Returns the square norm of the vector */
export function norm2(a: Vector): number {
  let result = 0;
  for (let i = 0; i < a.length; ++i) {
    result += a[i] * a[i];
  }
  return result;
}

/** Returns the euclidean distance between two vectors. */
export function dist(a: Vector, b: Vector): number {
  return Math.sqrt(dist2(a, b));
}

/** Returns the square euclidean distance between two vectors. */
export function dist2(a: Vector, b: Vector): number {
  assert(a.length === b.length, 'Vectors a and b must be of same length');
  let result = 0;
  for (let i = 0; i < a.length; ++i) {
    let diff = a[i] - b[i];
    result += diff * diff;
  }
  return result;
}

/** Returns the square euclidean distance between two 2D points. */
export function dist2_2D(a: Vector, b: Vector): number {
  let dX = a[0] - b[0];
  let dY = a[1] - b[1];
  return dX * dX + dY * dY;
}

/** Returns the square euclidean distance between two 3D points. */
export function dist2_3D(a: Vector, b: Vector): number {
  let dX = a[0] - b[0];
  let dY = a[1] - b[1];
  let dZ = a[2] - b[2];
  return dX * dX + dY * dY + dZ * dZ;
}

/** Returns the euclidean distance between 2 3D points. */
export function dist_3D(a: Vector, b: Vector): number {
  return Math.sqrt(dist2_3D(a, b));
}

/**
 * Returns the square euclidean distance between two vectors, with an early
 * exit (returns -1) if the distance is >= to the provided limit.
 */
export function dist2WithLimit(a: Vector, b: Vector, limit: number): number {
  assert(a.length === b.length, 'Vectors a and b must be of same length');
  let result = 0;
  for (let i = 0; i < a.length; ++i) {
    let diff = a[i] - b[i];
    result += diff * diff;
    if (result >= limit) {
      return -1;
    }
  }
  return result;
}

/** Returns the square euclidean distance between two 2D points. */
export function dist22D(a: Point2D, b: Point2D): number {
  let dX = a[0] - b[0];
  let dY = a[1] - b[1];
  return dX * dX + dY * dY;
}

/** Modifies the vector in-place to have unit norm. */
export function unit(a: Vector): void {
  let norm = Math.sqrt(norm2(a));
  assert(norm >= 0, 'Norm of the vector must be > 0');
  for (let i = 0; i < a.length; ++i) {
    a[i] /= norm;
  }
}

/**
 *  Projects the vectors to a lower dimension
 *
 * @param vectors Array of vectors to be projected.
 * @param newDim The resulting dimension of the vectors.
 */
export function projectRandom(vectors: Float32Array[], newDim: number):
    Float32Array[] {
  let dim = vectors[0].length;
  let N = vectors.length;
  let newVectors: Float32Array[] = new Array(N);
  for (let i = 0; i < N; ++i) {
    newVectors[i] = new Float32Array(newDim);
  }
  // Make nDim projections.
  for (let k = 0; k < newDim; ++k) {
    let randomVector = rn(dim);
    for (let i = 0; i < N; ++i) {
      newVectors[i][k] = dot(vectors[i], randomVector);
    }
  }
  return newVectors;
}

/**
 * Projects a vector onto a 2D plane specified by the two direction vectors.
 */
export function project2d(a: Vector, dir1: Vector, dir2: Vector): Point2D {
  return [dot(a, dir1), dot(a, dir2)];
}

/**
 * Computes the centroid of the data points. If the provided data points are not
 * vectors, an accessor function needs to be provided.
 */
export function centroid<T>(dataPoints: T[], accessor?: (a: T) => Vector):
    Vector {
  if (dataPoints.length === 0) {
    return null;
  }
  if (accessor == null) {
    accessor = (a: T) => <any>a;
  }
  assert(dataPoints.length >= 0, '`vectors` must be of length >= 1');
  let centroid = new Float32Array(accessor(dataPoints[0]).length);
  for (let i = 0; i < dataPoints.length; ++i) {
    let dataPoint = dataPoints[i];
    let vector = accessor(dataPoint);
    for (let j = 0; j < centroid.length; ++j) {
      centroid[j] += vector[j];
    }
  }
  for (let j = 0; j < centroid.length; ++j) {
    centroid[j] /= dataPoints.length;
  }
  return centroid;
}

/**
 * Generates a vector of the specified size where each component is drawn from
 * a random (0, 1) gaussian distribution.
 */
export function rn(size: number): Float32Array {
  const normal = d3.randomNormal();
  let result = new Float32Array(size);
  for (let i = 0; i < size; ++i) {
    result[i] = normal();
  }
  return result;
}

/**
 * Returns the cosine distance ([0, 2]) between two vectors
 * that have been normalized to unit norm.
 */
export function cosDistNorm(a: Vector, b: Vector): number {
  return 1 - dot(a, b);
}

/**
 * Returns the cosine distance ([0, 2]) between two vectors.
 */
export function cosDist(a: Vector, b: Vector): number {
  return 1 - cosSim(a, b);
}

/** Returns the cosine similarity ([-1, 1]) between two vectors. */
export function cosSim(a: Vector, b: Vector): number {
  return dot(a, b) / Math.sqrt(norm2(a) * norm2(b));
}

/**
 * Converts list of vectors (matrix) into a 1-dimensional
 * typed array with row-first order.
 */
export function toTypedArray<T>(
    dataPoints: T[], accessor: (dataPoint: T) => Float32Array): Float32Array {
  let N = dataPoints.length;
  let dim = accessor(dataPoints[0]).length;
  let result = new Float32Array(N * dim);
  for (let i = 0; i < N; ++i) {
    let vector = accessor(dataPoints[i]);
    for (let d = 0; d < dim; ++d) {
      result[i * dim + d] = vector[d];
    }
  }
  return result;
}

/**
 * Transposes an RxC matrix represented as a flat typed array
 * into a CxR matrix, again represented as a flat typed array.
 */
export function transposeTypedArray(
    r: number, c: number, typedArray: Float32Array) {
  let result = new Float32Array(r * c);
  for (let i = 0; i < r; ++i) {
    for (let j = 0; j < c; ++j) {
      result[j * r + i] = typedArray[i * c + j];
    }
  }
  return result;
}
