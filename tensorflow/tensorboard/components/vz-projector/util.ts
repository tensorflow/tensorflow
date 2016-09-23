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

import {DataSet} from './scatterPlot';
import {Point2D} from './vector';

/** Shuffles the array in-place in O(n) time using Fisher-Yates algorithm. */
export function shuffle<T>(array: T[]): T[] {
  let m = array.length;
  let t: T;
  let i: number;

  // While there remain elements to shuffle.
  while (m) {
    // Pick a remaining element
    i = Math.floor(Math.random() * m--);
    // And swap it with the current element.
    t = array[m];
    array[m] = array[i];
    array[i] = t;
  }
  return array;
}

/** Retrieves a projected point from the data set as a THREE.js vector */
export function getProjectedPointFromIndex(
    dataSet: DataSet, i: number): THREE.Vector3 {
  return new THREE.Vector3(
      dataSet.points[i].projectedPoint[0], dataSet.points[i].projectedPoint[1],
      dataSet.points[i].projectedPoint[2]);
}

/** Projects a 3d point into screen space */
export function vector3DToScreenCoords(
    cam: THREE.Camera, w: number, h: number, v: THREE.Vector3): Point2D {
  let dpr = window.devicePixelRatio;
  let pv = new THREE.Vector3().copy(v).project(cam);

  // The screen-space origin is at the middle of the screen, with +y up.
  let coords: Point2D =
      [((pv.x + 1) / 2 * w) * dpr, -((pv.y - 1) / 2 * h) * dpr];
  return coords;
}

/**
 * Gets the camera-space z coordinates of the nearest and farthest points.
 * Ignores points that are behind the camera.
 */
export function getNearFarPoints(
    dataSet: DataSet, cameraPos: THREE.Vector3,
    cameraTarget: THREE.Vector3): [number, number] {
  let shortestDist: number = Infinity;
  let furthestDist: number = 0;
  let camToTarget = new THREE.Vector3().copy(cameraTarget).sub(cameraPos);
  for (let i = 0; i < dataSet.points.length; i++) {
    let point = getProjectedPointFromIndex(dataSet, i);
    let camToPoint = new THREE.Vector3().copy(point).sub(cameraPos);
    if (camToTarget.dot(camToPoint) < 0) {
      continue;
    }
    let distToCam = cameraPos.distanceToSquared(point);
    furthestDist = Math.max(furthestDist, distToCam);
    shortestDist = Math.min(shortestDist, distToCam);
  }
  furthestDist = Math.sqrt(furthestDist);
  shortestDist = Math.sqrt(shortestDist);
  return [shortestDist, furthestDist];
}

/**
 * Generate a texture for the points/images and sets some initial params
 */
export function createTexture(image: HTMLImageElement|
                              HTMLCanvasElement): THREE.Texture {
  let tex = new THREE.Texture(image);
  tex.needsUpdate = true;
  // Used if the texture isn't a power of 2.
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  return tex;
}

/**
 * Assert that the condition is satisfied; if not, log user-specified message
 * to the console.
 */
export function assert(condition: boolean, message?: string) {
  if (!condition) {
    message = message || 'Assertion failed';
    throw new Error(message);
  }
}
