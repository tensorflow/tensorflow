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

import {DataPoint} from './data';
import * as logging from './logging';
import {Point2D} from './vector';

/**
 * Delay for running expensive tasks, in milliseconds.
 * The duration was empirically found so that it leaves enough time for the
 * browser to update its UI state before starting an expensive UI-blocking task.
 */
const TASK_DELAY_MS = 200;

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

export function range(count: number): number[] {
  const rangeOutput: number[] = [];
  for (let i = 0; i < count; i++) {
    rangeOutput.push(i);
  }
  return rangeOutput;
}

export function classed(
    element: HTMLElement, className: string, enabled: boolean) {
  const classNames = element.className.split(' ');
  if (enabled) {
    if (className in classNames) {
      return;
    } else {
      classNames.push(className);
    }
  } else {
    const index = classNames.indexOf(className);
    if (index === -1) {
      return;
    }
    classNames.splice(index, 1);
  }
  element.className = classNames.join(' ');
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

/** Loads 3 contiguous elements from a packed xyz array into a Vector3. */
export function vector3FromPackedArray(
    a: Float32Array, pointIndex: number): THREE.Vector3 {
  const offset = pointIndex * 3;
  return new THREE.Vector3(a[offset], a[offset + 1], a[offset + 2]);
}

/**
 * Gets the camera-space z coordinates of the nearest and farthest points.
 * Ignores points that are behind the camera.
 */
export function getNearFarPoints(
    worldSpacePoints: Float32Array, cameraPos: THREE.Vector3,
    cameraTarget: THREE.Vector3): [number, number] {
  let shortestDist: number = Infinity;
  let furthestDist: number = 0;
  const camToTarget = new THREE.Vector3().copy(cameraTarget).sub(cameraPos);
  const camPlaneNormal = new THREE.Vector3().copy(camToTarget).normalize();
  const n = worldSpacePoints.length / 3;
  let src = 0;
  let p = new THREE.Vector3();
  let camToPoint = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    p.x = worldSpacePoints[src];
    p.y = worldSpacePoints[src + 1];
    p.z = worldSpacePoints[src + 2];
    src += 3;

    camToPoint.copy(p).sub(cameraPos);
    const dist = camPlaneNormal.dot(camToPoint);
    if (dist < 0) {
      continue;
    }
    furthestDist = (dist > furthestDist) ? dist : furthestDist;
    shortestDist = (dist < shortestDist) ? dist : shortestDist;
  }
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

export type SearchPredicate = (p: DataPoint) => boolean;

export function getSearchPredicate(
    query: string, inRegexMode: boolean, fieldName: string): SearchPredicate {
  let predicate: SearchPredicate;
  if (inRegexMode) {
    let regExp = new RegExp(query, 'i');
    predicate = p => regExp.test(p.metadata[fieldName].toString());
  } else {
    // Doing a case insensitive substring match.
    query = query.toLowerCase();
    predicate = p => {
      let label = p.metadata[fieldName].toString().toLowerCase();
      return label.indexOf(query) >= 0;
    };
  }
  return predicate;
}

/**
 * Runs an expensive task asynchronously with some delay
 * so that it doesn't block the UI thread immediately.
 *
 * @param message The message to display to the user.
 * @param task The expensive task to run.
 * @param msgId Optional. ID of an existing message. If provided, will overwrite
 *     an existing message and won't automatically clear the message when the
 *     task is done.
 * @return The value returned by the task.
 */
export function runAsyncTask<T>(
    message: string, task: () => T, msgId: string = null): Promise<T> {
  let autoClear = (msgId == null);
  msgId = logging.setModalMessage(message, msgId);
  return new Promise<T>((resolve, reject) => {
    setTimeout(() => {
      try {
        let result = task();
        // Clearing the old message.
        if (autoClear) {
          logging.setModalMessage(null, msgId);
        }
        resolve(result);
      } catch (ex) {
        reject(ex);
      }
      return true;
    }, TASK_DELAY_MS);
  });
}


/**
 * Parses the URL for query parameters, e.g. ?foo=1&bar=2 will return
 *   {'foo': '1', 'bar': '2'}.
 * @param url The URL to parse.
 * @return A map of queryParam key to its value.
 */
export function getURLParams(url: string): {[key: string]: string} {
  if (!url) {
    return {};
  }

  let queryString = url.indexOf('?') !== -1 ? url.split('?')[1] : url;
  if (queryString.indexOf('#')) {
    queryString = queryString.split('#')[0];
  }

  const queryEntries = queryString.split('&');
  let queryParams: {[key: string]: string} = {};
  for (let i = 0; i < queryEntries.length; i++) {
    let queryEntryComponents = queryEntries[i].split('=');
    queryParams[queryEntryComponents[0].toLowerCase()] =
        decodeURIComponent(queryEntryComponents[1]);
  }
  return queryParams;
}

/** List of substrings that auto generated tensors have in their name. */
const SUBSTR_GEN_TENSORS = ['/Adagrad'];

/** Returns true if the tensor was automatically generated by TF API calls. */
export function tensorIsGenerated(tensorName: string): boolean {
  for (let i = 0; i < SUBSTR_GEN_TENSORS.length; i++) {
    if (tensorName.indexOf(SUBSTR_GEN_TENSORS[i]) >= 0) {
      return true;
    }
  }
  return false;
}

export function xor(cond1: boolean, cond2: boolean): boolean {
  return (cond1 || cond2) && !(cond1 && cond2);
}

/** Checks to see if the browser supports webgl. */
export function hasWebGLSupport(): boolean {
  try {
    let c = document.createElement('canvas');
    let gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    return gl != null && typeof weblas !== 'undefined';
  } catch (e) {
    return false;
  }
}
