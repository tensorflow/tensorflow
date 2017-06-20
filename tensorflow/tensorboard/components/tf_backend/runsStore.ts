/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
import {RequestManager} from './requestManager';
import {getRouter} from './router';

let runs: string[] = [];

export type Listener = () => void;
const listeners = new Set<Listener>();

const requestManager = new RequestManager(1 /* simultaneous request */);

/**
 * Register a listener (nullary function) to be called when new runs are
 * available.
 */
export function addListener(listener: Listener): void {
  listeners.add(listener);
}

/**
 * Remove a listener registered with `addListener`.
 */
export function removeListener(listener: Listener): void {
  listeners.delete(listener);
}

/**
 * Asynchronously load or reload the runs data. Listeners will be
 * invoked if this causes the runs data to change.
 *
 * @see addListener
 * @return {Promise<void>} a promise that resolves when the runs have
 * loaded
 */
export function fetchRuns(): Promise<void> {
  const url = getRouter().runs();
  return requestManager.request(url).then(newRuns => {
    if (!_.isEqual(runs, newRuns)) {
      runs = newRuns;
      listeners.forEach(listener => {
        listener();
      });
    }
  });
}

/**
 * Get the current list of runs. If no data is available, this will be
 * an empty array (i.e., there is no distinction between "no runs" and
 * "no runs yet").
 */
export function getRuns(): string[] {
  return runs.slice();
}
