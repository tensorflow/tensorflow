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

/** Delay for running async tasks, in milliseconds. */
const ASYNC_DELAY = 15;

/**
 * Runs an expensive task asynchronously with some delay
 * so that it doesn't block the UI thread immediately.
 */
export function runAsyncTask<T>(message: string, task: () => T): Promise<T> {
  updateMessage(message);
  return new Promise<T>((resolve, reject) => {
    d3.timer(() => {
      try {
        let result = task();
        // Clearing the old message.
        updateMessage();
        resolve(result);
      } catch (ex) {
        updateMessage('Error: ' + ex.message);
        reject(ex);
      }
      return true;
    }, ASYNC_DELAY);
  });
}

/**
 * Updates the user message at the top of the page. If the provided msg is
 * null, the message box is hidden from the user.
 */
export function updateMessage(msg?: string): void {
  if (msg == null) {
    d3.select('#notify-msg').style('display', 'none');
  } else {
    d3.select('#notify-msg').style('display', 'block').text(msg);
  }
}
