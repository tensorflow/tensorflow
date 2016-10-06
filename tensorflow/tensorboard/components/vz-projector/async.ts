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
const ASYNC_DELAY_MS = 25;

/** Duration in ms for showing warning messages to the user */
const WARNING_DURATION_MS = 5000;

/**
 * Animation duration for the user message which should align with `transition`
 * css property in `.notify-msg` in `vz-projector.html`.
 */
const MSG_ANIMATION_DURATION = 300;


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
export function runAsyncTask<T>(message: string, task: () => T,
    msgId: string = null): Promise<T> {
  let autoClear = (msgId == null);
  msgId = updateMessage(message, msgId);
  return new Promise<T>((resolve, reject) => {
    d3.timer(() => {
      try {
        let result = task();
        // Clearing the old message.
        if (autoClear) {
          updateMessage(null, msgId);
        }
        resolve(result);
      } catch (ex) {
        updateMessage('Error: ' + ex.message);
        reject(ex);
      }
      return true;
    }, ASYNC_DELAY_MS);
  });
}

let msgId = 0;

/**
 * Updates the user message with the provided id.
 *
 * @param msg The message shown to the user. If null, the message is removed.
 * @param id The id of an existing message. If no id is provided, a unique id
 *     is assigned.
 * @return The id of the message.
 */
export function updateMessage(msg: string, id: string = null): string {
  if (id == null) {
    id = (msgId++).toString();
  }
  let divId = `notify-msg-${id}`;
  let msgDiv = d3.select('#' + divId);
  let exists = msgDiv.size() > 0;
  if (!exists) {
    msgDiv = d3.select('#notify-msgs').insert('div', ':first-child')
      .attr('class', 'notify-msg')
      .attr('id', divId);
  }
  if (msg == null) {
    msgDiv.style('opacity', 0);
    setTimeout(() => msgDiv.remove(), MSG_ANIMATION_DURATION);
  } else {
    msgDiv.text(msg);
  }
  return id;
}

/**
 * Shows a warning message to the user for a certain amount of time.
 */
export function updateWarningMessage(msg: string): void {
  let warningDiv = d3.select('#warning-msg');
  warningDiv.style('display', 'block').text('Warning: ' + msg);

  // Hide the warning message after a certain timeout.
  setTimeout(() => {
    warningDiv.style('display', 'none');
  }, WARNING_DURATION_MS);
}
