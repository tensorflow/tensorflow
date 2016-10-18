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

/** Duration in ms for showing warning messages to the user */
const WARNING_DURATION_MS = 5000;

/**
 * Animation duration for the user message which should be +20ms more than the
 * `transition` css property in `.notify-msg` in `vz-projector.html`.
 */
const MSG_ANIMATION_DURATION_MSEC = 300 + 20;

let dom: HTMLElement = null;
let msgId = 0;
let numActiveMessages = 0;

export function setDomContainer(domElement: HTMLElement) {
  dom = domElement;
}

/**
 * Updates the user message with the provided id.
 *
 * @param msg The message shown to the user. If null, the message is removed.
 * @param id The id of an existing message. If no id is provided, a unique id
 *     is assigned.
 * @return The id of the message.
 */
export function setModalMessage(msg: string, id: string = null): string {
  if (dom == null) {
    console.warn('Can\'t show modal message before the dom is initialized');
    return;
  }
  if (id == null) {
    id = (msgId++).toString();
  }
  let dialog = dom.querySelector('#wrapper-notify-msg') as any;
  let msgsContainer = dom.querySelector('#notify-msgs') as HTMLElement;
  let divId = `notify-msg-${id}`;
  let msgDiv = d3.select(dom.querySelector('#' + divId));
  let exists = msgDiv.size() > 0;
  if (!exists) {
    msgDiv = d3.select(msgsContainer).insert('div', ':first-child')
      .attr('class', 'notify-msg')
      .attr('id', divId);
    numActiveMessages++;
  }
  if (msg == null) {
    numActiveMessages--;
    if (numActiveMessages === 0) {
      dialog.close();
    }
    msgDiv.style('opacity', 0);
    msgDiv.style('height', 0);
    setTimeout(() => msgDiv.remove(), MSG_ANIMATION_DURATION_MSEC);
  } else {
    msgDiv.text(msg);
    dialog.open();
  }
  return id;
}

/**
 * Shows a warning message to the user for a certain amount of time.
 */
export function setWarningMessage(msg: string): void {
  let warningMsg = dom.querySelector('#warning-msg') as HTMLElement;
  let warningDiv = d3.select(warningMsg);
  warningDiv.style('display', 'block').text('Warning: ' + msg);

  // Hide the warning message after a certain timeout.
  setTimeout(() => {
    warningDiv.style('display', 'none');
  }, WARNING_DURATION_MS);
}