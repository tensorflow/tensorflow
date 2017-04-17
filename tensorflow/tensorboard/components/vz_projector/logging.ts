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
const WARNING_DURATION_MS = 10000;

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
 * @param title The title of the notification.
 * @param isErrorMsg If true, the message is error and the dialog will have a
 *                   close button.
 * @return The id of the message.
 */
export function setModalMessage(
    msg: string, id: string = null, title = null, isErrorMsg = false): string {
  if (dom == null) {
    console.warn('Can\'t show modal message before the dom is initialized');
    return;
  }
  if (id == null) {
    id = (msgId++).toString();
  }
  let dialog = dom.querySelector('#notification-dialog') as any;
  dialog.querySelector('.close-button').style.display =
      isErrorMsg ? null : 'none';
  let spinner = dialog.querySelector('.progress');
  spinner.style.display = isErrorMsg ? 'none' : null;
  spinner.active = isErrorMsg ? null : true;
  dialog.querySelector('#notification-title').innerHTML = title;
  let msgsContainer = dialog.querySelector('#notify-msgs') as HTMLElement;
  if (isErrorMsg) {
    d3.select(msgsContainer).html('');
  } else {
    d3.select(msgsContainer).selectAll('.error').remove();
  }
  let divId = `notify-msg-${id}`;
  let msgDiv = d3.select(dialog.querySelector('#' + divId));
  let exists = msgDiv.size() > 0;
  if (!exists) {
    msgDiv = d3.select(msgsContainer)
                 .insert('div', ':first-child')
                 .attr('class', 'notify-msg')
                 .classed('error', isErrorMsg)
                 .attr('id', divId);
    if (!isErrorMsg) {
      numActiveMessages++;
    } else {
      numActiveMessages = 0;
    }
  }
  if (msg == null) {
    numActiveMessages--;
    if (numActiveMessages === 0) {
      dialog.close();
    }
    msgDiv.remove();
  } else {
    msgDiv.text(msg);
    dialog.open();
  }
  return id;
}

export function setErrorMessage(errMsg: string, task?: string) {
  setModalMessage(errMsg, null, 'Error ' + (task != null ? task : ''), true);
}

/**
 * Shows a warning message to the user for a certain amount of time.
 */
export function setWarningMessage(msg: string): void {
  let toast = dom.querySelector('#toast') as any;
  toast.text = msg;
  toast.duration = WARNING_DURATION_MS;
  toast.open();
}
