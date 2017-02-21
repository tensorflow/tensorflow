/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================================*/
import {attachChartGroup} from './vz-data-summary';
/**
 * Function which tries to attach a <g> element to the demo SVG, and waits
 * until the SVG element is created.
 */
function runDemoCode() {
  let element = d3.select('#chartGroupExample').node() as HTMLElement;
  if (element !== null) {
    let data = [200, 200, 200, 200, 200, 100];

    attachChartGroup(data, 300, element);
  } else {
    scheduleFunc(runDemoCode);  // Make code tail-recursive.
  }
}

/**
 * Function which is used to run a function in the future.
 * @param callback - The function to be called after the timeout.
 */
function scheduleFunc(callback) {
  setTimeout(callback, 200);
}

runDemoCode();
