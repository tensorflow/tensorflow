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

/**
 * A behavior that TensorBoard dashboards must implement. This behavior serves
 * the purpose of an interface.
 *
 * @polymerBehavior
 */
export function DashboardBehavior(dashboardName) {
  return {
    properties: {
      name: {
        type: String,
        value: dashboardName,
        readOnly: true,
      },
    },
    // This method is called when the dashboard reloads, either when the
    // dashboard is first visited, periodically reloaded, or manually reloaded
    // via the user clicking the button. Note that dashboard custom elements
    // that use TF.Dashboard.ReloadBehavior already implement a reload method.
    reload() {
      throw Error(
          'The ' + dashboardName + ' dashboard does not implement reload.');
    },
  };
}
