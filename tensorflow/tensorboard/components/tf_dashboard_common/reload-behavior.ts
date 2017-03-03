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

module TF.Dashboard {
  /**
   * ReloadBehavior: A simple behavior for dashboards where the
   * frontendReload() function should find every child element with a
   * given tag name (e.g. "tf-line-chart" or "tf-image-loader")
   * and call a `reload` method on that child.
   * May later extend it so it has more sophisticated logic, e.g. reloading
   * only tags that are in view.
   */
  export function ReloadBehavior(tagName) {
    return {
      properties: {
        reloadTag: {
          type: String,
          value: tagName,
        },
      },
      frontendReload: function() {
        var elements = this.getElementsByTagName(this.reloadTag);
        Array.prototype.forEach.call(elements, function(x) { x.reload(); });
      },
    };
  }
}
