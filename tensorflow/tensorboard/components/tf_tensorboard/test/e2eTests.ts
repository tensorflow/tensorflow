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

import {TABS} from '../../tf-globals/globals';

describe('end-to-end test', () => {
  window.HTMLImports.whenReady(() => {
    let tb = d3.select('tf-tensorboard');
    var tabs = (<any>tb.node()).$.tabs;

    function testTab(tabIndex: number) {
      it(`selecting ${TABS[tabIndex]} tab`, done => {
        // Every dashboard emits a rendered event when it is done rendering.
        tb.on('rendered', () => done());
        tabs.set('selected', tabIndex);
      });
    }
    // Listen for when the default tab has rendered and test other tabs after.
    tb.on('rendered', () => {
      // The default tab already rendered. Test everything else.
      // If a bug happened while rendering the default tab, the test would
      // have failed. Re-selecting the default tab and listening for
      // "rendered" event won't work since the content is not re-stamped.
      let selected = +tabs.get('selected');
      for (let i = 0; i < TABS.length; i++) {
        if (i !== selected) {
          testTab(i);
        }
      }
    });
  });
});
