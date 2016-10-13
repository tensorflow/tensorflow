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

describe('fast tab switch', () => {
  let assert = chai.assert;
  window.HTMLImports.whenReady(() => {
    let tb = d3.select('tf-tensorboard');
    var tabs = (<any>tb.node()).$.tabs;

    // This test will select the events tab. Once the events tab
    // renders, will select the graph tab, and immediately select
    // the images tab wihout waiting for the graph tab to finish
    // rendering. Finally, it finishes when the images tab
    // has rendered and no errors were thrown.
    let eventsTabIndex = TF.Globals.TABS.indexOf('events');
    let imagesTabIndex = TF.Globals.TABS.indexOf('images');
    let graphTabIndex = TF.Globals.TABS.indexOf('graphs');

    // Listen for when the events tab rendered.
    tb.on('rendered', () => {
      it('switching to graph tab and immediately to images', done => {
        // Select the graph tab.
        tabs.set('selected', graphTabIndex);
        // Interrupt graph rendering by immediately selecting the images tab
        // and finish when the images tab has rendered.
        tb.on('rendered', () => done());
        tabs.set('selected', imagesTabIndex);
      });
    });
    // Select the events tab.
    tabs.set('selected', eventsTabIndex);
  });
});
