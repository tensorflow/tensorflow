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

/* tslint:disable:no-namespace */
module TF.Globals {

  // The names of TensorBoard tabs.
  export var TABS = [
    'scalars', 'images', 'audio', 'graphs', 'distributions', 'histograms',
    'embeddings', 'text'
  ];

  // If true, TensorBoard stores its hash in the URI state.
  // If false, tab switching in TensorBoard will not update location hash,
  // because hash updates interfere with wct_tests.
  export var USE_HASH = false;

  // If USE_HASH is false, FAKE_HASH holds the hash contents.
  export var FAKE_HASH = '';
}
