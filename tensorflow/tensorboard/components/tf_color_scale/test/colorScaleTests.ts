/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

let assert = chai.assert;

import {ColorScale} from '../colorScale'

describe('ColorScale', function() {
  let ccs: ColorScale;

  beforeEach(function() {
    ccs = new ColorScale();
  });

  it('Returns consistent colors', function() {
    ccs.domain(['train', 'eval', 'test']);
    let trainColor = ccs.scale('train');
    let trainColor2 = ccs.scale('train');
    assert.equal(trainColor, trainColor2);
  });

  it('Returns consistent colors after new domain', function() {
    ccs.domain(['train', 'eval']);
    let trainColor = ccs.scale('train');
    ccs.domain(['train', 'eval', 'test']);
    let trainColor2 = ccs.scale('train');
    assert.equal(trainColor, trainColor2);
  });

  it('Throws an error if string is not in the domain', function() {
    ccs.domain(['red', 'yellow', 'green']);
    assert.throws(function() {
      ccs.scale('not in domain');
    }, 'String was not in the domain.');
  });
});
