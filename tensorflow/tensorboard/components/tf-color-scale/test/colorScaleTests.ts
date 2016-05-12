/* Copyright 2015 Google Inc. All Rights Reserved.

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

module TF {
  let assert = chai.assert;

  describe('ColorScale', function() {
    let ccs: ColorScale;

    beforeEach(function() { ccs = new ColorScale(); });

    it('No collisions with train, eval and test', function() {
      ccs.domain(['train']);
      let trainColor = ccs.scale('train');
      ccs.domain(['eval']);
      let evalColor = ccs.scale('eval');
      ccs.domain(['test']);
      let testColor = ccs.scale('test');
      assert.notEqual(trainColor, evalColor, testColor);
    });

    it('Returns consistent colors, given no hash collisions', function() {
      // These three colors don't have hash collisions
      ccs.domain(['red', 'yellow']);
      let firstRedColor = ccs.scale('red');
      ccs.domain(['red', 'yellow', 'blue']);
      let secondRedColor = ccs.scale('red');
      assert.deepEqual(firstRedColor, secondRedColor);
    });

    it('A 2-color scale returns the first and last colors of the palette',
       function() {
         let twoColorScale = new ColorScale(2, TF.palettes.googleStandard);
         // No hash collisions with these.
         twoColorScale.domain(['red', 'blue']);
         assert.deepEqual(
             twoColorScale.scale('blue'), TF.palettes.googleStandard[0]);
         assert.deepEqual(
             twoColorScale.scale('red'),
             TF.palettes.googleStandard[TF.palettes.googleStandard.length - 1]);
       });

    // This is testing that when we reset the domain with new colors, the old
    // domain doesn't influence the new color choices. Basically testing that we
    // get a fresh slate if we have a new domain. Basically testing that all the
    // internal bins are reset etc. and we aren't finding collisions with
    // previous colors.
    it('Colors don\'t nudge away from colors from an old domain.', function() {
      // at 12 breaks, 'orange' and 'blue' collide.
      ccs.domain(['red', 'blue']);
      let firstBlue = ccs.scale('blue');
      ccs.domain(['red', 'orange']);
      let firstOrange = ccs.scale('orange');
      assert.deepEqual(firstBlue, firstOrange);
    });

    it('Nudges all colors, given only one base color', function() {
      let ccsWithOneColor = new ColorScale(1);
      ccsWithOneColor.domain(['one', 'two', 'three']);
      assert.notEqual(
          ccsWithOneColor.scale('one'), ccsWithOneColor.scale('two'));
      assert.notEqual(
          ccsWithOneColor.scale('two'), ccsWithOneColor.scale('three'));
      assert.notEqual(
          ccsWithOneColor.scale('one'), ccsWithOneColor.scale('three'));
    });

    it('Nudges a color if it has a hash collision', function() {
      // at 12 breaks, 'orange' and 'blue' collide.
      ccs.domain(['red', 'blue']);
      let firstBlue = ccs.scale('blue');
      ccs.domain(['red', 'orange']);
      let firstOrange = ccs.scale('orange');
      ccs.domain(['red', 'blue', 'orange']);
      let secondBlue = ccs.scale('blue');
      let secondOrange = ccs.scale('orange');
      assert.deepEqual(firstBlue, secondBlue);
      assert.deepEqual(firstBlue, firstOrange);
      assert.notEqual(secondBlue, secondOrange);
    });

    it('Throws an error if string is not in the domain', function() {
      ccs.domain(['red', 'yellow', 'green']);
      assert.throws(function() {
        ccs.scale('not in domain');
      }, 'String was not in the domain.');
    });
  });
}
