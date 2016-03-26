/* Copyright 2015 Google Inc. All Rights Reserved.

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

module TF {
  let assert = chai.assert;

  describe("ColorScale", function() {
    let ccs: ColorScale;

    beforeEach(function() {
      ccs = new ColorScale();
    });

    it("No collisions with train, eval and test", function() {
      ccs.domain(["train"]);
      var trainColor = ccs.getColor("train");
      ccs.domain(["eval"]);
      var evalColor = ccs.getColor("eval");
      ccs.domain(["test"]);
      var testColor = ccs.getColor("test");
      assert.notEqual(trainColor, evalColor, testColor);
    });

    it("Returns consistent colors, given no hash collisions", function() {
      // These three colors don't have hash collisions
      ccs.domain(["red", "yellow"]);
      var firstRedColor = ccs.getColor("red");
      ccs.domain(["red", "yellow", "blue"]);
      var secondRedColor = ccs.getColor("red");
      assert.deepEqual(firstRedColor, secondRedColor);
    });

    it("A 2-color scale returns the first and last colors of the palette", function() {
      var twoColorScale = new ColorScale(2, TF.palettes.googleStandard);
      // No hash collisions with these.
      twoColorScale.domain(["red", "blue"]);
      assert.deepEqual(twoColorScale.getColor("blue"), TF.palettes.googleStandard[0]);
      assert.deepEqual(twoColorScale.getColor("red"), TF.palettes.googleStandard[TF.palettes.googleStandard.length - 1]);
    });

    // This is testing that when we reset the domain with new colors, the old
    // domain doesn't influence the new color choices. Basically testing that we
    // get a fresh slate if we have a new domain. Basically testing that all the
    // internal bins are reset etc. and we aren't finding collisions with
    // previous colors.
    it("Colors don't nudge away from colors from an old domain.", function() {
      // at 12 breaks, "orange" and "blue" collide.
      ccs.domain(["red", "blue"]);
      var firstBlue = ccs.getColor("blue");
      ccs.domain(["red", "orange"]);
      var firstOrange = ccs.getColor("orange");
      assert.deepEqual(firstBlue, firstOrange);
    });

    it("Nudges all colors, given only one base color", function() {
      var ccsWithOneColor = new ColorScale(1);
      ccsWithOneColor.domain(["one", "two", "three"]);
      assert.notEqual(ccsWithOneColor.getColor("one"), ccsWithOneColor.getColor("two"));
      assert.notEqual(ccsWithOneColor.getColor("two"), ccsWithOneColor.getColor("three"));
      assert.notEqual(ccsWithOneColor.getColor("one"), ccsWithOneColor.getColor("three"));
    });

    it("Nudges a color if it has a hash collision", function() {
      // at 12 breaks, "orange" and "blue" collide.
      ccs.domain(["red", "blue"]);
      var firstBlue = ccs.getColor("blue");
      ccs.domain(["red", "orange"]);
      var firstOrange = ccs.getColor("orange");
      ccs.domain(["red", "blue", "orange"]);
      var secondBlue = ccs.getColor("blue");
      var secondOrange = ccs.getColor("orange");
      assert.deepEqual(firstBlue, secondBlue);
      assert.deepEqual(firstBlue, firstOrange);
      assert.notEqual(secondBlue, secondOrange);
    });

    it("Throws an error if string is not in the domain", function() {
      ccs.domain(["red", "yellow", "green"]);
      assert.throws(function() {
        ccs.getColor("not in domain");
      }, "String was not in the domain.");
    });
  });
}
