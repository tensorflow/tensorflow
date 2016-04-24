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

module TF.Histogram {
  let assert = chai.assert;

  describe('Rebin', function() {

    var assertHistogramEquality = function(h1, h2) {
      h1.forEach(function(b1, i) {
        var b2 = h2[i];
        assert.closeTo(b1.x, b2.x, 1e-10);
        assert.closeTo(b1.dx, b2.dx, 1e-10);
        assert.closeTo(b1.y, b2.y, 1e-10);
      });
    };

    //
    // Rebinning
    //

    it("Returns an empty array if you don't have any bins", function() {
      assert.deepEqual(rebinHistogram([], 10), []);
    });

    it("Collapses two bins into one.", function() {
      var histogram = [
        {x: 0, dx: 1, y: 1},
        {x: 1, dx: 1, y: 2}
      ];
      var oneBin = [
        {x: 0, dx: 2, y: 3}
      ];
      assertHistogramEquality(rebinHistogram(histogram, 1), oneBin);
    });

    it("Splits one bin into two.", function() {
      var histogram = [
        {x: 0, dx: 1, y: 3}
      ];
      var twoBin = [
        {x: 0, dx: 0.5, y: 1.5},
        {x: 0.5, dx: 0.5, y: 1.5}
      ];
      assertHistogramEquality(rebinHistogram(histogram, 2), twoBin);
    });

    it("Regularizes non-uniform bins.", function() {
      var histogram = [
        {x: 0, dx: 2, y: 3},
        {x: 2, dx: 3, y: 3},
        {x: 5, dx: 1, y: 1}
      ];
      var twoBin = [
        {x: 0, dx: 3, y: 4},
        {x: 3, dx: 3, y: 3}
      ];
      assertHistogramEquality(rebinHistogram(histogram, 2), twoBin);
    });

  });
}
