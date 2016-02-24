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
var assert = chai.assert;

module TF.Backend {
  describe("urlPathHelpers", function() {
    var clean = UrlPathHelpers.clean;
    var encode = UrlPathHelpers.queryEncoder;
    it("clean works as expected", function() {
      var cleaned = clean(UrlPathHelpers.BAD_CHARACTERS);
      var all_clean = "";
      for (var i = 0; i < UrlPathHelpers.BAD_CHARACTERS.length; i++) {
        all_clean += "_";
      }
      assert.equal(cleaned, all_clean, "cleaning the BAD_CHARACTERS works");
      assert.equal(clean("foozod"), "foozod", "doesnt change safe string");
      assert.equal(clean("foo zod (2)"), "foo_zod__2_", "simple case");
    });

    it("queryEncoder works with clean on spaces and parens", function() {
      var params = {foo: "something with spaces and (parens)"};
      var actual = clean(encode(params));
      var expected = "_foo_something_with_spaces_and__28parens_29";
      assert.equal(actual, expected);
    });
  });

  function assertIsDatum(x) {
    assert.isNumber(x.step);
    assert.instanceOf(x.wall_time, Date);
  }

  describe("backend tests", function() {
    var backend: Backend;
    var rm: RequestManager;
    var base = "data";
    beforeEach(function() {
      // Construct a demo Backend (third param is true)
      backend = new Backend(base, new RequestManager(), true);
      rm = new RequestManager();
    });

    it("runs are loaded properly", function(done) {
      var runsResponse = backend.runs();
      var actualRuns = rm.request(base + "/runs");
      Promise.all([runsResponse, actualRuns]).then((values) => {
        assert.deepEqual(values[0], values[1]);
        done();
      });
    });

    it("scalars are loaded properly", function(done) {
      backend.scalars("run1", "cross_entropy (1)").then((s) => {
        // just check the data got reformatted properly
        var aScalar = s[s.length - 1];
        assertIsDatum(aScalar);
        assert.isNumber(aScalar.scalar);
        // verify date conversion works
        assert.equal(aScalar.wall_time.valueOf(), 40000);
        done();
      });
    });

    it("histograms are loaded properly", function(done) {
      backend.histograms("run1", "histo1").then((histos) => {
        var histo = histos[0];
        assertIsDatum(histo);
        assert.isNumber(histo.min);
        assert.isNumber(histo.max);
        assert.isNumber(histo.sum);
        assert.isNumber(histo.sumSquares);
        assert.isNumber(histo.nItems);
        assert.instanceOf(histo.bucketRightEdges, Array);
        assert.instanceOf(histo.bucketRightEdges, Array);
        done();
      });
    });

    it("images are loaded properly", function(done) {
      backend.images("run1", "im1").then((images) => {
        var image = images[0];
        assertIsDatum(image);
        assert.isNumber(image.width);
        assert.isNumber(image.height);
        var nonDemoQuery = "index=0&tag=im1&run=run1";
        var nonDemoUrl = "individualImage?" + nonDemoQuery;
        var expectedUrl = base + "/" + UrlPathHelpers.clean(nonDemoUrl);
        assert.equal(image.url, expectedUrl);
        done();
      });
    });

    it("trailing slash removed from base route", function() {
      var b = new TF.Backend.Backend("foo/");
      assert.equal(b.baseRoute, "foo");
    });
  });
}
