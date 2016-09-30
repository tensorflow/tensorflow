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
var assert = chai.assert;

module TF.Backend {
  describe('urlPathHelpers', function() {
    let demoify = TF.Backend.demoify;
    let encode = TF.Backend.queryEncoder;
    it('demoify works as expected', function() {
      let demoified = demoify(BAD_CHARACTERS);
      let all_clean = '';
      for (let i = 0; i < BAD_CHARACTERS.length; i++) {
        all_clean += '_';
      }
      assert.equal(demoified, all_clean, 'cleaning the BAD_CHARACTERS works');
      assert.equal(demoify('foozod'), 'foozod', 'doesnt change safe string');
      assert.equal(demoify('foo zod (2)'), 'foo_zod__2_', 'simple case');
    });

    it('queryEncoder works with demoify on spaces and parens', function() {
      let params = {foo: 'something with spaces and (parens)'};
      let actual = demoify(encode(params));
      let expected = '_foo_something_with_spaces_and__28parens_29';
      assert.equal(actual, expected);
    });
  });

  function assertIsDatum(x) {
    assert.isNumber(x.step);
    assert.instanceOf(x.wall_time, Date);
  }

  describe('backend tests', function() {
    let backend: Backend;
    let rm: RequestManager;
    let base = 'data';
    let demoRouter = TF.Backend.router(base, true);
    beforeEach(function() {
      // Construct a demo Backend (third param is true)
      backend = new Backend(demoRouter);
      rm = new RequestManager();
    });

    it('runs are loaded properly', function(done) {
      let runsResponse = backend.runs();
      let actualRuns = rm.request(demoRouter.runs());
      Promise.all([runsResponse, actualRuns]).then((values) => {
        assert.deepEqual(values[0], values[1]);
        done();
      });
    });

    it('scalars are loaded properly', function(done) {
      backend.scalar('cross_entropy (1)', 'run1').then((s) => {
        // just check the data got reformatted properly
        let aScalar = s[s.length - 1];
        assertIsDatum(aScalar);
        assert.isNumber(aScalar.scalar);
        // verify date conversion works
        assert.equal(aScalar.wall_time.valueOf(), 40000);
        done();
      });
    });

    it('histograms are loaded properly', function(done) {
      backend.histogram('histo1', 'run1').then((histos) => {
        let histo = histos[0];
        assertIsDatum(histo);
        assert.instanceOf(histo.bins, Array);
        done();
      });
    });

    it('all registered types have handlers', function() {
      TYPES.forEach((t: string) => {
        assert.isDefined(backend[t], t);
        assert.isDefined(backend[t + 'Runs'], t + 'Runs');
      });
    });

    it('images are loaded properly', function(done) {
      backend.image('im1', 'run1').then((images) => {
        let image = images[0];
        assertIsDatum(image);
        assert.isNumber(image.width);
        assert.isNumber(image.height);
        let nonDemoQuery = 'index=0&tag=im1&run=run1';
        let expectedUrl = demoRouter.individualImage(nonDemoQuery, 10.0);
        assert.equal(image.url, expectedUrl);
        done();
      });
    });

    it('audio is loaded properly', function(done) {
      backend.audio('audio1', 'run1').then((audio_clips) => {
        let audio = audio_clips[0];
        assertIsDatum(audio);
        assert.equal(audio.content_type, 'audio/wav');
        let nonDemoQuery = 'index=0&tag=audio1&run=run1';
        let expectedUrl = demoRouter.individualAudio(nonDemoQuery);
        assert.equal(audio.url, expectedUrl);
        done();
      });
    });

    it('trailing slash removed from base route', function() {
      let r = TF.Backend.router('foo/');
      assert.equal(r.runs(), 'foo/runs');
    });

    it('run helper methods work', function(done) {
      let scalar = {run1: ['cross_entropy (1)'], fake_run_no_data: ['scalar2']};
      let image = {run1: ['im1'], fake_run_no_data: ['im1', 'im2']};
      let audio = {run1: ['audio1'], fake_run_no_data: ['audio1', 'audio2']};
      let runMetadata = {run1: ['step99'], fake_run_no_data: ['step99']};
      let graph = ['fake_run_no_data'];
      let count = 0;
      function next() {
        count++;
        if (count === 4) {
          done();
        }
      }
      backend.scalarRuns().then((x) => {
        assert.deepEqual(x, scalar);
        next();
      });
      backend.imageRuns().then((x) => {
        assert.deepEqual(x, image);
        next();
      });
      backend.audioRuns().then((x) => {
        assert.deepEqual(x, audio);
        next();
      });
      backend.runMetadataRuns().then((x) => {
        assert.deepEqual(x, runMetadata);
        next();
      });
      backend.graphRuns().then((x) => {
        assert.deepEqual(x, graph);
        next();
      });
    });

    it('runToTag helpers work', function() {
      let r2t: RunToTag = {
        run1: ['foo', 'bar', 'zod'],
        run2: ['zod', 'zoink'],
        a: ['foo', 'zod']
      };
      let empty1: RunToTag = {};
      let empty2: RunToTag = {run1: [], run2: []};
      assert.deepEqual(getRuns(r2t), ['a', 'run1', 'run2']);
      assert.deepEqual(getTags(r2t), ['bar', 'foo', 'zod', 'zoink']);
      assert.deepEqual(filterTags(r2t, ['run1', 'run2']), getTags(r2t));
      assert.deepEqual(filterTags(r2t, ['run1']), ['bar', 'foo', 'zod']);
      assert.deepEqual(filterTags(r2t, ['run2', 'a']), ['foo', 'zod', 'zoink']);

      assert.deepEqual(getRuns(empty1), []);
      assert.deepEqual(getTags(empty1), []);

      assert.deepEqual(getRuns(empty2), ['run1', 'run2']);
      assert.deepEqual(getTags(empty2), []);
    });
  });

  describe('Verify that the histogram format conversion works.', function() {

    function assertHistogramEquality(h1, h2) {
      h1.forEach(function(b1, i) {
        let b2 = h2[i];
        assert.closeTo(b1.x, b2.x, 1e-10);
        assert.closeTo(b1.dx, b2.dx, 1e-10);
        assert.closeTo(b1.y, b2.y, 1e-10);
      });
    }

    it('Throws and error if the inputs are of different lengths', function() {
      assert.throws(function() {
        convertBins(
            {bucketRightEdges: [0], bucketCounts: [1, 2], min: 1, max: 2}, 1, 2,
            2);
      }, 'Edges and counts are of different lengths.');
    });

    it('Handles data with no bins', function() {
      assert.deepEqual(
          convertBins(
              {bucketRightEdges: [], bucketCounts: [], min: 0, max: 0}, 0, 0,
              0),
          []);
    });

    it('Handles data with one bin', function() {
      let counts = [1];
      let rightEdges = [1.21e-12];
      let histogram = [{x: 1.1e-12, dx: 1.21e-12 - 1.1e-12, y: 1}];
      let newHistogram = convertBins(
          {
            bucketRightEdges: rightEdges,
            bucketCounts: counts,
            min: 1.1e-12,
            max: 1.21e-12
          },
          1.1e-12, 1.21e-12, 1);
      assertHistogramEquality(newHistogram, histogram);
    });

    it('Handles data with two bins.', function() {
      let counts = [1, 2];
      let rightEdges = [1.1e-12, 1.21e-12];
      let histogram = [
        {x: 1.0e-12, dx: 1.05e-13, y: 1.09090909090909},
        {x: 1.105e-12, dx: 1.05e-13, y: 1.9090909090909}
      ];
      let newHistogram = convertBins(
          {
            bucketRightEdges: rightEdges,
            bucketCounts: counts,
            min: 1.0e-12,
            max: 1.21e-12
          },
          1.0e-12, 1.21e-12, 2);
      assertHistogramEquality(newHistogram, histogram);
    });

    it('Handles a domain that crosses zero, but doesn\'t include zero as ' +
           'an edge.',
       function() {
         let counts = [1, 2];
         let rightEdges = [-1.0e-12, 1.0e-12];
         let histogram = [
           {x: -1.1e-12, dx: 1.05e-12, y: 1.95},
           {x: -0.5e-13, dx: 1.05e-12, y: 1.05}
         ];
         let newHistogram = convertBins(
             {
               bucketRightEdges: rightEdges,
               bucketCounts: counts,
               min: -1.1e-12,
               max: 1.0e-12
             },
             -1.1e-12, 1.0e-12, 2);
         assertHistogramEquality(newHistogram, histogram);
       });

    it('Handles a right-most right edge that extends to very large number.',
       function() {
         let counts = [1, 2, 3];
         let rightEdges = [0, 1.0e-12, 1.0e14];
         let histogram = [
           {x: -1.0e-12, dx: 0.7e-12, y: 0.7},
           {x: -0.3e-12, dx: 0.7e-12, y: 1.1},
           {x: 0.4e-12, dx: 0.7e-12, y: 4.2}
         ];
         let newHistogram = convertBins(
             {
               bucketRightEdges: rightEdges,
               bucketCounts: counts,
               min: -1.0e-12,
               max: 1.1e-12
             },
             -1.0e-12, 1.1e-12, 3);
         assertHistogramEquality(newHistogram, histogram);
       });
  });
}
