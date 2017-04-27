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

import {Backend, getRuns, getTags, RunToTag} from './backend'
import {BackendBehavior} from './behavior'

declare function fixture(id: string): void;

window.addEventListener('WebComponentsReady', function() {
  Polymer({
    is: 'test-element',
    behaviors: [BackendBehavior],
    frontendReload: function() {
      // no-op
    },
  });
});

describe('data-behavior', function() {
  var testElement;
  var resolve;
  var reject;
  var fakeBackend = {
    scalarRuns: function() {
      return new Promise(function(_resolve, _reject) {
        resolve = (x) => _resolve(x);
        reject = (x) => _reject(x);
      });
    },
    scalar: function(x) {
      return this;
    },
  };
  beforeEach(function() {
    testElement = fixture('testElementFixture');
    testElement.autoLoad = false;
    testElement.backend = fakeBackend;
    testElement.dataType = 'scalar';
  });

  it('load states work as expected', function(done) {
    chai.assert.equal(testElement.loadState, 'noload');
    var reloaded = testElement.reload();
    chai.assert.equal(testElement.loadState, 'pending');
    resolve();
    reloaded
        .then(function() {
          chai.assert.equal(testElement.loadState, 'loaded');
          var reloaded2 = testElement.reload();
          chai.assert.equal(testElement.loadState, 'pending');
          reject();
          return reloaded2;
        })
        .then(function() {
          chai.assert.equal(testElement.loadState, 'failure');
          done();
        });
  });

  it('data provider set appropriately', function() {
    chai.assert.deepEqual(testElement.dataProvider(), testElement.backend);
  });

  it('loads data as expected', function(done) {
    var r2t: RunToTag = {
      run1: ['foo', 'bar', 'zod'],
      run2: ['zoink', 'zow'],
      run3: ['.'],
    };
    var tags = getTags(r2t);
    var runs = getRuns(r2t);
    testElement.backend = fakeBackend;
    testElement.dataType = 'scalar';
    testElement.reload().then(function(x) {
      chai.assert.deepEqual(testElement.run2tag, r2t);
      chai.assert.deepEqual(testElement.runs, runs);
      chai.assert.deepEqual(testElement.tags, tags);
      done();
    });
    resolve(r2t);
  });

  it('errors thrown on bad data types', function() {
    testElement.backend = undefined;
    chai.assert.throws(function() {
      testElement.dataType = 'foo';
    });
    testElement.dataType = 'scalar';
    testElement.dataType = 'graph';
    testElement.dataType = 'histogram';
  });

  it('dataNotFound flag works', function(done) {
    chai.assert.isFalse(testElement.dataNotFound, 'initially false');
    var next = testElement.reload();
    chai.assert.isFalse(testElement.dataNotFound, 'still false while pending');
    resolve({foo: [], bar: []});
    next.then(() => {
      chai.assert.isTrue(testElement.dataNotFound, 'true on empty data');
      var last = testElement.reload();
      chai.assert.isTrue(testElement.dataNotFound, 'still true while pending');
      resolve({foo: ['bar'], bar: ['zod']});
      last.then(() => {
        chai.assert.isFalse(
            testElement.dataNotFound, 'false now that we have data');
        done();
      });
    });
  });

  it('reloads as soon as setup, if autoReload is true', function(done) {
    var r2t = {foo: [], bar: []};
    var fakeBackend = {
      scalarRuns: () => Promise.resolve(r2t),
      scalar: () => null,
    };
    testElement = fixture('testElementFixture');
    testElement.dataType = 'scalar';
    testElement.backend = fakeBackend;
    setTimeout(() => {
      chai.assert.equal(testElement.run2tag, r2t);
      done();
    });
  });

  it('doesn\'t mutate props if backend returns same data', function(done) {
    var r2t_1 = {foo: ['1', '2'], bar: ['3', '4']};
    var r2t_2 = {foo: ['1', '2'], bar: ['3', '4']};
    var fakeBackend = {
      scalarRuns: () => Promise.resolve(r2t_1),
      scalar: () => null,
    };
    testElement.backend = fakeBackend;
    testElement.reload().then(() => {
      fakeBackend.scalarRuns = () => Promise.resolve(r2t_2);
      var tags = testElement.tags;
      testElement.reload().then(() => {
        // shallow equality ensures it wasn't recomputed
        chai.assert.equal(tags, testElement.tags, 'tags was not recomputed');
        done();
      });
    });
  });

  it('reload calls frontendReload', function(done) {
    testElement.frontendReload = function() {
      done();
    };
    testElement.reload();
  });

});
