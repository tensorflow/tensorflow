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

import {DataPoint, DataSet} from './data';


/**
 * Helper method that makes a list of points given an array of
 * trace indexes.
 *
 * @param traces The i-th entry holds the 'next' attribute for the i-th point.
 */
function makePointsWithTraces(traces: number[]) {
  let nextAttr = '__next__';
  let points: DataPoint[] = [];
  traces.forEach((t, i) => {
    let metadata: {[key: string]: any} = {};
    metadata[nextAttr] = t >= 0 ? t : null;
    points.push({
      vector: [],
      metadata: metadata,
      projections: {},
      projectedPoint: null,
      dataSourceIndex: i
    });
  });
  return points;
}

const assert = chai.assert;

it('Simple forward pointing traces', () => {
  // The input is: 0->2, 1->None, 2->3, 3->None. This should return
  // one trace 0->2->3.
  let points = makePointsWithTraces([2, -1, 3, -1]);
  let dataset = new DataSet(points);
  assert.equal(dataset.traces.length, 1);
  assert.deepEqual(dataset.traces[0].pointIndices, [0, 2, 3]);
});

it('No traces', () => {
  let points = makePointsWithTraces([-1, -1, -1, -1]);
  let dataset = new DataSet(points);
  assert.equal(dataset.traces.length, 0);
});

it('A trace that goes backwards and forward in the array', () => {
  // The input is: 0->2, 1->0, 2->nothing, 3->1. This should return
  // one trace 3->1->0->2.
  let points = makePointsWithTraces([2, 0, -1, 1]);
  let dataset = new DataSet(points);
  assert.equal(dataset.traces.length, 1);
  assert.deepEqual(dataset.traces[0].pointIndices, [3, 1, 0, 2]);
});
