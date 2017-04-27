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

import {SPTree} from './sptree';

const assert = chai.assert;

it('simple 2D data', () => {
  let data = [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
  ];
  let tree = new SPTree(data);
  // Check that each point is within the bound.
  tree.visit((node, low, high) => {
    assert.equal(low.length, 2);
    assert.equal(high.length, 2);
    let point = node.point;
    assert.equal(point.length, 2);
    // Each point should be in the node's bounding box.
    assert.equal(
        point[0] >= low[0] && point[0] <= high[0] && point[1] >= low[1] &&
            point[1] <= high[1],
        true);
    return false;
  });
});

it('simple 3D data', () => {
  let data = [
    [0, 1, 0],
    [1, 0.4, 2],
    [1, 1, 3],
    [0, 0, 5],
  ];
  let tree = new SPTree(data);
  // Check that each point is within the bound.
  tree.visit((node, low, high) => {
    assert.equal(low.length, 3);
    assert.equal(high.length, 3);
    let point = node.point;
    assert.equal(point.length, 3);
    // Each point should be in the node's bounding box.
    assert.equal(
        point[0] >= low[0] && point[0] <= high[0] && point[1] >= low[1] &&
            point[1] <= high[1] && point[2] >= low[2] && point[2] <= high[2],
        true);
    return false;
  });
});

it('Only visit root', () => {
  let data = [
    [0, 1, 0],
    [1, 0.4, 2],
    [1, 1, 3],
    [0, 0, 5],
  ];
  let tree = new SPTree(data);
  let numVisits = 0;
  tree.visit((node, low, high) => {
    numVisits++;
    return true;
  });
  assert.equal(numVisits, 1);
});

it('Search in random data', () => {
  let N = 10000;
  let data = new Array(N);
  for (let i = 0; i < N; i++) {
    data[i] = [Math.random(), Math.random()];
  }
  let tree = new SPTree(data);
  let numVisits = 0;
  let query = data[Math.floor(Math.random() * N)];
  let found = false;
  tree.visit((node, low, high) => {
    numVisits++;
    if (node.point === query) {
      found = true;
      return true;
    }
    let outOfBounds = query[0] < low[0] || query[0] > high[0] ||
        query[1] < low[1] || query[1] > high[1];
    return outOfBounds;
  });
  assert.equal(found, true);
  assert.isBelow(numVisits, N / 4);
});
