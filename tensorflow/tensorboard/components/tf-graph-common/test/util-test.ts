/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

suite('util', () => {
  let assert = chai.assert;

  test('remove common prefix', () => {

    // Empty array.
    let result = tf.graph.util.removeCommonPrefix([]);
    assert.deepEqual(result, []);

    // No common prefix.
    result = tf.graph.util.removeCommonPrefix(['a', 'b', 'c']);
    assert.deepEqual(result, ['a', 'b', 'c']);

    // One of the elements is empty string.
    result = tf.graph.util.removeCommonPrefix(['a/b', '', 'a/c']);
    assert.deepEqual(result, ['a/b', '', 'a/c']);

    // Only one string.
    result = tf.graph.util.removeCommonPrefix(['a/b/c']);
    assert.deepEqual(result, ['a/b/c']);

    // `q/w/` is the common prefix. Expect `q/w/` to be removed.
    result = tf.graph.util.removeCommonPrefix(['q/w/a', 'q/w/b', 'q/w/c/f']);
    assert.deepEqual(result, ['a', 'b', 'c/f']);

    // `q/w/` is the common prefix and also an element. Expect nothing to be
    // removed since the common prefix is also an element in the array.
    result = tf.graph.util.removeCommonPrefix(['q/w/', 'q/w/b', 'q/w/c/f']);
    assert.deepEqual(result, ['q/w/', 'q/w/b', 'q/w/c/f']);
  });

  test('query params', () => {
    // Starts with question mark.
    let queryParams = tf.graph.util.getQueryParams('?foo=1&bar=2');
    assert.deepEqual(queryParams, {'foo': '1', 'bar': '2'});

    // No question mark.
    queryParams = tf.graph.util.getQueryParams('foo=1&bar=2');
    assert.deepEqual(queryParams, {'foo': '1', 'bar': '2'});
  });
});
