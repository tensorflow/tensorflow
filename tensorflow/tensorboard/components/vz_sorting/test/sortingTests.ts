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

import {compareTagNames} from '../sorting';

describe('compareTagNames', () => {

  const assert = chai.assert;
  const sortTagNames = (a) => a.sort(compareTagNames);

  it('is asciibetical', () => {
    assert.deepEqual(sortTagNames(['a', 'b']), ['a', 'b']);
    assert.deepEqual(sortTagNames(['a', 'B']), ['B', 'a']);
  });

  it('sorts integer portions', () => {
    assert.deepEqual(['03', '1'].sort(), ['03', '1']);
    assert.deepEqual(sortTagNames(['03', '1']), ['1', '03']);
    assert.deepEqual(sortTagNames(['a03', 'a1']), ['a1', 'a03']);
    assert.deepEqual(sortTagNames(['a03', 'b1']), ['a03', 'b1']);
    assert.deepEqual(sortTagNames(['x0a03', 'x0a1']), ['x0a1', 'x0a03']);
    assert.deepEqual(sortTagNames(['a/b/03', 'a/b/1']), ['a/b/1', 'a/b/03']);
  });

  it('sorts fixed point numbers', () => {
    assert.deepEqual(sortTagNames(['a0.1', 'a0.01']), ['a0.01', 'a0.1']);
  });

  it('sorts engineering notation', () => {
    assert.deepEqual(sortTagNames(['a1e9', 'a9e8']), ['a9e8', 'a1e9']);
    assert.deepEqual(sortTagNames(['a1e+9', 'a9e+8']), ['a9e+8', 'a1e+9']);
    assert.deepEqual(sortTagNames(['a1e+5', 'a9e-6']), ['a9e-6', 'a1e+5']);
    assert.deepEqual(sortTagNames(['a1.0e9', 'a9.0e8']), ['a9.0e8', 'a1.0e9']);
    assert.deepEqual(
        sortTagNames(['a1.0e+9', 'a9.0e+8']), ['a9.0e+8', 'a1.0e+9']);
  });

  it('is componentized by slash', () => {
    assert.deepEqual(['a+/a', 'a/a', 'ab/a'].sort(), ['a+/a', 'a/a', 'ab/a']);
    assert.deepEqual(
        sortTagNames(['a+/a', 'a/a', 'ab/a']), ['a/a', 'a+/a', 'ab/a']);
  });

  it('is componentized by underscore', () => {
    assert.deepEqual(
        sortTagNames(['a+_a', 'a_a', 'ab_a']), ['a_a', 'a+_a', 'ab_a']);
    assert.deepEqual(
        sortTagNames(['a+/a', 'a_a', 'ab_a']), ['a_a', 'a+/a', 'ab_a']);
  });

  it('is componentized by number boundaries', () => {
    assert.deepEqual(
        sortTagNames(['a+0a', 'a0a', 'ab0a']), ['a0a', 'a+0a', 'ab0a']);
  });

  it('empty comes first', () => {
    assert.deepEqual(sortTagNames(['a', '//', '/', '']), ['', '/', '//', 'a']);
  });

  it('decimal parsed correctly', () => {
    assert.deepEqual(sortTagNames(['0.2', '0.03']), ['0.03', '0.2']);
    assert.deepEqual(sortTagNames(['0..2', '0..03']), ['0..2', '0..03']);
    assert.deepEqual(sortTagNames(['.2', '.03']), ['.2', '.03']);
  });
});
