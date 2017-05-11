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
import * as util from '../util';

describe('getURLParams', () => {
  it('search query with valid param returns correct object', () => {
    let urlParams = util.getURLParams('?config=http://google.com/');
    assert.deepEqual({'config': 'http://google.com/'}, urlParams);
  });

  it('search query with multiple valid params returns correct object', () => {
    let urlParams = util.getURLParams('?config=http://google.com/&foo=bar');
    assert.deepEqual({'config': 'http://google.com/', 'foo': 'bar'}, urlParams);
  });

  it('search query with valid param with URL encoded characters', () => {
    let urlParams = util.getURLParams('?config=http://google.com/%20search');
    assert.deepEqual({'config': 'http://google.com/ search'}, urlParams);
  });

  it('search query with pound sign', () => {
    let urlParams = util.getURLParams('?config=http://google.com/#foo');
    assert.deepEqual({'config': 'http://google.com/'}, urlParams);
  });

  it('no search query returns empty object', () => {
    let urlParams = util.getURLParams('');
    assert.deepEqual({}, urlParams);
  });
});
