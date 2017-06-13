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

import * as cat from '../tf-categorizer';

let assert = chai.assert;

describe('categorizer', () => {
  describe('topLevelNamespaceCategorizer', () => {
    it('returns empty array on empty tags', () => {
      assert.lengthOf(cat.topLevelNamespaceCategorizer([]), 0);
    });

    it('handles a simple case', () => {
      let simple = [
        'foo1/bar', 'foo1/zod', 'foo2/bar', 'foo2/zod', 'gosh/lod/mar',
        'gosh/lod/ned'
      ];
      let expected = [
        {name: 'foo1', tags: ['foo1/bar', 'foo1/zod']},
        {name: 'foo2', tags: ['foo2/bar', 'foo2/zod']},
        {name: 'gosh', tags: ['gosh/lod/mar', 'gosh/lod/ned']},
      ];
      assert.deepEqual(cat.topLevelNamespaceCategorizer(simple), expected);
    });

    it('orders the categories', () => {
      let test = ['e', 'f', 'g', 'a', 'b', 'c'];
      let expected = [
        {name: 'a', tags: ['a']},
        {name: 'b', tags: ['b']},
        {name: 'c', tags: ['c']},
        {name: 'e', tags: ['e']},
        {name: 'f', tags: ['f']},
        {name: 'g', tags: ['g']},
      ];
      assert.deepEqual(cat.topLevelNamespaceCategorizer(test), expected);
    });

    it('handles cases where category names overlap node names', () => {
      let test = ['a', 'a/a', 'a/b', 'a/c', 'b', 'b/a'];
      const actual = cat.topLevelNamespaceCategorizer(test);
      let expected = [
        {name: 'a', tags: ['a', 'a/a', 'a/b', 'a/c']},
        {name: 'b', tags: ['b', 'b/a']},
      ];
      assert.deepEqual(actual, expected);
    });

    it('handles singleton case', () => {
      assert.deepEqual(
          cat.topLevelNamespaceCategorizer(['a']), [{name: 'a', tags: ['a']}]);
    });
  });

  describe('customCategorizer', () => {
    function noFallbackCategorizer(tags: string[]): cat.Category[] {
      return [];
    }

    function testCategorizer(
        defs: string[], fallback: cat.Categorizer,
        tags: string[]): cat.Category[] {
      const catDefs = defs.map(cat.defineCategory);
      return cat._categorizer(catDefs, fallback)(tags);
    }

    it('categorizes by regular expression', () => {
      let defs = ['foo..', 'bar..'];
      let tags = ['fooab', 'fooxa', 'barts', 'barms'];
      const actual = testCategorizer(defs, noFallbackCategorizer, tags);
      let expected = [
        {name: 'foo..', tags: ['fooab', 'fooxa']},
        {name: 'bar..', tags: ['barms', 'barts']},
      ];
      assert.deepEqual(actual, expected);
    });

    it('matches non-exclusively', () => {
      let tags = ['abc', 'bar', 'zod'];
      const actual =
          testCategorizer(['...', 'bar'], noFallbackCategorizer, tags);
      let expected = [
        {name: '...', tags: ['abc', 'bar', 'zod']},
        {name: 'bar', tags: ['bar']},
      ];
      assert.deepEqual(actual, expected);
    });

    it('creates categories for unmatched rules', () => {
      const actual =
          testCategorizer(['a', 'b', 'c'], noFallbackCategorizer, []);
      let expected = [
        {name: 'a', tags: []},
        {name: 'b', tags: []},
        {name: 'c', tags: []},
      ];
      assert.deepEqual(actual, expected);
    });

    it('category regexs work with special characters', () => {
      let defs = ['^\\w+$', '^\\d+$', '^\\/..$'];
      let tags = ['foo', '3243', '/xa'];
      const actual = testCategorizer(defs, noFallbackCategorizer, tags);
      let expected = [
        {name: '^\\w+$', tags: ['3243', 'foo']},
        {name: '^\\d+$', tags: ['3243']},
        {name: '^\\/..$', tags: ['/xa']},
      ];
      assert.deepEqual(actual, expected);
    });

    it('category tags are sorted', () => {
      let tags = ['a', 'z', 'c', 'd', 'e', 'x', 'f', 'y', 'g'];
      let sorted = tags.slice().sort();
      let expected = [{name: '.*', tags: sorted}];
      const actual = testCategorizer(['.*'], noFallbackCategorizer, tags);
      assert.deepEqual(actual, expected);
    });

    it('if nonexclusive: all tags passed to fallback', () => {
      let passedToDefault = null;
      function defaultCategorizer(tags: string[]): cat.Category[] {
        passedToDefault = tags;
        return [];
      }
      let tags = ['foo', 'bar', 'foo123'];
      testCategorizer(['foo'], defaultCategorizer, tags);
      assert.deepEqual(passedToDefault, tags);
    });
  });
});
