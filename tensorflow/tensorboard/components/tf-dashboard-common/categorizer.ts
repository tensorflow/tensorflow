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

module Categorizer {
  /**
   * This module contains methods that allow sorting tags into 'categories'.
   * A category contains a name and a list of tags.
   * The sorting strategy is defined by a 'CustomCategorization', which contains
   * 'categoryDefinitions' which are regex rules used to construct a category.
   * E.g. the regex rule 'xent' will create a category called 'xent' that
   * contains values whose tags match the regex.
   *
   * After custom categories are evaluated, the tags are sorted by a hardcoded
   * fallback categorizer, which may, for example, group tags into categories
   * based on their top namespace.
   */

  export interface Category {
    // Categories that data is sorted into
    name: string;
    tags: string[];
  }

  export interface CustomCategorization {
    // Defines a categorization strategy
    categoryDefinitions: string[];
    fallbackCategorizer: string;
    /* {'TopLevelNamespaceCategorizer',
        'LegacyUnderscoreCategorizer'} */
  }

  export interface Categorizer {
    // Function that generates categories
    (tags: string[]): Category[];
  }

  /* Canonical TensorFlow ops are namespaced using forward slashes.
   * This fallback categorizer categorizes by the top-level namespace.
   */
  // Try to produce good categorizations on legacy graphs, which often
  // are namespaced like l1_foo/bar or l2_baz/bam.
  // If there is no leading underscore before the first forward slash,
  // then it behaves the same as topLevelNamespaceCategorizer
  export var rootNameUnderscoreCategorizer = rootNameCategorizer(/[\/_]/);

  /* An 'extractor' is a function that takes a tag name, and 'extracts' a
   * category name.
   * This function takes an extractor, and produces a categorizer.
   * Currently, it is just used for the fallbackCategorizer, but we may want to
   * refactor the general categorization logic to use the concept of extractors.
   */
  function extractorToCategorizer(extractor: (s: string) => string):
      Categorizer {
    return (tags: string[]): Category[] => {
      if (tags.length === 0) {
        return [];
      }
      let sortedTags = tags.slice().sort(VZ.Sorting.compareTagNames);
      let categories: Category[] = [];
      let currentCategory = {
        name: extractor(sortedTags[0]),
        tags: [],
      };
      sortedTags.forEach((t: string) => {
        let topLevel = extractor(t);
        if (currentCategory.name !== topLevel) {
          categories.push(currentCategory);
          currentCategory = {
            name: topLevel,
            tags: [],
          };
        }
        currentCategory.tags.push(t);
      });
      categories.push(currentCategory);
      return categories;
    };
  }

  /** Split on a regex, taking just the first element after splitting.
   * It's like getting the root directory. E.g. if you split on slash, then
   * 'foo/bar/zod' will go to 'foo'
   */
  function rootNameCategorizer(r: RegExp): Categorizer {
    let extractor = (t: string) => { return t.split(r)[0]; };
    return extractorToCategorizer(extractor);
  }

  /* Split on a regex, taking all the prefix until the last split.
   * It's like getting the dirname of a path. E.g. if you split on slash, then
   * 'foo/bar/zod' will go to 'foo/bar'.
   * In the case where there are no splits (e.g. 'foo') then it uses 'foo' as
   * the category name.
   */
  function dnameExtractor(t: string) {
    let splits = t.split('/');
    if (splits.length === 1) {
      return t;
    } else {
      let last = _.last(splits);
      return t.slice(0, t.length - last.length - 1);
    }
  }

  export var directoryNameCategorizer = extractorToCategorizer(dnameExtractor);

  export interface CategoryDefinition {
    name: string;
    matches: (t: string) => boolean;
  }

  export function defineCategory(ruledef: string): CategoryDefinition {
    let r = new RegExp(ruledef);
    let f = function(tag: string): boolean { return r.test(tag); };
    return { name: ruledef, matches: f };
  }

  export function _categorizer(
      rules: CategoryDefinition[], fallback: Categorizer) {
    return function(tags: string[]): Category[] {
      let remaining: d3.Set = d3.set(tags);
      let userSpecified = rules.map((def: CategoryDefinition) => {
        let tags: string[] = [];
        remaining.forEach((t: string) => {
          if (def.matches(t)) {
            tags.push(t);
          }
        });
        let cat = {name: def.name, tags: tags.sort(VZ.Sorting.compareTagNames)};
        return cat;
      });
      let defaultCategories = fallback(remaining.values());
      return userSpecified.concat(defaultCategories);
    };
  }

  export function fallbackCategorizer(s: string): Categorizer {
    switch (s) {
      case 'DirectoryNameCategorizer':
        return directoryNameCategorizer;
      case 'RootNameUnderscoreCategorizer':
        return rootNameUnderscoreCategorizer;
      default:
        throw new Error('Unrecognized categorization strategy: ' + s);
    }
  }

  export function categorizer(s: CustomCategorization): Categorizer {
    let rules = s.categoryDefinitions.map(defineCategory);
    let fallback = fallbackCategorizer(s.fallbackCategorizer);
    return _categorizer(rules, fallback);
  };
}
