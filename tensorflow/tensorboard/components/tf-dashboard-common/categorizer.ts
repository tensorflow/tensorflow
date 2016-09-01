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
  export var topLevelNamespaceCategorizer: Categorizer = splitCategorizer(/\//);

  // Try to produce good categorizations on legacy graphs, which often
  // are namespaced like l1_foo/bar or l2_baz/bam.
  // If there is no leading underscore before the first forward slash,
  // then it behaves the same as topLevelNamespaceCategorizer
  export var legacyUnderscoreCategorizer: Categorizer =
      splitCategorizer(/[\/_]/);

  export function fallbackCategorizer(s: string): Categorizer {
    switch (s) {
      case 'TopLevelNamespaceCategorizer':
        return topLevelNamespaceCategorizer;
      case 'LegacyUnderscoreCategorizer':
        return legacyUnderscoreCategorizer;
      default:
        throw new Error('Unrecognized categorization strategy: ' + s);
    }
  }

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
      var sortedTags = tags.slice().sort(VZ.Sorting.compareTagNames);
      var categories: Category[] = [];
      var currentCategory = {
        name: extractor(sortedTags[0]),
        tags: [],
      };
      sortedTags.forEach((t: string) => {
        var topLevel = extractor(t);
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

  function splitCategorizer(r: RegExp): Categorizer {
    var extractor = (t: string) => {
      return t.split(r)[0];
    };
    return extractorToCategorizer(extractor);
  }

  export interface CategoryDefinition {
    name: string;
    matches: (t: string) => boolean;
  }

  export function defineCategory(ruledef: string): CategoryDefinition {
    var r = new RegExp(ruledef);
    var f = function(tag: string): boolean {
      return r.test(tag);
    };
    return { name: ruledef, matches: f };
  }

  export function _categorizer(
      rules: CategoryDefinition[], fallback: Categorizer) {
    return function(tags: string[]): Category[] {
      var remaining: d3.Set = d3.set(tags);
      var userSpecified = rules.map((def: CategoryDefinition) => {
        var tags: string[] = [];
        remaining.forEach((t: string) => {
          if (def.matches(t)) {
            tags.push(t);
          }
        });
        var cat = {name: def.name, tags: tags.sort(VZ.Sorting.compareTagNames)};
        return cat;
      });
      var defaultCategories = fallback(remaining.values());
      return userSpecified.concat(defaultCategories);
    };
  }

  export function categorizer(s: CustomCategorization): Categorizer {
    var rules = s.categoryDefinitions.map(defineCategory);
    var fallback = fallbackCategorizer(s.fallbackCategorizer);
    return _categorizer(rules, fallback);
  };
}
