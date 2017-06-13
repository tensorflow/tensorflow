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

import {compareTagNames} from '../vz-sorting/sorting';

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

export function fallbackCategorizer(s: string): Categorizer {
  switch (s) {
    case 'TopLevelNamespaceCategorizer':
      return topLevelNamespaceCategorizer;
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
function extractorToCategorizer(extractor: (s: string) => string): Categorizer {
  return (tags: string[]): Category[] => {
    if (tags.length === 0) {
      return [];
    }

    // Maps between top-level name and category. We use the mapping to avoid
    // duplicating categories per run.
    const categoryMapping: {[key: string]: Category} = {};

    tags.forEach((t: string) => {
      const topLevel = extractor(t);
      if (!categoryMapping[topLevel]) {
        const newCategory = {
          name: topLevel,
          tags: [],
        };
        categoryMapping[topLevel] = newCategory;
      }

      categoryMapping[topLevel].tags.push(t);
    });

    // Sort categories into alphabetical order.
    const categories =
        _.map(_.keys(categoryMapping).sort(), key => categoryMapping[key]);
    _.forEach(categories, (category) => {
      // Sort the tags within each category.
      category.tags.sort(compareTagNames);
    });
    return categories;
  };
}

function splitCategorizer(r: RegExp): Categorizer {
  let extractor = (t: string) => {
    return t.split(r)[0];
  };
  return extractorToCategorizer(extractor);
}

export interface CategoryDefinition {
  name: string;
  matches: (t: string) => boolean;
}

export function defineCategory(ruledef: string): CategoryDefinition {
  let r = new RegExp(ruledef);
  let f = function(tag: string): boolean {
    return r.test(tag);
  };
  return {name: ruledef, matches: f};
}

export function _categorizer(
    rules: CategoryDefinition[], fallback: Categorizer) {
  return function(tags: string[]): Category[] {
    let remaining: d3.Set = d3.set(tags);
    let userSpecified = rules.map((def: CategoryDefinition) => {
      let tags: string[] = [];
      remaining.each((t: string) => {
        if (def.matches(t)) {
          tags.push(t);
        }
      });
      let cat = {name: def.name, tags: tags.sort(compareTagNames)};
      return cat;
    });
    let defaultCategories = fallback(remaining.values());
    return userSpecified.concat(defaultCategories);
  };
}

export function categorizer(s: CustomCategorization): Categorizer {
  let rules = s.categoryDefinitions.map(defineCategory);
  let fallback = fallbackCategorizer(s.fallbackCategorizer);
  return _categorizer(rules, fallback);
};

Polymer({
  is: 'tf-categorizer',
  properties: {
    regexes: {type: Array},
    tags: {type: Array},
    categoriesAreExclusive: {type: Boolean, value: true},
    fallbackCategorizer: {
      type: String,
      value: 'TopLevelNamespaceCategorizer',
    },
    categorizer: {
      type: Object,
      computed:
          'computeCategorization(regexes.*, categoriesAreExclusive, fallbackCategorizer)',
    },
    categories: {
      type: Array,
      value: function() {
        return [];
      },
      notify: true,
      readOnly: true
    },
  },
  observers: ['recategorize(tags.*, categorizer)'],
  computeCategorization: function(
      regexes, categoriesAreExclusive, fallbackCategorizer) {
    var categorizationStrategy = {
      categoryDefinitions: regexes.base,
      categoriesAreExclusive: categoriesAreExclusive,
      fallbackCategorizer: fallbackCategorizer,
    };
    return categorizer(categorizationStrategy);
  },
  recategorize: function() {
    this.debounce('tf-categorizer-recategorize', function() {
      var categories = this.categorizer(this.tags);
      this._setCategories(categories);
    })
  },
});
