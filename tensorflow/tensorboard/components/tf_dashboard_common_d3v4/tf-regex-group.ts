/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

import * as storage from '../tf-storage/storage';

Polymer({
  is: 'tf-regex-group',
  properties: {
    rawRegexes: {
      type: Array,
      value: storage.getObjectInitializer(
          'rawRegexes', [{regex: '', valid: true}]),
    },
    regexes:
        {type: Array, computed: 'usableRegexes(rawRegexes.*)', notify: true},
  },
  observers: [
    'addNewRegexIfNeeded(rawRegexes.*)',
    'checkValidity(rawRegexes.*)',
    '_uriStoreRegexes(rawRegexes.*)',
  ],
  _uriStoreRegexes:
      storage.getObjectObserver('rawRegexes', [{regex: '', valid: true}]),
  checkValidity: function(x) {
    var match = x.path.match(/rawRegexes\.(\d+)\.regex/);
    if (match) {
      var idx = match[1];
      this.set('rawRegexes.' + idx + '.valid', this.isValid(x.value));
    }
  },
  isValid: function(s) {
    try {
      new RegExp(s);
      return true;
    } catch (e) {
      return false;
    }
  },
  usableRegexes: function(regexes) {
    var isValid = this.isValid;
    return regexes.base
        .filter(function(r) {
          // Checking validity here (rather than using the data property)
          // is necessary because otherwise we might send invalid regexes due
          // to the fact that this function can call before the observer does
          return r.regex !== '' && isValid(r.regex);
        })
        .map(function(r) {
          return r.regex;
        });
  },
  addNewRegexIfNeeded: function() {
    var last = this.rawRegexes[this.rawRegexes.length - 1];
    if (last.regex !== '') {
      this.push('rawRegexes', {regex: '', valid: true});
    }
  },
  deleteRegex: function(e) {
    if (this.rawRegexes.length > 1) {
      this.splice('rawRegexes', e.model.index, 1);
    }
  },
  moveFocus: function(e) {
    if (e.keyCode === 13) {
      var idx = e.model.index;
      var inputs = Polymer.dom(this.root).querySelectorAll('.regex-input');
      if (idx < this.rawRegexes.length - 1) {
        (inputs[idx + 1] as any).$.input.focus();
      } else {
        (document.activeElement as HTMLElement).blur();
      }
    }
  }
});
