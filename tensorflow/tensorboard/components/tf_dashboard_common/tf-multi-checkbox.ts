/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

import * as storage from '../tf-storage/storage';

Polymer({
  is: 'tf-multi-checkbox',
  properties: {
    names: {
      type: Array,
      value: function() {
        return [];
      },
    },  // All the runs in consideration
    regexInput: {
      type: String,
      value: storage.getStringInitializer('regexInput', ''),
      observer: '_regexInputObserver',
    },  // Regex for filtering the runs
    regex: {type: Object, computed: '_makeRegex(regexInput)'},
    namesMatchingRegex: {
      type: Array,
      computed: 'computeNamesMatchingRegex(names.*, regex)'
    },  // Runs that match the regex
    runSelectionState: {
      // if a run is explicitly enabled, True, if explicitly disabled, False.
      // if undefined, default value (enable for first k runs, disable after).
      type: Object,
      value: storage.getObjectInitializer('runSelectionState', {}),
      observer: '_storeRunToIsCheckedMapping',
    },
    // (Allows state to persist across regex filtering)
    outSelected: {
      type: Array,
      notify: true,
      computed: 'computeOutSelected(namesMatchingRegex.*, runSelectionState.*)'
    },
    colorScale: {
      type: Object,
      observer: 'synchronizeColors',
    },  // map from run name to css class
    maxRunsToEnableByDefault: {
      // When TB first loads, if it has k or fewer runs, they are all enabled
      // by default. If there are more, then they are all disabled.
      type: Number,
      value: 40,
    },
    _debouncedRegexChange: {
      type: Object,
      // Updating the regex can be slow, because it involves updating styles
      // on a large number of Polymer paper-checkboxes. We don't want to do
      // this while the user is typing, as it may make a bad, laggy UI.
      // So we debounce the updates that come from user typing.
      value: function() {
        const _this = this;
        var debounced = _.debounce(function(r) {
          _this.regexInput = r;
        }, 150, {leading: false});
        return function() {
          var r = this.$$('#runs-regex').value;
          if (r == '') {
            // If the user cleared the field, they may be done typing, so
            // update more quickly.
            this.async(function() {
              _this.regexInput = r;
            }, 30);
          } else {
            debounced(r);
          };
        };
      },
    },
  },
  listeners: {
    'dom-change': 'synchronizeColors',
  },
  observers: [
    '_setIsolatorIcon(runSelectionState, names)',
  ],
  _storeRunToIsCheckedMapping:
      storage.getObjectObserver('runSelectionState', {}),
  _makeRegex: function(regex) {
    try {
      return new RegExp(regex)
    } catch (e) {
      return null;
    }
  },
  _setIsolatorIcon: function() {
    var runMap = this.runSelectionState;
    var numChecked = _.filter(_.values(runMap)).length;
    var buttons =
        Array.prototype.slice.call(this.querySelectorAll('.isolator'));

    buttons.forEach(function(b) {
      if (numChecked === 1 && runMap[b.name]) {
        b.icon = 'radio-button-checked';
      } else {
        b.icon = 'radio-button-unchecked';
      }
    });
  },
  computeNamesMatchingRegex: function(__, ___) {
    var regex = this.regex;
    return this.names.filter(function(n) {
      return regex == null || regex.test(n);
    });
  },
  computeOutSelected: function(__, ___) {
    var runSelectionState = this.runSelectionState;
    var num = this.maxRunsToEnableByDefault;
    var allEnabled = this.namesMatchingRegex.length <= num;
    return this.namesMatchingRegex.filter(function(n, i) {
      return runSelectionState[n] == null ? allEnabled : runSelectionState[n];
    });
  },
  synchronizeColors: function(e) {
    if (!this.colorScale) return;

    this._setIsolatorIcon();

    var checkboxes =
        Array.prototype.slice.call(this.querySelectorAll('paper-checkbox'));
    var scale = this.colorScale;
    checkboxes.forEach(function(p) {
      var color = scale.scale(p.name);
      p.customStyle['--paper-checkbox-checked-color'] = color;
      p.customStyle['--paper-checkbox-checked-ink-color'] = color;
      p.customStyle['--paper-checkbox-unchecked-color'] = color;
      p.customStyle['--paper-checkbox-unchecked-ink-color'] = color;
    });
    var buttons =
        Array.prototype.slice.call(this.querySelectorAll('.isolator'));
    buttons.forEach(function(p) {
      var color = scale.scale(p.name);
      p.style['color'] = color;
    });
    // The updateStyles call fails silently if the browser doesn't have focus,
    // e.g. if TensorBoard was opened into a new tab that isn't visible.
    // So we wait for requestAnimationFrame.
    var _this = this;
    window.requestAnimationFrame(function() {
      _this.updateStyles();
    });
  },
  _isolateRun: function(e) {
    // If user clicks on the label for one run, enable it and disable all other
    // runs.

    var name = (Polymer.dom(e) as any).localTarget.name;
    var selectionState = {};
    this.names.forEach(function(n) {
      selectionState[n] = n == name;
    });
    this.runSelectionState = selectionState;
  },
  _checkboxChange: function(e) {
    var target = (Polymer.dom(e) as any).localTarget;
    this.runSelectionState[target.name] = target.checked;
    // n.b. notifyPath won't work because run names may have periods.
    this.runSelectionState = _.clone(this.runSelectionState);
  },
  _isChecked: function(item, outSelectedChange) {
    return this.outSelected.indexOf(item) != -1;
  },
  _regexInputObserver: storage.getStringObserver('regexInput', ''),
  toggleAll: function() {
    var _this = this;
    var anyToggledOn = this.namesMatchingRegex.some(function(n) {
      return _this.runSelectionState[n]
    });


    var runSelectionStateIsDefault =
        Object.keys(this.runSelectionState).length == 0;

    var defaultOff =
        this.namesMatchingRegex.length > this.maxRunsToEnableByDefault;
    // We have runs toggled either if some were explicitly toggled on, or if
    // we are in the default state, and there are few enough that we default
    // to toggling on.
    anyToggledOn = anyToggledOn || runSelectionStateIsDefault && !defaultOff;

    // If any are toggled on, we turn everything off. Or, if none are toggled
    // on, we turn everything on.

    var newRunsDisabled = {};
    this.names.forEach(function(n) {
      newRunsDisabled[n] = !anyToggledOn;
    });
    this.runSelectionState = newRunsDisabled;
  },
});
