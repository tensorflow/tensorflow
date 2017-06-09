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
import {getRuns, getTags, TYPES} from './backend';

/** @polymerBehavior */
export const BackendBehavior = {
  properties: {
    /** *** Required properties *** */
    /** Data type. One of Backend.TYPES */
    dataType: {
      type: String,
      observer: '_throwErrorOnUnrecognizedType',
    },

    /** Backend for data loading. */
    backend: {
      type: Object,
    },

    /** Should it automatically load when configured ready? Default true. */
    autoLoad: {
      type: Boolean,
      value: true,
    },

    /** *** Component-provided properties *** */
    /** Every tag available for data type (sorted, dedpulicated) */
    tags: {
      type: Array,
      readOnly: true,
      notify: true,
    },

    /** Every run available for data type (sorted) */
    runs: {
      type: Array,
      readOnly: true,
      notify: true,
    },

    /** Mapping from runs to tags for the data type */
    run2tag: {
      type: Object,
      readOnly: true,
      notify: true,
    },

    /** Promise provider for the data. Useful for passing to subcomponents */
    dataProvider:
        {type: Function, computed: '_getDataProvider(dataType, backend)'},

    /** Has the dashboard loaded yet? */
    loadState: {
      type: String,
      value: 'noload',  // [noload, pending, loaded, failure]
      readOnly: true,
    },

    /**
     * True if dashboard has loaded, and no tags were found.
     * Persists through subsequent reloads (ie. still true while
     * next load is pending) so warning won't flash away every reload
     * when there is no data.
     */
    dataNotFound: {
      type: Boolean,
      value: false,
      readOnly: true,
    }

  },
  observers: ['_do_autoLoad(dataType, backend, autoLoad)'],
  /**
   * Reloading works in two steps:
   * Backend reload, which gets metadata on available runs, tags, etc from
   *   the backend.
   * Frontend reload, which loads new data for each chart or visual display.
   * Backend reload logic is provided by this behavior. The frontend reload
   *   logic should be provided elsewhere, since it is component-specific.
   * To keep things simple and consistent, we do the backend reload first,
   *   and the frontend reload afterwards.
   */
  reload() {
    return this.backendReload().then((x) => {
      return this.frontendReload();
    });
  },
  /**
   * Load data from backend and then set run2tag, tags, runs, and loadState.
   * Returns a promise that resolves/rejects when data is loaded.
   */
  backendReload() {
    if (this.dataType == null) {
      throw new Error('BackendBehavior: Need a dataType to reload.');
    }
    if (this.backend == null) {
      throw new Error('BackendBehavior: Need a backend to reload.');
    }
    const runsRoute = (this.backend[this.dataType + 'Runs'] ||
                       this.backend[this.dataType + 'Tags'])
                          .bind(this.backend);
    this._setLoadState('pending');
    return runsRoute().then(
        (x) => {
          this._setLoadState('loaded');
          if (_.isEqual(x, this.run2tag)) {
            // If x and run2tag are equal, let's avoid updating everything
            // since that can needlessly trigger run changes, reloads, etc
            return x;
          }
          this._setRun2tag(x);
          const tags = getTags(x);
          this._setDataNotFound(tags.length === 0);
          this._setTags(tags);
          this._setRuns(getRuns(x));
          return x;
        },
        (fail) => {
          this._setLoadState('failure');
          return fail;
        });
  },
  _do_autoLoad(type, backend, autoLoad) {
    if (autoLoad) {
      this.reload();
    }
  },
  _getDataProvider(dataType, backend) {
    return this.backend[this.dataType].bind(this.backend);
  },
  _throwErrorOnUnrecognizedType(dataType) {
    if (TYPES.indexOf(dataType) === -1) {
      throw new Error('BackendBehavior: Unknown dataType ' + dataType);
    }
  },
};
