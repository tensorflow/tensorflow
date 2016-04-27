module TF.Backend {
  export var Behavior = {
    properties: {
      /** *** Required properties *** */
      /** Data type. One of TF.Backend.TYPES */
      dataType: {
        type: String,
        observer: '_throwErrorOnUnrecognizedType',
      },

      /** TF.Backend.Backend for data loading. */
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
     * Backend reload logic is provided by this behaivor. The frontend reload
     *   logic should be provided elsewhere, since it is component-specific.
     * To keep things simple and consistent, we do the backend reload first,
     *   and the frontend reload afterwards.
     */
    reload: function() {
      return this.backendReload().then(
          (x) => { return this.frontendReload(); });
    },
    /**
     * Load data from backend and then set run2tag, tags, runs, and loadState.
     * Returns a promise that resolves/rejects when data is loaded.
     */
    backendReload: function() {
      if (this.dataType == null) {
        throw new Error('TF.Backend.Behavior: Need a dataType to reload.');
      }
      if (this.backend == null) {
        throw new Error('TF.Backend.Behavior: Need a backend to reload.');
      }
      var runsRoute = this.backend[this.dataType + 'Runs'].bind(this.backend);
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
            var tags = TF.Backend.getTags(x);
            this._setDataNotFound(tags.length === 0);
            this._setTags(tags);
            this._setRuns(TF.Backend.getRuns(x));
            return x;
          },
          (fail) => {
            this._setLoadState('failure');
            return fail;
          });
    },
    _do_autoLoad: function(type, backend, autoLoad) {
      if (autoLoad) {
        this.reload();
      };
    },
    _getDataProvider: function(dataType, backend) {
      return this.backend[this.dataType].bind(this.backend);
    },
    _throwErrorOnUnrecognizedType: function(dataType) {
      if (TF.Backend.TYPES.indexOf(dataType) === -1) {
        throw new Error('TF.Backend.Behavior: Unknown dataType ' + dataType);
      }
    },
  };
}
