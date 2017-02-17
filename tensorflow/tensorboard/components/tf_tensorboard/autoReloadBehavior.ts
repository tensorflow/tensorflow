/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
module TF.TensorBoard {
  export var AUTORELOAD_LOCALSTORAGE_KEY = 'TF.TensorBoard.autoReloadEnabled';

  var getAutoReloadFromLocalStorage: () => boolean = () => {
    var val = window.localStorage.getItem(AUTORELOAD_LOCALSTORAGE_KEY);
    return val === 'true' || val == null;  // defaults to true
  };

  export var AutoReloadBehavior = {
    properties: {
      autoReloadEnabled: {
        type: Boolean,
        observer: '_autoReloadObserver',
        value: getAutoReloadFromLocalStorage,
      },
      _autoReloadId: {
        type: Number,
      },
      autoReloadIntervalSecs: {
        type: Number,
        value: 120,
      },
    },
    detached: function() { window.clearTimeout(this._autoReloadId);},
    _autoReloadObserver: function(autoReload) {
      window.localStorage.setItem(AUTORELOAD_LOCALSTORAGE_KEY, autoReload);
      if (autoReload) {
        var _this = this;
        this._autoReloadId = window.setTimeout(
            this._doAutoReload.bind(this), this.autoReloadIntervalSecs * 1000);
      } else {
        window.clearTimeout(this._autoReloadId);
      }
    },
    _doAutoReload: function() {
      if (this.reload == null) {
        throw new Error('AutoReloadBehavior requires a reload method');
      }
      this.reload();
      this._autoReloadId = window.setTimeout(
          this._doAutoReload.bind(this), this.autoReloadIntervalSecs * 1000);
    }
  };
}
