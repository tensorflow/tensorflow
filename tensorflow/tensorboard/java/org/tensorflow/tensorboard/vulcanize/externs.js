// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Miscellaneous JSCompiler externs needed for TensorBoard.
 * @externs
 */

/** @type {!Object} */ var _;
/** @type {!Object} */ var d3;
/** @type {!Object} */ var dagre;
/** @type {!Object} */ var weblas;
/** @type {!Object} */ var graphlib;
/** @type {!Object} */ var Plottable;
/** @type {!Object} */ var GroupEffect;
/** @type {!Function|undefined} */ var ga;
/** @type {!Function|undefined} */ var KeyframeEffect;

/**
 * Some weird webcomponents-lite.js thing.
 * @type {!Function|undefined}
 */
var wrap;

/**
 * Some weird webcomponents-lite.js thing.
 * @type {!Function|undefined}
 */
window.wrap;

var HTMLImports;

/**
 * @param {function()} callback
 * @param {!HTMLDocument=} opt_doc
 */
HTMLImports.whenReady = function(callback, opt_doc) {};
