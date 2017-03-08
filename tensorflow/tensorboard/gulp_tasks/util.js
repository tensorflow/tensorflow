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

var fs = require('fs');
var path = require('path');

/**
 * Returns a list of web components inside the components directory for which
 * the name predicate is true.
 */
exports.getComponents = function(namePredicate) {
  return fs.readdirSync('components')
      .filter(function(file) {
        return fs.statSync(path.join('components', file)).isDirectory() &&
            namePredicate(file);
      })
      .map(function(dir) { return '/' + dir + '/'; });
};

/**
 * Returns a list of tensorboard web components that are inside the components
 * directory.
 */
exports.tbComponents = exports.getComponents(function(name) {
  var prefix = name.slice(0, 3);
  return prefix == 'tf_' || prefix == 'vz_';
});
