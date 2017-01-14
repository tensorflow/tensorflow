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

var gulp = require('gulp');
var ts = require('gulp-typescript');
var typescript = require('typescript');
var gutil = require('gulp-util');
var filter = require('gulp-filter');
var merge = require('merge2');

var tsProject = ts.createProject('./tsconfig.json', {
  typescript: typescript,
  noExternalResolve: true, // opt-in for faster compilation!
});


module.exports = function() {
  var isComponent = filter([
    'components/tf-*/**/*.ts',
    'components/vz-*/**/*.ts',
    'typings/**/*.ts',
    // TODO(danmane): add plottable to the typings registry after updating it
    // and remove this.
    'components/plottable/plottable.d.ts'
  ]);

  return tsProject.src()
           .pipe(isComponent)
           .pipe(ts(tsProject))
           .js
            .pipe(gulp.dest('.'));

}
