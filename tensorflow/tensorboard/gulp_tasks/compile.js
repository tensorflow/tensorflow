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
var browserify = require('browserify');
var tsify = require('tsify');
var source = require('vinyl-source-stream');
var glob = require('glob').sync;
var concat = require('gulp-concat');

var tsProject = ts.createProject('./tsconfig.json', {
  typescript: typescript,
  noExternalResolve: true, // opt-in for faster compilation!
});

/** List of components (and their external deps) that are using es6 modules. */
var ES6_COMPONENTS = [{
  name: 'vz_projector',
  deps: [
    'd3/d3.min.js', 'weblas/dist/weblas.js', 'three.js/build/three.min.js',
    'three.js/examples/js/controls/OrbitControls.js',
    'numericjs/lib/numeric-1.2.6.js'
  ]
}];

module.exports = function(includeDeps) {
  return function() {
    // Compile all components that are using ES6 modules into a bundle.js
    // using browserify.
    var entries = ['typings/index.d.ts'];
    var deps = {};
    ES6_COMPONENTS.forEach(function(component) {
      // Collect all the typescript files across the components.
      entries = entries.concat(glob(
          'components/' + component.name + '/**/*.ts',
          // Do not include tests or IDE-purposed files.
          {ignore: ['**/*_test.ts', '**/deps.d.ts']}));
      // Collect the unique external deps across all components using es6
      // modules.
      component.deps.forEach(function(dep) {
        deps['components/' + dep] = true;
      });
    });
    deps = Object.keys(deps);

    // Compile, bundle all the typescript files and prepend their deps.
    browserify(entries)
        .plugin(tsify)
        .bundle()
        .on('error', function(error) { console.error(error.toString()); })
        .pipe(source('bundle.js'))
        .pipe(gulp.dest('components'))
        .on('end', function() {
          // Typescript was compiled and bundled. Now we need to prepend
          // the external dependencies.
          if (includeDeps) {
            gulp.src(deps.concat(['components/bundle.js']))
                .pipe(concat('bundle.js'))
                .pipe(gulp.dest('components'));
          }
        });

    // Compile components that are using global namespaces producing 1 js file
    // for each ts file.
    var isComponent = filter([
      'components/tf_*/**/*.ts', 'components/vz_*/**/*.ts', 'typings/**/*.ts',
      'components/plottable/plottable.d.ts'
      // Ignore components that use es6 modules.
    ].concat(ES6_COMPONENTS.map(function(component) {
      return '!components/' + component.name + '/**/*.ts';
    })));

    return tsProject.src()
        .pipe(isComponent)
        .pipe(ts(tsProject))
        .js.pipe(gulp.dest('.'));
  };
};
