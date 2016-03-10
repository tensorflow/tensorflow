/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Based on the gulpfile provided by angular team
// (https://github.com/angular/ts2dart/blob/master/gulpfile.js)
var gulp = require('gulp');
var tester = require('web-component-tester').test;
var ts = require('gulp-typescript');
var typescript = require('typescript');
var gutil = require('gulp-util');
var tslint = require('gulp-tslint');
var server = require('gulp-server-livereload');
var merge = require('merge2');
var gulpFilter = require('gulp-filter');
var vulcanize = require('gulp-vulcanize');
var minimist = require('minimist');
var replace = require('gulp-replace');
var rename = require('gulp-rename');
var header = require('gulp-header');
var fs = require('fs');
var path = require('path');
var typings = require('gulp-typings');
var bower = require('gulp-bower');
var options = minimist(process.argv.slice(2), {
  default: {
    p: 8000,  // port for gulp server
    h: '0.0.0.0', // host to serve on
  }
});

var tsProject = ts.createProject('tsconfig.json', {
  typescript: typescript,
  noExternalResolve: true, // opt-in for faster compilation!
});

var hasError;
var failOnError = true; // Is set to false when watching.

var onError = function(err) {
  hasError = true;
  gutil.log(err.message);
  if (failOnError) {
    process.exit(1);
  }
};

// These constants should always be in sync with the path in the .gitignore
// file.
var TF_COMPONENTS_PREFIX = 'tf-';
var TF_COMPONENTS_TYPESCRIPT_GLOB = 'components/' + TF_COMPONENTS_PREFIX +
    '*/**/*.ts';

var TF_LIB_TYPESCRIPT_GLOB = 'lib/js/**/*.ts';

gulp.task('typings', function() {
  // This task will create a typings directory at root level, with all typings
  // installed in it.
  return gulp.src('./typings.json')
      .pipe(typings());
});

// TODO(danmane): Wire this up once bower.json specifies all resolutions
gulp.task('bower', function() {
  return bower();
});

gulp.task('compile.all', ['typings'], function() {
  hasError = false;
  var isComponent = gulpFilter(['components/**/*.js']);
  var isLib = gulpFilter(['lib/js/**/*.js']);
  var isApp = gulpFilter(['app/**/*.js']);

  var tsResult = tsProject.src()
                     .pipe(ts(tsProject))
                     .on('error', onError);
  return merge([
    // Duplicate all component code to live next to the ts file
    // (makes polymer imports very clean)
    tsResult.js
            .pipe(isComponent)
            .pipe(gulp.dest('.')),
    tsResult.js
            .pipe(isLib)
            .pipe(gulp.dest('.')),
  ]);
});

gulp.task('test', ['tslint-strict', 'compile.all'], function(done) {
  tester({}, function(error) {
    if (error) {
      // Pretty error for gulp.
      error = new Error(error.message || error);
      error.showStack = false;
    }
    done(error);
  });
});

var tslintTask = function(strict) {
  return function(done) {
    if (hasError) {
      done();
      return;
    }
    return gulp.src([TF_COMPONENTS_TYPESCRIPT_GLOB, TF_LIB_TYPESCRIPT_GLOB])
               .pipe(tslint())
               .pipe(tslint.report('verbose', {
                  emitError: strict,
               }));
 };
};

// Since constructs like console.log are disabled by tslint
// but very useful while developing, create a "permissive"
// version of tslint that warns without erroring, for the
// watch task.
gulp.task('tslint-permissive', [], tslintTask(false));
gulp.task('tslint-strict', [], tslintTask(true));

gulp.task('watch', ['compile.all', 'tslint-permissive'], function() {
  failOnError = false;
  // Avoid watching generated .d.ts in the build (aka output) directory.
  return gulp.watch([TF_COMPONENTS_TYPESCRIPT_GLOB, TF_LIB_TYPESCRIPT_GLOB],
          {ignoreInitial: true},
          ['compile.all', 'tslint-permissive']);
});

gulp.task('server', function() {
  gulp.src('.').pipe(server({
    host: options.h,
    port: options.p,
    livereload: {
      enable: true,
      // Don't livereload on .ts changes, since they aren't loaded by browser.
      filter: function(filePath, cb) { cb(!(/\.ts$/.test(filePath))); },
      port: 27729 + options.p
    },
    directoryListing: true,
  }));
});

/**
 * Returns a list of non-tensorboard components inside the components
 * directory, i.e. components that don't begin with 'tf-'.
 */
function getNonTensorBoardComponents() {
  return fs.readdirSync('components')
      .filter(function(file) {
        var filePrefix = file.slice(0, TF_COMPONENTS_PREFIX.length);
        return fs.statSync(path.join('components', file)).isDirectory() &&
            filePrefix !== TF_COMPONENTS_PREFIX;
      })
      .map(function(dir) { return '/' + dir + '/'; });
}


var linkRegex = /<link rel="[^"]*" (type="[^"]*" )?href="[^"]*">\n/g;
var scriptRegex = /<script src="[^"]*"><\/script>\n/g;
gulp.task('vulcanize', ['compile.all', 'tslint-strict'], function() {
  // Vulcanize TensorBoard without external libraries.
  gulp.src('components/tf-tensorboard/tf-tensorboard.html')
      .pipe(vulcanize({
        inlineScripts: true,
        inlineCss: true,
        stripComments: true,
        excludes: getNonTensorBoardComponents(),
      }))
      // TODO(danmane): Remove this worrisome brittleness when vulcanize
      // fixes https://github.com/Polymer/vulcanize/issues/273
      .pipe(replace(linkRegex, ''))
      .pipe(replace(scriptRegex, ''))
      .pipe(header('// AUTOGENERATED FILE - DO NOT MODIFY \n'))
      .pipe(rename('tf-tensorboard.html.OPENSOURCE'))
      .pipe(gulp.dest('./dist'));


  gulp.src('components/tf-tensorboard/tf-tensorboard-demo.html')
      .pipe(vulcanize({
        inlineScripts: true,
        inlineCss: true,
        stripComments: true,
      }))
      .pipe(header('// AUTOGENERATED FILE - DO NOT MODIFY \n'))
      .pipe(gulp.dest('dist'));
});

gulp.task('serve', ['server']); // alias
gulp.task('default', ['compile.all', 'watch', 'serve']);
