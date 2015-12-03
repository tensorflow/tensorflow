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
var concat = require('gulp-concat');
var merge = require('merge2');
var gulpFilter = require('gulp-filter');
var vulcanize = require('gulp-vulcanize');
var rename = require('gulp-rename');
var minimist = require('minimist');
var replace = require('gulp-replace');
var tfserve = require('./scripts/tfserve.js');
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

gulp.task('compile.all', function() {
  hasError = false;
  var isComponent = gulpFilter(['components/**/*.js']);
  var isApp = gulpFilter(['app/**/*.js']);

  var srcs = ['components/**/*.ts', 'test/**/*.ts', 'app/**/*.ts',
              'typings/**/*.d.ts', 'bower_components/**/*.d.ts'];

  var tsResult = gulp.src(srcs, {base: '.'})
                     .pipe(ts(tsProject))
                     .on('error', onError);
  return merge([
    // Duplicate all component code to live next to the ts file
    // (makes polymer imports very clean)
    tsResult.js
            .pipe(isComponent)
            .pipe(gulp.dest('.'))
  ]);
});

gulp.task('test', ['tslint-strict', 'compile.all'], function(done) {
  tester({suites: ['components/test/'],
          plugins: {local: {}, sauce: false}}, function(error) {
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
    return gulp.src(['components/**/*.ts', 'test/**/*.ts'])
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
  return gulp.watch(['test/**/*.ts', 'components/**/*.ts'],
          {ignoreInitial: true},
          ['compile.all', 'tslint-permissive']);
});

gulp.task('server', function() {
  tfserve({
    port: options.p,
    host: options.h,
    verbose: options.v,
  });
});


var linkRegex = /<link rel="[^"]*" (type="[^"]*" )?href=".*bower_components[^"]*">\n/g;
var scriptRegex = /<script src=".*bower_components[^"]*"><\/script>\n/g;
gulp.task('vulcanize', ['compile.all', 'tslint-strict'], function() {
      gulp.src('app/tf-tensorboard.html')
          .pipe(vulcanize({
            inlineScripts: true,
            inlineCss: true,
            stripComments: true,
            excludes: ['/bower_components/'],
          }))
          // TODO(danmane): Remove this worrysome brittleness when vulcanize
          // fixes https://github.com/Polymer/vulcanize/issues/273
          .pipe(replace(linkRegex, ''))
          .pipe(replace(scriptRegex, ''))
          .pipe(gulp.dest('dist'));

      gulp.src('app/index.html')
          .pipe(vulcanize({
            inlineScripts: true,
            inlineCss: true,
            stripComments: true,
          }))
          .pipe(gulp.dest('dist'));

      gulp.src('app/tf-tensorboard-demo.html')
          .pipe(vulcanize({
            inlineScripts: true,
            inlineCss: true,
            stripComments: true,
          }))
          .pipe(gulp.dest('dist'));
});

gulp.task('serve', ['server']); // alias
gulp.task('default', ['watch', 'serve']);
