// Based on the gulpfile provided by angular team
// (https://github.com/angular/ts2dart/blob/master/gulpfile.js)
var gulp = require('gulp');
var ts = require('gulp-typescript');
var typescript = require('typescript');
var gutil = require('gulp-util');
var mochaPhantomJS = require('gulp-mocha-phantomjs');
var tslint = require('gulp-tslint');
var server = require('gulp-server-livereload');
var concat = require('gulp-concat');
var merge = require('merge2');
var gulpFilter = require('gulp-filter');
var vulcanize = require('gulp-vulcanize');
var rename = require('gulp-rename');
var minimist = require('minimist');
var replace = require('gulp-replace');

var options = minimist(process.argv.slice(2), {
  default: {
    p: 8000  // port for gulp server
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
  var isComponent = gulpFilter(['components/**/*.js', '!components/**/test/*']);
  var isApp = gulpFilter(['app/**/*.js']);
  var isTest = gulpFilter(['test/**/*', 'components/**/test/*']);

  var srcs = ['components/**/*.ts', 'test/**/*.ts', 'app/**/*.ts',
              'typings/**/*.d.ts', 'bower_components/**/*.d.ts'];

  var tsResult = gulp.src(srcs, {base: '.'})
                     .pipe(ts(tsProject))
                     .on('error', onError);
  return merge([
    // Send concatenated component code to build/component
    tsResult.js
            .pipe(isComponent)
            .pipe(concat('components.js'))
            .pipe(gulp.dest('build')),

    // Duplicate all component code to live next to the ts file
    // (makes polymer imports very clean)
    tsResult.js
            .pipe(isComponent)
            .pipe(gulp.dest('.')),

    tsResult.js
            .pipe(isApp)
            .pipe(gulp.dest('.')),

    // Send all test code to build/test.js
    tsResult.js
            .pipe(isTest)
            .pipe(concat('test.js'))
            .pipe(gulp.dest('build')),

    // Create a unified defintions file at build/all.d.ts
    tsResult.dts
            .pipe(concat('all.d.ts'))
            .pipe(gulp.dest('build')),
  ]);
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


gulp.task('run-tests', ['compile.all'], function(done) {
  if (hasError) {
    done();
    return;
  }
  return gulp.src('tests.html')
    .pipe(mochaPhantomJS({reporter: 'dot'}));
});

gulp.task('test', ['run-tests', 'tslint-strict']);
gulp.task('watch', ['run-tests', 'tslint-permissive'], function() {
  failOnError = false;
  // Avoid watching generated .d.ts in the build (aka output) directory.
  return gulp.watch(['test/**/*.ts', 'components/**/*.ts'],
          {ignoreInitial: true},
          ['run-tests', 'tslint-permissive']);
});

gulp.task('server', function() {
  gulp.src('.')
      .pipe(server({
        host: '0.0.0.0',
        port: options.p,
        livereload: {
          enable: true,
          port: 27729 + options.p
        },
        directoryListing: true,
      }));
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
gulp.task('default', ['watch']);
