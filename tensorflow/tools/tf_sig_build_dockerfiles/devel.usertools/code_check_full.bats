# vim: filetype=bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
setup_file() {
    cd /tf/tensorflow
    bazel version  # Start the bazel server
}

# Do a bazel query specifically for the licenses checker. It searches for
# targets matching the provided query, which start with // or @ but not
# //tensorflow (so it looks for //third_party, //external, etc.), and then
# gathers the list of all packages (i.e. directories) which contain those
# targets.
license_query() {
 bazel cquery --experimental_cc_shared_library "$1" --keep_going \
  | grep -e "^//" -e "^@" \
  | grep -E -v "^//tensorflow" \
  | sed -e 's|:.*||' \
  | sort -u
}

# Verify that, given a build target and a license-list generator target, all of
# the dependencies of that target which include a license notice file are then
# included when generating that license. Necessary because the license targets
# in TensorFlow are manually enumerated rather than generated automatically.
do_external_licenses_check(){
  BUILD_TARGET="$1"
  LICENSES_TARGET="$2"

  # grep patterns for targets which are allowed to be missing from the licenses
  cat > $BATS_TEST_TMPDIR/allowed_to_be_missing <<EOF
@absl_py//absl
@bazel_tools//platforms
@bazel_tools//third_party/
@bazel_tools//tools
@local
@com_google_absl//absl
@pybind11_abseil//pybind11_abseil
@org_tensorflow//
@com_github_googlecloudplatform_google_cloud_cpp//google
@com_github_grpc_grpc//src/compiler
@platforms//os
@ruy//
EOF

  # grep patterns for targets which are allowed to be extra licenses
  cat > $BATS_TEST_TMPDIR/allowed_to_be_extra <<EOF
//third_party/mkl
//third_party/mkl_dnn
@absl_py//
@bazel_tools//src
@bazel_tools//platforms
@bazel_tools//tools/
@org_tensorflow//tensorflow
@com_google_absl//
@pybind11_abseil//pybind11_abseil
//external
@local
@com_github_googlecloudplatform_google_cloud_cpp//
@embedded_jdk//
^//$
@ruy//
EOF

  license_query "attr('licenses', 'notice', deps($BUILD_TARGET))" > $BATS_TEST_TMPDIR/expected_licenses
  license_query "deps($LICENSES_TARGET)" > $BATS_TEST_TMPDIR/actual_licenses

  # Column 1 is left only, Column 2 is right only, Column 3 is shared lines
  # Select lines unique to actual_licenses, i.e. extra licenses.
  comm -1 -3 $BATS_TEST_TMPDIR/expected_licenses $BATS_TEST_TMPDIR/actual_licenses | grep -v -f $BATS_TEST_TMPDIR/allowed_to_be_extra > $BATS_TEST_TMPDIR/actual_extra_licenses || true
  # Select lines unique to expected_licenses, i.e. missing licenses
  comm -2 -3 $BATS_TEST_TMPDIR/expected_licenses $BATS_TEST_TMPDIR/actual_licenses | grep -v -f $BATS_TEST_TMPDIR/allowed_to_be_missing > $BATS_TEST_TMPDIR/actual_missing_licenses || true

  if [[ -s $BATS_TEST_TMPDIR/actual_extra_licenses ]]; then
    echo "Please remove the following extra licenses from $LICENSES_TARGET:"
    cat $BATS_TEST_TMPDIR/actual_extra_licenses
  fi

  if [[ -s $BATS_TEST_TMPDIR/actual_missing_licenses ]]; then
    echo "Please include the missing licenses for the following packages in $LICENSES_TARGET:"
    cat $BATS_TEST_TMPDIR/actual_missing_licenses
  fi

  # Fail if either of the two "extras" or "missing" lists are present. If so,
  # then the user will see the above error messages.
  [[ ! -s $BATS_TEST_TMPDIR/actual_extra_licenses ]] && [[ ! -s $BATS_TEST_TMPDIR/actual_missing_licenses ]]
}

@test "Pip package generated license includes all dependencies' licenses" {
  do_external_licenses_check \
    "//tensorflow/tools/pip_package:build_pip_package" \
    "//tensorflow/tools/pip_package:licenses"
}

@test "Libtensorflow generated license includes all dependencies' licenses" {
  do_external_licenses_check \
    "//tensorflow:libtensorflow.so" \
    "//tensorflow/tools/lib_package:clicenses_generate"
}

@test "Java library generated license includes all dependencies' licenses" {
  do_external_licenses_check \
    "//tensorflow/java:libtensorflow_jni.so" \
    "//tensorflow/tools/lib_package:jnilicenses_generate"
}

# This test ensures that all the targets built into the Python package include
# their dependencies. It's a rewritten version of the "smoke test", an older
# Python script that was very difficult to understand. See
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/pip_smoke_test.py
@test "Pip package includes all required //tensorflow dependencies" {
  # grep patterns for packages whose dependencies can be ignored
  cat > $BATS_TEST_TMPDIR/ignore_deps_for_these_packages <<EOF
//tensorflow/lite
//tensorflow/compiler/mlir/lite
//tensorflow/compiler/mlir/tfrt
//tensorflow/core/runtime_fallback
//tensorflow/core/tfrt
//tensorflow/python/kernel_tests/signal
//tensorflow/examples
//tensorflow/tools/android
//tensorflow/python/eager/benchmarks
EOF

  # grep patterns for files and targets which don't need to be in the pip
  # package, ever.
  cat > $BATS_TEST_TMPDIR/ignore_these_deps <<EOF
benchmark
_test$
_test.py$
_test_cpu$
_test_cpu.py$
_test_gpu$
_test_gpu.py$
_test_lib$
//tensorflow/cc/saved_model:saved_model_test_files
//tensorflow/cc/saved_model:saved_model_half_plus_two
//tensorflow:no_tensorflow_py_deps
//tensorflow/tools/pip_package:win_pip_package_marker
//tensorflow/core:image_testdata
//tensorflow/core/lib/lmdb:lmdb_testdata
//tensorflow/core/lib/lmdb/testdata:lmdb_testdata
//tensorflow/core/kernels/cloud:bigquery_reader_ops
//tensorflow/python:extra_py_tests_deps
//tensorflow/python:mixed_precision
//tensorflow/python:tf_optimizer
//tensorflow/python:compare_test_proto_py
//tensorflow/python/framework:test_ops_2
//tensorflow/python/framework:test_file_system.so
//tensorflow/python/debug:grpc_tensorflow_server.par
//tensorflow/python/feature_column:vocabulary_testdata
//tensorflow/python/util:nest_test_main_lib
//tensorflow/lite/experimental/examples/lstm:rnn_cell
//tensorflow/lite/experimental/examples/lstm:rnn_cell.py
//tensorflow/lite/experimental/examples/lstm:unidirectional_sequence_lstm_test
//tensorflow/lite/experimental/examples/lstm:unidirectional_sequence_lstm_test.py
//tensorflow/lite/python:interpreter
//tensorflow/lite/python:interpreter_test
//tensorflow/lite/python:interpreter.py
//tensorflow/lite/python:interpreter_test.py
EOF

  # Get the full list of files and targets which get included into the pip
  # package
  bazel query --keep_going 'deps(//tensorflow/tools/pip_package:build_pip_package)' | sort -u > $BATS_TEST_TMPDIR/pip_deps
  # Find all Python py_test targets not tagged "no_pip" or "manual", excluding
  # any targets in ignored packages. Combine this list of targets into a bazel
  # query list (e.g. the list becomes "target+target2+target3")
  bazel query --keep_going 'kind(py_test, //tensorflow/python/...) - attr("tags", "no_pip|manual", //tensorflow/python/...)' | grep -v -f $BATS_TEST_TMPDIR/ignore_deps_for_these_packages | paste -sd "+" - > $BATS_TEST_TMPDIR/deps
  # Find all one-step dependencies of those tests which are from //tensorflow
  # (since external deps will come from Python-level pip dependencies),
  # excluding dependencies and files that are known to be unneccessary.
  # This creates a list of targets under //tensorflow that are required for
  # TensorFlow python tests.
  bazel query --keep_going "deps($(cat $BATS_TEST_TMPDIR/deps), 1)" | grep "^//tensorflow" | grep -v -f $BATS_TEST_TMPDIR/ignore_these_deps | sort -u > $BATS_TEST_TMPDIR/required_deps


  # Find if any required dependencies are missing from the list of dependencies
  # included in the pip package.
  # (comm: Column 1 is left, Column 2 is right, Column 3 is shared lines)
  comm -2 -3 $BATS_TEST_TMPDIR/required_deps $BATS_TEST_TMPDIR/pip_deps > $BATS_TEST_TMPDIR/missing_deps || true

  if [[ -s $BATS_TEST_TMPDIR/missing_deps ]]; then
    cat <<EOF
One or more test dependencies are not in the pip package.
If these test dependencies need to be in the TensorFlow pip package, please
add them to //tensorflow/tools/pip_package/BUILD. Otherwise, add the no_pip tag
to the test, or change code_check_full.bats in the SIG Build repository. That's
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/code_check_full.bats
Here are the affected tests:
EOF
    while read dep; do
      echo "For dependency $dep:"
      # For every missing dependency, find the tests which directly depend on
      # it, and print that list for debugging. Not really clear if this is
      # helpful since the only examples I've seen are enormous.
      bazel query "rdeps(kind(py_test, $(cat $BATS_TEST_TMPDIR/deps)), $dep, 1)"
    done < $BATS_TEST_TMPDIR/missing_deps
    exit 1
  fi
}

# The Python package is not allowed to depend on any CUDA packages.
@test "Pip package doesn't depend on CUDA" {
  bazel cquery \
    --experimental_cc_shared_library \
    --@local_config_cuda//:enable_cuda \
    "somepath(//tensorflow/tools/pip_package:build_pip_package, " \
    "@local_config_cuda//cuda:cudart + "\
    "@local_config_cuda//cuda:cudart + "\
    "@local_config_cuda//cuda:cuda_driver + "\
    "@local_config_cuda//cuda:cudnn + "\
    "@local_config_cuda//cuda:curand + "\
    "@local_config_cuda//cuda:cusolver + "\
    "@local_config_tensorrt//:tensorrt)" --keep_going > $BATS_TEST_TMPDIR/out

  cat <<EOF
There was a path found connecting //tensorflow/tools/pip_package:build_pip_package
to a banned CUDA dependency. Here's the output from bazel query:
EOF
  cat $BATS_TEST_TMPDIR/out
  [[ ! -s $BATS_TEST_TMPDIR/out ]]
}

@test "Pip package doesn't depend on CUDA for static builds (i.e. Windows)" {
  bazel cquery \
    --experimental_cc_shared_library \
    --@local_config_cuda//:enable_cuda \
    --define framework_shared_object=false \
    "somepath(//tensorflow/tools/pip_package:build_pip_package, " \
    "@local_config_cuda//cuda:cudart + "\
    "@local_config_cuda//cuda:cudart + "\
    "@local_config_cuda//cuda:cuda_driver + "\
    "@local_config_cuda//cuda:cudnn + "\
    "@local_config_cuda//cuda:curand + "\
    "@local_config_cuda//cuda:cusolver + "\
    "@local_config_tensorrt//:tensorrt)" --keep_going > $BATS_TEST_TMPDIR/out

  cat <<EOF
There was a path found connecting //tensorflow/tools/pip_package:build_pip_package
to a banned CUDA dependency when '--define framework_shared_object=false' is set.
This means that a CUDA target was probably included via an is_static condition,
used when targeting platforms like Windows where we build statically instead
of dynamically. Here's the output from bazel query:
EOF
  cat $BATS_TEST_TMPDIR/out
  [[ ! -s $BATS_TEST_TMPDIR/out ]]
}

@test "All tensorflow.org/code links point to real files" {
    for i in $(grep -onI 'https://www.tensorflow.org/code/[a-zA-Z0-9/._-]\+' -r tensorflow); do
        target=$(echo $i | sed 's!.*https://www.tensorflow.org/code/!!g')

        if [[ ! -f $target ]] && [[ ! -d $target ]]; then
            echo "$i" >> errors.txt
        fi
        if [[ -e errors.txt ]]; then
            echo "Broken links found:"
            cat errors.txt
            rm errors.txt
            false
        fi
    done
}

@test "No duplicate files on Windows" {
    cat <<EOF
Please rename files so there are no repeats. For example, README.md and
Readme.md would be the same file on Windows. In this test, you would get a
warning for "readme.md" because it makes everything lowercase. There are
repeats of these filename(s) with different casing:
EOF
    find . | tr '[A-Z]' '[a-z]' | sort | uniq -d | tee $BATS_FILE_TMPDIR/repeats
    [[ ! -s $BATS_FILE_TMPDIR/repeats ]]
}

# It's unclear why, but running this on //tensorflow/... is faster than running
# only on affected targets, usually. There are targets in //tensorflow/lite that
# don't pass --nobuild, so they're on their own.
#
# Although buildifier checks for formatting as well, "bazel build nobuild"
# checks for cross-file issues like bad includes or missing BUILD definitions.
#
# We can't test on the windows toolchains because they're using a legacy
# toolchain format (or something) that specifies the toolchain directly instead
# of as a "repository". They can't be valid on Linux because Linux can't do
# anything with a Windows-only toolchain, and bazel errors if trying to build
# that directory.
@test "bazel nobuild passes on all of TF except TF Lite and win toolchains" {
    bazel build --experimental_cc_shared_library --nobuild --keep_going -- //tensorflow/... -//tensorflow/lite/... -//tensorflow/tools/toolchains/win/... -//tensorflow/tools/toolchains/win_1803/...
}


teardown_file() {
    bazel shutdown
}
