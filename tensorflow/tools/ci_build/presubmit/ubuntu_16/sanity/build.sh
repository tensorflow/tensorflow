#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

set -e

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

# From this point on, logs can be publicly available
set -x

function install_pylint () {
  # TODO(gunan): figure out why we get stuck with later versions of pylint.
  # TODO(mihaimaruseac): this is used in the release build in the same way,
  # maybe extract out to a common?
  sudo python2 -m pip install pylint==1.6.4
  sudo python3 -m pip install pylint==1.6.4
}

function run_sanity_checks () {
  SANITY_OUT=ci_sanity.out
  SANITY_OUT_TARGET=gen_ci_sanity_out

  # Run tensorflow sanity checks.
  tensorflow/tools/ci_build/ci_sanity.sh 2>&1 | tee tensorflow/tools/ci_build/${SANITY_OUT}
  RC=${PIPESTATUS[0]}

  # Since we are running the sanity build remotely (rbe), we need to build a bazel
  # target that would output the log generated above and return the expected
  # error code.
  cat << EOF > tensorflow/tools/ci_build/BUILD
package(default_visibility = ["//tensorflow:internal"])

sh_test(
    name = "${SANITY_OUT_TARGET}",
    srcs = ["${SANITY_OUT_TARGET}.sh"],
    data = ["${SANITY_OUT}"],
    tags = ["local"],
)
EOF

  cat << EOF > tensorflow/tools/ci_build/${SANITY_OUT_TARGET}.sh
#!/bin/bash
cat tensorflow/tools/ci_build/${SANITY_OUT}
exit ${RC}
EOF

  # Now trigger the rbe build that outputs the log
  chmod +x tensorflow/tools/ci_build/${SANITY_OUT_TARGET}.sh

  # Run bazel test command. Double test timeouts to avoid flakes.
  # //tensorflow/core:platform_setround_test is not supported. See b/64264700
  "${BAZEL_WRAPPER_PATH}" \
    --host_jvm_args=-Dbazel.DigestFunction=SHA256 \
    test \
    --incompatible_list_based_execution_strategy_selection \
    --test_output=all \
    tensorflow/tools/ci_build:${SANITY_OUT_TARGET}

  # Copy log to output to be available to GitHub
  ls -la "$(bazel info output_base)/java.log"
  cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"
}


source tensorflow/tools/ci_build/release/common.sh
update_bazel_linux
which bazel

install_pylint

run_sanity_checks
