#!/usr/bin/env bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# explicitly call third_party_downloads since we need pigweed for the license
# and clang-format checks.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Explicitly disable exit on error so that we can report all the style errors in
# one pass and clean up the temporary git repository even when one of the
# scripts fail with an error code.
set +e

# The pigweed scripts only work from a git repository and the Tensorflow CI
# infrastructure does not always guarantee that. As an ugly workaround, we
# create our own git repo when running on the CI servers.
pushd tensorflow/lite/
if [[ ${1} == "PRESUBMIT" ]]; then
  git init .
  git config user.email "tflm@google.com"
  git config user.name "TensorflowLite Micro"
  git add *
  git commit -a -m "Commit for a temporary repository." > /dev/null
fi

############################################################
# License Check
############################################################
micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/pigweed_presubmit.py \
  kernels/internal/reference/ \
  micro/ \
  -p copyright_notice \
  -e kernels/internal/reference/integer_ops/ \
  -e kernels/internal/reference/reference_ops.h \
  -e tools/make/downloads \
  -e tools/make/targets/ecm3531 \
  -e BUILD\
  -e leon_commands \
  -e "\.bzl" \
  -e "\.h5" \
  -e "\.ipynb" \
  -e "\.inc" \
  -e "\.patch" \
  -e "\.properties" \
  -e "\.txt" \
  -e "\.tpl" \
  --output-directory /tmp

LICENSE_CHECK_RESULT=$?

############################################################
# Formatting Check
############################################################
# We are currently ignoring Python files (with yapf as the formatter) because
# that needs additional setup.  We are also ignoring the markdown files to allow
# for a more gradual rollout of this presubmit check.
micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/format_code.py \
  kernels/internal/reference/ \
  micro/ \
  -e kernels/internal/reference/integer_ops/ \
  -e kernels/internal/reference/reference_ops.h \
  -e "\.inc" \
  -e "\.md" \
  -e "\.py"

CLANG_FORMAT_RESULT=$?

#############################################################################
# Avoided specific-code snippets for TFLM
#############################################################################

CHECK_CONTENTS_PATHSPEC=\
"micro "\
":(exclude)micro/tools/ci_build/test_code_style.sh"

# See https://github.com/tensorflow/tensorflow/issues/46297 for more context.
check_contents "gtest|gmock" "${CHECK_CONTENTS_PATHSPEC}" \
  "These matches can likely be deleted."
GTEST_RESULT=$?

# See http://b/175657165 for more context.
ERROR_REPORTER_MESSAGE=\
"TF_LITE_REPORT_ERROR should be used instead, so that log strings can be "\
"removed to save space, if needed."

check_contents "error_reporter.*Report\(|context->ReportError\(" \
  "${CHECK_CONTENTS_PATHSPEC}" "${ERROR_REPORTER_MESSAGE}"
ERROR_REPORTER_RESULT=$?

# See http://b/175657165 for more context.
ASSERT_PATHSPEC=\
"${CHECK_CONTENTS_PATHSPEC}"\
" :(exclude)micro/examples/micro_speech/esp/ringbuf.c"\
" :(exclude)*\.ipynb"\
" :(exclude)*\.py"\
" :(exclude)*zephyr_riscv/Makefile.inc"

check_contents "\<assert\>" "${ASSERT_PATHSPEC}" \
  "assert should not be used in TFLM code.."
ASSERT_RESULT=$?

###########################################################################
# All checks are complete, clean up.
###########################################################################

popd
if [[ ${1} == "PRESUBMIT" ]]; then
  rm -rf tensorflow/lite/.git
fi

# Re-enable exit on error now that we are done with the temporary git repo.
set -e

if [[ ${LICENSE_CHECK_RESULT}  != 0 || \
      ${CLANG_FORMAT_RESULT}   != 0 || \
      ${GTEST_RESULT}          != 0 || \
      ${ERROR_REPORTER_RESULT} != 0 || \
      ${ASSERT_RESULT}         != 0    \
   ]]
then
  exit 1
fi
