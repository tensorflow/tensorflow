#!/usr/bin/env bash
#
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
#
# Usage: collect_additional_wheel.sh <wheel name>
# This script is aware of TFCI_ variables, so it only needs the project name of
# the wheel to collect.
#
# Copies the wheel that the additional-WHEEL_NAME build in wheel.sh just
# produced out of bazel-bin and into TFCI_OUTPUT_DIR, and gives it the same
# platform tag that rename_and_verify_wheels.sh gives the primary wheel.
#
# rename_and_verify_wheels.sh cannot do this itself: it expects exactly one
# wheel to be present, and it has already run against the primary wheel by the
# time this script is called.
set -exo pipefail

wheel_name="$1"

# `find -printf` is a GNU extension, and macOS ships BSD find, so let `ls -t`
# do the sorting instead and take the newest match.
built_wheel=$("$TFCI_FIND_BIN" ./bazel-bin/tensorflow/tools/pip_package \
  -iname "${wheel_name}*.whl" -exec ls -t {} + | head -n 1)
cp "$built_wheel" "$TFCI_OUTPUT_DIR"

# Repair the wheel with auditwheel, the same way the primary wheel was
# repaired. Bazel tags Linux wheels `linux_<arch>`, and PyPI only accepts the
# `manylinux_*` tag that auditwheel puts on in its place.
if [[ "$TFCI_WHL_AUDIT_ENABLE" == "1" ]]; then
  cd "$TFCI_OUTPUT_DIR"
  python3 -m auditwheel repair --plat "$TFCI_WHL_AUDIT_PLAT" --wheel-dir . \
    "$(basename "$built_wheel")"
  # Drop the unrepaired wheel, keeping the newest one, the same way
  # rename_and_verify_wheels.sh does for the primary wheel. If the wheel was
  # already named correctly then auditwheel did not rename it, nothing is left
  # over and `rm` gets no arguments, which is why it needs -f. Matching on the
  # project name leaves the primary wheel alone.
  ls -t "${wheel_name}"-*.whl | tail -n +2 | xargs rm -f
fi
