# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================

# Drawn from https://github.com/openxla/iree/blob/0246bbfd7955fcd858f8467182404060ccd2e9ae/build_tools/docker/context/install_bazel.sh

set -euo pipefail

if ! [[ -f .bazelversion ]]; then
  echo "Couldn't find .bazelversion file in current directory" >&2
  exit 1
fi

# TODO(b/277241075): avoid duplicating .bazelversion (repo level from TF, local)
BAZEL_VERSION="$(cat .bazelversion)"

# We could do the whole apt install dance, but this technique works across a
# range of platforms, allowing us to use a single script. See
# https://bazel.build/install/ubuntu#binary-installer

curl --silent --fail --show-error --location \
  "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION?}/bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" \
  --output bazel-installer.sh
chmod +x bazel-installer.sh
./bazel-installer.sh

if [[ "$(bazel --version)" != "bazel ${BAZEL_VERSION}" ]]; then
  echo "Bazel installation failed" >&2
  exit 1
fi
