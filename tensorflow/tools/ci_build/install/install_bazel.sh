#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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

# Select bazel version.
BAZEL_VERSION="0.1.1"

# Install bazel.
mkdir /bazel
cd /bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt
chmod +x /bazel/bazel-*.sh
/bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Enable bazel auto completion.
echo "source /usr/local/lib/bazel/bin/bazel-complete.bash" >> ~/.bashrc

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
echo "startup --batch" >>/root/.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/root/.bazelrc
# Force bazel output to use colors (good for jenkins).
echo "common --color=yes" >>/root/.bazelrc
# Configure tests - increase timeout, print errors and timeout warnings
echo "test" \
    " --test_timeout=3600" \
    " --test_output=errors" \
    " --test_verbose_timeout_warnings" \
    >>/root/.bazelrc
