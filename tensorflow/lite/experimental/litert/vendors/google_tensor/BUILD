<<<<<<< HEAD:tensorflow/tools/tf_sig_build_dockerfiles/setup.packages.rocm.cs7.sh
#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
=======
# Copyright 2024 Google LLC.
>>>>>>> upstream/master:tensorflow/lite/experimental/litert/vendors/google_tensor/BUILD
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
# setup.packages.sh: Given a list of Ubuntu packages, install them and clean up.
# Usage: setup.packages.sh <package_list.txt>
set -e
set -x

<<<<<<< HEAD:tensorflow/tools/tf_sig_build_dockerfiles/setup.packages.rocm.cs7.sh
yum clean all
yum update -y
PKG_LIST=$(sed -e '/^\s*#.*$/d' -e '/^\s*$/d' "$1" | sort -u)
yum --enablerepo=extras install -y $PKG_LIST
=======
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)
>>>>>>> upstream/master:tensorflow/lite/experimental/litert/vendors/google_tensor/BUILD
