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
# =============================================================================

"""Provides the repo macro to import ml_dtypes.

ml_dtypes provides machine-learning-specific data-types like bfloat16,
float8 varieties, and int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ML_DTYPES_COMMIT = "00d98cd92ade342fef589c0470379abb27baebe9"
    ML_DTYPES_SHA256 = "f6e5880666661351e6cd084ac4178ddc4dabcde7e9a73722981c0d1500cf5937"
    tf_http_archive(
        name = "ml_dtypes_py",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes_py.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.BUILD": "ml_dtypes/BUILD.bazel",
        },
        sha256 = ML_DTYPES_SHA256,
        strip_prefix = "ml_dtypes-{commit}".format(commit = ML_DTYPES_COMMIT),
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/{commit}/ml_dtypes-{commit}.tar.gz".format(commit = ML_DTYPES_COMMIT)),
    )
