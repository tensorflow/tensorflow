# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    tf_http_archive(
        name = "rules_proto",
        sha256 = "14a225870ab4e91869652cfd69ef2028277fc1dc4910d65d353b62d6e0ae21f4",
        strip_prefix = "rules_proto-7.1.0",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/7.1.0.tar.gz",
        ),
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.9.0/bazel-skylib-1.9.0.tar.gz",
        ),
    )

    tf_http_archive(
        name = "rules_license",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        ),
        sha256 = "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
    )

    tf_http_archive(
        name = "rules_pkg",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        ),
        sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
    )

    tf_http_archive(
        name = "bazel_features",
        sha256 = "4fd9922d464686820ffd8fcefa28ccffa147f7cdc6b6ac0d8b07fde565c65d66",
        strip_prefix = "bazel_features-1.25.0",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz"),
    )

    tf_http_archive(
        name = "rules_cc",
        sha256 = "69e05df29f0010ba248ef8dafc1f084c8fd2f5c553da634422d8167f5c4b277b",
        strip_prefix = "rules_cc-0.2.20",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_cc/releases/download/0.2.20/rules_cc-0.2.20.tar.gz",
        ),
    )

    # Toolchains for ML projects hermetic builds.
    # Details: https://github.com/google-ml-infra/rules_ml_toolchain
    tf_http_archive(
        name = "rules_ml_toolchain",
        sha256 = "66d823b6c6b4ea6f09d79ef6fe49e181e5301759ee07d176a6c82a01e1227d6b",
        strip_prefix = "rules_ml_toolchain-e8709f15382e4da1de5ca15672cb4412da3d989d",
        urls = tf_mirror_urls(
            "https://github.com/google-ml-infra/rules_ml_toolchain/archive/e8709f15382e4da1de5ca15672cb4412da3d989d.tar.gz",
        ),
    )

    # Maven dependencies.
    RULES_JVM_EXTERNAL_TAG = "4.3"
    tf_http_archive(
        name = "rules_jvm_external",
        strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
        sha256 = "6274687f6fc5783b589f56a2f1ed60de3ce1f99bc4e8f9edef3de43bdf7c6e74",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG),
    )

    # Platforms
    tf_http_archive(
        name = "platforms",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        ),
        sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    )

    # clang-tidy checks.
    tf_http_archive(
        name = "bazel_clang_tidy",
        strip_prefix = "bazel_clang_tidy-c4d35e0d0b838309358e57a2efed831780f85cd0",
        sha256 = "96da6e935ccc91045cf928dbc57f22508a2729c51f7fb3f56178017b0deb9b3c",
        urls = tf_mirror_urls("https://github.com/erenon/bazel_clang_tidy/archive/c4d35e0d0b838309358e57a2efed831780f85cd0.tar.gz"),
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace3 = workspace
