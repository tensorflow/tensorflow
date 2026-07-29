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

"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@com_google_benchmark//:bazel/benchmark_deps.bzl", "benchmark_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("@xla//third_party/llvm:setup.bzl", "llvm_setup")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/android:android_configure.bzl", "android_configure")

# buildifier: disable=unnamed-macro
def workspace(with_rules_cc = True):
    """Loads a set of TensorFlow dependencies. To be used in a WORKSPACE file.

    Args:
      with_rules_cc: Unused, to be removed soon.
    """
    llvm_setup(name = "llvm-project")
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()

    closure_repositories()

    tf_http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
        ),
    )

    android_configure(name = "local_config_android")

    grpc_deps()
    benchmark_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace1 = workspace
