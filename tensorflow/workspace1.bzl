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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_google_benchmark//:bazel/benchmark_deps.bzl", "benchmark_deps")
load("@grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@llvm-raw//utils/bazel:linux_uapi.bzl", "linux_uapi_setup")
load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("@xla//third_party/llvm:setup.bzl", "llvm_setup")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/android:android_configure.bzl", "android_configure")

_PYYAML_CONTENT = """\
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for PyYAML)
    licenses = ["notice"],
)

py_library(
    name = "yaml",
    srcs = glob(["yaml/*.py"]),
)
"""

# buildifier: disable=unnamed-macro
def workspace(with_rules_cc = True):
    """Loads a set of TensorFlow dependencies. To be used in a WORKSPACE file.

    Args:
      with_rules_cc: Unused, to be removed soon.
    """
    linux_uapi_setup(name = "linux_uapi")
    http_archive(
        name = "pyyaml",
        urls = tf_mirror_urls(
            "https://github.com/yaml/pyyaml/archive/refs/tags/5.1.zip",
        ),
        sha256 = "f0a35d7f282a6d6b1a4f3f3965ef5c124e30ed27a0088efb97c0977268fd671f",
        strip_prefix = "pyyaml-5.1/lib3",
        build_file_content = _PYYAML_CONTENT,
    )

    llvm_setup(name = "llvm-project")
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()
    if "cc_compatibility_proxy" not in native.existing_rules():
        compatibility_proxy_repo()

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
