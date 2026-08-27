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

"""TODO(jakeharmon): Write module docstring."""

load("@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl", "use_pywrap_rules")
load("@local_config_syslibs//:build_defs.bzl", "if_system_lib")

def tf_system_libs_linkopts():
    """Returns linker flags for system libraries configured via TF_SYSTEM_LIBS."""
    return (
        if_system_lib("boringssl", ["-lssl", "-lcrypto"]) +
        if_system_lib("com_github_googlecloudplatform_google_cloud_cpp", ["-lgoogle_cloud_cpp_common", "-lgoogle_cloud_cpp_bigtable", "-lgoogle_cloud_cpp_storage"]) +
        if_system_lib("com_github_grpc_grpc", ["-lgrpc++", "-lgrpc", "-lgpr"]) +
        if_system_lib("com_google_protobuf", ["-lprotobuf"]) +
        if_system_lib("com_googlesource_code_re2", ["-lre2"]) +
        if_system_lib("curl", ["-lcurl"]) +
        if_system_lib("flatbuffers", ["-lflatbuffers"]) +
        if_system_lib("gif", ["-lgif"]) +
        if_system_lib("hwloc", ["-lhwloc"]) +
        if_system_lib("icu", ["-licui18n", "-licuuc", "-licudata"]) +
        if_system_lib("jsoncpp_git", ["-ljsoncpp"]) +
        if_system_lib("libjpeg_turbo", ["-ljpeg"]) +
        if_system_lib("org_sqlite", ["-lsqlite3"]) +
        if_system_lib("png", ["-lpng"]) +
        if_system_lib("snappy", ["-lsnappy"]) +
        if_system_lib("zlib", ["-lz"])
    )

# unused in TSL
def tf_additional_plugin_deps():
    return select({
        str(Label("@xla//xla/tsl:with_xla_support")): [
            str(Label("//tensorflow/compiler/jit")),
        ],
        "//conditions:default": [],
    })

def if_dynamic_kernels(extra_deps, otherwise = []):
    # TODO(b/356020232): remove after migration is done
    if use_pywrap_rules():
        return otherwise

    return select({
        str(Label("//tensorflow:dynamic_loaded_kernels")): extra_deps,
        "//conditions:default": otherwise,
    })
