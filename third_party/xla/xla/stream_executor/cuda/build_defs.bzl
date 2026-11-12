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

"""
This module contains custom build rules for CUDA assembly compiler tests.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss")

# Internally this loads a macro, but in OSS this is a function
# buildifier: disable=out-of-order-load
def register_extension_info(**_kwargs):
    pass

visibility(DEFAULT_LOAD_VISIBILITY + stream_executor_friends())

def _stage_in_bin_subdirectory_impl(ctx):
    if len(ctx.files.data) != 1:
        fail("Expected exactly one data dependency.")
    symlinks = {}
    symlinks["bin/" + ctx.label.name] = ctx.files.data[0]
    return [DefaultInfo(
        runfiles = ctx.runfiles(symlinks = symlinks),
    )]

# This rules takes a data dependency and makes it available under bin/<rule_name> in the runfiles
# directory. This is useful for some of our CUDA logic which expects to find binaries in a bin/
# subdirectory.
stage_in_bin_subdirectory = rule(
    implementation = _stage_in_bin_subdirectory_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
    },
)

def _find_objcopy(cc_toolchain):
    if cc_toolchain.objcopy_executable:
        return cc_toolchain.objcopy_executable
    compiler_dir = cc_toolchain.compiler_executable.rsplit("/", 1)[0] + "/"
    fallback = None
    for f in cc_toolchain.all_files.to_list():
        if f.basename == "llvm-objcopy":
            if f.path.startswith(compiler_dir):
                return f
            if not fallback:
                fallback = f
    return fallback or "/usr/bin/objcopy"

def _redefine_obj(ctx, cc_toolchain, objcopy, obj, register_with_cudart):
    out = ctx.actions.declare_file(ctx.label.name + "/" + obj.short_path)
    args = ctx.actions.args()
    if register_with_cudart:
        args.add_all([
            "--redefine-sym=__cudaRegisterFatBinary=__xla_cudart_cudaRegisterFatBinary",
            "--redefine-sym=__cudaRegisterFunction=__xla_cudart_cudaRegisterFunction",
        ])
    else:
        args.add_all([
            "--redefine-sym=__cudaRegisterFatBinary=__xla_isolated_cudaRegisterFatBinary",
            "--redefine-sym=__cudaRegisterFunction=__xla_isolated_cudaRegisterFunction",
            "--redefine-sym=__cudaRegisterFatBinaryEnd=__xla_isolated_cudaRegisterFatBinaryEnd",
            "--redefine-sym=__cudaUnregisterFatBinary=__xla_isolated_cudaUnregisterFatBinary",
            "--redefine-sym=__cudaRegisterVar=__xla_isolated_cudaRegisterVar",
            "--redefine-sym=__cudaRegisterManagedVar=__xla_isolated_cudaRegisterManagedVar",
            "--redefine-sym=__cudaInitModule=__xla_isolated_cudaInitModule",
        ])
    args.add(obj)
    args.add(out)
    ctx.actions.run(
        executable = objcopy,
        arguments = [args],
        inputs = depset(
            direct = [obj],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [out],
        mnemonic = "RedefineCudaRegisterSyms",
    )
    return out

def _redefine_cuda_register_syms_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    objcopy = _find_objcopy(cc_toolchain)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    src_cc_info = ctx.attr.src[CcInfo]
    src_name = ctx.attr.src.label.name

    new_objs = []
    new_pic_objs = []
    for li in src_cc_info.linking_context.linker_inputs.to_list():
        if li.owner.package == ctx.label.package and (
            li.owner.name == src_name or li.owner.name == src_name + "_impl"
        ):
            for lib in li.libraries:
                for obj in (lib.objects or []):
                    new_objs.append(_redefine_obj(ctx, cc_toolchain, objcopy, obj, ctx.attr.register_with_cudart))
                for pic_obj in (lib.pic_objects or []):
                    new_pic_objs.append(_redefine_obj(ctx, cc_toolchain, objcopy, pic_obj, ctx.attr.register_with_cudart))

    compilation_outputs = cc_common.create_compilation_outputs(
        objects = depset(new_objs),
        pic_objects = depset(new_pic_objs),
    )
    dep_linking_contexts = [
        dep[CcInfo].linking_context
        for dep in ctx.attr.deps
        if CcInfo in dep
    ]
    (linking_context, _) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = dep_linking_contexts,
        disallow_dynamic_library = ctx.attr.linkstatic,
        alwayslink = ctx.attr.alwayslink,
    )

    return [
        CcInfo(
            compilation_context = src_cc_info.compilation_context,
            linking_context = linking_context,
        ),
    ]

_redefine_cuda_register_syms = rule(
    implementation = _redefine_cuda_register_syms_impl,
    attrs = {
        "alwayslink": attr.bool(default = False),
        "deps": attr.label_list(providers = [CcInfo]),
        "linkstatic": attr.bool(default = False),
        "register_with_cudart": attr.bool(default = True),
        "src": attr.label(mandatory = True, providers = [CcInfo]),
    } | CC_TOOLCHAIN_ATTRS,
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    provides = [CcInfo],
)

def embeddable_cuda_library(name, register_with_cudart = True, **kwargs):
    """Wrapper around cuda_library that rewrites __cudaRegister* symbols for kernel registration.

    Args:
        name: Target name.
        register_with_cudart: If True (the default), kernels are registered with
          both XLA's CUDA runtime kernel registry and libcudart, so they can
          also be launched with <<<...>>> syntax. If False, kernels are only
          registered with XLA's CUDA runtime kernel registry and never
          with libcudart.
        **kwargs: Additional arguments to pass to cuda_library.
    """
    if not register_with_cudart and kwargs.get("rdc"):
        fail("embeddable_cuda_library does not support rdc = True with register_with_cudart = False")

    deps = kwargs.pop("deps", [])
    implicit_deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:cudart_kernel_registry",
    ]
    if register_with_cudart:
        implicit_deps.append("//xla/tsl/cuda:cudart")

    if type(deps) == "list":
        link_deps = list(deps)
        for dep in implicit_deps:
            short_dep = ":" + dep.split(":")[1]
            if dep not in link_deps and short_dep not in link_deps:
                link_deps.append(dep)
    else:
        link_deps = implicit_deps + deps
    kwargs["deps"] = link_deps

    copts = kwargs.pop("copts", [])
    no_compress_opts = if_oss(["-no-compress", "-Xcuda-fatbinary=--compress=false"])
    kwargs["copts"] = copts + no_compress_opts
    kwargs.update(if_google({"cuda_compress": False}, {}))

    features = kwargs.pop("features", []) + ["-thin_lto", "cfi_opt_out"]

    raw_name = name + "_raw"
    redefined_name = name + "_redefined"
    hdrs = kwargs.pop("hdrs", None)
    textual_hdrs = kwargs.pop("textual_hdrs", None)
    include_prefix = kwargs.pop("include_prefix", None)
    strip_include_prefix = kwargs.pop("strip_include_prefix", None)
    linkopts = kwargs.get("linkopts", None)
    visibility = kwargs.pop("visibility", None)
    alwayslink = kwargs.get("alwayslink", False)
    linkstatic = kwargs.get("linkstatic", False)
    testonly = kwargs.get("testonly", None)
    compatible_with = kwargs.get("compatible_with", None)
    target_compatible_with = kwargs.get("target_compatible_with", None)
    tags = kwargs.get("tags", None)
    cuda_library(
        name = raw_name,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        features = features,
        visibility = ["//visibility:private"],
        **kwargs
    )
    _redefine_cuda_register_syms(
        name = redefined_name,
        src = ":" + raw_name,
        deps = link_deps,
        alwayslink = alwayslink,
        linkstatic = linkstatic,
        register_with_cudart = register_with_cudart,
        features = features,
        visibility = ["//visibility:private"],
        testonly = testonly,
        compatible_with = compatible_with,
        target_compatible_with = target_compatible_with,
        tags = tags,
    )
    cc_library(
        name = name,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        deps = [":" + redefined_name] + link_deps,
        linkopts = linkopts,
        alwayslink = alwayslink,
        linkstatic = linkstatic,
        features = features,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        target_compatible_with = target_compatible_with,
        tags = tags,
    )

register_extension_info(
    extension = embeddable_cuda_library,
    label_regex_for_dep = "{extension_name}_raw",
)
