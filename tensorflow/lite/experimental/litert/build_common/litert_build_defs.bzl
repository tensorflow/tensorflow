# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common LiteRT Build Utilities."""

####################################################################################################
# Util

_LRT_SO_PREFIX = "libLiteRt"
_SO_EXT = ".so"
_SHARED_LIB_SUFFIX = "_so"

# Public

def make_linkopt(opt):
    return "-Wl,{}".format(opt)

def make_rpaths(rpaths):
    return make_linkopt("-rpath={}".format(":".join(rpaths)))

def append_rule_kwargs(rule_kwargs, **append):
    for k, v in append.items():
        append_to = rule_kwargs.pop(k, [])
        append_to += v
        rule_kwargs[k] = append_to

def absolute_label(label, package_name = None):
    """Get the absolute label for a given label.

    Args:
      label: The label to convert to absolute.
      package_name: The package name to use if the label is relative.

    Returns:
      The absolute label.
    """
    if label.startswith("//"):
        if ":" in label:
            return label
        return "%s:%s" % (label, label.rsplit("/", 1)[-1])
    if not package_name:
        package_name = native.package_name()
    if label.startswith(":"):
        return "//%s%s" % (package_name, label)
    if ":" in label:
        return "//%s/%s" % (package_name, label)
    return "//%s:%s" % (package_name, label)

# Private

def _valid_shared_lib_name(name):
    return name.endswith(_SHARED_LIB_SUFFIX)

def _valid_so_name(name):
    return name.startswith(_LRT_SO_PREFIX) and name.endswith(_SO_EXT)

def _make_target_ref(name):
    return ":{}".format(name)

####################################################################################################
# Explicitly Link System Libraries ("ungrte")

_SYS_RPATHS_X86_64 = [
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
]
_SYS_RPATHS_LINKOPT_X86_64 = make_rpaths(_SYS_RPATHS_X86_64)

_SYS_ELF_INTERPRETER_X86_64 = "/lib64/ld-linux-x86-64.so.2"
_SYS_ELF_INTERPRETER_LINKOPT_X86_64 = make_linkopt("--dynamic-linker={}".format(_SYS_ELF_INTERPRETER_X86_64))

####################################################################################################
# Symbol Hiding

_EXPORT_LRT_ONLY_SCRIPT_LINUX = "//tensorflow/lite/experimental/litert/build_common:export_litert_only_linux.lds"
_EXPORT_LRT_ONLY_SCRIPT_DARWIN = "//tensorflow/lite/experimental/litert/build_common:export_litert_only_darwin.lds"
_EXPORT_LRT_ONLY_LINKOPT_LINUX = make_linkopt("--version-script=$(location {})".format(_EXPORT_LRT_ONLY_SCRIPT_LINUX))
_EXPORT_LRT_ONLY_LINKOPT_DARWIN = make_linkopt("-exported_symbols_list,$(location {})".format(_EXPORT_LRT_ONLY_SCRIPT_DARWIN))

def symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        "//tensorflow:debug": [],
        "//conditions:default": [
            # Omit symbol table, for all non debug builds
            "-Wl,-s",
        ],
    })

def export_lrt_only_script():
    return select({
        "//tensorflow:linux_x86_64": [_EXPORT_LRT_ONLY_SCRIPT_LINUX],
        "//tensorflow:android": [_EXPORT_LRT_ONLY_SCRIPT_LINUX],
        "//tensorflow:macos": [_EXPORT_LRT_ONLY_SCRIPT_DARWIN],
        "//tensorflow:ios": [_EXPORT_LRT_ONLY_SCRIPT_DARWIN],
        "//conditions:default": [],
    })

def export_lrt_only_linkopt():
    return select({
        "//tensorflow:linux_x86_64": [_EXPORT_LRT_ONLY_LINKOPT_LINUX],
        "//tensorflow:android": [_EXPORT_LRT_ONLY_LINKOPT_LINUX],
        "//tensorflow:macos": [_EXPORT_LRT_ONLY_LINKOPT_DARWIN],
        "//tensorflow:ios": [_EXPORT_LRT_ONLY_LINKOPT_DARWIN],
        "//conditions:default": [],
    }) + symbol_opts()

_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX = "//tensorflow/lite/experimental/litert/build_common:export_litert_runtime_only_linux.lds"
_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN = "//tensorflow/lite/experimental/litert/build_common:export_litert_runtime_only_darwin.lds"
_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX = make_linkopt("--version-script=$(location {})".format(_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX))
_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN = make_linkopt("-exported_symbols_list,$(location {})".format(_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN))

# TODO b/391390553: Add "-Wl,--no-undefined" to make sure all symbols are defined.
_EXPORT_LRT_COMMON_LINKOPTS_LINUX = [
    "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
    "-Wl,--gc-sections",  # Eliminate unused code and data.
    "-Wl,--as-needed",  # Don't link unused libs.a
]

def export_lrt_runtime_only_script():
    return select({
        "//tensorflow:linux_x86_64": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX],
        "//tensorflow:android": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_LINUX],
        "//tensorflow:macos": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN],
        "//tensorflow:ios": [_EXPORT_LRT_RUNTIME_ONLY_SCRIPT_DARWIN],
        "//conditions:default": [],
    })

def export_lrt_runtime_only_linkopt():
    return select({
        "//tensorflow:linux_x86_64": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX],
        "//tensorflow:android": _EXPORT_LRT_COMMON_LINKOPTS_LINUX + [
            "-Wl,-z,max-page-size=16384",
            _EXPORT_LRT_RUNTIME_ONLY_LINKOPT_LINUX,
        ],
        "//tensorflow:macos": [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN],
        "//tensorflow:ios": [_EXPORT_LRT_RUNTIME_ONLY_LINKOPT_DARWIN],
        "//conditions:default": [],
    }) + symbol_opts()

####################################################################################################
# Macros

# Private

def _litert_base(
        rule,
        ungrte = False,
        **cc_rule_kwargs):
    """
    Base rule for LiteRT targets.

    Args:
      rule: The underlying rule to use (e.g., cc_test, cc_library).
      ungrte: Whether to link against system libraries ("ungrte").
      **cc_rule_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if ungrte:
        append_rule_kwargs(
            cc_rule_kwargs,
            linkopts = select({
                "//tensorflow:linux_x86_64": [_SYS_ELF_INTERPRETER_LINKOPT_X86_64, _SYS_RPATHS_LINKOPT_X86_64],
                "//conditions:default": [],
            }),
        )
    rule(**cc_rule_kwargs)

# Public

def litert_test(
        ungrte = False,
        use_sys_malloc = False,
        **cc_test_kwargs):
    """
    LiteRT test rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      use_sys_malloc: Whether to use the system malloc.
      **cc_test_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if use_sys_malloc:
        # copybara:uncomment cc_test_kwargs["malloc"] = "//base:system_malloc"
        pass

    append_rule_kwargs(
        cc_test_kwargs,
        deps = ["@com_google_googletest//:gtest_main"],
    )

    _litert_base(
        native.cc_test,
        ungrte,
        **cc_test_kwargs
    )

def litert_lib(
        ungrte = False,
        **cc_lib_kwargs):
    """
    LiteRT library rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      **cc_lib_kwargs: Keyword arguments to pass to the underlying rule.
    """
    _litert_base(
        native.cc_library,
        ungrte,
        **cc_lib_kwargs
    )

def litert_bin(
        ungrte = False,
        export_litert_only = False,
        **cc_bin_kwargs):
    """
    LiteRT binary rule.

    Args:
      ungrte: Whether to link against system libraries ("ungrte").
      export_litert_only: Whether to export only LiteRT symbols.
      **cc_bin_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if export_litert_only:
        append_rule_kwargs(
            cc_bin_kwargs,
            linkopts = export_lrt_only_linkopt(),
            deps = export_lrt_only_script(),
        )

    _litert_base(
        native.cc_binary,
        ungrte,
        **cc_bin_kwargs
    )

def litert_dynamic_lib(
        name,
        shared_lib_name,
        so_name,
        export_litert_only = False,
        ungrte = False,
        **cc_lib_kwargs):
    """
    LiteRT dynamic library rule.

    Args:
      name: The name of the library.
      shared_lib_name: The name of the shared library.
      so_name: The name of the shared object file.
      export_litert_only: Whether to export only LiteRT symbols.
      ungrte: Whether to link against system libraries ("ungrte").
      **cc_lib_kwargs: Keyword arguments to pass to the underlying rule.
    """
    if not _valid_shared_lib_name(shared_lib_name):
        fail("\"shared_lib_name\" must end with \"_so\"")
    if not _valid_so_name(so_name):
        fail("\"so_name\" must be \"libLiteRt*.so\"")

    lib_name = name
    cc_lib_kwargs["name"] = lib_name

    lib_target_ref = _make_target_ref(lib_name)

    vis = cc_lib_kwargs.get("visibility", None)

    # Share tags for all targets.
    tags = cc_lib_kwargs.get("tags", [])

    litert_lib(
        ungrte = ungrte,
        **cc_lib_kwargs
    )

    user_link_flags = []
    additional_linker_inputs = []
    if export_litert_only:
        user_link_flags = export_lrt_only_linkopt()
        additional_linker_inputs = export_lrt_only_script()

    native.cc_shared_library(
        name = shared_lib_name,
        shared_lib_name = so_name,
        user_link_flags = user_link_flags,
        additional_linker_inputs = additional_linker_inputs,
        tags = tags,
        visibility = vis,
        deps = [lib_target_ref],
    )

def copy_file(name, src, target, visibility = None):
    input_path = "$(location %s)" % src
    output_path = "$(@D)/" + target

    native.genrule(
        name = name,
        srcs = [src],
        outs = [target],
        visibility = visibility,
        cmd = "cp %s %s" % (input_path, output_path),
    )

def gtest_main_no_heapcheck_deps():
    # copybara:uncomment_begin(google-only)
    # return ["//testing/base/public:gunit_main_no_heapcheck"]
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["@com_google_googletest//:gtest_main"]
    # copybara:comment_end
