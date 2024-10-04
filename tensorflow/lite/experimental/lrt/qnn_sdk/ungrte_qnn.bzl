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

"""Build utilities for configuring targets with sys linux dependencies and qnn sdk libs."""

def _make_linkopt(opt):
    return "-Wl,{}".format(opt)

def _make_rpaths(rpaths):
    return _make_linkopt("-rpath={}".format(":".join(rpaths)))

def _append_rule_kwargs(rule_kwargs, **append):
    for k, v in append.items():
        append_to = rule_kwargs.pop(k, [])
        append_to += v
        rule_kwargs[k] = append_to

####################################################################################################
# Symbol Hiding

# TODO: Figure out what symbols actually need to be hidden.
SYMBOL_HIDE_LD_SCRIPT = "//tensorflow/lite/experimental/lrt/qnn:lrt.lds"
SYMBOL_HIDE_LINKOPT = _make_linkopt("--version-script=$(location {})".format(SYMBOL_HIDE_LD_SCRIPT))

####################################################################################################
# Custom libc++

QNN_LIBCC = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt:lib/x86_64-linux-clang/libc++.so.1",
    # "//third_party/qairt:lib/x86_64-linux-clang/libc++abi.so.1",
    # copybara:uncomment_end
]

# TODO: Make rpaths dynamic with "$(location {})".
QNN_LIB_RPATHS = [
    # copybara:uncomment_begin(google-only)
    # "third_party/qairt/lib/x86_64-linux-clang",
    # copybara:uncomment_end
]

####################################################################################################
# QNN SDK Lib(s) and settings

QNN_HTP_LINUX_LIBS = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt:lib/x86_64-linux-clang/libQnnHtp.so",
    # copybara:uncomment_end
]
QNN_COMMON_LINKOPTS = ["-ldl"]

####################################################################################################
# ungrte (system libs)

SYS_RPATHS = ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
SYS_ELF_INTERPRETER = "/lib64/ld-linux-x86-64.so.2"
SYS_ELF_INTERPRETER_LINKOPT = _make_linkopt("--dynamic-linker={}".format(SYS_ELF_INTERPRETER))

####################################################################################################
# Private Macros

UNGRTE_TEST_DEPS = [
    "@com_google_googletest//:gtest_main",
]

# TODO: Relax nobuilder and no_oss tags one we assert OS environment works as expected. These
# tags hides targets from OS CI.
UNGRTE_BASE_TAGS = [
    "nobuilder",
    "no_oss",
]

UNGRTE_TEST_TAGS = [
    "no-remote-exec",
    "notap",
]

# TODO: Update this to switch on dependencies/configurations based on target platform. Currently
# Only linux supported.
# TODO: Explicitly set GLIBC version in similar fashion (check which one QNN is tested against).
RESTRICTED_TO = [
    "//third_party/bazel_platforms/os:linux",
    "//third_party/bazel_platforms/cpu:x86_64",
]

def _ungrte_cc_base(rule, rpaths, **cc_rule_kwargs):
    linkopts = [
        _make_rpaths(rpaths + SYS_RPATHS),
        SYS_ELF_INTERPRETER_LINKOPT,
    ]

    _append_rule_kwargs(
        cc_rule_kwargs,
        linkopts = linkopts,
        tags = UNGRTE_BASE_TAGS,
    )

    rule(**cc_rule_kwargs)

def _ungrte_cc_binary(rpaths, **cc_binary_kwargs):
    _ungrte_cc_base(
        native.cc_binary,
        rpaths,
        # copybara:uncomment malloc = "//base:system_malloc",
        **cc_binary_kwargs
    )

def _ungrte_cc_test(rpaths, **cc_test_kwargs):
    _append_rule_kwargs(
        cc_test_kwargs,
        deps = UNGRTE_TEST_DEPS,
        tags = UNGRTE_TEST_TAGS,
    )

    cc_test_kwargs["linkstatic"] = 1
    _ungrte_cc_base(
        native.cc_test,
        rpaths,
        # copybara:uncomment malloc = "//base:system_malloc",
        **cc_test_kwargs
    )

def _ungrte_cc_library(rpaths, **cc_library_kwargs):
    _ungrte_cc_base(native.cc_library, rpaths, **cc_library_kwargs)

def _ungrte_cc_base_with_qnn(ungrte_rule, backend, hide_symbols, use_custom_libcc, **cc_rule_kwargs):
    # Only htp backend currently supported.
    if backend != "htp":
        fail()

    _append_rule_kwargs(
        cc_rule_kwargs,
        data = QNN_HTP_LINUX_LIBS,
        linkopts = QNN_COMMON_LINKOPTS,
    )

    if hide_symbols:
        _append_rule_kwargs(
            cc_rule_kwargs,
            deps = [SYMBOL_HIDE_LD_SCRIPT],
            linkopts = [SYMBOL_HIDE_LINKOPT],
        )

    if use_custom_libcc:
        _append_rule_kwargs(
            cc_rule_kwargs,
            data = QNN_LIBCC,
        )

    ungrte_rule(QNN_LIB_RPATHS, **cc_rule_kwargs)

####################################################################################################
# Public Macros

def ungrte_cc_binary(**cc_binary_kwargs):
    _ungrte_cc_binary([], **cc_binary_kwargs)

def ungrte_cc_test(**cc_test_kwargs):
    _ungrte_cc_test([], **cc_test_kwargs)

def ungrte_cc_library(**cc_library_kwargs):
    _ungrte_cc_library([], **cc_library_kwargs)

def ungrte_cc_binary_with_qnn(
        backend = "htp",
        hide_symbols = False,
        use_custom_libcc = False,
        **cc_binary_kwargs):
    _ungrte_cc_base_with_qnn(
        ungrte_rule = _ungrte_cc_binary,
        backend = backend,
        hide_symbols = hide_symbols,
        use_custom_libcc = use_custom_libcc,
        **cc_binary_kwargs
    )

def ungrte_cc_test_with_qnn(
        backend = "htp",
        hide_symbols = False,
        use_custom_libcc = False,
        **cc_test_kwargs):
    _ungrte_cc_base_with_qnn(
        ungrte_rule = _ungrte_cc_test,
        backend = backend,
        hide_symbols = hide_symbols,
        use_custom_libcc = use_custom_libcc,
        **cc_test_kwargs
    )

def ungrte_cc_library_with_qnn(
        backend = "htp",
        hide_symbols = False,
        use_custom_libcc = False,
        **cc_library_kwargs):
    _ungrte_cc_base_with_qnn(
        ungrte_rule = _ungrte_cc_library,
        backend = backend,
        hide_symbols = hide_symbols,
        use_custom_libcc = use_custom_libcc,
        **cc_library_kwargs
    )
