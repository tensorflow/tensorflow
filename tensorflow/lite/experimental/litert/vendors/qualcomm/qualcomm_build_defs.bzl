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

"""Build definitions for QualComm backend."""

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "append_rule_kwargs", "litert_bin", "litert_lib", "make_rpaths")

_QNN_LIBCC_X86_64 = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt/latest:lib/x86_64-linux-clang/libc++.so.1",
    # "//third_party/qairt/latest:lib/x86_64-linux-clang/libc++abi.so.1",
    # copybara:uncomment_end
]  # @unused

# TODO: Make rpaths dynamic with "$(location {})".
_QNN_LIB_RPATHS_X86_64 = [
    # copybara:uncomment_begin(google-only)
    # "third_party/qairt/latest/lib/x86_64-linux-clang",
    # copybara:uncomment_end
]

_QNN_LIB_HTP_X86_64 = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt/latest:lib/x86_64-linux-clang/libQnnHtp.so",
    # copybara:uncomment_end
]

_QNN_LIB_SYSTEM_X86_64 = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt/latest:lib/x86_64-linux-clang/libQnnSystem.so",
    # copybara:uncomment_end
]

def _litert_with_qnn_base(
        litert_rule,
        backend,
        include_system,
        use_custom_libcc,
        **litert_rule_kwargs):
    if backend != "htp":
        fail("Only htp currently supported")

    if use_custom_libcc:
        # TODO: Figure out strategy for custom libcc.
        fail("Custom libcc not yet supported")

    data_x86_64 = []
    data_x86_64.extend(_QNN_LIB_HTP_X86_64)
    if include_system:
        data_x86_64.extend(_QNN_LIB_SYSTEM_X86_64)
    data = select({
        "//tensorflow:linux_x86_64": data_x86_64,
        "//conditions:default": [],
    })

    append_rule_kwargs(
        litert_rule_kwargs,
        data = data,
        linkopts = select({
            "//tensorflow:linux_x86_64": [make_rpaths(_QNN_LIB_RPATHS_X86_64)],
            "//conditions:default": [],
        }),
    )

    litert_rule(**litert_rule_kwargs)

def litert_cc_lib_with_qnn(
        backend = "htp",
        include_system = False,
        use_custom_libcc = False,
        **litert_lib_kwargs):
    """Creates a litert_lib target with QualComm backend dependencies.

    Args:
        backend: The backend to use. Currently only "htp" is supported.
        include_system: Whether to include libQnnSystem.so.
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_lib_kwargs: Keyword arguments passed to litert_lib.
    """
    _litert_with_qnn_base(
        litert_lib,
        backend,
        include_system,
        use_custom_libcc,
        **litert_lib_kwargs
    )

def litert_cc_bin_with_qnn(
        backend = "htp",
        include_system = False,
        use_custom_libcc = False,
        **litert_bin_kwargs):
    """Creates a litert_bin target with QualComm backend dependencies.

    Args:
        backend: The backend to use. Currently only "htp" is supported.
        include_system: Whether to include libQnnSystem.so.
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_bin_kwargs: Keyword arguments passed to litert_bin.
    """
    _litert_with_qnn_base(
        litert_bin,
        backend,
        include_system,
        use_custom_libcc,
        **litert_bin_kwargs
    )
