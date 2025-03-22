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

"""Build definitions for Mediatek backend."""

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "append_rule_kwargs", "litert_bin", "litert_lib", "make_rpaths")

_MTK_STD_LIBS_HOST = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/neuro_pilot:latest/host/lib/libc++.so.1",
    # "//third_party/neuro_pilot:latest/host/lib/libstdc++.so.6",
    # copybara:uncomment_end
]  # @unused

_MTK_NEURON_ADAPTER_SO = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/neuro_pilot:latest/host/lib/libneuron_adapter.so",
    # copybara:uncomment_end
]

# TODO: Make rpaths dynamic with "$(location {})".
_MTK_HOST_RPATHS = [
    # copybara:uncomment_begin(google-only)
    # "third_party/neuro_pilot/latest/host/lib",
    # copybara:uncomment_end
]

def _litert_with_mtk_base(
        litert_rule,
        use_custom_libcc,
        **litert_rule_kwargs):
    if use_custom_libcc:
        # TODO: Figure out strategy for custom libcc.
        fail("Custom libcc not yet supported")

    data_x86_64 = []
    data_x86_64.extend(_MTK_NEURON_ADAPTER_SO)
    append_rule_kwargs(
        litert_rule_kwargs,
        data = select({
            "//tensorflow:linux_x86_64": data_x86_64,
            "//conditions:default": [],
        }),
        linkopts = select({
            "//tensorflow:linux_x86_64": [make_rpaths(_MTK_HOST_RPATHS)],
            "//conditions:default": [],
        }),
    )

    litert_rule(**litert_rule_kwargs)

def litert_cc_lib_with_mtk(
        use_custom_libcc = False,
        **litert_lib_kwargs):
    """Creates a litert_lib target with Mediatek backend dependencies.

    Args:
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_lib_kwargs: Keyword arguments passed to litert_lib.
    """
    _litert_with_mtk_base(
        litert_lib,
        use_custom_libcc,
        **litert_lib_kwargs
    )

def litert_cc_bin_with_mtk(
        use_custom_libcc = False,
        **litert_bin_kwargs):
    """Creates a litert_bin target with Mediatek backend dependencies.

    Args:
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **litert_bin_kwargs: Keyword arguments passed to litert_bin.
    """
    _litert_with_mtk_base(
        litert_bin,
        use_custom_libcc,
        **litert_bin_kwargs
    )
