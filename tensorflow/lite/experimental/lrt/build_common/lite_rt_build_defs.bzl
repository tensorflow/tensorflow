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

_HIDE_ABSL_LD_SCRIPT = "//tensorflow/lite/experimental/lrt/build_common:hide_absl.lds"
_LRT_SO_TEMPLATE = "libLrt{}.so"

def hide_absl_linkopt():
    return "-Wl,--version-script=$(location {})".format(_HIDE_ABSL_LD_SCRIPT)

def lite_rt_cc_lib_and_so(**kwargs):
    """Creates a cc_library target as well as a cc_shared_library target for LRT.

    Hides neccessary symbols from the .so. So target will the same name as
    the library suffixed with "_so". Must pass "shared_lib_name" in additional to
    standard cc_library kwargs, which will be prefixed with "libLrt" and given .so extension.

    Args:
        **kwargs: Starndard cc_library kwargs plus "shared_lib_name".
    """
    shared_lib_name = kwargs.pop("shared_lib_name")

    kwargs["linkstatic"] = True
    native.cc_library(**kwargs)

    vis = kwargs["visibility"]
    cc_lib_name = kwargs["name"]
    so_target_name = "{}_so".format(cc_lib_name)
    cc_lib_target_label = ":{}".format(cc_lib_name)
    shared_lib_name = _LRT_SO_TEMPLATE.format(shared_lib_name)

    native.cc_shared_library(
        name = so_target_name,
        shared_lib_name = shared_lib_name,
        user_link_flags = [hide_absl_linkopt()],
        visibility = vis,
        deps = [cc_lib_target_label],
        additional_linker_inputs = [_HIDE_ABSL_LD_SCRIPT],
    )
