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

"""XLA package_group definitions"""

def xla_package_groups(name = "xla_package_groups"):
    """Defines visibility groups for XLA.

    Args:
     name: package groups name
    """

    native.package_group(
        name = "friends",
        packages = ["//..."],
    )

    native.package_group(
        name = "internal",
        packages = ["//..."],
    )

    native.package_group(
        name = "backends",
        packages = ["//..."],
    )

    native.package_group(
        name = "codegen",
        packages = ["//..."],
    )

    native.package_group(
        name = "collectives",
        packages = ["//..."],
    )

    native.package_group(
        name = "runtime",
        packages = ["//..."],
    )

def xla_test_friend_package_group(name):
    """Defines visibility group for XLA tests.

    Args:
     name: package group name
    """

    native.package_group(
        name = name,
        packages = ["//..."],
    )
