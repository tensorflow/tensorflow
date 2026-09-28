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

"""Support for working with patterns and matching."""

def Pattern(pattern):
    """Compiles pattern into a Pattern struct.

    Args:
        pattern: Bazel Target pattern

    Returns:
        Pattern struct
    """
    if pattern.endswith("/..."):
        return struct(
            label = Label(pattern.removesuffix("/...")),
            subpackages = True,
        )
    return struct(
        label = Label(pattern),
        subpackages = False,
    )

def compile_patterns(patterns):
    """Compiles each string into a Pattern struct.

    Args:
        patterns: Iterable of Bazel Target pattern strings

    Returns:
        List of Pattern structs
    """
    return [Pattern(pattern) for pattern in patterns]

def matches(pattern, label):
    """Checks if patterns includes label.

    Args:
        pattern: A Pattern struct
        label: Bazel Label object

    Returns:
        True if pattern includes label, False otherwise
    """
    if pattern.label.workspace_name != label.workspace_name:
        return False
    if pattern.subpackages:
        return label.package == pattern.label.package or label.package.startswith(pattern.label.package + "/")
    if pattern.label.package != label.package:
        return False
    if pattern.label.name == "all" or pattern.label.name == "*":
        return True
    return pattern.label.name == label.name

def any_match(patterns, label):
    """Whether any Pattern in patterns include labels.

    Args:
        patterns: An iterable of Pattern structs
        label: Bazel Label object

    Returns:
        True if any pattern includes label, False otherwise
    """
    for pattern in patterns:
        if matches(pattern, label):
            return True
    return False
