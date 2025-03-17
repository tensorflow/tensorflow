# Copyright 2024 The OpenXLA Authors.
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
# ============================================================================

"""A collection of Bazel aspects that can help detecting dependency violations

The dependency violation detection works by iterating through all targets in XLA
and comparing the applied tags of each target to the tags of all its dependencies.

If a target is tagged `gpu` it means it can only be used in an XLA build with the
GPU backend enabled. Hence all targets that are NOT tagged `gpu` may never depend
on a target that IS tagged `gpu` if we are building XLA with only the CPU backend
enabled.

The Bazel aspect runs after Bazel's analysis phase. That means all `select` expressions
(and its derivatives like the `if_gpu_is_configured` macro) have been evaluated and
the actual build configuration is taken into account.

The easiest way to run the aspect is during a build:
`bazel build --aspects build_tools/dependencies/aspects.bzl%validate_gpu_tag //xla/...`

But a cquery expression also works:
`bazel cquery --aspects build_tools/dependencies/aspects.bzl%validate_gpu_tag //xla/...`

The results are reported as debug prints and need to be fished out of stderr. There
are ways to make it less hacky but the complexity of the aspect would also increase
quite a bit.
"""

DependencyViolationInfo = provider(
    "Internal provider needed by the dependency violation check",
    fields = {
        # We can't access the tags of a dependency through the context, so instead we
        # "send" the tags to the dependee through this provider.
        "tags": "Tags of the dependecy",
    },
)

def _dependency_violation_aspect_impl(_, ctx, tag):
    if not hasattr(ctx.rule.attr, "deps"):
        return [DependencyViolationInfo(tags = ctx.rule.attr.tags)]

    for dep in ctx.rule.attr.deps:
        if DependencyViolationInfo not in dep:
            continue
        dep_tags = dep[DependencyViolationInfo].tags
        if tag in dep_tags and tag not in ctx.rule.attr.tags:
            print("[Violation] {} (not tagged {}) depends on {} (tagged {})".format(
                ctx.label,
                tag,
                dep.label,
                tag,
            ))  # buildifier: disable=print

    return [DependencyViolationInfo(tags = ctx.rule.attr.tags)]

def _gpu_tag_violation_aspect_impl(target, ctx):
    return _dependency_violation_aspect_impl(target, ctx, "gpu")

validate_gpu_tag = aspect(
    implementation = _gpu_tag_violation_aspect_impl,
    attr_aspects = ["deps"],
)

def _cuda_only_tag_violation_aspect_impl(target, ctx):
    return _dependency_violation_aspect_impl(target, ctx, "cuda-only")

validate_cuda_only_tag = aspect(
    implementation = _cuda_only_tag_violation_aspect_impl,
    attr_aspects = ["deps"],
)

def _rocm_only_tag_violation_aspect_impl(target, ctx):
    return _dependency_violation_aspect_impl(target, ctx, "rocm-only")

validate_rocm_only_tag = aspect(
    implementation = _rocm_only_tag_violation_aspect_impl,
    attr_aspects = ["deps"],
)
