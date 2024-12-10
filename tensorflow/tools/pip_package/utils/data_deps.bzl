# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Rule to collect data files.

It recursively traverses `deps` attribute of the target and collects paths to files that are
in `data` attribute. Then it filters all files that do not match the provided extensions.
"""

FilePathInfo = provider(
    "Returns path of selected files.",
    fields = {
        "files": "requested files from data attribute",
    },
)

def _collect_data_aspect_impl(_, ctx):
    files = {}
    extensions = ctx.attr._extensions
    if hasattr(ctx.rule.attr, "data"):
        for data in ctx.rule.attr.data:
            for f in data.files.to_list():
                if not any([f.path.endswith(ext) for ext in extensions]):
                    continue
                if "pypi" in f.path:
                    continue
                files[f] = True

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if dep[FilePathInfo].files:
                for file in dep[FilePathInfo].files.to_list():
                    files[file] = True

    return [FilePathInfo(files = depset(files.keys()))]

collect_data_aspect = aspect(
    implementation = _collect_data_aspect_impl,
    attr_aspects = ["deps"],
    attrs = {
        "_extensions": attr.string_list(
            default = [".so", ".pyd", ".pyi", ".dll", ".dylib", ".lib", ".pd"],
        ),
    },
)

def _collect_data_files_impl(ctx):
    files = []
    for dep in ctx.attr.deps:
        files.extend((dep[FilePathInfo].files.to_list()))
    return [DefaultInfo(files = depset(
        files,
    ))]

collect_data_files = rule(
    implementation = _collect_data_files_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [collect_data_aspect],
        ),
    },
)
