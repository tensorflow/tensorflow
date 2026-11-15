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
# =============================================================================

"""
A much simplified version of third_party/py/python_repo.bzl, which generates the "python_version_repo" repo.

This is just to keep the current build compatible with both WORKSPACE and Bzlmod, we may not need this in future.
"""

_PY_VERSION_BZL = """
HERMETIC_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION_KIND = "{py_kind}"
USE_PYWRAP_RULES = {use_pywrap_rules}
MACOSX_DEPLOYMENT_TARGET = "{macosx_deployment_target}"
WHEEL_COLLAB = "{wheel_collab}"
WHEEL_NAME = "{wheel_name}"
REQUIREMENTS = "{requirements_lock}"
"""

def _python_version_repo_impl(repository_ctx):
    version = repository_ctx.os.environ.get(
        "HERMETIC_PYTHON_VERSION",
        repository_ctx.os.environ.get("TF_PYTHON_VERSION", "3.11"),
    )
    py_kind = repository_ctx.os.environ.get("HERMETIC_PYTHON_VERSION_KIND", "")
    for suffix in ("-freethreaded", "-ft"):
        if version.endswith(suffix):
            version = version[:-len(suffix)]
            py_kind = "freethreaded"
            break
    if py_kind == "ft":
        py_kind = "freethreaded"
    wheel_name = repository_ctx.os.environ.get("WHEEL_NAME", "tensorflow")
    wheel_collab = repository_ctx.os.environ.get("WHEEL_COLLAB", False)
    macosx_deployment_target = repository_ctx.os.environ.get("MACOSX_DEPLOYMENT_TARGET", "")

    use_pywrap_rules = (
        repository_ctx.os.environ.get("USE_PYWRAP_RULES", "").lower() in ("true", "1")
    )
    requirements_lock = "//:requirements_lock_{}{}.txt".format(
        version.replace(".", "_"),
        "_freethreaded" if py_kind == "freethreaded" else "",
    )
    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file(
        "py_version.bzl",
        _PY_VERSION_BZL.format(
            version = version,
            py_kind = py_kind,
            use_pywrap_rules = use_pywrap_rules,
            wheel_name = wheel_name,
            wheel_collab = wheel_collab,
            macosx_deployment_target = macosx_deployment_target,
            requirements_lock = requirements_lock,
        ),
    )

python_version_repo = repository_rule(
    implementation = _python_version_repo_impl,
    environ = [
        "HERMETIC_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION_KIND",
        "MACOSX_DEPLOYMENT_TARGET",
        "TF_PYTHON_VERSION",
        "USE_PYWRAP_RULES",
        "WHEEL_COLLAB",
        "WHEEL_NAME",
    ],
)
