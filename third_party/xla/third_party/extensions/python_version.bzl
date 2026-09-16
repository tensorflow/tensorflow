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
# =============================================================================

"""
A much simplified version of third_party/py/python_repo.bzl, which generates the "python_version_repo" repo.

This is just to keep the current build compatible with both WORKSPACE and Bzlmod, we may not need this in future.
"""

_PY_VERSION_BZL = """
HERMETIC_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION_KIND = "{py_kind}"
USE_PYWRAP_RULES = {use_pywrap_rules}
# TODO(pcloudy): Figure out how to support requirements_lock in Bzlmod.
REQUIREMENTS = "//:requirements.txt"
"""

def _python_version_repo_impl(repository_ctx):
    version = repository_ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "3.11")
    use_pywrap_rules = bool(
        repository_ctx.os.environ.get("USE_PYWRAP_RULES", False),
    )
    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file(
        "py_version.bzl",
        _PY_VERSION_BZL.format(
            version = version,
            py_kind = "",  # TODO(pcloudy): introduce this value properly.
            use_pywrap_rules = use_pywrap_rules,
        ),
    )

python_version_repo = repository_rule(
    implementation = _python_version_repo_impl,
    environ = [
        "HERMETIC_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION_KIND",
        "USE_PYWRAP_RULES",
    ],
)

python_version_ext = module_extension(
    implementation = lambda mctx: python_version_repo(name = "python_version_repo"),
)
