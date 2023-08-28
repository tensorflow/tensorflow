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
# =============================================================================
"""Binary for generating the API for tensorflow."""

from collections.abc import Sequence

from absl import app
from absl import flags

from tensorflow.python.tools.api.generator2.generator import generator

_OUTPUT_FILES = flags.DEFINE_list(
    'output_files', None, 'List of files expected to generate.', required=True
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Directory where the generated output files are placed. This should be a'
    ' prefix of every directory in "output_files".',
    required=True,
)
_ROOT_INIT_TEMPLATE = flags.DEFINE_string(
    'root_init_template',
    None,
    'Template for top level __init__.py file.  "#API IMPORTS PLACEHOLDER"'
    ' comment will be replaced with imports.',
)
_API_VERSION = flags.DEFINE_integer(
    'apiversion', 2, 'The API version to generate. (1 or 2)'
)
_COMPAT_API_VERSIONS = flags.DEFINE_list(
    'compat_api_versions',
    [],
    'Additional versions to generate in compat/ subdirectory.',
)
_COMPAT_INIT_TEMPLATES = flags.DEFINE_list(
    'compat_init_templates',
    [],
    'Template for top-level __init__.py files under compat modules. This list'
    ' must be in the same order as the list of versions in'
    ' "compat_apiversions".',
)
_OUTPUT_PACKAGE = flags.DEFINE_string(
    'output_package', 'tensorflow', 'Root output package.'
)
_USE_LAZY_LOADING = flags.DEFINE_bool(
    'use_lazy_loading',
    True,
    'If true, lazily load imports rather than loading them all in the'
    ' __init__.py files. Defaults to true.',
)
_PROXY_MODULE_ROOT = flags.DEFINE_string(
    'proxy_module_root',
    None,
    'Module root for proxy-import format. If specified, proxy files with `from'
    ' proxy_module_root.proxy_module import *` will be created to enable import'
    ' resolution under TensorFlow.',
)
_FILE_PREFIXES_TO_STRIP = flags.DEFINE_list(
    'file_prefixes_to_strip',
    [],
    "File prefixes to strip from the import paths. Ex: bazel's bin and genfile"
    ' directories.',
)
_PACKAGES_TO_IGNORE = flags.DEFINE_list(
    'packages_to_ignore',
    [],
    'Comma seperated list of packages to ignore tf_exports from. Ex:'
    ' packages_to_ignore="tensorflow.python.framework.test_ops"'
    ' will not export any tf_exports from test_ops',
)
_MODULE_PREFIX = flags.DEFINE_string(
    'module_prefix',
    '',
    'Prefix to append to all imported modules.'
)


def main(argv: Sequence[str]) -> None:
  if _PROXY_MODULE_ROOT.value:
    generator.generate_proxy_api_files(
        _OUTPUT_FILES.value, _PROXY_MODULE_ROOT.value, _OUTPUT_DIR.value
    )
    return

  for out_file in _OUTPUT_FILES.value:
    with open(out_file, 'w') as f:
      f.write('')

  generator.gen_public_api(
      _OUTPUT_DIR.value,
      _OUTPUT_PACKAGE.value,
      _ROOT_INIT_TEMPLATE.value,
      _API_VERSION.value,
      _COMPAT_API_VERSIONS.value,
      _COMPAT_INIT_TEMPLATES.value,
      _USE_LAZY_LOADING.value,
      _FILE_PREFIXES_TO_STRIP.value,
      argv[1:],
      _PACKAGES_TO_IGNORE.value,
      _MODULE_PREFIX.value
  )


if __name__ == '__main__':
  app.run(main)
