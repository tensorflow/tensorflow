# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Generate TensorFlow Lite Java reference docs for TensorFlow.org."""
import os
import pathlib
import shutil
import tempfile
from typing import Iterable, Union

from absl import app
from absl import flags

from tensorflow_docs.api_generator import gen_java

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
_OUT_DIR = flags.DEFINE_string('output_dir', '/tmp/lite_api/',
                               "Use this branch as the root version and don't "
                               'create in version directory')

_SITE_PATH = flags.DEFINE_string('site_path', 'lite/api_docs/java',
                                 'Path prefix in the _toc.yaml')

flags.DEFINE_string('code_url_prefix', None,
                    '[UNUSED] The url prefix for links to code.')

flags.DEFINE_bool(
    'search_hints', True,
    '[UNUSED] Include metadata search hints in the generated files')

# This tool expects to live within a directory structure that hosts multiple
# repositories, like so:
# /root (name not important)
#   /tensorflow (github.com/tensorflow/tensorflow - home of this script)
#   /tensorflow_lite_support (github.com/tensorflow/tflite-support)
#   /android/sdk (android.googlesource.com/platform/prebuilts/sdk
#     - Note that this needs a branch with an api/ dir, such as *-release)
# Internally, both the monorepo and the external build system do this for you.
SOURCE_PATH_CORE = pathlib.Path('tensorflow/lite/java/src/main/java')
SOURCE_PATH_SUPPORT = pathlib.Path('tensorflow_lite_support/java/src/java')
SOURCE_PATH_METADATA = pathlib.Path(
    'tensorflow_lite_support/metadata/java/src/java')
SOURCE_PATH_ODML = pathlib.Path('tensorflow_lite_support/odml/java/image/src')
SOURCE_PATH_ANDROID_SDK = pathlib.Path('android/sdk/api/26.txt')

# This (key) ordering is preserved in the TOC output.
SECTION_LABELS = {
    'org.tensorflow.lite': 'Core',
    'org.tensorflow.lite.support': 'Support Library',
    'org.tensorflow.lite.task': 'Task Library',
    # If we ever need other ODML packages, drop the `.image` here.
    'com.google.android.odml.image': 'ODML',
}

EXTERNAL_APIS = {'https://developer.android.com': SOURCE_PATH_ANDROID_SDK}


def overlay(from_root: Union[str, os.PathLike[str]],
            to_root: Union[str, os.PathLike[str]]) -> None:
  """Recursively copy from_root/* into to_root/."""
  shutil.copytree(from_root, to_root, dirs_exist_ok=True)


def resolve_nested_dir(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
  """Returns the path that exists, out of foo/... and foo/foo/..., with root."""
  nested = path.parts[0] / path
  if (root_path := root / path).exists():
    return root_path
  elif (root_nested_path := root / nested).exists():
    return root_nested_path
  raise ValueError(f'Could not find {path} or {nested}')


def exists_maybe_nested(paths: Iterable[pathlib.Path],
                        root: pathlib.Path) -> bool:
  """Evaluates whether all paths exist, either as-is, or nested."""
  # Due to differing directory structures between GitHub & Google, we need to
  # check if a path exists as-is, or with the first section repeated.
  for path in paths:
    try:
      resolve_nested_dir(path, root)
    except ValueError:
      return False
  return True


def main(unused_argv):
  root = pathlib.Path(__file__).resolve()
  all_deps = [SOURCE_PATH_CORE, SOURCE_PATH_SUPPORT, SOURCE_PATH_ODML]
  # Keep searching upwards for a root that hosts the various dependencies. We
  # test `root.name` to ensure we haven't hit /.
  while root.name and not (exists := exists_maybe_nested(all_deps, root)):
    root = root.parent
  if not exists:
    raise FileNotFoundError('Could not find dependencies.')

  with tempfile.TemporaryDirectory() as merge_tmp_dir:
    # Merge the combined API sources into a single location.
    merged_temp_dir = pathlib.Path(merge_tmp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_CORE, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_SUPPORT, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_METADATA, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_ODML, root), merged_temp_dir)

    gen_java.gen_java_docs(
        package=['org.tensorflow.lite', 'com.google.android.odml'],
        source_path=merged_temp_dir,
        output_dir=pathlib.Path(_OUT_DIR.value),
        site_path=pathlib.Path(_SITE_PATH.value),
        section_labels=SECTION_LABELS,
        federated_docs={k: root / v for k, v in EXTERNAL_APIS.items()})


if __name__ == '__main__':
  app.run(main)
