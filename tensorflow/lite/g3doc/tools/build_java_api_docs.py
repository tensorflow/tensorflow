# Lint as: python3
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
import pathlib
import shutil
import tempfile

from absl import app
from absl import flags

from tensorflow_docs.api_generator import gen_java

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
flags.DEFINE_string('output_dir', '/tmp/lite_api/',
                    ("Use this branch as the root version and don't"
                     ' create in version directory'))

flags.DEFINE_string('site_path', 'lite/api_docs/java',
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
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE_PATH_CORE = REPO_ROOT / 'tensorflow/lite/java/src/main/java'
SOURCE_PATH_SUPPORT = REPO_ROOT / 'tensorflow_lite_support/java/src/java'
SOURCE_PATH_ODML = REPO_ROOT / 'tensorflow_lite_support/odml/java/image/src'

# This (key) ordering is preserved in the TOC output.
SECTION_LABELS = {
    'org.tensorflow.lite': 'Core',
    'org.tensorflow.lite.support': 'Support Library',
    'org.tensorflow.lite.task': 'Task Library',
    # If we ever need other ODML packages, drop the `.image` here.
    'com.google.android.odml.image': 'ODML',
}

EXTERNAL_APIS = {
    'https://developer.android.com': REPO_ROOT / 'android/sdk/api/26.txt'
}


def overlay(from_root: pathlib.Path, to_root: pathlib.Path):
  """Recursively copy from_root/* into to_root/."""
  # When Python3.8 lands, replace with shutil.copytree(..., dirs_exist_ok=True)
  for from_path in from_root.rglob('*'):
    to_path = to_root / from_path.relative_to(from_root)
    if from_path.is_file():
      assert not to_path.exists(), f'{to_path} exists!'
      shutil.copyfile(from_path, to_path)
    else:
      to_path.mkdir(exist_ok=True)


def main(unused_argv):
  with tempfile.TemporaryDirectory() as merge_tmp_dir:
    # Merge the combined API sources into a single location.
    merged_temp_dir = pathlib.Path(merge_tmp_dir)
    overlay(SOURCE_PATH_CORE, merged_temp_dir)
    overlay(SOURCE_PATH_SUPPORT, merged_temp_dir)
    overlay(SOURCE_PATH_ODML, merged_temp_dir)

    gen_java.gen_java_docs(
        package=['org.tensorflow.lite', 'com.google.android.odml'],
        source_path=merged_temp_dir,
        output_dir=pathlib.Path(FLAGS.output_dir),
        site_path=pathlib.Path(FLAGS.site_path),
        section_labels=SECTION_LABELS,
        federated_docs=EXTERNAL_APIS)


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
