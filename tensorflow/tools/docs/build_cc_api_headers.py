# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Generate C++ reference docs for TensorFlow.org."""
import os
import pathlib
import subprocess

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
flags.DEFINE_string('output_dir', None,
                    ("Use this branch as the root version and don't"
                     ' create in version directory'))

# __file__ is the path to this file
DOCS_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
TENSORFLOW_ROOT = DOCS_TOOLS_DIR.parents[2]


def build_headers(output_dir):
  """Builds the headers files for TF."""
  os.makedirs(output_dir, exist_ok=True)

  # `$ yes | configure`
  yes = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
  configure = subprocess.Popen([TENSORFLOW_ROOT / 'configure'],
                               stdin=yes.stdout,
                               cwd=TENSORFLOW_ROOT)
  configure.communicate()

  subprocess.check_call(['bazel', 'build', 'tensorflow/cc:cc_ops'],
                        cwd=TENSORFLOW_ROOT)
  subprocess.check_call(
      ['cp', '--dereference', '-r', 'bazel-bin', output_dir / 'bazel-genfiles'],
      cwd=TENSORFLOW_ROOT)


def main(argv):
  del argv
  build_headers(pathlib.Path(FLAGS.output_dir))


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
