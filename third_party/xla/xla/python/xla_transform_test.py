# Copyright 2026 The OpenXLA Authors.
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

from absl.testing import absltest
from xla.python import xla_extension
from xla.python import xla_transform


class XlaTransformTest(absltest.TestCase):

  def test_register_hlo_xla_transform(self):
    stage = xla_transform.PipelineStage.kPreScheduler

    def trivial_transform(module):
      self.assertIsInstance(module, xla_extension.HloModule)
      return True

    xla_transform.RegisterHloXlaTransform(
        stage, "python_test_transform", trivial_transform
    )


if __name__ == "__main__":
  absltest.main()
