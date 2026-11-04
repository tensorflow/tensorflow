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
"""Tests the Python HLO runner's FFI registration and execution."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from xla.tools.multihost_hlo_runner import py_hlo_multihost_runner as runner
from xla.tools.multihost_hlo_runner import python_hlo_runner_ffi_test_lib as test_lib

_DEVICE = flags.DEFINE_enum(
    "device", "host", ["host", "gpu"], "Device to test."
)


class PythonHloRunnerFfiTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("without_record", False), ("with_record", True)
  )
  def test_handler_bundle(self, with_record):
    is_gpu = _DEVICE.value == "gpu"
    platform = "CUDA" if is_gpu else "Host"
    name = self._testMethodName
    handlers = test_lib.handlers()
    if not with_record:
      del handlers["record"]
    runner.register_custom_call_target(name, handlers, platform, api_version=1)
    self.assertCountEqual(runner.custom_call_targets(platform)[name], handlers)
    test_lib.reset()

    hlo = self.create_tempfile(
        "ffi.hlo",
        content=f"""m {{
  r = f32[0] custom-call(), custom_call_target="{name}",
    custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI
}}
""",
    )
    config = runner.PyHloRunnerConfig()
    config.device_type = (
        runner.DeviceType.Gpu if is_gpu else runner.DeviceType.Host
    )
    config.input_format = runner.InputFormat.Text
    config.num_replicas = 1
    config.num_repeats = 3
    config.gpu_client_mem_fraction = 0.01
    runner.RunHloFiles([hlo.full_path], config)

    executed, recorded = test_lib.calls()
    self.assertTrue(executed)
    self.assertEqual(recorded, is_gpu and with_record)


if __name__ == "__main__":
  absltest.main()
