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
# ==============================================================================

"""Checks the PTX limit and probe failure across two GPU modules."""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PTXAS_SCRIPT = """#!/bin/sh
case "$1" in
  --version)
    echo 'Cuda compilation tools, release 12.5, V12.5.82'
    ;;
  --input-as-string)
    echo probe >> "$PTX_PROBE_LOG"
    echo "$PTX_PROBE_RESPONSE" >&2
    exit 1
    ;;
  *) exit 1 ;;
esac
"""

# Each entry function becomes a separate GPU module.
_KERNELS = """
func.func @first(%input: tensor<4xf32>) -> tensor<4xf32> attributes {tf_entry} {
  %result = mhlo.negate %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
func.func @second(%input: tensor<4xf32>) -> tensor<4xf32> attributes {tf_entry} {
  %result = mhlo.negate %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
"""

_PTX_VERSIONS = re.compile(r"\.version\s+(\d+\.\d+)")


class PtxVersionTest(unittest.TestCase):
  kernel_generator = None

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.cuda_dir = Path(directory.name)
    (self.cuda_dir / "bin").mkdir()
    ptxas = self.cuda_dir / "bin" / "ptxas"
    ptxas.write_text(_PTXAS_SCRIPT)
    ptxas.chmod(0o755)
    self.source = self.cuda_dir / "kernels.mlir"
    self.source.write_text(_KERNELS)
    self.output = self.cuda_dir / "kernels.o"
    self.probe_log = self.cuda_dir / "probes"

  def compile_kernels(self, probe_response):
    return subprocess.run(
        [
            self.kernel_generator,
            "--input=" + str(self.source),
            "--output=" + str(self.output),
            "--arch=compute_80",
            "--tile_sizes=256",
            "--print-ptx",
        ],
        env=dict(
            os.environ,
            XLA_FLAGS="--xla_gpu_cuda_data_dir=" + str(self.cuda_dir),
            PTX_PROBE_LOG=str(self.probe_log),
            PTX_PROBE_RESPONSE=probe_response,
        ),
        capture_output=True,
        text=True,
        check=False,
    )

  def test_compiler_limit_caps_both_modules_with_one_probe(self):
    result = self.compile_kernels(
        "Unsupported .version 99.99; current version is '8.0'"
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(_PTX_VERSIONS.findall(result.stderr), ["8.0", "8.0"])
    self.assertEqual(self.probe_log.read_text(), "probe\n")

  def test_failed_probe_still_compiles_both_modules_without_retrying(self):
    result = self.compile_kernels("Failed to query PTX version")

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertGreater(self.output.stat().st_size, 0)
    self.assertEqual(len(_PTX_VERSIONS.findall(result.stderr)), 2)
    self.assertEqual(self.probe_log.read_text(), "probe\n")


if __name__ == "__main__":
  PtxVersionTest.kernel_generator = str(Path(sys.argv.pop(1)).resolve())
  unittest.main()
