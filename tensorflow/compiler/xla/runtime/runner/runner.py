# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Python helper for running Xla runtime runner tools."""

import os
import subprocess
import tempfile

from typing import Sequence, Any

from tensorflow.compiler.xla.runtime.runner import runner_pb2
from tensorflow.python.platform import resource_loader


class Runner:
  """Python helper for running Xla runtime runner tools."""

  def __init__(self, runner: str):
    self.runner = runner

  def execute(self, module: str, function: str,
              arguments: Sequence[Any]) -> Sequence[Any]:
    """Executes `module` with user-provided arguments."""
    temp = tempfile.mkdtemp()

    # Write input mlir module to a file.
    module_file = os.path.join(temp, "module.mlir")
    with open(module_file, "w") as f:
      f.write(module)

    # Pack arguments into a proto message.
    args_proto = runner_pb2.ArgumentsProto()
    for arg in arguments:
      if isinstance(arg, int):
        args_proto.arguments.append(
            runner_pb2.ArgumentProto(scalar=runner_pb2.ScalarProto(i32=arg)))
        continue

      raise TypeError("Unsupported argument type")

    # Serialize argument proto message to a file.
    arguments_file = os.path.join(temp, "arguments.pb")
    with open(arguments_file, "wb") as f:
      f.write(args_proto.SerializeToString())

    # Expected results file path.
    results_file = os.path.join(temp, "results.pb")

    # Execute the runner tool.
    runner = resource_loader.get_path_to_datafile(self.runner)
    runner_cmd = [
        runner, "--logtostderr", f"--function={function}",
        f"--module={module_file}", f"--arguments={arguments_file}",
        f"--results={results_file}"
    ]
    result = subprocess.run(runner_cmd, capture_output=True, check=True)

    if result.returncode != 0:
      err = result.stderr.decode("utf-8")
      raise RuntimeError(f"failed to execute runner tool: {err}")

    # Read returned results.
    with open(results_file, "rb") as f:
      results_proto = runner_pb2.ResultsProto.FromString(f.read())

    # Convert results from proto back to python objects.
    results = []

    for res in results_proto.results:
      # Convert ScalarProto to scalar object
      if hasattr(res, "scalar"):
        scalar = res.scalar

        if hasattr(scalar, "i32"):
          results.append(scalar.i32)
          continue
        if hasattr(scalar, "i64"):
          results.append(scalar.i64)
          continue

      raise ValueError(f"Unknown result {res}")

    return results
