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

"""Wraps clang-cl.exe with retry logic and bypasses cmd.exe limits."""

import os
import subprocess
import sys
import time


def invoke_clang() -> int:
  """Wraps clang-cl.exe with retry logic and bypasses cmd.exe limits."""
  # Find clang-cl.exe.
  clang_path = "C:/tools/LLVM/bin/clang-cl.exe"

  # Prepend clang_path to arguments
  args = [clang_path] + sys.argv[1:]

  max_retries = 5
  retry_delay = 1  # seconds

  # Set environment variable if needed
  os.environ["CYGWIN"] = "nodosfilewarning"

  for i in range(max_retries):
    # Bazel streams output, so we should not capture it unless we have to.
    # We use shell=False to bypass cmd.exe limits.
    result = subprocess.run(args, shell=False, check=False)
    if result.returncode == 0:
      return 0

    sys.stderr.write(
        f"Attempt {i+1} failed with errorlevel {result.returncode}. "
        f"Retrying in {retry_delay} seconds...\n"
    )
    time.sleep(retry_delay)

  sys.stderr.write("All clang-cl.exe attempts failed.\n")
  return 1


if __name__ == "__main__":
  sys.exit(invoke_clang())
