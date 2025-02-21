# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Calculates full TensorFlow version."""

import argparse
import time


def parse_args():
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for TensorFlow version calculation",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--wheel-type",
      required=True,
      choices=["nightly", "release"],
      help="Type of the wheel",
  )
  parser.add_argument("--wheel-version", required=True, help="Wheel version")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  tensorflow_version_suffix = ""
  if args.wheel_type == "nightly":
    tensorflow_version_suffix = "-dev{}".format(time.strftime("%Y%m%d"))

  print(
      '"{wheel_version}{tensorflow_version_suffix}"'.format(
          wheel_version=args.wheel_version,
          tensorflow_version_suffix=tensorflow_version_suffix,
      )
  )
