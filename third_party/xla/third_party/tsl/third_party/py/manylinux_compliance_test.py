# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import platform
import third_party.py.manylinux_compliance_utils as manylinux_compliance_utils


def parse_args():
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for auditwheel", fromfile_prefix_chars="@"
  )
  parser.add_argument(
      "--wheel-path", required=True, help="Path of the wheel, mandatory"
  )
  parser.add_argument(
      "--aarch64-compliance-tag",
      required=True,
      help="ManyLinux compliance tag for aarch64",
  )
  parser.add_argument(
      "--x86_64-compliance-tag",
      required=True,
      help="ManyLinux compliance tag for x86_64",
  )
  return parser.parse_args()


def test_manylinux_compliance(args):
  machine_type = platform.uname().machine
  if machine_type == "x86_64":
    compliance_tag = args.x86_64_compliance_tag
  else:
    compliance_tag = args.aarch64_compliance_tag
  auditwheel_output = manylinux_compliance_utils.get_auditwheel_output(
      args.wheel_path
  )
  manylinux_compliance_utils.verify_manylinux_compliance(
      auditwheel_output,
      compliance_tag,
      "auditwheel_show.log",
  )


if __name__ == "__main__":
  test_manylinux_compliance(parse_args())
