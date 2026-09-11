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
"""Utility function for updating setup.py with NVIDIA wheel versions.

The content of the setup.py file is updated with the NVIDIA wheel versions
provided in the nvidia_wheel_versions_data string.

The setup.py file is expected to have the following lines:

```
# Mandatory placeholders
cuda_version = 0  # placeholder
cuda_wheel_suffix = ''  # placeholder

# Optional placeholders (add only those that are needed)
nvidia_cublas_version = ''  # placeholder

EXTRA_PACKAGES = {
    'and-cuda': [
        f'nvidia-cublas{cuda_wheel_suffix}{nvidia_cublas_version}',
        # add more wheels here
    ],
}
```
"""

import re

# Regex to capture wheel name and its version constraint
# Example: "nvidia-cublas-cu12>=12.1.3.1 ; sys_platform == 'linux'"
NVIDIA_WHEEL_VERSIONS_PATTERN = re.compile(r"^([a-z0-9_-]+)(\W*[0-9\.]*.*)$")


def get_setup_py_content_with_nvidia_wheel_versions(
    setup_py_content: str, cuda_version: str, nvidia_wheel_versions_data: str
) -> str:
  nvidia_wheel_versions = {"12": {}, "13": {}}
  for line in nvidia_wheel_versions_data.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    match = NVIDIA_WHEEL_VERSIONS_PATTERN.match(line)
    if match and match.group(1).startswith("nvidia"):
      wheel_name = match.group(1).replace("-", "_")
      for suffix, version in {"_cu12": "12", "_cu13": "13", "": "13"}.items():
        if not wheel_name.endswith(suffix):
          continue
        if suffix:
          wheel_name = wheel_name[:-len(suffix)]
        wheel_name = wheel_name + "_version"
        nvidia_wheel_versions[version][wheel_name] = match.group(2).strip()
        break

  cuda_version_str = str(cuda_version) if cuda_version else "12"
  setup_py_content = setup_py_content.replace(
      "cuda_version = 0  # placeholder", f"cuda_version = {cuda_version_str}"
  )
  setup_py_content = setup_py_content.replace(
      "cuda_wheel_suffix = ''  # placeholder",
      (
          "cuda_wheel_suffix = '-cu12'"
          if cuda_version_str == "12"
          else "cuda_wheel_suffix = ''"
      ),
  )
  setup_py_content = setup_py_content.replace(
      "cuda_major_version = '12'  # placeholder",
      f"cuda_major_version = '{cuda_version_str}'",
  )
  for version_name, version_value in nvidia_wheel_versions.get("12", {}).items():
    setup_py_content = setup_py_content.replace(
        f"cuda12_{version_name} = ''  # placeholder",
        f"cuda12_{version_name} = '{version_value}'",
    )
  for version_name, version_value in nvidia_wheel_versions.get("13", {}).items():
    setup_py_content = setup_py_content.replace(
        f"cuda13_{version_name} = ''  # placeholder",
        f"cuda13_{version_name} = '{version_value}'",
    )

  return setup_py_content
