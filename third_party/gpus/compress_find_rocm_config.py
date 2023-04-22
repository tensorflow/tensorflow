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
"""Compresses the contents of 'find_rocm_config.py'.

The compressed file is what is actually being used. It works around remote
config not being able to upload files yet.
"""
import base64
import zlib


def main():
  with open('find_rocm_config.py', 'rb') as f:
    data = f.read()

  compressed = zlib.compress(data)
  b64encoded = base64.b64encode(compressed)

  with open('find_rocm_config.py.gz.base64', 'wb') as f:
    f.write(b64encoded)


if __name__ == '__main__':
  main()
