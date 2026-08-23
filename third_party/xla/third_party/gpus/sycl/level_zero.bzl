# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

level_zero_redist = {
    "ubuntu_24.10": {
        "2025.1": {
            "level_zero": {
                "root": "dl_essential_root",
                "archives": [
                    {
                        "url": "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/level-zero-1.21.10.tar.gz",
                        "sha256": "e0ff1c6cb9b551019579a2dd35c3a611240c1b60918c75345faf9514142b9c34",
                    },
                    {
                        "url": "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/ze_loader_libs.tar.gz",
                        "sha256": "71cbfd8ac59e1231f013e827ea8efe6cf5da36fad771da2e75e202423bd6b82e",
                    },
                ],
            },
        },
    },
}
