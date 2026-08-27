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

sycl_redist = {
    "ubuntu_24.10": {
        "2025.1": {
            "sycl_dl_essential": {
                "root": "dl_essential_root",
                "archives": [
                    {
                        "url": "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-2025.1.3.7.tar.gz",
                        "sha256": "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a",
                    },
                ],
            },
        },
    },
    "ubuntu_24.04": {
        "2025.1": {
            "sycl_dl_essential": {
                "root": "dl_essential_root",
                "archives": [
                    {
                        "url": "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-2025.1.3.7.tar.gz",
                        "sha256": "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a",
                    },
                ],
            },
        },
    },
    "ubuntu_22.04": {
        "2025.1": {
            "sycl_dl_essential": {
                "root": "dl_essential_root",
                "archives": [
                    {
                        "url": "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-2025.1.3.7.tar.gz",
                        "sha256": "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a",
                    },
                ],
            },
        },
    },
}
