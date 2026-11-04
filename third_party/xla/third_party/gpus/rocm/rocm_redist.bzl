# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Rocm distribution list, used as a hermetic rocm dependency."""

rocm_redist = {
    "rocm_latest_multiarch": struct(
        packages = [
            {
                "url": "https://dev.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.1.0.dev0%2Be54d902c2bc89baf912f69149e47d9ddc5102c4d.tar.gz",
                "sha256": "682075648697e4aa8fe7cc9ec2a56e9207a1fc47f74c4f10440f8dadad89f5e6",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
}

def _parse_rocm_distro_links(distro_links):
    result = []
    if distro_links == "":
        return result

    for pair in distro_links.split(","):
        link = pair.split(":")
        result.append(struct(target = link[0], link = link[1]))
    return result

def create_rocm_distro(distro_url, distro_hash, symlinks):
    return struct(
        packages = [
            {
                "url": distro_url,
                "sha256": distro_hash,
            },
        ],
        required_softlinks = _parse_rocm_distro_links(symlinks),
        rocm_root = "",
    )
