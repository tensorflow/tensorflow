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
    "rocm_7.13.0_gfx94X": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "sha256": "db5543de096fb175ff2ece19dacc28b2a3201df48b38051cc505e508d84e35ab",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
    "rocm_7.13.0_gfx908": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx908-7.13.0.tar.gz",
                "sha256": "5d84753a8d8895ff2f6137a2a922ee8f36ce9c2e01b60a99d3ee776a683bfc34",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
    "rocm_7.13.0_gfx90a": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx90a-7.13.0.tar.gz",
                "sha256": "b2d3c49ef936b3b24b10a25bae3e60df7ccc9c5134095a080bbc721d5062b4c7",
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
