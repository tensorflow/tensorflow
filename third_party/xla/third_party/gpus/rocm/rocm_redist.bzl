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
                "url": "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz",
                "sha256": "1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6",
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
