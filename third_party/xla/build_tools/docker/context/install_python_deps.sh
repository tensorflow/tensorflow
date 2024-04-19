# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================

# Drawn from https://github.com/openxla/iree/blob/0246bbfd7955fcd858f8467182404060ccd2e9ae/build_tools/docker/context/install_python_deps.sh

set -euo pipefail

if ! [[ -f python_build_requirements.txt ]]; then
  echo "Couldn't find python_build_requirements.txt in current directory" >&2
  exit 1
fi

PYTHON_VERSION="$1"

apt-get update

apt-get install -y \
  "python${PYTHON_VERSION}" \
  "python${PYTHON_VERSION}-dev"

update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1

apt-get install -y \
  python3-pip \
  python3-setuptools \
  python3-distutils \
  python3-venv \
  "python${PYTHON_VERSION}-venv"
python3 -m pip install --ignore-installed --upgrade -r python_build_requirements.txt
