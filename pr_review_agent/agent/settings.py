# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=bad-indentation,line-too-long

import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_BASE_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = GITHUB_BASE_URL + "/graphql"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER")
REPO = os.getenv("REPO")
PULL_REQUEST_NUMBER = os.getenv("PULL_REQUEST_NUMBER")

IS_INTERACTIVE = False
