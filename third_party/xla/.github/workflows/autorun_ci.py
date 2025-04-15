# Copyright 2024 The OpenXLA Authors. All Rights Reserved.
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
"""Autoruns CI for OpenXLA org members with membership set to public."""
import logging
import os

import github_api

_OPENXLA_ORG_ID = 107584881  # https://api.github.com/orgs/107584881


def main():
  username = os.getenv("PR_AUTHOR_USERNAME")
  pr_number = os.getenv("PR_NUMBER")
  api = github_api.GitHubAPI(os.getenv("GH_TOKEN"))

  orgs = api.get_user_orgs(username)
  logging.info("Found public organizations for user %s: %s", username, orgs)

  if _OPENXLA_ORG_ID in {org["id"] for org in orgs}:
    logging.info(
        "Found OpenXLA org in public memberships, so adding kokoro:force-run"
        " label."
    )
    api.add_issue_labels("openxla/xla", pr_number, ["kokoro:force-run"])


if __name__ == "__main__":
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)
  main()
