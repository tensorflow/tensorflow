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
# ============================================================================
"""Checks PR size and posts a comment if it's too large."""

import os
import sys

import github_api

THRESHOLD_MESSAGES = {
    1000: (
        "🔴 This PR has a very large delta of over 1000. In order to enable an"
        " effective code reivew, please break the PR down into smaller and more"
        " focused PRs.\nSee the [Small"
        " CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)"
        " Google Eng practice for more details on how to write compact PRs."
    ),
    500: (
        "⚠️ This PR has a large delta of over 500. Consider breaking the PR"
        " down into smaller PRs for a faster code review.\nSee the [Small"
        " CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)"
        " Google Eng practice for more details on how to write compact PRs."
    ),
}


def main():
  pr_number = os.getenv("PR_NUMBER")
  if not pr_number:
    print("PR_NUMBER is not set.")
    sys.exit(1)

  additions = int(os.getenv("ADDITIONS", "0"))
  deletions = int(os.getenv("DELETIONS", "0"))
  total_delta = additions + deletions

  print(
      f"PR #{pr_number} additions: {additions}, deletions: {deletions}, total:"
      f" {total_delta}"
  )

  comment = (
      f"PR #{pr_number} additions: {additions}, deletions: {deletions}, total:"
      f" {total_delta}"
  )
  exceeded = False

  for threshold in sorted(THRESHOLD_MESSAGES.keys(), reverse=True):
    if total_delta >= threshold:
      comment += f"\n\n{THRESHOLD_MESSAGES[threshold]}"
      exceeded = True
      break

  if exceeded:
    token = os.getenv("GH_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not repo:
      print("GITHUB_REPOSITORY is not set.")
      sys.exit(1)

    try:
      api = github_api.GitHubAPI(token)
      api.write_issue_comment(repo, int(pr_number), comment)
      print("Comment posted.")
    except Exception as e:  # pylint: disable=broad-except
      print(f"Failed to post comment: {e}")
      sys.exit(1)
  else:
    print("PR size is within limits.")


if __name__ == "__main__":
  main()
