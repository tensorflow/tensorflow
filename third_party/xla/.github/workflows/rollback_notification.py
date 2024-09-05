# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Notifies PR authors of rollbacks on committed PRs."""
import os
import re
from typing import Generator, Sequence

import github_api


def get_reverted_commit_hashes(message: str) -> list[str]:
  """Searches a commit message for `reverts <commit hash>` and returns the found SHAs.

  Arguments:
    message: the commit message to search

  Returns:
    A list of SHAs as strings.
  """
  print("Head commit message:", message, sep="\n")
  regex = re.compile(r"reverts ([0-9a-f]{5,40})", flags=re.IGNORECASE)
  commit_hashes = regex.findall(message)
  print(f"Found commit hashes reverted in this commit: {commit_hashes}")
  return commit_hashes


def get_associated_prs(
    api: github_api.GitHubAPI,
    commit_hashes: Sequence[str],
) -> Generator[int, None, None]:
  """Finds PRs associated with commits.

  Arguments:
    api: GitHubAPI object which will be used to make requests
    commit_hashes: A sequence of SHAs which may have PRs associated with them

  Yields:
    Associated pairs of (PR number, SHA), both as strings
  """
  regex = re.compile(r"PR #(\d+)")
  for commit_hash in commit_hashes:
    response = api.get_commit("openxla/xla", commit_hash)
    message = response["commit"]["message"]
    if maybe_match := regex.match(message):
      pr_number = maybe_match.group(1)
      print(f"Found PR #{pr_number} associated with commit hash {commit_hash}")
      yield int(pr_number)
  print(f"Didn't find any PRs associated with commit hashes: {commit_hashes}")


def main():
  api = github_api.GitHubAPI(os.getenv("GH_TOKEN"))
  head_commit = api.get_commit("openxla/xla", "HEAD")
  commit_hashes = get_reverted_commit_hashes(head_commit["commit"]["message"])

  for pr_number in get_associated_prs(api, commit_hashes):
    sha = head_commit["sha"]
    api.write_issue_comment(
        "openxla/xla", pr_number, f"This PR was rolled back in {sha}!"
    )
    api.set_issue_status("openxla/xla", pr_number, "open")


if __name__ == "__main__":
  main()
