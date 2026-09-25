# Copyright 2026
# TensorFlow PR Review Agent
# pylint: disable=bad-indentation,line-too-long,unused-import

from __future__ import annotations

import asyncio
import logging
import time
import requests
from os import environ

from agent import agent
from agent.settings import OWNER, REPO, PULL_REQUEST_NUMBER, GITHUB_BASE_URL
from agent.utils import (
    call_agent_async,
    has_agent_reviewed_commit,
    parse_number_string,
    run_pylint_on_changed_files,
)

from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai.errors import APIError, ClientError, ServerError

APP_NAME = "tensorflow_pr_review_app"
USER_ID = "tensorflow_pr_review_user"

logs.setup_adk_logger(level=logging.DEBUG)


def is_fallback_eligible_error(err: Exception) -> bool:
    """Returns True only for Gemini API model-availability (404) or transient service (503/5xx/429) errors."""
    if not isinstance(err, (APIError, ClientError, ServerError)):
        return False

    err_code = getattr(err, "code", None)
    err_status = str(getattr(err, "status", "") or "").upper()
    err_text = str(err).upper()

    if err_code in (400, 401, 403):
        return False
    if any(
        token in err_status or token in err_text
        for token in (
            "400",
            "INVALID_ARGUMENT",
            "FAILED_PRECONDITION",
            "401",
            "UNAUTHENTICATED",
            "403",
            "PERMISSION_DENIED",
        )
    ):
        return False

    if (
        err_code == 404
        or "404" in err_text
        or "NOT_FOUND" in err_status
        or "NOT_FOUND" in err_text
    ):
        return True

    if err_code in (429, 500, 502, 503, 504):
        return True
    if any(
        token in err_status or token in err_text
        for token in (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "500",
            "INTERNAL",
            "502",
            "BAD_GATEWAY",
            "504",
            "DEADLINE_EXCEEDED",
        )
    ):
        return True

    return False


def get_first_comment_id(pr_number: int) -> int | None:
    """Fetches the ID of the very first issue comment to attach reactions to."""
    token = environ.get("GITHUB_TOKEN")
    if not token or not pr_number:
        return None

    # Pull requests are treated as issues for top-level comment threads
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json().get("id")
    except Exception as e:
        print(f"Failed to fetch issue metadata: {e}")
    return None


def clear_and_set_reaction(pr_number: int, add_content: str = "eyes"):
    """Cleans up previous runtime reactions and establishes the new active emoji."""
    token = environ.get("GITHUB_TOKEN")
    if not token or not pr_number:
        return

    # Use the specific issue ID endpoint to isolate the root comment reaction block
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/reactions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.squirrel-girl-preview+json"
    }

    try:
        # Step 1: Read all existing reactions on this thread
        existing_res = requests.get(url, headers=headers, timeout=10)
        if existing_res.status_code == 200:
            reactions_list = existing_res.json()
            # Loop through and remove any active 'eyes' reactions posted by this agent integration
            for reaction in reactions_list:
                if reaction.get("content") == "eyes":
                    reaction_id = reaction.get("id")
                    delete_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/reactions/{reaction_id}"
                    requests.delete(delete_url, headers=headers, timeout=10)
                    print(f"Cleared stale 'eyes' reaction ID: {reaction_id}")

        # Step 2: Post the fresh structural reaction status
        requests.post(url, headers=headers, json={"content": add_content}, timeout=10)
        print(f"Successfully posted final state reaction: {add_content}")
    except Exception as e:
        print(f"Failed to balance PR reaction status lifecycle: {e}")


async def main():
    pr_number = parse_number_string(PULL_REQUEST_NUMBER)
    if not pr_number:
        print(f"Error: Invalid pull request number received: {PULL_REQUEST_NUMBER}")
        return

    # 1. Clean old states and put down the looking eyes emoji
    clear_and_set_reaction(pr_number, add_content="eyes")

    # Fetch metadata once at startup
    pr_details_response = agent.get_pull_request_details(pr_number)
    if pr_details_response.get("status") != "success":
        print(f"Error: Failed to retrieve PR details: {pr_details_response.get('error_message')}")
        return

    # Inject into the pre-fetched state storage so tools can reuse it
    agent._PREFETCHED_PR_DETAILS = pr_details_response

    pr_data = pr_details_response.get("pull_request", {})
    head_sha = pr_data.get("headRefOid", "")
    reviews_url = (
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews"
    )
    if head_sha and has_agent_reviewed_commit(reviews_url, head_sha):
        print(
            f"Commit {head_sha} has already been reviewed by TensorFlow "
            f"PR Review Agent. Skipping duplicate review."
        )
        clear_and_set_reaction(pr_number, add_content="rocket")
        return

    files = pr_data.get("files", {}).get("nodes", [])
    title = pr_data.get("title", "")
    body = pr_data.get("body", "")
    diff = pr_data.get("diff", "")

    # Perform scoring-based categorization
    category, reason = agent.classify_pr_with_scoring(files, title, body, diff)
    print(f"Detected Category: {category}")
    print(f"Reason: {reason}")

    focus_areas, skip_areas = agent.get_focus_skip_areas(category)

    # Run static analysis once per newly reviewed commit before model fallback loop
    print("Running Pylint static analysis on changed Python files...")
    pylint_output = run_pylint_on_changed_files(files, diff, head_sha=head_sha)
    pr_data["pylint_output"] = pylint_output
    print(f"Pylint Analysis Summary:\n{pylint_output}\n")

    last_error: Exception | None = None

    # Reference the custom pool dynamically from agent.py
    for model_name in agent.MODELS_POOL:
        print(f"Attempting execution using engine target: {model_name}...")
        
        try:
            response = await agent.run_pr_review(
                model_name=model_name,
                pr_number=pr_number,
                category=category,
                reason=reason,
                focus_areas=focus_areas,
                skip_areas=skip_areas,
                pylint_output=pylint_output
            )
            print(f"<<<< Agent Final Output: {response}\n")
            if head_sha and not has_agent_reviewed_commit(
                reviews_url, head_sha
            ):
                print(
                    f"⚠️ Review submission for commit {head_sha} was not "
                    f"confirmed on GitHub with {model_name}. "
                    f"Attempting alternative fallback..."
                )
                last_error = RuntimeError(
                    f"Review submission not confirmed on GitHub for model {model_name}"
                )
                continue
            print(f"Processing complete successfully with {model_name}!")
            
            # 2. Execution complete! Swap eyes out for a final rocket emoji
            clear_and_set_reaction(pr_number, add_content="rocket")
            return 

        except (APIError, ClientError, ServerError) as e:
            if head_sha and has_agent_reviewed_commit(reviews_url, head_sha):
                print(
                    f"Review for commit {head_sha} was already submitted "
                    f"before API error. Completing successfully."
                )
                clear_and_set_reaction(pr_number, add_content="rocket")
                return
            if is_fallback_eligible_error(e):
                last_error = e
                print(
                    f"⚠️ Model target {model_name} failed with fallback-eligible error ({e}). "
                    f"Attempting alternative fallback..."
                )
                continue
            raise e
        except Exception as e:
            raise e

    error_msg = (
        f"All models in MODELS_POOL failed to complete PR review. "
        f"Last error: {last_error}"
    )
    print(f"Error: {error_msg}")
    raise RuntimeError(error_msg)


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start reviewing {OWNER}/{REPO} pull request #{PULL_REQUEST_NUMBER}")
    print("-" * 80)
    asyncio.run(main())
    print("-" * 80)
