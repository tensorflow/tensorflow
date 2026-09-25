# Copyright 2026
#
# TensorFlow PR Review Agent - Utility Functions
# pylint: disable=bad-indentation,line-too-long

from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from agent.settings import GITHUB_BASE_URL
from agent.settings import GITHUB_GRAPHQL_URL
from agent.settings import GITHUB_TOKEN
from agent.settings import OWNER
from agent.settings import REPO
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.genai import types
import requests

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

diff_headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}


def run_graphql_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Executes a GitHub GraphQL query."""
    payload = {"query": query, "variables": variables}
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_request(
    url: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Executes a GitHub GET request."""
    if params is None:
        params = {}

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60,
    )
    if response.status_code == 401:
        unauth_headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(url, headers=unauth_headers, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def get_diff(url: str) -> str:
    """Retrieves PR diff content."""
    response = requests.get(
        url,
        headers=diff_headers,
        timeout=60,
    )
    if response.status_code == 401:
        unauth_diff_headers = {"Accept": "application/vnd.github.v3.diff"}
        response = requests.get(url, headers=unauth_diff_headers, timeout=60)
    response.raise_for_status()
    return response.text


def annotate_diff_with_line_numbers(raw_diff: str) -> str:
    """Annotates a unified diff with explicit right-side ([L...]) and left-side ([LEFT L...]) line numbers."""
    lines = raw_diff.split('\n')
    annotated = []
    current_left = None
    current_right = None
    hunk_pattern = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
    for line in lines:
        m = hunk_pattern.match(line)
        if m:
            current_left = int(m.group(1))
            current_right = int(m.group(2))
            annotated.append(line)
            continue
        if current_right is not None and current_left is not None:
            if line.startswith('+') and not line.startswith('+++'):
                annotated.append(f'[L{current_right}] {line}')
                current_right += 1
            elif line.startswith('-') and not line.startswith('---'):
                annotated.append(f'[LEFT L{current_left}] {line}')
                current_left += 1
            elif line.startswith(' ') or line == '':
                annotated.append(f'[L{current_right}] {line}')
                current_right += 1
                current_left += 1
            else:
                annotated.append(line)
        else:
            annotated.append(line)
    return '\n'.join(annotated)


def post_request(url: str, payload: Any) -> dict[str, Any]:
    """Executes a GitHub POST request."""
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def format_commit_review_marker(commit_sha: str) -> str:
    """Returns an HTML comment marker identifying the reviewed commit SHA."""
    return (
        f"\n\n<!-- tensorflow-pr-review-agent: commit_sha={commit_sha} -->"
    )


def has_agent_reviewed_commit(reviews_url: str, commit_sha: str) -> bool:
    """Checks PR reviews to see if this agent already reviewed commit_sha."""
    if not commit_sha:
        return False
    marker = f"<!-- tensorflow-pr-review-agent: commit_sha={commit_sha} -->"
    try:
        reviews = get_request(reviews_url, params={"per_page": 100})
        if not isinstance(reviews, list):
            return False
        for review in reviews:
            body = review.get("body") or ""
            review_commit_id = review.get("commit_id") or ""
            if review_commit_id == commit_sha and marker in body:
                return True
    except Exception as e:  # pylint: disable=broad-except
        print(
            f"Warning: Failed to check PR reviews for commit {commit_sha}: {e}"
        )
    return False


def post_pull_request_review(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Executes a GitHub POST request to create a pull request review with inline comments."""
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    if response.status_code == 422:
        # If GitHub returns 422 Unprocessable Entity (e.g. invalid inline line numbers outside diff hunks),
        # gracefully fall back by moving inline comments into the top-level summary body so feedback is not lost.
        fallback_body = payload.get("body", "")
        inline_comments = payload.get("comments", [])
        if inline_comments:
            fallback_body += "\n\n### Inline Comments (Fallback)\n"
            for ic in inline_comments:
                fallback_body += f"\n- **File**: `{ic.get('path', '')}` (Line {ic.get('line', 'N/A')}):\n{ic.get('body', '')}\n"
        fallback_payload: dict[str, Any] = {
            "body": fallback_body,
            "event": payload.get("event", "COMMENT"),
        }
        if payload.get("commit_id"):
            fallback_payload["commit_id"] = payload["commit_id"]
        response = requests.post(url, headers=headers, json=fallback_payload, timeout=60)
    response.raise_for_status()
    return response.json()


def error_response(error_message: str) -> dict[str, Any]:
    """Returns a standardized error response."""
    return {
        "status": "error",
        "error_message": error_message,
    }


def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return ""


def parse_number_string(
    number_str: str | None,
    default_value: int = 0,
) -> int:
    """Parse integer values safely."""
    if not number_str:
        return default_value

    try:
        return int(number_str)
    except ValueError:
        print(
            f"Warning: Invalid number string: {number_str}. "
            f"Defaulting to {default_value}.",
            file=sys.stderr,
        )
        return default_value


async def call_agent_async(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> str:
    """Execute an ADK agent asynchronously."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    final_response_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(
            save_input_blobs_as_artifacts=False,
        ),
    ):
        if event.content and event.content.parts:
            text = "".join(
                part.text or ""
                for part in event.content.parts
            )

            if text and event.author != "user":
                final_response_text += text

    return final_response_text


def extract_modified_lines_by_file(raw_diff: str) -> dict[str, set[int]]:
    """Extracts right-side added/modified line numbers per file from a unified diff (raw or annotated)."""
    modified_lines: dict[str, set[int]] = {}
    if not raw_diff:
        return modified_lines

    current_file: str | None = None
    current_right: int | None = None

    file_header_pattern = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    annotated_add_pattern = re.compile(r"^\[L(\d+)\]\s*\+(?!\+\+)")
    annotated_ctx_pattern = re.compile(r"^\[L(\d+)\]\s*(?: |$)")

    for line in raw_diff.splitlines():
        clean_line = re.sub(r"^\[(?:LEFT )?L\d+\]\s*", "", line)
        if clean_line.startswith("+++ "):
            m_file = file_header_pattern.match(clean_line)
            if m_file:
                path = m_file.group(1).strip()
                if path == "/dev/null":
                    current_file = None
                else:
                    current_file = path
                    if current_file not in modified_lines:
                        modified_lines[current_file] = set()
            current_right = None
            continue

        m_hunk = hunk_pattern.match(line)
        if m_hunk:
            current_right = int(m_hunk.group(1))
            continue

        if current_file is None:
            continue

        m_ann_add = annotated_add_pattern.match(line)
        if m_ann_add:
            line_num = int(m_ann_add.group(1))
            modified_lines[current_file].add(line_num)
            current_right = line_num + 1
            continue

        m_ann_ctx = annotated_ctx_pattern.match(line)
        if m_ann_ctx:
            current_right = int(m_ann_ctx.group(1)) + 1
            continue

        if line.startswith("[LEFT L"):
            continue

        if current_right is not None:
            if line.startswith("+") and not line.startswith("+++"):
                modified_lines[current_file].add(current_right)
                current_right += 1
            elif line.startswith("-") and not line.startswith("---"):
                pass
            elif line.startswith(" ") or line == "":
                current_right += 1

    return modified_lines


def _extract_deleted_text_by_file(raw_diff: str) -> dict[str, str]:
    """Collects deleted lines per file from a unified diff to verify unused-import regressions."""
    deleted_chunks: dict[str, list[str]] = {}
    if not raw_diff:
        return {}

    current_file: str | None = None
    file_header_pattern = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
    annotated_del_pattern = re.compile(r"^\[LEFT L\d+\]\s*-(?!---)(.*)$")

    for line in raw_diff.splitlines():
        clean_line = re.sub(r"^\[(?:LEFT )?L\d+\]\s*", "", line)
        if clean_line.startswith("+++ "):
            m_file = file_header_pattern.match(clean_line)
            if m_file:
                path = m_file.group(1).strip()
                current_file = None if path == "/dev/null" else path
            continue

        if current_file is None:
            continue

        m_ann_del = annotated_del_pattern.match(line)
        if m_ann_del:
            deleted_chunks.setdefault(current_file, []).append(m_ann_del.group(1))
        elif line.startswith("-") and not line.startswith("---"):
            deleted_chunks.setdefault(current_file, []).append(line[1:])

    return {k: "\n".join(v) for k, v in deleted_chunks.items()}


def _is_safe_relative_path(rel_path: str) -> bool:
    """Validates that a file path is safe, relative, and outside the review agent directory."""
    if not rel_path or rel_path.startswith(("/", "\\")) or os.path.isabs(rel_path):
        return False
    path_obj = Path(rel_path)
    if path_obj.is_absolute() or ".." in path_obj.parts:
        return False
    norm = rel_path.replace("\\", "/")
    if norm.startswith("pr_review_agent/"):
        return False
    return True


def _fetch_file_content_at_commit(
    repo_root: Path,
    head_sha: str,
    rel_path: str,
) -> str | None:
    """Safely fetches a single file's content at head_sha without checking out untrusted PR code."""
    if not _is_safe_relative_path(rel_path):
        return None

    if head_sha and re.match(r"^[0-9a-fA-F]{7,40}$", head_sha):
        try:
            res = subprocess.run(
                ["git", "show", f"{head_sha}:{rel_path}"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout
        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: git show failed for {head_sha}:{rel_path}: {e}")

        if OWNER and REPO:
            try:
                url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/contents/{rel_path}"
                data = get_request(url, params={"ref": head_sha})
                if (
                    isinstance(data, dict)
                    and data.get("encoding") == "base64"
                    and data.get("content")
                ):
                    return base64.b64decode(data["content"]).decode(
                        "utf-8", errors="replace"
                    )
            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"Warning: GitHub API content fetch failed for {rel_path} at {head_sha}: {e}"
                )

    try:
        local_path = (repo_root / rel_path).resolve()
        if local_path.is_relative_to(repo_root.resolve()) and local_path.is_file():
            return local_path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # pylint: disable=broad-except
        pass

    return None


def filter_pylint_output_by_diff(
    pylint_stdout: str,
    modified_lines_by_file: dict[str, set[int]],
    raw_diff: str = "",
) -> list[str]:
    """Filters Pylint output to retain only diagnostics on PR-modified lines or PR-caused import/module issues."""
    if not pylint_stdout:
        return []

    deleted_text_by_file = _extract_deleted_text_by_file(raw_diff)
    diag_pattern = re.compile(
        r"^([^:\n]+):(\d+):(\d+):\s+([A-Z]\d+)\s+\(([^)]+)\):\s+(.+)$"
    )
    unused_import_sym_pattern = re.compile(
        r"(?:Unused import\s+([a-zA-Z0-9_]+)|Unused\s+([a-zA-Z0-9_]+)\s+imported from)"
    )

    retained: list[str] = []
    for line in pylint_stdout.splitlines():
        m = diag_pattern.match(line.strip())
        if not m:
            continue
        path = m.group(1).strip().lstrip("./")
        line_num = int(m.group(2))
        col_num = int(m.group(3))
        msg_id = m.group(4)
        symbol = m.group(5)
        msg = m.group(6)

        mod_lines = modified_lines_by_file.get(path, set())
        keep = False

        if line_num in mod_lines:
            keep = True
        elif msg_id.startswith("F") or msg_id == "E0001" or line_num == 0:
            keep = True
        elif symbol == "unused-import" or msg_id == "W0611":
            m_sym = unused_import_sym_pattern.search(msg)
            sym_name = (m_sym.group(1) or m_sym.group(2)) if m_sym else None
            deleted_text = deleted_text_by_file.get(path, "")
            if sym_name and deleted_text and re.search(rf"\b{re.escape(sym_name)}\b", deleted_text):
                keep = True

        if keep:
            retained.append(
                f"{path}:{line_num}:{col_num}: {msg_id} ({symbol}): {msg}"
            )

    return retained


def run_pylint_on_changed_files(
    files: list[dict[str, Any]],
    raw_diff: str,
    head_sha: str = "",
    repo_root: Path | None = None,
) -> str:
    """Materializes changed Python files into an isolated temp dir and runs Pylint with diff-line filtering."""
    changed_py_files: list[str] = []
    for f in (files or []):
        path = f.get("path", "")
        change_type = (f.get("changeType") or "").upper()
        if (
            path.endswith(".py")
            and change_type != "DELETED"
            and _is_safe_relative_path(path)
        ):
            changed_py_files.append(path)

    if not changed_py_files:
        return "No Python files were modified in this pull request."

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root.resolve()

    rcfile_path = repo_root / "tensorflow" / "tools" / "ci_build" / "pylintrc"
    if not rcfile_path.is_file():
        fallback_rc = repo_root / ".pylintrc"
        if fallback_rc.is_file():
            rcfile_path = fallback_rc.resolve()

    try:
        with tempfile.TemporaryDirectory(prefix="tf_pr_pylint_") as temp_dir:
            temp_dir_path = Path(temp_dir).resolve()
            materialized_paths: list[str] = []

            for rel_path in changed_py_files:
                dest_file = (temp_dir_path / rel_path).resolve()
                if not dest_file.is_relative_to(temp_dir_path):
                    continue
                content = _fetch_file_content_at_commit(repo_root, head_sha, rel_path)
                if content is None:
                    continue
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(content, encoding="utf-8")
                materialized_paths.append(rel_path)

            if not materialized_paths:
                return "No changed Python files could be materialized for Pylint analysis."

            cmd = [
                sys.executable,
                "-m",
                "pylint",
            ]
            if rcfile_path.is_file():
                cmd.append(f"--rcfile={rcfile_path}")
            cmd.extend([
                "--disable=unrecognized-option,useless-option-value,unknown-option-value,bad-option-value",
                "--overgeneral-exceptions=builtins.Exception,builtins.BaseException",
                "--output-format=text",
                "--score=no",
                "--msg-template={path}:{line}:{column}: {msg_id} ({symbol}): {msg}",
                *materialized_paths,
            ])

            res = subprocess.run(
                cmd,
                cwd=str(temp_dir_path),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            modified_lines_by_file = extract_modified_lines_by_file(raw_diff)
            retained = filter_pylint_output_by_diff(
                res.stdout,
                modified_lines_by_file,
                raw_diff=raw_diff,
            )

            if not retained:
                return "No Pylint issues detected on modified lines."

            return "\n".join(retained[:50])[:5000]

    except subprocess.TimeoutExpired:
        print("Warning: Pylint execution timed out after 120 seconds.")
        return "Pylint static analysis timed out and was skipped."
    except Exception as e:  # pylint: disable=broad-except
        print(f"Warning: Pylint execution failed: {e}")
        return f"Pylint static analysis could not be completed: {e}"

