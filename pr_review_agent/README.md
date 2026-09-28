# TensorFlow PR Review Agent

Automated pull request review agent for TensorFlow. The agent combines static analysis (`pylint`) with semantic review via Google ADK (`google-adk`) and Gemini (`google-genai`) to post structured, evidence-based code reviews on GitHub pull requests.

## Purpose

The TensorFlow PR Review Agent evaluates pull request diffs and modified files against the principle-based guidelines referenced in `styleguide/tensorflow_pr_review.md`. It submits a GitHub pull request review containing:
- A top-level review summary prefixed with the detected PR `Category` and `Reason`.
- A GitHub pull request review `event` that is always `"COMMENT"` (`submit_pr_code_review` in `agent/agent.py` and the HTTP 422 fallback in `post_pull_request_review` in `agent/utils.py`). While the model generates `overall_assessment` text (`"No actionable review comments identified."`, `"Minor improvements suggested."`, or `"Changes required."`), `summary_comment`, and `inline_comments`, it never controls the GitHub review state (`APPROVE` or `REQUEST_CHANGES`).
- Up to 5 line-anchored inline comments (`Priority 1` through `Priority 3`) with optional GitHub `suggestion` code blocks.

## High-Level Workflow

1. **Trigger**: `.github/workflows/pr_review.yml` runs on `pull_request_target` (`types: [labeled]`) when the applied label is `Needs Review`.
2. **Reaction Status (`eyes`)**: `clear_and_set_reaction` in `agent/main.py` removes any existing `eyes` reactions on the PR issue thread and posts an `eyes` reaction while processing.
3. **PR Metadata & Diff Retrieval**: `get_pull_request_details` queries the GitHub GraphQL API (with a REST API fallback) for the configured `PULL_REQUEST_NUMBER` metadata, up to 100 changed files, and the unified diff. The diff is annotated with explicit right-side (`[L...]`) and left-side (`[LEFT L...]`) line numbers and capped at 30,000 characters.
4. **Labeled Commit BindingCheck**: `main.py` compares `PR_HEAD_SHA` (`github.event.pull_request.head.sha`) against the live PR `headRefOid`. If `PR_HEAD_SHA` is missing/invalid or does not match `headRefOid`, the agent logs the SHA mismatch and aborts immediately before running Pylint, invoking Gemini, or posting a review.
5. **Commit Idempotency Check**: `has_agent_reviewed_commit` checks whether the verified commit SHA (`verified_head_sha`) already has a review authored by `github-actions[bot]` with a matching `commit_id` and HTML commit marker. If so, the agent skips analysis, posts a `rocket` reaction, and exits.
6. **Deterministic Categorization**: `classify_pr_with_scoring` classifies the PR into one of 13 categories (`Documentation`, `Test-only`, `Build / CI`, `TensorFlow Lite`, `XLA`, `Keras`, `API change`, `oneDNN / MKL`, `Bug fix`, `Feature`, `Refactor`, `Performance`, or `General TensorFlow`) using weighted rules over modified file paths, change types, diff patterns, and PR title/body keywords. `get_focus_skip_areas` maps the category to specific `Focus Areas` and `Skip Areas`.
7. **Pylint Static Analysis**: `run_pylint_on_changed_files` runs once on modified Python files at `verified_head_sha` and collects diff-filtered diagnostics.
8. **Semantic Review & Submission**: `run_pr_review` runs the ADK `LlmAgent` (`InMemoryRunner`) through a five-stage evaluation pipeline (`Stage 0: PR Understanding`, `Stage 1: General Engineering`, `Stage 2: TensorFlow Repository Review`, `Stage 3: Category-Specific Review`, and `Stage 4: Evidence & Quality Validation Gate`) referencing `styleguide/tensorflow_pr_review.md`, and calls `submit_pr_code_review_orchestrated` to post a `"COMMENT"` review bound to `PULL_REQUEST_NUMBER` and `verified_head_sha` to `/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews`. If GitHub returns `422 Unprocessable Entity` for inline comment positions, `post_pull_request_review` falls back to appending the inline comments into the top-level review body with `event="COMMENT"`.
9. **Completion Reaction (`rocket`)**: After verifying that the review for `verified_head_sha` is recorded on GitHub, the agent clears the `eyes` reaction and posts a `rocket` reaction.

## Pylint Static-Analysis Integration

- **Scope**: Runs only on non-deleted `.py` files modified by the PR (`run_pylint_on_changed_files` in `agent/utils.py`). If no Python files were modified, Pylint is skipped.
- **Configuration**: Uses `tensorflow/tools/ci_build/pylintrc` if present, falling back to `.pylintrc` in the repository root.
- **Isolated Materialization**: Each changed Python file's content at `head_sha` is retrieved via `git show <head_sha>:<path>` (with a GitHub Contents API fallback at `ref=<head_sha>`; when `head_sha` is supplied, it never falls back to reading base-branch local files) and written into an isolated `tempfile.TemporaryDirectory(prefix="tf_pr_pylint_")`.
- **Environment Isolation**: The Pylint subprocess runs with `sys.executable -P -m pylint` (`shell=False`), `--` option termination before file paths, and a minimal environment containing only `PATH` and `HOME` (excluding `GITHUB_TOKEN`, `GEMINI_API_KEY`, and workflow metadata variables).
- **Diff Filtering**: `filter_pylint_output_by_diff` retains only:
  - Diagnostics on right-side lines added or modified in the PR diff (`extract_modified_lines_by_file`).
  - Fatal (`F*`), syntax (`E0001`), or module-level (`line 0`) errors.
  - `unused-import` (`W0611`) warnings where the unused symbol appears in lines deleted by the PR.
- **Limits & Resilience**: Output is capped at 50 diagnostics and 5,000 characters. Pylint runs with a 120-second timeout; timeouts or execution errors are caught and reported as status text without aborting the review.

## Gemini Semantic Review and Model Fallback

The agent constructs an `LlmAgent` with category context, Pylint evidence, and the guidelines loaded from `styleguide/tensorflow_pr_review.md`. `agent/agent.py` defines `MODELS_POOL` in priority order:

1. `gemini-3.1-pro-preview`
2. `gemini-3-flash-preview`
3. `gemini-flash-latest`
4. `gemini-3.1-flash-lite`

During execution (`agent/main.py`):
- Pylint runs once before the model loop, and its output is reused across any fallback attempts.
- `is_fallback_eligible_error` triggers fallback to the next model in `MODELS_POOL` on model-availability errors (`404` / `NOT_FOUND`), rate-limit errors (`429` / `RESOURCE_EXHAUSTED`), or transient server errors (`500`, `502`, `503` / `UNAVAILABLE`, `504` / `DEADLINE_EXCEEDED`), as well as when a model finishes without a confirmed review on GitHub.
- Non-fallback client/auth errors (`400`, `401`, `403`) raise immediately without retrying additional models.
- If all models in `MODELS_POOL` fail, `main.py` raises a `RuntimeError`.

## Commit Idempotency and Re-Review Behavior

- **Marker Format**: Every submitted review includes `commit_id: <verified_head_sha>` in the API payload and appends an HTML comment marker to the review body:
  ```html
  <!-- tensorflow-pr-review-agent: commit_sha=<verified_head_sha> -->
  ```
- **Same-Commit Idempotency**: Before running Pylint or Gemini, `has_agent_reviewed_commit` inspects `/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews?per_page=100`. A review counts only when `(review.get("user") or {}).get("login") == "github-actions[bot]"`, `commit_id == verified_head_sha`, and the exact HTML marker is present in `body`.
- **New-Commit Re-Review & Retry**: Pushing a new commit updates `headRefOid`; re-applying the `Needs Review` label triggers a fresh review for the new commit SHA. If a previous review submission failed, no marker is persisted on GitHub and the commit remains eligible for retry.

## GitHub Actions Permissions and Secrets

### Permissions (`.github/workflows/pr_review.yml`)
- `contents: read` — Checkout the base repository and fetch the PR head commit object.
- `pull-requests: write` — Submit pull request reviews and inline review comments via `/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews`.
- `issues: write` — List existing reactions, delete stale `eyes` reactions, and post `eyes` / `rocket` status reactions on the PR issue thread (`/repos/{OWNER}/{REPO}/issues/{pr_number}/reactions` and `/repos/{OWNER}/{REPO}/issues/reactions/{reaction_id}`) via `clear_and_set_reaction`. The active workflow does not create issue comments.

### API Key Handling (`API_NEEDS_REVIEW`)
- The Gemini API key is stored in the GitHub Actions secret `API_NEEDS_REVIEW` and injected into the workflow step as the `GEMINI_API_KEY` environment variable:
  ```yaml
  GEMINI_API_KEY: ${{ secrets.API_NEEDS_REVIEW }}
  ```
- GitHub API authentication uses the workflow's `GITHUB_TOKEN` (`${{ secrets.GITHUB_TOKEN }}`).

## Security Approach

The workflow uses `pull_request_target` so it can access `secrets.API_NEEDS_REVIEW` and write PR reviews on fork PRs, while enforcing strict trust boundaries:
- **Base Checkout Only**: `actions/checkout` (SHA-pinned with `persist-credentials: false`) checks out the repository's base branch, running only the trusted code under `pr_review_agent/`.
- **No PR Branch Checkout or Execution**: The workflow fetches the labeled commit SHA (`git fetch origin "$PR_HEAD_SHA" --depth=1`) without checking out the PR worktree or executing PR code/tests.
- **Exact Labeled Commit Binding**: `main.py` verifies `PR_HEAD_SHA == headRefOid` before analysis and binds all downstream operations to that verified SHA.
- **Trusted PR Number & Review Event Enforcement**: LLM-facing tools do not accept `pr_number`, binding all operations to the trusted `PULL_REQUEST_NUMBER` environment variable, and always submit reviews with `event="COMMENT"`.
- **Path-Validated & Secret-Isolated Static Analysis**: `_is_safe_relative_path` rejects absolute paths, `..` traversal, path components starting with `-`, and paths inside `pr_review_agent/`. Changed Python files are extracted as blobs at `verified_head_sha` into a temporary directory (`is_relative_to`-checked) and analyzed by `pylint` using a minimal `PATH`/`HOME` subprocess environment.

## Setup and Configuration

### Dependencies
Configured for Python `3.11` with packages listed in `pr_review_agent/requirements.txt`:
```bash
cd pr_review_agent
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables
`agent/settings.py` (using `python-dotenv`) and `agent/main.py` read the following environment variables:
- `GITHUB_TOKEN` (required; raises `ValueError` if unset)
- `GEMINI_API_KEY` (populated from `secrets.API_NEEDS_REVIEW` in GitHub Actions)
- `OWNER` (repository owner, e.g. `tensorflow`)
- `REPO` (repository name, e.g. `tensorflow`)
- `PULL_REQUEST_NUMBER` (target PR number)
- `PR_HEAD_SHA` (expected PR head commit SHA from the triggering `pull_request_target` event)
- `PYTHONPATH` (set to the `pr_review_agent` directory path)

### Running the Agent
From the `pr_review_agent` directory:
```bash
python -m agent.main
```

## Testing

Unit tests are provided in `pr_review_agent/test_idempotency.py` and cover:
- Commit-level idempotency (`TestCommitIdempotency`)
- Diff line extraction and Pylint integration/filtering (`TestPylintIntegration`)
- Gemini model pool fallback behavior (`TestModelFallback`)

Run the test suite from `pr_review_agent/`:
```bash
python -m unittest test_idempotency.py
```