# TensorFlow PR Review Agent

Automated pull request review agent for TensorFlow. The agent combines static analysis (`pylint`) with semantic review via Google ADK (`google-adk`) and Gemini (`google-genai`) to post structured, evidence-based code reviews on GitHub pull requests.

## Purpose

The TensorFlow PR Review Agent evaluates pull request diffs and modified files against the principle-based guidelines referenced in `styleguide/tensorflow_pr_review.md`. It submits a GitHub pull request review containing:
- A top-level review summary prefixed with the detected PR `Category` and `Reason`.
- A GitHub pull request review `event` determined by `submit_pr_code_review` in `agent/agent.py` from `overall_assessment`:
  - `"No actionable review comments"` in `overall_assessment` -> `APPROVE`
  - `"Changes required"` in `overall_assessment` -> `REQUEST_CHANGES`
  - Otherwise (default, including `"Minor improvements suggested."`) -> `COMMENT`
- Up to 5 line-anchored inline comments (`Priority 1` through `Priority 3`) with optional GitHub `suggestion` code blocks.

## High-Level Workflow

1. **Trigger**: `.github/workflows/pr_review.yml` runs on `pull_request_target` (`types: [labeled]`) when the applied label is `Needs Review`.
2. **Reaction Status (`eyes`)**: `clear_and_set_reaction` in `agent/main.py` removes any existing `eyes` reactions on the PR issue thread and posts an `eyes` reaction while processing.
3. **PR Metadata & Diff Retrieval**: `get_pull_request_details` queries the GitHub GraphQL API (with a REST API fallback) for PR metadata, up to 100 changed files, and the unified diff. The diff is annotated with explicit right-side (`[L...]`) and left-side (`[LEFT L...]`) line numbers and capped at 30,000 characters.
4. **Commit Idempotency Check**: `has_agent_reviewed_commit` checks whether the current `headRefOid` already has a review from this agent. If so, the agent skips analysis, posts a `rocket` reaction, and exits.
5. **Deterministic Categorization**: `classify_pr_with_scoring` classifies the PR into one of 13 categories (`Documentation`, `Test-only`, `Build / CI`, `TensorFlow Lite`, `XLA`, `Keras`, `API change`, `oneDNN / MKL`, `Bug fix`, `Feature`, `Refactor`, `Performance`, or `General TensorFlow`) using weighted rules over modified file paths, change types, diff patterns, and PR title/body keywords. `get_focus_skip_areas` maps the category to specific `Focus Areas` and `Skip Areas`.
6. **Pylint Static Analysis**: `run_pylint_on_changed_files` runs once on modified Python files and collects diff-filtered diagnostics.
7. **Semantic Review & Submission**: `run_pr_review` runs the ADK `LlmAgent` (`InMemoryRunner`) through a five-stage evaluation pipeline (`Stage 0: PR Understanding`, `Stage 1: General Engineering`, `Stage 2: TensorFlow Repository Review`, `Stage 3: Category-Specific Review`, and `Stage 4: Evidence & Quality Validation Gate`) referencing `styleguide/tensorflow_pr_review.md`, and calls `submit_pr_code_review_orchestrated` to post the review to `/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews`. If GitHub returns `422 Unprocessable Entity` for inline comment positions, `post_pull_request_review` falls back to appending the inline comments into the top-level review body with the same review `event`.
8. **Completion Reaction (`rocket`)**: After verifying that the review for `headRefOid` is recorded on GitHub, the agent clears the `eyes` reaction and posts a `rocket` reaction.

## Pylint Static-Analysis Integration

- **Scope**: Runs only on non-deleted `.py` files modified by the PR (`run_pylint_on_changed_files` in `agent/utils.py`). If no Python files were modified, Pylint is skipped.
- **Configuration**: Uses `tensorflow/tools/ci_build/pylintrc` if present, falling back to `.pylintrc` in the repository root.
- **Isolated Materialization**: Each changed Python file's content at `head_sha` is retrieved via `git show <head_sha>:<path>` (with a GitHub Contents API fallback and local file fallback) and written into an isolated `tempfile.TemporaryDirectory(prefix="tf_pr_pylint_")`.
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

- **Marker Format**: Every submitted review includes `commit_id: <headRefOid>` in the API payload and appends an HTML comment marker to the review body:
  ```html
  <!-- tensorflow-pr-review-agent: commit_sha=<headRefOid> -->
  ```
- **Same-Commit Idempotency**: Before running Pylint or Gemini, `has_agent_reviewed_commit` inspects `/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews?per_page=100`. If a review matching both `commit_id == head_sha` and the HTML marker exists, duplicate execution is skipped.
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

The workflow uses `pull_request_target` so it can access `secrets.API_NEEDS_REVIEW` and write PR reviews on fork PRs, while preventing untrusted code execution:
- **Base Checkout Only**: `actions/checkout@v5` checks out the repository's base branch, running only the trusted code under `pr_review_agent/`.
- **No PR Branch Checkout or Execution**: The workflow runs `git fetch origin ${{ github.event.pull_request.head.sha }} --depth=1` without checking out the PR worktree or executing PR code/tests.
- **Path-Validated Isolated Inspection**: `_is_safe_relative_path` rejects absolute paths, `..` traversal, and paths inside `pr_review_agent/`. Changed Python files are extracted as blobs via `git show <head_sha>:<path>` into a temporary directory (`is_relative_to`-checked) for static analysis by `pylint`.

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