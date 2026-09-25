# pylint: disable=bad-indentation,line-too-long

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.settings import GITHUB_BASE_URL
from agent.settings import OWNER
from agent.settings import REPO
from agent.utils import error_response
from agent.utils import get_diff
from agent.utils import post_request
from agent.utils import read_file
from agent.utils import run_graphql_query

from google.adk.agents import LlmAgent
import requests

STYLE_GUIDE = read_file(
    Path(__file__).resolve().parents[1]
    / "styleguide"
    / "tensorflow_pr_review.md"
)

# Centralized Model Pool for Fallbacks (Sorted by reasoning capability)
MODELS_POOL = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
]

import re

_PREFETCHED_PR_DETAILS = None

def get_pull_request_details(pr_number: int) -> dict[str, Any]:
    """Fetch TensorFlow PR details along with file structural metadata."""
    global _PREFETCHED_PR_DETAILS
    if _PREFETCHED_PR_DETAILS is not None:
        return _PREFETCHED_PR_DETAILS

    query = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $prNumber) {
          id
          number
          title
          body
          state
          headRefOid
          author { login }
          files(first: 100) {
            nodes {
              path
              additions
              deletions
              changeType
            }
          }
          comments(last: 50) { nodes { body createdAt author { login } } }
          commits(last: 50) { nodes { commit { url message } } }
        }
      }
    }
    """
    variables = {"owner": OWNER, "repo": REPO, "prNumber": pr_number}
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/pulls/{pr_number}"

    try:
        from agent.utils import annotate_diff_with_line_numbers
        response = run_graphql_query(query, variables)
        if "errors" in response:
            raise requests.exceptions.RequestException(str(response["errors"]))
        pr = response.get("data", {}).get("repository", {}).get("pullRequest")
        if not pr:
            raise requests.exceptions.RequestException(f"Pull Request #{pr_number} not found.")
        pr["diff"] = annotate_diff_with_line_numbers(get_diff(url))[:30000]
        return {"status": "success", "pull_request": pr}
    except Exception as e:
        try:
            from agent.utils import get_request, annotate_diff_with_line_numbers
            pr_data = get_request(url)
            files_data = get_request(f"{url}/files")
            files_nodes = [
                {
                    "path": f.get("filename", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "changeType": f.get("status", "").upper()
                }
                for f in files_data
            ]
            pr = {
                "id": str(pr_data.get("id", "")),
                "number": pr_data.get("number", pr_number),
                "title": pr_data.get("title", ""),
                "body": pr_data.get("body", "") or "",
                "state": pr_data.get("state", "").upper(),
                "headRefOid": (pr_data.get("head") or {}).get("sha", ""),
                "author": {"login": pr_data.get("user", {}).get("login", "")},
                "files": {"nodes": files_nodes},
                "comments": {"nodes": []},
                "commits": {"nodes": []},
                "diff": annotate_diff_with_line_numbers(get_diff(url))[:30000]
            }
            return {"status": "success", "pull_request": pr}
        except Exception as e2:
            return error_response(f"GraphQL error ({e}) and REST fallback error ({e2})")


def add_comment_to_pr(pr_number: int, comment: str) -> dict[str, Any]:
    """Post review feedback to the PR."""
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{pr_number}/comments"
    payload = {"body": comment}
    try:
        post_request(url, payload)
    except requests.exceptions.RequestException as e:
        return error_response(str(e))
    return {"status": "success", "added_comment": comment}


def submit_pr_code_review(
    pr_number: int,
    overall_assessment: str,
    summary_comment: str,
    inline_comments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Post structured pull request review with top-level summary and line-level inline comments."""
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews"
    formatted_comments = []
    for ic in (inline_comments or []):
        path = ic.get("path", "")
        line = ic.get("line")
        body = ic.get("body", "")
        suggestion = ic.get("suggestion_code", "")
        if suggestion and "```suggestion" not in body:
            body = f"{body.rstrip()}\n\n```suggestion\n{suggestion.rstrip()}\n```"
        if path and line:
            formatted_comments.append({
                "path": path,
                "line": int(line),
                "side": ic.get("side", "RIGHT"),
                "body": body
            })

    head_sha = (
        (_PREFETCHED_PR_DETAILS or {})
        .get("pull_request", {})
        .get("headRefOid", "")
    )
    if head_sha:
        from agent.utils import format_commit_review_marker
        marker = format_commit_review_marker(head_sha)
        if marker.strip() not in summary_comment:
            summary_comment = summary_comment.rstrip() + marker

    event = "COMMENT"
    if "Changes required" in overall_assessment:
        event = "REQUEST_CHANGES"
    elif "No actionable review comments" in overall_assessment:
        event = "APPROVE"

    payload: dict[str, Any] = {
        "body": summary_comment,
        "event": event,
        "comments": formatted_comments
    }
    if head_sha:
        payload["commit_id"] = head_sha
    try:
        from agent.utils import post_pull_request_review
        response = post_pull_request_review(url, payload)
        return {"status": "success", "response": response}
    except Exception as e:
        return error_response(str(e))


def get_focus_skip_areas(category: str) -> tuple[str, str]:
    """Returns focus and skip areas for a given PR category."""
    if category == "Documentation":
        return (
            "Markdown structure, document formatting, spelling, links, correctness of code examples, readability.",
            "Testing requirements, API design, performance optimization, model training best practices, numerical stability."
        )
    elif category == "Test-only":
        return (
            "Test coverage, reliability (avoiding flakiness), test execution speed, platform compatibility, edge case verification, clear assertions.",
            "Production API design, model training efficiency, deployment constraints, style guide rules unrelated to tests."
        )
    elif category == "Build / CI":
        return (
            "Bazel configurations (BUILD, WORKSPACE), GitHub Actions workflow accuracy, toolchains, Python/C++ compiler flags, setup scripts.",
            "Python/C++ code style guidelines, ML modeling practices, API design compatibility."
        )
    elif category == "TensorFlow Lite":
        return (
            "Mobile/embedded deployment constraints, quantization, FlatBuffer schema compatibility, hardware delegate integration, binary size optimization.",
            "Large-scale distributed training, server-side data pipelining, non-Lite API stability."
        )
    elif category == "XLA":
        return (
            "Compiler optimization (HLO passes, operator fusion), shape/dimension constraints (dynamic shapes limitations), JIT compilation, TPU/GPU code generation.",
            "Python API style guidelines, Keras high-level callback designs, standard CPU performance."
        )
    elif category == "Keras":
        return (
            "Keras APIs (layers, callbacks, losses, custom training loops), serialization/saving models, functional vs sequential design.",
            "Low-level C++ kernel optimizations, XLA compilation, TFLite mobile delegation."
        )
    elif category == "API change":
        return (
            "Public API (tf.*) backward compatibility, deprecation lifecycle, naming conventions, argument validation, decorator placements (@tf_export).",
            "Test helper details, internal implementation refactoring, build configs."
        )
    elif category == "oneDNN / MKL":
        return (
            "oneDNN/MKL layout buffer awareness (IsMklTensor(), logical_shape vs layout geometry, GetTfShape()), memory safety, broadcasting rank alignment, thread safety.",
            "Python API docstrings, mobile TFLite constraints, high-level Keras callbacks."
        )
    elif category == "Bug fix":
        return (
            "Correctness, edge cases, error handling, preventing segfaults/nullptr dereferences, regression testing.",
            "Stylistic refactoring, feature enhancements, documentation details."
        )
    elif category == "Feature":
        return (
            "Design necessity, API cleanliness, comprehensive test coverage, robust error handling, documentation.",
            "Subjective styling, optimization of unrelated code paths."
        )
    elif category == "Refactor":
        return (
            "Code readability, maintainability, formatting, removing dead code, preserving identical behavior.",
            "Performance enhancement, new feature design, api signature changes."
        )
    elif category == "Performance":
        return (
            "Vectorization (preventing python loops), tf.function retracing prevention, memory usage, CPU/GPU utilization, caching.",
            "Documentation spelling, test style, api backward compatibility."
        )
    else:
        return (
            "General TensorFlow contribution guidelines, code quality, test coverage, correctness, performance.",
            "None."
        )


def classify_pr_with_scoring(files: list[dict[str, Any]], title: str, body: str, diff: str) -> tuple[str, str]:
    """Determines the PR category and concise reason using a weighted scoring system."""
    scores = {
        "Documentation": 0,
        "Test-only": 0,
        "Bug fix": 0,
        "Feature": 0,
        "Refactor": 0,
        "Performance": 0,
        "API change": 0,
        "Build / CI": 0,
        "TensorFlow Lite": 0,
        "XLA": 0,
        "Keras": 0,
        "oneDNN / MKL": 0,
        "General TensorFlow": 1,
    }
    reasons = []

    num_files = len(files)
    if num_files > 0:
        is_md_file = lambda f: f.endswith('.md') or '/docs/' in f
        is_test_file = lambda f: '_test.' in f or 'test_' in f or f.endswith('test.py') or f.endswith('test.cc') or f.endswith('test.h') or '/testdata/' in f
        is_build_ci_file = lambda f: f.startswith('.github/') or f.endswith('.yml') or f.endswith('.yaml') or f.endswith('BUILD') or f.endswith('WORKSPACE') or f.endswith('MODULE.bazel') or f.endswith('.bazelrc') or f.endswith('.bazelversion') or f.endswith('.bzl') or 'ci/' in f

        paths = [f.get("path", "") for f in files]

        if all(is_md_file(p) for p in paths):
            scores["Documentation"] += 100
            reasons.append("All modified files are documentation files.")
        if all(is_test_file(p) for p in paths):
            scores["Test-only"] += 100
            reasons.append("All modified files are test files.")
        if all(is_build_ci_file(p) for p in paths):
            scores["Build / CI"] += 100
            reasons.append("All modified files are build or CI configuration files.")

        tflite_files = 0
        xla_files = 0
        keras_files = 0
        build_ci_files = 0
        api_files_with_changes = 0
        renamed_files = 0
        balanced_edit_files = 0

        sig_pattern = re.compile(r'^[+-]\s*(def |class |@tf_export)', re.MULTILINE)

        for f in files:
            p = f.get("path", "")
            additions = f.get("additions", 0)
            deletions = f.get("deletions", 0)
            change_type = f.get("changeType", "")

            if 'tensorflow/lite/' in p or 'lite/' in p:
                tflite_files += 1
            if 'tensorflow/compiler/xla/' in p or 'xla/' in p or 'third_party/xla/' in p:
                xla_files += 1
            if 'tensorflow/python/keras/' in p or 'keras/' in p:
                keras_files += 1
            if 'tensorflow/core/kernels/mkl/' in p or '/mkl/' in p or 'onednn' in p.lower() or p.split('/')[-1].startswith('mkl_'):
                scores["oneDNN / MKL"] += 50
                reasons.append(f"File {p} modified under oneDNN/MKL kernel directory or naming structure.")
            if is_build_ci_file(p):
                build_ci_files += 1
            
            if (p.startswith('tensorflow/python/') and 
                not is_test_file(p) and 
                not '/internal/' in p and 
                not p.split('/')[-1].startswith('_')):
                if sig_pattern.search(diff):
                    api_files_with_changes += 1
                if change_type in ("ADDED", "DELETED"):
                    scores["API change"] += 30
                    scores["Feature"] += 20 if change_type == "ADDED" else 0
                    reasons.append(f"Public API file {p} was {change_type.lower()}.")

            if change_type == "RENAMED":
                renamed_files += 1
            elif additions > 10 and deletions > 10:
                total_edits = additions + deletions
                if abs(additions - deletions) / total_edits < 0.15:
                    balanced_edit_files += 1

        if tflite_files > 0:
            scores["TensorFlow Lite"] += tflite_files * 50
            reasons.append(f"{tflite_files} file(s) modified under tensorflow/lite/.")
        if xla_files > 0:
            scores["XLA"] += xla_files * 50
            reasons.append(f"{xla_files} file(s) modified under xla/ or compiler/xla/.")
        if keras_files > 0:
            scores["Keras"] += keras_files * 50
            reasons.append(f"{keras_files} file(s) modified under keras/.")
        if build_ci_files > 0 and not all(is_build_ci_file(f.get("path", "")) for f in files):
            scores["Build / CI"] += build_ci_files * 30
            reasons.append(f"{build_ci_files} build/CI file(s) modified.")
        if api_files_with_changes > 0:
            scores["API change"] += api_files_with_changes * 40
            reasons.append(f"{api_files_with_changes} public API file(s) modified with signature changes.")
        if renamed_files > 0:
            scores["Refactor"] += renamed_files * 25
            reasons.append(f"{renamed_files} file(s) renamed/moved.")
        if balanced_edit_files > 0:
            scores["Refactor"] += balanced_edit_files * 15
            reasons.append(f"{balanced_edit_files} file(s) edited with balanced additions/deletions.")

    combined_text = f"{title}\n{body}".lower()

    keyword_map = {
        "Performance": (["performance", "speedup", "latency", "throughput", "faster", "optimize", "optimization", "efficient", "cache", "memory usage", "benchmark", "profiler", "allocator"], 15),
        "Refactor": (["refactor", "cleanup", "clean up", "simplify", "restructure", "formatting", "rename", "unused", "dead code", "deprecation"], 15),
        "Bug fix": (["fix", "bug", "issue", "crash", "error", "leak", "resolve", "close", "prevent", "incorrect", "wrong", "nullptr", "oob", "out of bounds", "segfault", "overflow", "check fails", "division by zero", "check", "validation", "bounds", "mismatch", "corrupt", "data loss", "slice"], 15),
        "oneDNN / MKL": (["onednn", "mkl", "mkldnn", "intel mkl", "batch matmul helper"], 10),
        "Feature": (["feature", "implement", "introduce", "support for", "add support", "new api", "new op", "adding op", "enable support"], 15),
        "TensorFlow Lite": (["tflite", "lite", "micro", "tensorflow lite", "mobile", "android", "ios", "quantization", "delegate"], 10),
        "XLA": (["xla", "hlo", "jit", "compiler", "lower", "lowering", "gpu compiler", "tpu compiler"], 10),
        "Keras": (["keras", "layer", "model", "callback", "loss", "metric", "sequential", "functional api"], 10),
        "Documentation": (["docs", "documentation", "readme", "markdown", "tutorial", "guide"], 10),
        "Build / CI": (["ci", "workflow", "github actions", "bazel", "build", "toolchain", "cmake", "docker"], 10),
    }

    for cat, (keywords, weight) in keyword_map.items():
        matches = [kw for kw in keywords if kw in combined_text]
        if matches:
            scores[cat] += weight
            reasons.append(f"PR title/description matches keywords: {', '.join(matches[:3])} (+{weight} pts).")

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_cat, best_score = sorted_scores[0]

    if best_cat == "General TensorFlow" and best_score == 1:
        explanation = "No specific category matches based on modified files or keywords; falling back to General TensorFlow."
    else:
        if best_cat == "Documentation":
            if any("All modified files are documentation" in r for r in reasons):
                explanation = "All modified files are documentation files (such as Markdown or guides)."
            else:
                explanation = "The pull request modifies documentation and inline documentation files."
        elif best_cat == "Test-only":
            if any("All modified files are test files" in r for r in reasons):
                explanation = "The changes are entirely limited to test files."
            else:
                explanation = "The changes are focused on test coverage and test suites."
        elif best_cat == "Build / CI":
            if any("All modified files are build or CI" in r for r in reasons):
                explanation = "All modified files are build configurations or CI workflow configurations."
            else:
                explanation = "The pull request modifies build scripts, configurations, or CI files."
        elif best_cat == "TensorFlow Lite":
            if tflite_files > 0:
                explanation = "The modified files are located under the TensorFlow Lite directory structure."
            else:
                explanation = "The pull request context and keywords refer to TensorFlow Lite/TFLite development."
        elif best_cat == "XLA":
            if xla_files > 0:
                explanation = "The modified files are located under the XLA compiler directory structure."
            else:
                explanation = "The pull request context and keywords refer to XLA compiler optimization."
        elif best_cat == "Keras":
            if keras_files > 0:
                explanation = "The modified files are located under the Keras directory structure."
            else:
                explanation = "The pull request context and keywords refer to Keras API or model configurations."
        elif best_cat == "API change":
            if api_files_with_changes > 0 or any("Public API file" in r and "deleted" in r for r in reasons):
                explanation = "The pull request modifies public API signatures under the tensorflow/python/ namespace."
            else:
                explanation = "The pull request context and keywords indicate public API modifications."
        elif best_cat == "oneDNN / MKL":
            explanation = "The modified files and context indicate oneDNN / MKL kernel optimization or bug fix."
        elif best_cat == "Performance":
            explanation = "The pull request is focused on performance optimization, latency reduction, or resource efficiency."
        elif best_cat == "Refactor":
            if renamed_files > 0:
                explanation = "The pull request contains renamed or moved source files, suggesting code refactoring."
            elif balanced_edit_files > 0:
                explanation = "The modifications involve balanced additions and deletions in existing source files, typical of refactoring."
            else:
                explanation = "The pull request context indicates code refactoring or cleanup."
        elif best_cat == "Bug fix":
            explanation = "The pull request description and keywords indicate it addresses a bug, issue, or crash fix."
        elif best_cat == "Feature":
            explanation = "The pull request description and keywords indicate it implements a feature or new capability."
        else:
            explanation = f"The pull request was classified as {best_cat}."

    return best_cat, explanation


IMMUTABLE_BASE_INSTRUCTIONS = f"""
# Identity

You are an experienced TensorFlow maintainer performing pull request reviews.

# Review Objective

Produce high-signal, code-specific, evidence-based review feedback that helps maintain TensorFlow quality while minimizing false positives and unnecessary comments. You must separate high-level summary overview from exact line-level code suggestions.

# Review Guidelines

{STYLE_GUIDE}

# Required Review Process

1. Call get_pull_request_details.
2. Read the pull request title, description, modified files metadata (additions, deletions, changeType), and diff (annotated with right-side `[L...]` and left-side `[LEFT L...]` line numbers).
3. Note that the pull request category has already been programmatically determined for this run. Do not attempt to recalculate it.
4. Execute the Five-Stage Evaluation Pipeline sequentially:
   - **Stage 0 (PR Understanding & Scope Verification)**: Summarize the architectural objective of the pull request, map out the data paths and boundaries modified, and explicitly identify which hunks require detailed verification.
   - **Stage 1 (General Engineering Review)**: Evaluate universal software engineering soundness across all modified files and relevant surrounding source context. Start with the changed lines, then inspect the surrounding definitions/imports/control flow needed to verify each potential finding. Do not report issues that are disproved by the surrounding source.
   - **Stage 2 (TensorFlow Repository Review)**: Evaluate alignment with repository conventions (`absl::Status` reporting, early-return macro semantics like `OP_REQUIRES`, Bazel `BUILD` target grouping/minimalism, `jit_compile=True` boundaries, deterministic reproducibility).
   - **Stage 3 (Category-Specific Review)**: Apply the specialized domain rules from the style guide relevant to the detected category and focus areas.
   - **Stage 4 (Evidence & Quality Validation Gate)**: Before including any candidate finding in your review output, verify it passes all mandatory checks:

      1. **Exact source evidence**
         - The finding must be directly supported by the current PR diff and/or the actual modified-file contents returned by `get_pull_request_details`.
         - Anchor every localized finding to an exact annotated right-side `[L...]` line or deletion `[LEFT L...]` line.
         - Never infer a defect merely because something "could" be wrong.

      2. **Symbol, import, and definition verification**
         - Before reporting an undefined name, missing import, incorrect type annotation, unresolved symbol, or similar Python/C++ semantic issue, inspect the relevant source context.
         - Verify whether the referenced symbol is already imported, defined locally, inherited, available from an enclosing scope, or otherwise valid.
         - For Python, explicitly check relevant `import` and `from ... import ...` statements before reporting a missing import or undefined name.
         - For example, if a file contains `from typing import Any`, do NOT report `Any` as undefined or recommend adding an import.
         - Never create a hypothetical finding based only on what might happen if an import or definition were absent.

      3. **Technical correctness**
         - The finding and proposed fix must be technically correct under the actual C++/Python language semantics and applicable `-Werror` rules.
         - Do not recommend changing valid code merely because another form is also possible.
         - If multiple valid forms exist, do not present one as a correctness defect unless the PR introduces a concrete compatibility, correctness, or repository-convention problem.

      4. **Actionability**
         - Provide a concrete, drop-in `suggestion_code` only when a safe, compiling replacement is possible.
         - The suggestion must actually fix the identified defect.
         - Never provide a suggestion that introduces a new error or changes valid code unnecessarily.

      5. **PR-local verification**
         - Verify that the issue is not already handled elsewhere in the pull request.
         - Consider the complete relevant diff and surrounding source context before reporting the finding.

      6. **No duplication or filler**
         - Do not report duplicate findings.
         - Prefer NO findings over speculative, weak, stylistic, or filler comments.

      7. **Category and skip-area validation**
         - The finding must be relevant to the pre-detected PR category and Focus Areas.
         - It must not fall inside the configured Skip Areas.

      8. **Priority validation**
         - Assign Priority 1 only to confirmed correctness, security, or memory-safety defects.
         - Assign Priority 2 to confirmed portability, compiler, or performance issues.
         - Assign Priority 3 to confirmed maintainability or repository-convention issues.
         - Do not escalate a speculative or hypothetical issue to Priority 1.

      9. **Counterfactual verification**
         - Before submitting a finding, ask:
           "If the suggested change were NOT made, is there concrete evidence in the current source that the PR is actually defective?"
         - If the answer is no, discard the finding.
      10. **Static-analysis verification**
          - When Pylint output is provided, treat Pylint diagnostics as concrete static-analysis evidence.
          - Do not invent Pylint warnings or claim that Pylint reports an issue unless actual Pylint output confirms it.
          - Verify that the reported Pylint diagnostic applies to code modified by the pull request.
          - Do not report unrelated Pylint warnings from unchanged legacy code.
          - Use the actual Pylint message ID and diagnostic location when available.
          - If Pylint reports an issue but the issue is already fixed elsewhere in the PR, discard it.
          - If Pylint reports an issue that is irrelevant to the PR's scope or Focus Areas, discard it.

      [CRITICAL ACTION]: Any candidate finding that fails ANY mandatory validation check MUST be discarded immediately.
5. Base all findings primarily on the modified code and diff.
6. Use the title, description, and commit messages only as supporting context.
7. Ignore unchanged files and topics listed in `Skip Areas`.

# Critical Rules

- Do not speculate.
- Do not hallucinate findings or line numbers.
- If the available evidence is insufficient to support a finding, do not generate that finding.
- If you are uncertain whether a behavior is intentional or incorrect based on the available evidence, do not present it as a defect. Instead, either omit the finding or clearly phrase it as a question for the author.
- Do not recommend repository-wide refactoring when the pull request modifies only a localized implementation.
- Do not assume missing tests unless the modified code clearly requires them.
- Do not suggest tf.data.Dataset, callbacks, validation datasets, batching, reproducibility, or training improvements unless the PR actually modifies training code.
- Do not discuss API stability unless public TensorFlow APIs are modified.
- Do not discuss performance unless performance-sensitive code is modified.
- Documentation-only changes should receive documentation-focused review only.
- Configuration-only changes should receive configuration-focused review only.
- README-only changes should not trigger TensorFlow model-training recommendations.
- Every finding must primarily reference evidence from the modified diff or files. Use the title and description only as supporting context.
- If evidence cannot be found in the pull request, do not mention it.
- Every inline finding must target an exact file path and line number from the annotated diff (`[L...]`).
- Findings without supporting evidence must not be included.

# Semantic Verification Rules

- When a finding depends on whether a symbol, import, function, class, variable, attribute, or type exists, verify its existence from the actual source context before reporting the finding.
- Do not report "missing import" issues using hypothetical language such as "if X is not imported" when the relevant file contents are available.
- Do not recommend replacing a valid Python type annotation with a different syntax unless the current syntax causes a concrete technical problem.
- In Python, distinguish between a genuinely unresolved name and a name that is already imported or defined.
- When `from typing import Any` is present, `Any` is a valid available typing symbol and must not be reported as undefined.
- A suggestion must be evaluated against the source state before the suggestion is emitted. Do not produce a suggestion that merely reverses or changes valid code.
- If source inspection resolves the concern, discard the candidate finding rather than mentioning the concern as a hypothetical.

# Review Quality Bar

If no meaningful issues are identified across any stage, output exactly `inline_comments = []` and provide a clean `summary_comment`:

## Summary
[One sentence describing the PR]

## Overall Assessment
No actionable review comments identified. The modified files and diff were reviewed against the TensorFlow PR Review Guidelines and no evidence-based concerns were found.

Prefer NO findings over weak, filler, or duplicate recommendations. 

# Output Format and Tool Calling (`submit_pr_code_review_orchestrated`)

When you complete your evaluation, you MUST call the `submit_pr_code_review_orchestrated` tool with:
1. `pr_number`: The integer ID of the pull request.
2. `summary_comment`: The top-level summary of the pull request, positive observations (if any), and overall architectural assessment. DO NOT put localized code defects or line-level suggestions in `summary_comment`! Keep `summary_comment` strictly for high-level overview.
3. `overall_assessment`: Exactly one of: "No actionable review comments identified.", "Minor improvements suggested.", or "Changes required."
4. `inline_comments`: An array of structured objects for every localized defect that passed the Stage 4 Validation Gate. Maximum 5 items total across all files. For each inline comment provide:
   - `path`: The exact relative file path (e.g. `tensorflow/core/util/tensor_bundle/tensor_bundle.cc`).
   - `line`: The exact integer right-side line number extracted from the `[L...]` annotation in the diff (or `[LEFT L...]` if targeting deletion with `side="LEFT"`).
   - `side`: "RIGHT" for additions/context (default), or "LEFT" for deletions.
   - `body`: A clear, concise explanation starting with the priority badge (e.g. `**[Priority 1: Correctness]** ...`).
   - `suggestion_code`: The exact, compiling drop-in replacement code for that line/block without surrounding backticks. If a drop-in code edit is not possible, leave empty string.
"""

root_agent = LlmAgent(
    model=MODELS_POOL[0],
    name="tensorflow_pr_review_agent",
    description="Reviews TensorFlow pull requests using style guidelines.",
    instruction=IMMUTABLE_BASE_INSTRUCTIONS,
    tools=[get_pull_request_details, submit_pr_code_review],
)


def make_review_agent(
    model_name: str,
    category: str,
    reason: str,
    focus_areas: str,
    skip_areas: str,
    post_comment_callback=None,
    pylint_output: str = ""
) -> LlmAgent:
    """Creates a fresh, immutable-base agent instance configured with Category Context and Pylint evidence."""
    pylint_evidence = (
        pylint_output.strip()
        if pylint_output and pylint_output.strip()
        else "No Pylint issues detected on modified lines."
    )
    context_block = (
        f"# PR Categorization Context\n"
        f"- Detected Category: {category}\n"
        f"- Reason: {reason}\n"
        f"- Focus Areas: {focus_areas}\n"
        f"- Skip Areas: {skip_areas}\n\n"
        f"# Static Analysis (Pylint) Evidence\n"
        f"The following Pylint diagnostics were collected using TensorFlow's pylintrc against changed Python files and filtered to lines modified by this PR:\n"
        f"```text\n"
        f"{pylint_evidence}\n"
        f"```\n"
        f"Treat these diagnostics as concrete static-analysis evidence under Stage 4 Check #10. Validate whether each finding is applicable to the changed code before reporting.\n\n"
        f"This pull request has already been programmatically classified. "
        f"Review it using the Five-Stage Evaluation Pipeline below, ensuring you evaluate Stage 0 (PR Understanding), Stage 1 (General Engineering), Stage 2 (Repository Conventions), Stage 3 (Category-Specific Rules for {category}), and pass all findings through the Stage 4 Validation Gate.\n"
        f"Do not comment on or evaluate topics listed in the 'Skip Areas'.\n\n"
        f"---\n\n"
    )
    
    agent_instruction = context_block + IMMUTABLE_BASE_INSTRUCTIONS
    
    def submit_pr_code_review_orchestrated(
        pr_number: int,
        overall_assessment: str,
        summary_comment: str,
        inline_comments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Post structured pull request review with pre-classified summary header and inline comments."""
        header = f"Category: {category}\nReason: {reason}\n\n"
        if not summary_comment.startswith("Category:"):
            summary_comment = header + summary_comment
            
        formatted_comments = []
        for ic in (inline_comments or []):
            path = ic.get("path", "")
            line = ic.get("line")
            body = ic.get("body", "")
            suggestion = ic.get("suggestion_code", "")
            if suggestion and "```suggestion" not in body:
                body = f"{body.rstrip()}\n\n```suggestion\n{suggestion.rstrip()}\n```"
            if path and line:
                formatted_comments.append({
                    "path": path,
                    "line": int(line),
                    "side": ic.get("side", "RIGHT"),
                    "body": body
                })

        if post_comment_callback:
            full_mock_output = f"{summary_comment}\n\nOverall Assessment: {overall_assessment}\n\n### Inline Comments:\n"
            for fc in formatted_comments:
                full_mock_output += f"\n- [{fc['path']} L{fc['line']} ({fc['side']})]: {fc['body']}"
            return post_comment_callback(pr_number, full_mock_output)
            
        return submit_pr_code_review(pr_number, overall_assessment, summary_comment, formatted_comments)

    return LlmAgent(
        model=model_name,
        name="tensorflow_pr_review_agent",
        description="Reviews TensorFlow pull requests using style guidelines.",
        instruction=agent_instruction,
        tools=[get_pull_request_details, submit_pr_code_review_orchestrated],
    )


async def run_pr_review(
    model_name: str,
    pr_number: int,
    category: str,
    reason: str,
    focus_areas: str,
    skip_areas: str,
    pylint_output: str = ""
) -> str:
    """Orchestrates and executes the PR review runner using a fresh agent instance."""
    review_agent = make_review_agent(
        model_name=model_name,
        category=category,
        reason=reason,
        focus_areas=focus_areas,
        skip_areas=skip_areas,
        pylint_output=pylint_output
    )
    
    from google.adk.runners import InMemoryRunner
    from agent.utils import call_agent_async
    
    APP_NAME = "tensorflow_pr_review_app"
    USER_ID = "tensorflow_pr_review_user"
    
    runner = InMemoryRunner(agent=review_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    
    prompt = (
        f"Execute your full workflow for pull request #{pr_number}:\n"
        f"1. Run the `get_pull_request_details` tool to fetch the diff and content.\n"
        f"2. Note that your pre-detected category is: {category}.\n"
        f"3. Analyze the code changes using the TensorFlow PR Review Guidelines style guide and the Static Analysis (Pylint) Evidence provided, "
        f"focusing strictly on the 'Focus Areas' and ignoring the 'Skip Areas'.\n"
        f"4. Generate your architectural feedback review findings.\n"
        f"5. Run the `submit_pr_code_review_orchestrated` tool to post your final review feedback comment onto the PR."
    )
    
    return await call_agent_async(runner, USER_ID, session.id, prompt)
