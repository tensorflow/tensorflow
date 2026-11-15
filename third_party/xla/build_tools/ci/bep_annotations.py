# Copyright 2026 The OpenXLA Authors.
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
"""Emits GitHub Actions ::error annotations from Bazel BEP JSON files."""

from __future__ import annotations

import argparse
import base64
import bisect
from collections.abc import Sequence
import dataclasses
import json
import logging
import os
import sys
from typing import TextIO
import urllib.parse
import urllib.request

DEFAULT_MAX_ANNOTATIONS = 10
DEFAULT_MAX_MESSAGE_CHARS = 3500
_MAX_TITLE_CHARS = 120
_TAIL_LINES = 50
_MAX_LOG_BYTES = 65536
_PASSING_STATUSES = frozenset({"PASSED", "FLAKY", ""})


@dataclasses.dataclass(frozen=True)
class _Failure:
  """Single failing build/test entry parsed from a BEP file."""

  title: str
  invocation_ref: str = ""
  detail_text: str = ""
  log_files: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True)
class _ParsedBep:
  """Parsed invocation summary and failures from one BEP file."""

  invocation_ref: str
  exit_code: int | None
  failures: tuple[_Failure, ...]


def _as_dict(value: object) -> dict[str, object]:
  return value if isinstance(value, dict) else {}


def _escape(text: str, *, is_property: bool = False) -> str:
  text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
  return text.replace(":", "%3A").replace(",", "%2C") if is_property else text


def _escape_prefix(
    text: str, max_chars: int, *, is_property: bool = False
) -> str:
  """Escapes `text` and truncates at a character boundary to fit `max_chars`."""
  if max_chars <= 0:
    return ""
  escaped = _escape(text, is_property=is_property)
  if len(escaped) <= max_chars:
    return escaped
  candidate = text[:max_chars]
  bad_index = bisect.bisect_left(
      range(len(candidate) + 1),
      True,
      key=lambda k: len(_escape(candidate[:k], is_property=is_property))
      > max_chars,
  )
  return _escape(candidate[: max(0, bad_index - 1)], is_property=is_property)


def _escape_tail(text: str, max_chars: int) -> str:
  """Escapes `text`, keeping the longest suffix fitting within `max_chars`."""
  escaped = _escape(text)
  if len(escaped) <= max_chars:
    return escaped
  prefix = "...%0A"
  limit = max_chars - len(prefix)
  if limit <= 0:
    return ""
  candidate = text[-limit:]
  start = bisect.bisect_left(
      range(len(candidate) + 1),
      True,
      key=lambda index: len(_escape(candidate[index:])) <= limit,
  )
  return prefix + _escape(candidate[start:])


def _read_log_tail(file_obj: object) -> str:
  """Returns up to the last `_TAIL_LINES` lines from a BEP file artifact."""
  if not isinstance(file_obj, dict):
    return ""
  contents = file_obj.get("contents")
  if isinstance(contents, str) and contents:
    try:
      raw = base64.b64decode(contents).decode("utf-8", errors="replace")
    except ValueError:
      return ""
    return "\n".join(raw.splitlines()[-_TAIL_LINES:])
  uri = file_obj.get("uri")
  if not isinstance(uri, str) or not uri.startswith("file://"):
    return ""
  path = urllib.request.url2pathname(urllib.parse.urlparse(uri).path)
  try:
    with open(path, "rb") as f:
      f.seek(0, os.SEEK_END)
      f.seek(max(0, f.tell() - _MAX_LOG_BYTES))
      raw = f.read().decode("utf-8", errors="replace")
  except OSError:
    return ""
  return "\n".join(raw.splitlines()[-_TAIL_LINES:])


def _parse_bep(bep_path: str) -> _ParsedBep:
  """Parses invocation metadata and failure entries from one BEP JSONL file."""
  invocation_id = ""
  bes_url = ""
  exit_code: int | None = None
  test_failures: dict[str, _Failure] = {}
  other_failures: list[_Failure] = []

  with open(bep_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
      if not line.strip():
        continue
      try:
        event = json.loads(line)
      except json.JSONDecodeError:
        continue
      if not isinstance(event, dict):
        continue
      event_id = _as_dict(event.get("id"))

      if isinstance(event.get("started"), dict):
        invocation_id = str(event["started"].get("uuid") or "") or invocation_id
      elif isinstance(event.get("buildMetadata"), dict) and not invocation_id:
        metadata = _as_dict(event["buildMetadata"].get("metadata"))
        invocation_id = str(metadata.get("INVOCATION_ID") or "")
      elif isinstance(event.get("optionsParsed"), dict) and not bes_url:
        cmd_line = event["optionsParsed"].get("cmdLine")
        if isinstance(cmd_line, list):
          for option in cmd_line:
            if isinstance(option, str) and option.startswith(
                "--bes_results_url="
            ):
              bes_url = option.split("=", 1)[1].strip()
      elif isinstance(event.get("testResult"), dict):
        test_result = event["testResult"]
        status = str(test_result.get("status") or "")
        label = str(_as_dict(event_id.get("testResult")).get("label") or "")
        if label and status not in _PASSING_STATUSES:
          outputs = test_result.get("testActionOutput")
          log_file = None
          if isinstance(outputs, list):
            log_file = next(
                (
                    output
                    for output in outputs
                    if isinstance(output, dict)
                    and output.get("name") == "test.log"
                ),
                None,
            )
          prev_files = (
              test_failures[label].log_files if label in test_failures else ()
          )
          log_files = (log_file,) if log_file is not None else prev_files
          test_failures[label] = _Failure(
              title=f"{label} ({status})", log_files=log_files
          )
      elif isinstance(event.get("testSummary"), dict):
        label = str(_as_dict(event_id.get("testSummary")).get("label") or "")
        if event["testSummary"].get("overallStatus") in _PASSING_STATUSES:
          test_failures.pop(label, None)
      elif isinstance(event.get("action"), dict):
        action = event["action"]
        if not action.get("success"):
          label = str(
              _as_dict(event_id.get("actionCompleted")).get("label")
              or action.get("label")
              or "Build action"
          )
          log_files = tuple(
              candidate
              for candidate in (action.get("stderr"), action.get("stdout"))
              if isinstance(candidate, dict)
          )
          other_failures.append(
              _Failure(title=f"{label} (BUILD_FAILED)", log_files=log_files)
          )
      elif isinstance(event.get("aborted"), dict):
        aborted = event["aborted"]
        reason = str(aborted.get("reason") or "ABORTED")
        target = next(
            (
                str(id_part["label"])
                for id_part in event_id.values()
                if isinstance(id_part, dict) and id_part.get("label")
            ),
            "Build aborted",
        )
        description = str(aborted.get("description") or "").strip()
        if reason != "SKIPPED" and (description or target != "Build aborted"):
          other_failures.append(
              _Failure(title=f"{target} ({reason})", detail_text=description)
          )
      elif isinstance(event.get("finished"), dict):
        code = _as_dict(event["finished"].get("exitCode")).get("code")
        if isinstance(code, int):
          exit_code = code

  invocation_ref = (
      f"{bes_url.rstrip('/')}/{invocation_id}"
      if bes_url and invocation_id
      else invocation_id
  )
  failures = tuple(
      dataclasses.replace(failure, invocation_ref=invocation_ref)
      for failure in (*test_failures.values(), *other_failures)
  )
  return _ParsedBep(
      invocation_ref=invocation_ref, exit_code=exit_code, failures=failures
  )


def _format_error(
    title: str, header: str, detail: str, max_message_chars: int
) -> str:
  """Escapes and formats a single bounded GitHub `::error` workflow command."""
  safe_title = _escape_prefix(title, _MAX_TITLE_CHARS, is_property=True)
  body = _escape_prefix(header or title, max_message_chars)
  if detail:
    tail = _escape_tail(detail, max_message_chars - len(body) - 6)
    if tail:
      body = f"{body}%0A%0A{tail}"
  return f"::error title={safe_title}::{body}"


def _resolve_detail(failure: _Failure) -> str:
  if failure.detail_text:
    return failure.detail_text
  for log_file in failure.log_files:
    tail = _read_log_tail(log_file)
    if tail:
      return tail
  return ""


def emit_github_annotations(
    bep_paths: Sequence[str],
    *,
    label: str = "",
    max_annotations: int = DEFAULT_MAX_ANNOTATIONS,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    exit_code: int | None = None,
    stream: TextIO | None = None,
) -> int:
  """Parses BEP files and writes bounded `::error` annotations best-effort."""
  try:
    failures: list[_Failure] = []
    last_invocation = ""
    last_code = exit_code
    for path in bep_paths:
      if not os.path.exists(path):
        continue
      parsed = _parse_bep(path)
      last_invocation = parsed.invocation_ref or last_invocation
      if parsed.exit_code is not None:
        last_code = parsed.exit_code
      failures.extend(parsed.failures)

    if not failures and last_code not in (None, 0):
      failures.append(
          _Failure(
              title=f"{label or 'Bazel build'} failed (exit {last_code})",
              invocation_ref=last_invocation,
          )
      )
    if not failures or max_annotations <= 0:
      return 0

    overflow = 0
    if len(failures) > max_annotations:
      keep = max(1, max_annotations - 1)
      overflow = len(failures) - keep
      failures = failures[:keep]

    lines: list[str] = []
    for failure in failures:
      header_parts: list[str] = []
      if label:
        header_parts.append(label)
      if failure.invocation_ref:
        header_parts.append(f"Invocation: {failure.invocation_ref}")
      header = " | ".join(header_parts)
      lines.append(
          _format_error(
              failure.title,
              header,
              _resolve_detail(failure),
              max_message_chars,
          )
      )

    if overflow and len(lines) < max_annotations:
      summary_parts = [f"... and {overflow} more failing targets"]
      if last_invocation:
        summary_parts.append(f"Invocation: {last_invocation}")
      lines.append(
          _format_error(
              label or "More BEP failures",
              " | ".join(summary_parts),
              "",
              max_message_chars,
          )
      )

    out = stream if stream is not None else sys.stdout
    out.write("\n".join(lines) + "\n")
    out.flush()
    return len(lines)
  except (OSError, ValueError, TypeError) as e:
    logging.warning("Failed to emit BEP GitHub annotations: %s", e)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint; always returns 0 so CI exit status is preserved."""
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser(description="Emit ::error from Bazel BEP.")
  parser.add_argument("bep_files", nargs="+", help="Bazel BEP JSONL file(s).")
  parser.add_argument("--label", default="", help="Optional build/job label.")
  try:
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
  except SystemExit as e:
    if e.code not in (None, 0):
      logging.warning("bep_annotations CLI failed (ignored): %s", e)
    return 0
  emit_github_annotations(args.bep_files, label=args.label)
  return 0


if __name__ == "__main__":
  sys.exit(main())
