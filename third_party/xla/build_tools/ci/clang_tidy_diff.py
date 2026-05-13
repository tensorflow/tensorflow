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

"""Filters Clang-Tidy errors based on modified lines in a Git patch.

This script reads a unified diff (patch file) to determine which lines of which
files have been modified. It then parses the Bazel Build Event Protocol (BEP)
JSON file to find the paths of all generated Clang-Tidy YAML reports. For each
report, it checks if any reported errors fall within the modified line ranges
and prints the matching errors.
"""

from __future__ import annotations

import argparse
import bisect
import collections
from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import IO, Protocol, TypedDict

from build_tools.lint import diff_parser


@dataclasses.dataclass(frozen=True)
class AppConfig:
  """Configuration for the ClangTidyDiffFilter application.

  Attributes:
    patch: Path to the patch file containing the diff.
    repo_root: Absolute path to the repository root.
    bep_file: Path to the Bazel Build Event Protocol JSON file.
    warnings_as_errors: If True, treat Clang-Tidy warnings as errors.
    fix: If True, apply fixes to the source files.
  """

  patch: str
  repo_root: str
  bep_file: str
  warnings_as_errors: bool
  fix: bool = False


@dataclasses.dataclass(frozen=True)
class Diagnostic:
  """Represents a single Clang-Tidy diagnostic.

  Attributes:
    file_path: The path to the file where the diagnostic was found, relative to
      the repo root.
    line_num: The 1-based line number of the diagnostic.
    col_num: The 1-based column number of the diagnostic.
    level: The severity level of the diagnostic (e.g., "warning", "error").
    name: The name of the Clang-Tidy check (e.g., "clang-diagnostic-error").
    message: The diagnostic message.
    yaml_file: The path to the .clang-tidy.yaml file where this diagnostic was
      reported.
    has_replacements: Whether the diagnostic has replacements available. For
      some diagnostics, clang-tidy can automatically fix the issue, and this
      attribute will be True if there are one or more replacements available.
  """

  file_path: str
  line_num: int
  col_num: int
  level: str
  name: str
  message: str
  yaml_file: str
  has_replacements: bool


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiagnosticSummary:
  """Summary for all Clang-Tidy diagnostics for a file.

  Attributes:
    file_path: The path to the file where the diagnostics were found, relative
      to the repo root.
    was_skipped: Whether the file was skipped due to not being in the git diff.
    total: The total number of diagnostics found in the report.
    matched: The number of diagnostics that matched the diff.
  """

  file_path: str
  was_skipped: bool
  total: int
  matched: int


# Can't use TypedDict with classes because linting will complain about fields
# starting with capital letters.
ClangTidyDiagnostic = TypedDict(
    "ClangTidyDiagnostic",
    {
        "DiagnosticName": str,
        "Message": str,
        "FilePath": str,
        "FileOffset": int,
        "Level": str,
        "HasReplacements": bool,
    },
    total=False,
)


ClangTidyReport = TypedDict(
    "ClangTidyReport",
    {
        "MainSourceFile": str,
        "Diagnostics": list[ClangTidyDiagnostic],
    },
    total=False,
)


class ApplyFixes(Protocol):
  """A callable for applying fixes from clang-tidy diagnostics."""

  def __call__(self, temp_dir: pathlib.Path) -> None:
    """Applies fixes to the source files in the given temporary directory.

    Args:
      temp_dir: The temporary directory containing the source files to apply
        fixes to.
    """
    ...


def clang_apply_fixes(bin_path: pathlib.Path, temp_dir: pathlib.Path) -> None:
  """Applies fixes to the source files in the given temporary directory.

  Args:
    bin_path: Path to the clang-apply-replacements binary.
    temp_dir: The temporary directory containing the source files to apply fixes
      to.
  """
  cmd = [
      bin_path.as_posix(),
      "-remove-change-desc-files",
      temp_dir.as_posix(),
  ]
  _logger().info("Running command: %r", " ".join(cmd))
  subprocess.check_output(cmd)


def _logger() -> logging.Logger:
  """Returns the logger for this module."""
  return logging.getLogger(__name__)


def _set_log_level(log_level: str) -> None:
  """Sets the log level for the application."""
  logging.basicConfig(
      format=(
          "[%(asctime)s] [%(levelname)s][%(filename)s:%(funcName)s:%(lineno)d]"
          " %(message)s"
      ),
      datefmt="%H:%M:%S",
  )

  def _set_level(level: int) -> None:
    _logger().setLevel(level)

  match log_level:
    case "DEBUG":
      _set_level(logging.DEBUG)
    case "INFO":
      _set_level(logging.INFO)
    case "WARNING":
      _set_level(logging.WARNING)
    case "ERROR":
      _set_level(logging.ERROR)
    case _:
      raise ValueError(f"Unsupported log level: {log_level}")


def _resolve_clang_apply_replacements(
    clang_apply_replacements_bin: pathlib.Path | None,
) -> pathlib.Path | None:
  """Resolve path to the clang-apply-replacements binary.

  If the path is explicitly provided, use that. Otherwise, try to find it using
  bazel-runfiles.

  Args:
    clang_apply_replacements_bin: The path to the clang-apply-replacements
      binary, or None if it should be resolved from the environment.

  Returns:
    The path to the clang-apply-replacements binary, or None if it could not be
    resolved.
  """
  if clang_apply_replacements_bin:
    return clang_apply_replacements_bin
  runfiles_dir = os.environ.get("RUNFILES_DIR")
  if not runfiles_dir:
    return None
  runfiles_path = (
      pathlib.Path(runfiles_dir)
      / "xla/build_tools/ci/clang_apply_replacements_bin"
  )
  if runfiles_path.exists():
    return runfiles_path
  _logger().warning(
      "clang-apply-replacements binary not found in RUNFILES_DIR: %s",
      runfiles_path,
  )
  return None


def parse_diff(diff_path: str) -> dict[str, set[int]]:
  """Parses a unified diff file using diff_parser and returns a dictionary mapping filenames to a set of modified line numbers."""
  with open(diff_path, "r") as f:
    diff_str = f.read()
  hunks = diff_parser.parse_hunks(diff_str)
  file_to_lines: dict[str, set[int]] = collections.defaultdict(set)
  for hunk in hunks:
    for line_no, _ in hunk.added_lines():
      file_to_lines[hunk.file].add(line_no)
  return file_to_lines


def get_line_offsets(file_path: str) -> tuple[int, ...]:
  """Returns a list of byte offsets for the start of each line in the file."""
  offsets = [0]
  with open(file_path, "rb") as f:
    while f.readline():
      offsets.append(f.tell())
  return tuple(offsets)


def offset_to_line(offsets: Sequence[int], offset: int) -> int:
  """Converts a byte offset to a 1-based line number using binary search."""
  if not offsets:
    return -1
  # bisect_right returns the index where the offset would be inserted after
  # existing entries. Since offsets contains start of lines, bisect_right - 1
  # gives the line index (0-based).
  return bisect.bisect_right(offsets, offset)


def normalize_path(path: str, repo_root: str) -> str:
  """Normalizes a path to be relative to the repo root.

  This is not foolproof for all possible path formats,
  but it handles common cases seen locally and in CI.

  Args:
    path: The path to normalize.
    repo_root: The absolute path to the repository root.

  Returns:
      The normalized path as a string.
  """
  if not path:
    return ""
  p = pathlib.Path(path)
  # Handle local absolute paths under repo_root (CI Runner paths)
  # This handles /__w/xla/xla by removing the prefix.
  if p.is_absolute() and p.is_relative_to(repo_root):
    return p.relative_to(repo_root).as_posix()
  # Handle bazel execroot paths
  if "execroot" in p.parts:
    idx = p.parts.index("execroot")
    if idx + 2 < len(p.parts):
      return pathlib.Path(*p.parts[idx + 2 :]).as_posix()
  # Handle remote execution paths
  # p is like "/b/f/w/xla/..."
  # NB: We don't quite know the top level directory to look for in the remote
  # path, but since all CPP sources live mostly under "xla/" we use it as the
  # anchor. We also include "third_party" as an anchor since some files may
  # be under that directory.
  _top_level_pkgs = ("xla", "third_party")
  for pkg in _top_level_pkgs:
    if pkg in p.parts:
      parts = list(p.parts)
      idx = parts.index(pkg)  # Find FIRST occurrence in remote path
      return pathlib.Path(*parts[idx:]).as_posix()
  return path


def parse_bep(bep_path: str, repo_root: str) -> list[str]:
  """Parses a Bazel BEP JSON file and returns a list of paths to .clang-tidy.yaml files.

  Args:
    bep_path: Path to the Bazel BEP JSON file.
    repo_root: Absolute path to the repository root.

  Returns:
    A list of paths to .clang-tidy.yaml files.

  Raises:
    ValueError: If a file entry in the BEP is missing 'name' or 'pathPrefix'
      fields.
  """
  yaml_files: list[str] = []
  with open(bep_path, "r") as f:
    for line in f:
      try:
        event = json.loads(line)
        if "namedSetOfFiles" not in event:
          continue
        files = event["namedSetOfFiles"].get("files", [])
        for file_info in files:
          name = file_info.get("name")
          prefix = file_info.get("pathPrefix")
          if name is None:
            raise ValueError("File entry in BEP is missing 'name' field.")
          if not name.endswith(".clang-tidy.yaml"):
            continue
          if prefix is None:
            raise ValueError("File entry in BEP is missing 'pathPrefix' field.")
          path = (
              pathlib.Path(repo_root) / pathlib.Path(*prefix) / name
          ).as_posix()
          yaml_files.append(path)
      except json.JSONDecodeError:
        _logger().warning(
            "Skipping invalid JSON line in BEP file: %s", line.strip()
        )
        continue

  return yaml_files


def parse_clang_tidy_yaml(yaml_path: str) -> ClangTidyReport:
  """A simple, specialized parser for clang-tidy YAML reports to avoid PyYAML dependency."""

  def extract(s: str) -> str:
    _, value = s.split(":", 1)
    return value.strip().strip("'\"")

  result: ClangTidyReport = {"Diagnostics": []}

  current_diag: ClangTidyDiagnostic | None = None
  in_diag_message = False

  with open(yaml_path, "r") as f:
    for line in f:
      stripped = line.strip()
      if stripped.startswith("MainSourceFile:"):
        result["MainSourceFile"] = extract(stripped)
      elif stripped.startswith("- DiagnosticName:"):
        current_diag = {"DiagnosticName": extract(stripped)}
        result["Diagnostics"].append(current_diag)
        in_diag_message = False
      elif stripped.startswith("DiagnosticMessage:"):
        in_diag_message = True
      elif in_diag_message and stripped.startswith("Message:"):
        if current_diag is not None:
          current_diag["Message"] = extract(stripped)
      elif in_diag_message and stripped.startswith("FilePath:"):
        if current_diag is not None:
          current_diag["FilePath"] = extract(stripped)
      elif in_diag_message and stripped.startswith("FileOffset:"):
        if current_diag is not None:
          current_diag["FileOffset"] = int(extract(stripped))
      elif stripped.startswith("Level:"):
        if current_diag is not None:
          current_diag["Level"] = extract(stripped)
      elif stripped.startswith("Replacements:"):
        in_diag_message = False
        if current_diag is not None:
          current_diag["HasReplacements"] = not stripped.endswith("[]")
  return result


RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"


def ansiformat(text: str, color: str = "", bold: bool = False) -> str:
  """Returns text wrapped in ANSI color codes."""
  bold_str = BOLD if bold else ""
  return f"{bold_str}{color}{text}{RESET}"


def print_diagnostic(
    diag: Diagnostic,
    repo_root: str,
    *,
    warnings_as_errors: bool = False,
    stream: IO[str] = sys.stderr,
) -> None:
  """Prints a diagnostic message with code snippet and color."""

  # Clang-tidy format: file:line:col: level: message [name]
  def get_level_str() -> str:
    level_str = diag.level.lower() if diag.level else "warning"
    if warnings_as_errors and level_str == "warning":
      return "error"
    return level_str

  level_str = get_level_str()
  diag_line = " ".join((
      ansiformat(
          ":".join((diag.file_path, str(diag.line_num), str(diag.col_num))),
          bold=True,
      ),
      ansiformat(
          f"{level_str}:",
          color=RED if level_str == "error" else YELLOW,
          bold=True,
      ),
      ansiformat(diag.message, bold=True),
      ansiformat(f"[{diag.name}]", color=CYAN, bold=True),
  ))
  stream.write(f"{diag_line}\n")
  abs_path = pathlib.Path(repo_root) / diag.file_path
  try:
    with open(abs_path, "r") as f:
      lines = f.readlines()
  except FileNotFoundError:
    _logger().warning(
        "Could not read file %r to print diagnostic snippet.",
        abs_path.as_posix(),
    )
    return

  context_lines = 5
  start = max(0, diag.line_num - context_lines - 1)
  end = min(len(lines), diag.line_num + context_lines)
  for linenum, line in enumerate(lines[start:end], start=start + 1):
    line_content = line.rstrip("\n")
    prefix = f"{linenum:5d} | "
    if linenum == diag.line_num:
      stream.write(f"{ansiformat(prefix, bold=True)}{line_content}\n")
      # Print caret
      spaces = "".join(
          "\t" if ch == "\t" else " " for ch in line_content[: diag.col_num - 1]
      )
      stream.write(
          f"{' ' * len(prefix)}{spaces}{ansiformat('^', color=GREEN, bold=True)}\n"
      )
    else:
      stream.write(f"{prefix}{line_content}\n")


def _find_with_prefix(
    parts: Sequence[str], prefix: str
) -> tuple[int, str | None]:
  """Finds the index and value of the first part starting with the prefix.

  Args:
    parts: The path parts to search.
    prefix: The prefix to search for.

  Returns:
    A tuple of (index, dir) where dir is the directory starting with the
    prefix or None if not found.
  """
  for idx, part in enumerate(parts):
    if part.startswith(prefix):
      return idx, part
  return 0, None


def _get_report_source(yaml_file_path: pathlib.Path, repo_root: str) -> str:
  """Returns the source file path for a Clang-Tidy YAML report file."""
  # yaml_path looks like:
  # x/y/z/bazel_clang_tidy/path/to/file.cc.<target>.clang-tidy.yml.
  expected_prefix = "bazel_clang_tidy_"
  idx, aspect_dir = _find_with_prefix(yaml_file_path.parts, expected_prefix)
  if aspect_dir is not None:
    toplevel = [aspect_dir.removeprefix(expected_prefix)]
    rel_parts = yaml_file_path.parts[idx + 1 :]
    # Remove <target>.clang-tidy.yml suffix.
    filename = rel_parts[-1].rsplit(".", 3)[0]
  else:
    toplevel = []
    rel_parts = yaml_file_path.parts
    # Remove .clang-tidy.yml without the <target>.
    filename = rel_parts[-1].rsplit(".", 2)[0]
  return normalize_path(
      pathlib.Path(*toplevel, *rel_parts[:-1], filename).as_posix(),
      repo_root,
  )


class ClangTidyDiffFilter:
  """Filters Clang-Tidy diagnostics based on a diff."""

  def __init__(
      self,
      config: AppConfig,
      *,
      offset_provider: Callable[[str], Sequence[int]] = get_line_offsets,
      apply_fixes_fn: ApplyFixes | None = None,
  ):
    """Initializes the ClangTidyDiffFilter.

    Args:
      config: An AppConfig object containing the application configuration.
      offset_provider: A callable that takes a file path and returns a list of
        byte offsets for the start of each line in the file. Defaults to
        `get_line_offsets`.
      apply_fixes_fn: Method to call to apply fixes, if --fix is enabled. It
        will be called with the path to the temp directory where the relevant
        YAML files have been staged. If None, fixes will not be applied.
    """
    self.diff_ranges = parse_diff(config.patch)
    self.yaml_files = parse_bep(config.bep_file, config.repo_root)
    self.repo_root = config.repo_root
    self.warnings_as_errors = config.warnings_as_errors
    self.offset_provider = offset_provider
    self.file_offsets_cache: dict[str, Sequence[int]] = {}
    self.seen_files: set[str] = set()
    self.apply_fixes_fn = apply_fixes_fn

  def process_file(
      self, yaml_file: str
  ) -> tuple[Sequence[Diagnostic], DiagnosticSummary]:
    """Processes a single Clang-Tidy YAML report file.

    Args:
      yaml_file: The path to the Clang-Tidy YAML report file.

    Returns:
      A tuple of (matched_diagnostics, summary).
    """
    matched_diagnostics: list[Diagnostic] = []
    data = parse_clang_tidy_yaml(yaml_file)
    _logger().debug(
        "Processing clang-tidy report file: %s with content:\n%s",
        yaml_file,
        data,
    )
    yaml_file_path = pathlib.Path(yaml_file)
    report_source = _get_report_source(yaml_file_path, self.repo_root)
    if not data:
      return [], DiagnosticSummary(
          file_path=report_source,
          was_skipped=True,
          total=0,
          matched=0,
      )

    main_source = data.get("MainSourceFile")
    if main_source:
      norm_main_source = normalize_path(main_source, self.repo_root)
      self.seen_files.add(norm_main_source)
    if report_source in self.diff_ranges:
      self.seen_files.add(report_source)
    if "Diagnostics" not in data:
      return [], DiagnosticSummary(
          file_path=report_source,
          was_skipped=True,
          total=0,
          matched=0,
      )

    for diag in data["Diagnostics"]:
      file_path = diag.get("FilePath")
      offset = diag.get("FileOffset")

      if not file_path or offset is None:
        continue

      norm_path = normalize_path(file_path, self.repo_root)
      if norm_path not in self.diff_ranges:
        continue

      abs_path = pathlib.Path(self.repo_root) / norm_path
      if norm_path not in self.file_offsets_cache:
        self.file_offsets_cache[norm_path] = self.offset_provider(
            abs_path.as_posix()
        )

      offsets = self.file_offsets_cache[norm_path]
      if not offsets:
        _logger().warning(
            "Could not read file %r to calculate line number.",
            abs_path.as_posix(),
        )
        continue

      line_num = offset_to_line(offsets, offset)

      line_start_offset = offsets[line_num - 1]
      col_num = offset - line_start_offset + 1

      lines = self.diff_ranges[norm_path]
      if line_num in lines:
        matched_diagnostics.append(
            Diagnostic(
                file_path=norm_path,
                line_num=line_num,
                col_num=col_num,
                level=diag.get("Level") or "",
                name=diag.get("DiagnosticName") or "",
                message=diag.get("Message") or "",
                yaml_file=yaml_file,
                has_replacements=diag.get("HasReplacements", False),
            )
        )
    return matched_diagnostics, DiagnosticSummary(
        file_path=report_source,
        was_skipped=report_source not in self.diff_ranges,
        total=len(data["Diagnostics"]),
        matched=len(matched_diagnostics),
    )

  def report_missing(self) -> None:
    """Reports any touched files that were not processed using the logger."""
    touched_files = set(self.diff_ranges.keys())
    missing_files = [
        f for f in touched_files - self.seen_files if f.endswith((".h", ".cc"))
    ]
    if missing_files:
      _logger().warning(
          "No Clang-Tidy reports were processed for the following modified"
          " files:"
      )
      for f in sorted(missing_files):
        _logger().warning("  - %s", f)

  def _copy_and_normalize_yaml(
      self, src: pathlib.Path, dest: pathlib.Path
  ) -> None:
    """Copies a YAML report and normalizes all remote paths to local absolute paths."""
    with src.open("r") as f_source, dest.open("w") as f_dest:
      for line in f_source:
        stripped = line.strip()
        # Paths in clang-tidy YAMLs only appear in these two fields.
        if not stripped.startswith(("MainSourceFile:", "FilePath:")):
          f_dest.write(line)
          continue
        _, val = line.split(":", 1)
        raw_path = val.strip()
        if not raw_path:
          continue
        norm_rel_path = normalize_path(raw_path, self.repo_root)
        local_abs_path = (
            pathlib.Path(self.repo_root) / norm_rel_path
        ).as_posix()
        line = line.replace(raw_path, local_abs_path)
        f_dest.write(line)

  def apply_fixes(self, yaml_files: Sequence[str]) -> None:
    """Stages YAML files in temp directory and calls apply_fixes_fn."""
    if not self.apply_fixes_fn or not yaml_files:
      return
    _logger().info("Applying fixes automatically.")
    with tempfile.TemporaryDirectory() as temp_dir_str:
      temp_dir = pathlib.Path(temp_dir_str)
      counter = itertools.count()
      for y in yaml_files:
        dest_path = (
            pathlib.Path(temp_dir) / f"{next(counter)}_{pathlib.Path(y).name}"
        )
        self._copy_and_normalize_yaml(pathlib.Path(y), dest_path)
        _logger().info(
            "Copied YAML report to temp dir for fixes: %s", dest_path
        )
      self.apply_fixes_fn(temp_dir)

  def run(self) -> bool:
    """Runs the Clang-Tidy diff filter.

    Returns:
      True if the check was successful (no errors found), False if errors were
      found or running the check failed.
    """
    if not self.diff_ranges or not self.yaml_files:
      _logger().error("No YAML files provided or found in BEP.")
      return False

    found_diagnostics = False
    yaml_files_to_fix: set[str] = set()
    for y in self.yaml_files:
      diagnostics, summary = self.process_file(y)
      if summary.was_skipped and summary.total > 0:
        _logger().info(
            "Skipping %r with %d diagnostics because it is not in the diff.",
            summary.file_path,
            summary.total,
        )
      if summary.total > summary.matched and not summary.was_skipped:
        _logger().info(
            "Not all diagnostics will be reported for %r because they are not"
            " part of the diff. Found %d diagnostics but reported %d.",
            summary.file_path,
            summary.total,
            summary.matched,
        )
      for d in diagnostics:
        if d.has_replacements and self.apply_fixes_fn is not None:
          yaml_files_to_fix.add(y)
        else:
          print_diagnostic(
              d, self.repo_root, warnings_as_errors=self.warnings_as_errors
          )
          found_diagnostics = True

    self.report_missing()
    self.apply_fixes(sorted(yaml_files_to_fix))
    return not found_diagnostics


def main() -> None:
  """Main entry point for the Clang-Tidy diff filter."""

  parser = argparse.ArgumentParser(
      description="Filter Clang-Tidy errors by Git diff."
  )
  parser.add_argument("--patch", required=True, help="Path to the patch file.")
  parser.add_argument(
      "--repo-root", required=True, help="Absolute path to the repo root."
  )
  parser.add_argument(
      "--bep-file",
      required=True,
      help="Path to Bazel Build Event Protocol JSON file.",
  )
  parser.add_argument(
      "--warnings-as-errors",
      default="true",
      choices=["true", "false"],
      help="Treat warnings as errors.",
  )
  parser.add_argument(
      "--log-level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Set the log level.",
  )
  parser.add_argument(
      "--fix",
      action="store_true",
      help="Apply automatic fixes to touched files.",
  )
  parser.add_argument(
      "--clang-apply-replacements",
      type=pathlib.Path,
      default=None,
      help="Path to clang-apply-replacements binary.",
  )

  args = parser.parse_args()
  _set_log_level(args.log_level)

  config = AppConfig(
      patch=args.patch,
      repo_root=args.repo_root,
      bep_file=args.bep_file,
      warnings_as_errors=args.warnings_as_errors == "true",
      fix=args.fix,
  )
  apply_replacements_bin_path = (
      _resolve_clang_apply_replacements(args.clang_apply_replacements)
      if args.fix
      else None
  )
  if args.fix and apply_replacements_bin_path is None:
    _logger().error(
        "--clang-apply-replacements is required when --fix is enabled "
        "and the binary could not be resolved automatically in runfiles."
    )
    sys.exit(1)
  apply_fixes_fn = (
      functools.partial(clang_apply_fixes, apply_replacements_bin_path)
      if apply_replacements_bin_path
      else None
  )
  filterer = ClangTidyDiffFilter(config, apply_fixes_fn=apply_fixes_fn)
  if not filterer.run():
    sys.exit(1)


if __name__ == "__main__":
  main()
