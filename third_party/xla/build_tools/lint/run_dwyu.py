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
# ============================================================================
"""Run BANT with consistent repository identities for XLA and vendored TSL.

BANT 0.3.6 discovers external repositories from Bazel's output directory, but
does not resolve the root module's own name. Analyze both XLA and TSL through
their repository names in a temporary workspace so @xla// references resolve
and the same target is not indexed as both //... and @xla//.... The workspace
links to the checkout and existing dependencies without modifying either.
Pass Bazel's output base explicitly: analysis-only builds do not create the
checkout's bazel-out symlink.
"""

import argparse
import contextlib
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence


@contextlib.contextmanager
def bant_workspace(
    root: pathlib.Path, output_base: pathlib.Path
) -> Iterator[pathlib.Path]:
  """Expose the checkout, TSL, and fetched dependencies to BANT by repo name."""
  root = root.resolve()
  external = (output_base / "external").resolve(strict=True)
  with tempfile.TemporaryDirectory(prefix="xla-dwyu-") as temporary:
    workspace = pathlib.Path(temporary)
    repositories = workspace / "external"
    repositories.mkdir()
    for entry in external.iterdir():
      # Bazel's _main alias must not shadow @xla during BANT discovery.
      if entry.name not in ("_main", "xla", "tsl"):
        (repositories / entry.name).symlink_to(entry)
    # Always use this checkout, including in worktrees where Bazel's cached
    # @tsl symlink may point to a different checkout.
    (repositories / "xla").symlink_to(root, target_is_directory=True)
    (repositories / "tsl").symlink_to(
        root / "third_party/tsl", target_is_directory=True
    )
    # BANT looks for repositories at bazel-out/../../../external.
    output = workspace / "execroot/xla/bazel-out"
    output.mkdir(parents=True)
    (workspace / "bazel-out").symlink_to(output, target_is_directory=True)
    for name in ("MODULE.bazel", ".bant-macros"):
      (workspace / name).symlink_to(root / name)
    yield workspace


def run_dwyu(
    targets: Sequence[str], *, output_base: pathlib.Path, bant: str = "bant"
) -> int:
  """Check exactly the selected targets and preserve BANT's diagnostics/status."""
  if not targets:
    return 0
  executable = shutil.which(bant)
  if executable is None:
    raise FileNotFoundError(f"BANT executable not found: {bant}")
  # Use one identity for the root repository, including targets passed from CI.
  labels = [
      "@xla" + label if label.startswith("//") else label for label in targets
  ]
  with bant_workspace(pathlib.Path.cwd(), output_base) as workspace:
    return subprocess.run(
        [
            str(pathlib.Path(executable).resolve()),
            "-C",
            str(workspace),
            # Index potential providers even when the missing dependency is
            # not reachable through the target's existing deps. These patterns
            # expand the graph; only labels below are checked for DWYU.
            "--graph-augment=@xla//xla/...",
            "--graph-augment=@tsl//tsl/...",
            "dwyu",
            *labels,
        ],
        check=False,
    ).returncode


def main(argv: Sequence[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--bant", default="bant", help="BANT executable")
  parser.add_argument(
      "--output-base",
      type=pathlib.Path,
      required=True,
      help="Bazel output base from `bazel info output_base`",
  )
  parser.add_argument("targets", nargs="*", help="Exact Bazel target labels")
  args = parser.parse_args(argv[1:])
  return run_dwyu(args.targets, output_base=args.output_base, bant=args.bant)


if __name__ == "__main__":
  sys.exit(main(sys.argv))
