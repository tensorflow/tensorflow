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

from collections.abc import Sequence
import io
import json
import pathlib
import tempfile
import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import clang_tidy_diff


class TestClangTidyDiff(parameterized.TestCase):

  @parameterized.parameters(
      (0, 1),
      (5, 1),
      (10, 2),
      (15, 2),
      (20, 3),
      (25, 3),
      (30, 4),
      (35, 4),
  )
  def test_offset_to_line(self, offset, expected_line):
    offsets = [0, 10, 20, 30]
    self.assertEqual(
        clang_tidy_diff.offset_to_line(offsets, offset), expected_line
    )

  def test_offset_to_line_empty(self):
    self.assertEqual(clang_tidy_diff.offset_to_line([], 10), -1)

  def test_normalize_path_relative(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("foo/bar.cc", "/root"), "foo/bar.cc"
    )

  def test_normalize_path_absolute_in_repo(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("/root/foo/bar.cc", "/root"),
        "foo/bar.cc",
    )

  def test_normalize_path_absolute_outside_repo(self):
    self.assertEqual(
        clang_tidy_diff.normalize_path("/other/foo/bar.cc", "/root"),
        "/other/foo/bar.cc",
    )

  def test_normalize_path_execroot(self):
    path = "/usr/local/google/home/user/.cache/bazel/_bazel_user/a708d4fc59660ccd295a76cce84d113c/execroot/xla/xla/stream_executor/cuda/cuda_status.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/root"),
        "xla/stream_executor/cuda/cuda_status.h",
    )

  def test_normalize_path_remote_worker(self):
    path = "/b/f/w/xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
    # repo_root is /__w/xla/xla, so workspace name is 'xla'
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h",
    )

  def test_normalize_path_local_ci_runner(self):
    path = "/__w/xla/xla/xla/backends/gpu/codegen/triton/transforms/lowering_utils.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "xla/backends/gpu/codegen/triton/transforms/lowering_utils.h",
    )

  def test_normalize_path_remote_worker_third_party(self):
    path = "/b/f/w/third_party/gpus/cuda/include/cuda.h"
    self.assertEqual(
        clang_tidy_diff.normalize_path(path, "/__w/xla/xla"),
        "third_party/gpus/cuda/include/cuda.h",
    )

  def test_parse_diff(self):
    tmpdir = self.create_tempdir().full_path
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    with open(diff_path, "w") as f:

      f.write(textwrap.dedent("""\
                  diff --git a/file1.cc b/file1.cc
                  index 123456..789012 100644
                  --- a/file1.cc
                  +++ b/file1.cc
                  @@ -1,2 +1,3 @@
                   line1
                  +line2
                   line3
                  """))
    ranges = clang_tidy_diff.parse_diff(str(diff_path))
    self.assertEqual(ranges, {"file1.cc": {2}})

  def test_parse_bep(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            '{"namedSetOfFiles": {"files": [{"name": "file1.clang-tidy.yaml",'
            ' "pathPrefix": ["bazel-out", "k8-opt", "bin"]}]}}\n'
        )
      yaml_files = clang_tidy_diff.parse_bep(str(bep_path), "/root")
      self.assertEqual(
          yaml_files, ["/root/bazel-out/k8-opt/bin/file1.clang-tidy.yaml"]
      )

  def test_parse_clang_tidy_yaml(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "test.yaml"
      with open(yaml_path, "w") as f:

        f.write(textwrap.dedent("""\
                    ---
                    MainSourceFile:  '/root/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '/root/file1.cc'
                          FileOffset:      15
                    ...
                    """))

      data = clang_tidy_diff.parse_clang_tidy_yaml(str(yaml_path))

      self.assertEqual(data.get("MainSourceFile"), "/root/file1.cc")
      diagnostics = data.get("Diagnostics", [])
      with self.subTest("Diagnostics"):
        self.assertLen(diagnostics, 1)
        self.assertEqual(diagnostics[0].get("DiagnosticName"), "misc-unused")
        self.assertEqual(diagnostics[0].get("Message"), "unused variable")
        self.assertEqual(diagnostics[0].get("FilePath"), "/root/file1.cc")
        self.assertEqual(diagnostics[0].get("FileOffset"), 15)

  def test_process_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "file1.cc.clang-tidy.yaml"
      with open(yaml_path, "w") as f:
        f.write(textwrap.dedent(f"""\
                    ---
                    MainSourceFile:  '{tmpdir}/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '{tmpdir}/file1.cc'
                          FileOffset:      15
                        Level:           Error
                    ...
                    """))

      diff_path = pathlib.Path(tmpdir) / "test.diff"
      with open(diff_path, "w") as f:

        f.write(textwrap.dedent("""\
                    diff --git a/file1.cc b/file1.cc
                    index 123456..789012 100644
                    --- a/file1.cc
                    +++ b/file1.cc
                    @@ -1,2 +1,3 @@
                     line1
                    +line2
                     line3
                    """))

      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            json.dumps({
                "namedSetOfFiles": {
                    "files": [{
                        "name": "file1.cc.clang-tidy.yaml",
                        "pathPrefix": [],
                    }]
                }
            })
            + "\n"
        )

      config = clang_tidy_diff.AppConfig(
          patch=str(diff_path),
          repo_root=str(tmpdir),
          bep_file=str(bep_path),
          warnings_as_errors=True,
      )

      def mock_offset_provider(_: str) -> list[int]:
        return [0, 10, 20, 30]

      filterer = clang_tidy_diff.ClangTidyDiffFilter(
          config, offset_provider=mock_offset_provider
      )
      diagnostics, summary = filterer.process_file(str(yaml_path))

      with self.subTest("Diagnostics"):
        self.assertLen(diagnostics, 1)
        self.assertEqual(diagnostics[0].file_path, "file1.cc")
        self.assertEqual(diagnostics[0].line_num, 2)
        self.assertEqual(
            diagnostics[0].col_num, 6
        )  # Offset 15 - Line 2 start 10 + 1 = 6
        self.assertEqual(diagnostics[0].level, "Error")
        self.assertEqual(diagnostics[0].name, "misc-unused")
        self.assertEqual(diagnostics[0].message, "unused variable")

      with self.subTest("DiagnosticSummary"):
        self.assertEqual(
            summary,
            clang_tidy_diff.DiagnosticSummary(
                file_path="file1.cc",
                was_skipped=False,
                total=1,
                matched=1,
            ),
        )

  def test_process_file_no_substring_false_positives(self):
    """Tests that we don't get false positives from diff file paths being substrings of other file paths."""
    tmpdir = self.create_tempdir().full_path
    yaml_path = pathlib.Path(tmpdir) / "xla/long_util.cc.clang-tidy.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
      f.write(textwrap.dedent(f"""\
                  ---
                  MainSourceFile:  '{tmpdir}/xla/long_util.cc'
                  Diagnostics: []
                  """))
    # util and long_util are both in the diff but only long_util has a report.
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    with open(diff_path, "w") as f:
      f.write(textwrap.dedent("""\
                  diff --git a/util.cc b/util.cc
                  index 123456..789012 100644
                  --- a/util.cc
                  +++ b/util.cc
                  @@ -1,1 +1,2 @@
                   line1
                  +line2
                  diff --git a/xla/long_util.cc b/xla/long_util.cc
                  index 123456..789012 100644
                  --- a/xla/long_util.cc
                  +++ b/xla/long_util.cc
                  @@ -1,1 +1,2 @@
                   line1
                  +line2
                  """))
    bep_path = pathlib.Path(tmpdir) / "test.bep"
    with open(bep_path, "w") as f:
      f.write(
          json.dumps({
              "namedSetOfFiles": {
                  "files": [{
                      "name": "xla/long_util.cc.clang-tidy.yaml",
                      "pathPrefix": [],
                  }]
              }
          })
          + "\n"
      )
    config = clang_tidy_diff.AppConfig(
        patch=diff_path.as_posix(),
        repo_root=tmpdir,
        bep_file=bep_path.as_posix(),
        warnings_as_errors=True,
    )
    filterer = clang_tidy_diff.ClangTidyDiffFilter(config)
    _, summary = filterer.process_file(str(yaml_path))
    self.assertIn("xla/long_util.cc", filterer.seen_files)
    self.assertFalse(summary.was_skipped)
    self.assertNotIn("util.cc", filterer.seen_files)

  def test_process_file_empty_yaml_aspect_path(self):
    tmpdir = self.create_tempdir().full_path
    yaml_path = pathlib.Path(tmpdir) / (
        "bazel-out/k8-opt/bin/xla/backends/bazel_clang_tidy_xla/"
        "backends/source.cc.target.clang-tidy.yaml"
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    self.create_tempfile(yaml_path.as_posix(), content="")
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    self.create_tempfile(
        diff_path.as_posix(),
        content=textwrap.dedent("""\
        diff --git a/xla/backends/source.cc b/xla/backends/source.cc
        index 123456..789012 100644
        --- a/xla/backends/source.cc
        +++ b/xla/backends/source.cc
        @@ -1,1 +1,2 @@
         line1
        +line2
        """),
    )
    bep_path = pathlib.Path(tmpdir) / "test.bep"
    self.create_tempfile(
        bep_path.as_posix(),
        content=json.dumps({
            "namedSetOfFiles": {
                "files": [{
                    "name": "xla/backends/source.cc.target.clang-tidy.yaml",
                    "pathPrefix": [
                        "bazel-out",
                        "k8-opt",
                        "bin",
                        "xla",
                        "backends",
                        "bazel_clang_tidy_xla",
                    ],
                }]
            }
        })
        + "\n",
    )
    config = clang_tidy_diff.AppConfig(
        patch=diff_path.as_posix(),
        repo_root=tmpdir,
        bep_file=bep_path.as_posix(),
        warnings_as_errors=False,
    )
    filterer = clang_tidy_diff.ClangTidyDiffFilter(config)
    _, summary = filterer.process_file(str(yaml_path))
    # File should be marked as seen despite the empty YAML.
    self.assertIn(
        "xla/backends/source.cc",
        filterer.seen_files,
    )
    with self.subTest("DiagnosticSummary"):
      self.assertEqual(
          summary,
          clang_tidy_diff.DiagnosticSummary(
              file_path="xla/backends/source.cc",
              was_skipped=False,
              total=0,
              matched=0,
          ),
      )

  def test_run(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yaml_path = pathlib.Path(tmpdir) / "test.clang-tidy.yaml"
      with open(yaml_path, "w") as f:
        f.write(textwrap.dedent(f"""\
                    ---
                    MainSourceFile:  '{tmpdir}/file1.cc'
                    Diagnostics:
                      - DiagnosticName:  misc-unused
                        DiagnosticMessage:
                          Message:         'unused variable'
                          FilePath:        '{tmpdir}/file1.cc'
                          FileOffset:      15
                        Level:           Error
                    ...
                    """))

      diff_path = pathlib.Path(tmpdir) / "test.diff"
      with open(diff_path, "w") as f:

        f.write(textwrap.dedent("""\
                    diff --git a/file1.cc b/file1.cc
                    index 123456..789012 100644
                    --- a/file1.cc
                    +++ b/file1.cc
                    @@ -1,2 +1,3 @@
                     line1
                    +line2
                     line3
                    """))

      bep_path = pathlib.Path(tmpdir) / "test.bep"
      with open(bep_path, "w") as f:
        f.write(
            json.dumps({
                "namedSetOfFiles": {
                    "files": [{
                        "name": "test.clang-tidy.yaml",
                        "pathPrefix": [],
                    }]
                }
            })
            + "\n"
        )

      config = clang_tidy_diff.AppConfig(
          patch=str(diff_path),
          repo_root=str(tmpdir),
          bep_file=str(bep_path),
          warnings_as_errors=True,
      )

      def mock_offset_provider(_: str) -> list[int]:
        return [0, 10, 20, 30]

      filterer = clang_tidy_diff.ClangTidyDiffFilter(
          config, offset_provider=mock_offset_provider
      )

      self.assertFalse(filterer.run())

  def test_print_diagnostic_sanity(self):
    diag = clang_tidy_diff.Diagnostic(
        file_path="file1.cc",
        line_num=2,
        col_num=3,
        level="warning",
        name="misc-unused",
        message="unused variable",
        yaml_file="test.clang-tidy.yaml",
        has_replacements=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      src_file = pathlib.Path(tmpdir) / "file1.cc"
      with open(src_file, "w") as f:
        f.write("line1\nline2\nline3\n")
      captured_stderr = io.StringIO()
      # Run it with warnings_as_errors=True to test that path too
      clang_tidy_diff.print_diagnostic(
          diag,
          repo_root=tmpdir,
          warnings_as_errors=True,
          stream=captured_stderr,
      )
      output = captured_stderr.getvalue()
      with self.subTest("diagnostic_string_sanity"):
        self.assertIn("file1.cc:2:3", output)
        self.assertIn("error:", output)
        self.assertIn("unused variable", output)
        self.assertIn("[misc-unused]", output)
        self.assertIn("  2 |", output)  # Snippet line
        self.assertIn("line2", output)
        self.assertIn("^", output)  # Caret

  @parameterized.parameters(
      dict(replacement_text="", expected=None),
      dict(replacement_text="Replacements: []", expected=False),
      dict(
          replacement_text=textwrap.dedent("""\
            Replacements:
              - FilePath:        '/root/file1.cc'
                Offset:          15
                Length:          6
                ReplacementText: 'blehblehbleh'
            """),
          expected=True,
      ),
  )
  def test_parse_clang_tidy_yaml_with_replacements(
      self, replacement_text: str, expected: bool | None
  ):
    tmpdir = self.create_tempdir().full_path
    yaml_path = pathlib.Path(tmpdir) / "test.yaml"
    self.create_tempfile(
        yaml_path.as_posix(),
        content=textwrap.dedent(f"""\
            ---
            MainSourceFile:  '/root/file1.cc'
            Diagnostics:
              - DiagnosticName:  misc-unused
                DiagnosticMessage:
                  Message:         'unused variable'
                  FilePath:        '/root/file1.cc'
                  FileOffset:      15
                {replacement_text}
            ...
            """),
    )
    data = clang_tidy_diff.parse_clang_tidy_yaml(str(yaml_path))
    diagnostics = data.get("Diagnostics", [])
    self.assertLen(diagnostics, 1)
    self.assertEqual(diagnostics[0].get("HasReplacements"), expected)

  def _setup_temp_files(
      self,
  ) -> tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Setup for a simple yaml, diff, and BEP file in a temporary directory.

    Returns:
      A tuple of (tmpdir, yaml_path, diff_path, bep_path).
    """
    tmpdir = self.create_tempdir().full_path
    yaml_path = pathlib.Path(tmpdir) / "file1.cc.clang-tidy.yaml"
    self.create_tempfile(
        yaml_path.as_posix(),
        content=textwrap.dedent(f"""\
            ---
            MainSourceFile:  '{tmpdir}/file1.cc'
            Diagnostics:
              - DiagnosticName:  misc-unused
                DiagnosticMessage:
                  Message:         'unused variable'
                  FilePath:        '{tmpdir}/file1.cc'
                  FileOffset:      15
                Level:           Warning
                Replacements:
                  - FilePath:        '{tmpdir}/file1.cc'
                    Offset:          15
                    Length:          6
                    ReplacementText: ''
            ...
            """),
    )

    diff_path = pathlib.Path(tmpdir) / "test.diff"
    self.create_tempfile(
        diff_path.as_posix(),
        content=textwrap.dedent("""\
            diff --git a/file1.cc b/file1.cc
            index 123456..789012 100644
            --- a/file1.cc
            +++ b/file1.cc
            @@ -1,2 +1,3 @@
             line1
            +line2
             line3
            """),
    )

    bep_path = pathlib.Path(tmpdir) / "test.bep"
    self.create_tempfile(
        bep_path.as_posix(),
        content=json.dumps({
            "namedSetOfFiles": {
                "files": [{
                    "name": "file1.cc.clang-tidy.yaml",
                    "pathPrefix": [],
                }]
            }
        })
        + "\n",
    )
    return tmpdir, yaml_path, diff_path, bep_path

  def test_process_file_detects_replacements(self):
    tmpdir, yaml_path, diff_path, bep_path = self._setup_temp_files()
    config = clang_tidy_diff.AppConfig(
        patch=str(diff_path),
        repo_root=tmpdir,
        bep_file=str(bep_path),
        warnings_as_errors=False,
        fix=False,
    )

    def mock_offset_provider(_: str) -> Sequence[int]:
      return [0, 10, 20, 30]

    filterer = clang_tidy_diff.ClangTidyDiffFilter(
        config, offset_provider=mock_offset_provider
    )
    diagnostics, _ = filterer.process_file(yaml_path.as_posix())

    self.assertLen(diagnostics, 1)
    self.assertTrue(diagnostics[0].has_replacements)

  def test_run_collects_files_to_fix(self):
    tmpdir, _, diff_path, bep_path = self._setup_temp_files()
    config = clang_tidy_diff.AppConfig(
        patch=str(diff_path),
        repo_root=tmpdir,
        bep_file=str(bep_path),
        warnings_as_errors=False,
        fix=True,
    )
    staged_dirs = []

    def fake_apply_fixes(temp_dir: pathlib.Path):
      staged_dirs.append(temp_dir)
      copied_files = list(
          pathlib.Path(temp_dir).glob("*_file1.cc.clang-tidy.yaml")
      )
      self.assertLen(copied_files, 1)
      self.assertTrue(copied_files[0].exists())

    def mock_offset_provider(_: str) -> Sequence[int]:
      return [0, 10, 20, 30]

    filterer = clang_tidy_diff.ClangTidyDiffFilter(
        config,
        offset_provider=mock_offset_provider,
        apply_fixes_fn=fake_apply_fixes,
    )
    filterer.run()
    self.assertLen(staged_dirs, 1)

  def test_apply_fixes_stages_files_correctly(self):
    staged_dirs = []

    def fake_apply_fixes(temp_dir: pathlib.Path) -> None:
      staged_dirs.append(temp_dir.as_posix())
      copied_files = list(pathlib.Path(temp_dir).glob("*_file1.yaml"))
      self.assertLen(copied_files, 1)
      copied_file = copied_files[0]
      self.assertTrue(copied_file.exists())
      expected_normalized = textwrap.dedent(f"""\
            MainSourceFile: {tmpdir}/xla/file1.cc
            SomeOtherField: some_value
            FilePath: {tmpdir}/third_party/file1.cc
            """)
      self.assertEqual(copied_file.read_text(), expected_normalized)

    tmpdir = self.create_tempdir().full_path
    src_yaml = pathlib.Path(tmpdir) / "file1.yaml"
    src_yaml_content = textwrap.dedent("""\
        MainSourceFile: /b/f/w/xla/file1.cc
        SomeOtherField: some_value
        FilePath: /somepath/_bazel_user/1234/execroot/xla/third_party/file1.cc
        """)
    self.create_tempfile(src_yaml.as_posix(), content=src_yaml_content)
    diff_path = pathlib.Path(tmpdir) / "test.diff"
    self.create_tempfile(diff_path.as_posix(), content="")
    bep_path = pathlib.Path(tmpdir) / "test.bep"
    self.create_tempfile(bep_path.as_posix(), content="")

    config = clang_tidy_diff.AppConfig(
        patch=str(diff_path),
        repo_root=tmpdir,
        bep_file=str(bep_path),
        warnings_as_errors=False,
        fix=True,
    )
    filterer = clang_tidy_diff.ClangTidyDiffFilter(
        config,
        apply_fixes_fn=fake_apply_fixes,
    )

    filterer.apply_fixes([src_yaml])

    self.assertLen(staged_dirs, 1)


if __name__ == "__main__":
  absltest.main()
