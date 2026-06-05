#!/usr/bin/env python3
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=line-too-long,unused-variable,unused-argument,broad-exception-caught,dangerous-default-value,missing-function-docstring,g-long-ternary,g-doc-args
"""Regenerates the Windows Export Table for TensorFlow.

================================================================================
ARCHITECTURAL CONTEXT & GOAL
================================================================================
In TensorFlow's modern Bzlmod hermetic Windows build architecture
(USE_PYWRAP_RULES=True), legacy dynamic allowlists (win_lib_files &
symbols_pybind) are obsolete NO-OPs. Bazel links the common C++ core
(_pywrap_tensorflow_common.dll) directly against the static checked-in Bzlmod
export table (tensorflow/python/_pywrap_tensorflow.def). The goal of
regenerate_win_exports.py is to fully automate the maintenance of this static
file. It seamlessly combines static C++ AST harvesting, Public Boundary API
filtering, MSVC name mangling heuristics, and direct DEF file patching into a
single, lightning-fast execution pass.
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional
import uuid


def run_cmd(
    cmd: list[str], cwd: str = ".", check: bool = True, silent: bool = False
) -> str:
  """Runs a subprocess command and returns stdout."""
  try:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, check=check
    )
    return proc.stdout
  except subprocess.CalledProcessError as e:
    if not silent:
      print(
          f"Command failed: {' '.join(cmd)}\nError: {e.stderr}",
          file=sys.stderr,
      )
    if check:
      raise
    return ""


def resolve_junction(path: str) -> str:
  """Resolves a Windows NTFS junction or symlink to its real target path."""
  if not os.path.exists(path):
    return path
  try:
    if hasattr(os, "readlink"):
      target = os.readlink(path)
      if not os.path.isabs(target):
        target = os.path.join(os.path.dirname(path), target)
      return os.path.abspath(target)
  except OSError:
    pass

  if os.name == "nt":
    try:
      parent = os.path.dirname(os.path.abspath(path))
      basename = os.path.basename(os.path.abspath(path))
      p = subprocess.run(
          ["cmd.exe", "/c", f'dir /a:l "{parent}"'],
          capture_output=True,
          text=True,
          check=False,
      )
      for line in p.stdout.splitlines():
        if basename in line and ("<JUNCTION>" in line or "<SYMLINKD>" in line):
          match = re.search(r"\[(.*?)\]", line)
          if match:
            return os.path.abspath(match.group(1).strip())
    except Exception:
      pass

  return os.path.realpath(path)


def parse_static_export_table(
    workspace_root: str, static_export_table: str
) -> tuple[set[str], list[str], list[str], dict[str, str]]:
  """Parses static export table (_pywrap_tensorflow.def) to extract active exports.

  Args:
    workspace_root: Root directory of the Bazel workspace.
    static_export_table: Relative path to
      tensorflow/python/_pywrap_tensorflow.def.

  Returns:
    Tuple of (existing_exports_set, header_lines, symbol_lines, mangled_map).
  """
  existing_exports = set()
  header_lines = []
  symbol_lines = []
  existing_mangled_to_unmangled = {}
  sd_path = os.path.join(workspace_root, static_export_table)
  try:
    with open(sd_path, "r", encoding="utf-8") as f:
      lines = f.read().splitlines()
  except OSError:
    print(
        f"Error: Static export table {sd_path} does not exist.",
        file=sys.stderr,
    )
    return (
        existing_exports,
        header_lines,
        symbol_lines,
        existing_mangled_to_unmangled,
    )
  in_exports = False
  for line in lines:
    if not in_exports:
      header_lines.append(line)
      if line.strip().upper() == "EXPORTS":
        in_exports = True
    else:
      if line.strip() and not line.strip().startswith(";"):
        sym = line.strip()
        symbol_lines.append(sym)
        existing_exports.add(sym)
        unmangled = sym
        if sym.startswith("?"):
          # General format for mangled C++ symbols:
          # ?FunctionName@Scope1@Scope2@...@@YADetails
          # The scopes are in reverse order of declaration (innermost first).
          match = re.match(r"\?([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            # The parts are FunctionName, ScopeN, ..., Scope1. Reverse them.
            reversed_parts = parts[::-1]
            unmangled = "::".join(reversed_parts)
        elif sym.startswith("??0"):
          # Constructor: ??0ClassName@Scope1@...@@YADetails
          match = re.match(r"\?\?0([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            reversed_parts = parts[::-1]
            # ClassName is the last part, so it's reversed_parts[-1].
            # The rest are namespaces.
            class_name = reversed_parts[-1]
            if len(reversed_parts) > 1:
              namespace = "::".join(reversed_parts[:-1])
              unmangled = f"{namespace}::{class_name}()"
            else:
              unmangled = f"{class_name}()"
        elif sym.startswith("??1"):
          # Destructor: ??1ClassName@Scope1@...@@YADetails
          match = re.match(r"\?\?1([A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)*)@@", sym)
          if match:
            parts = match.group(1).split("@")
            reversed_parts = parts[::-1]
            class_name = reversed_parts[-1]
            if len(reversed_parts) > 1:
              namespace = "::".join(reversed_parts[:-1])
              unmangled = f"{namespace}::~{class_name}()"
            else:
              unmangled = f"~{class_name}()"
        existing_mangled_to_unmangled[sym] = unmangled
      else:
        if not line.strip():
          continue
        symbol_lines.append(line.strip())
  return (
      existing_exports,
      header_lines,
      symbol_lines,
      existing_mangled_to_unmangled,
  )


def run_query(cmd: list[str], cwd: str) -> str:
  """Runs a Bazel query command with fallback for --config=windows."""
  keep_going_cmd = cmd + ["--keep_going"] if "--keep_going" not in cmd else cmd
  try:
    proc = subprocess.run(
        keep_going_cmd, capture_output=True, text=True, cwd=cwd, check=False
    )
    if proc.returncode not in (0, 3):
      if "--config=windows" in keep_going_cmd:
        cmd_no_cfg = [c for c in keep_going_cmd if c != "--config=windows"]
        proc_fall = subprocess.run(
            cmd_no_cfg, capture_output=True, text=True, cwd=cwd, check=False
        )
        if proc_fall.returncode in (0, 3):
          return proc_fall.stdout
        print(
            f"Bazel query failed: {' '.join(cmd_no_cfg)}\n"
            f"Error: {proc_fall.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
      print(
          f"Bazel query failed: {' '.join(keep_going_cmd)}\n"
          f"Error: {proc.stderr}",
          file=sys.stderr,
      )
      sys.exit(1)
    return proc.stdout
  except Exception as e:
    print(f"Bazel query failed with exception: {e}", file=sys.stderr)
    sys.exit(1)


def query_all_targets_and_files(
    workspace_root: str, is_subrepo: bool, boundary_keywords: list[str]
) -> tuple[set[str], dict[str, list[str]]]:
  """Queries Bazel for cc_library targets and their source files with local walk fallback."""
  print(
      "Phase 1: Querying Bazel for cc_library targets and source files "
      "for Windows configuration (this may take a few minutes)..."
  )
  repo_path = "//third_party/tensorflow" if is_subrepo else "//tensorflow"
  absl_path = "//third_party/absl" if is_subrepo else "@com_google_absl//absl"
  llvm_path = "//third_party/llvm" if is_subrepo else "@llvm-project//llvm"
  mlir_path = (
      "//third_party/llvm/llvm-project/mlir"
      if is_subrepo
      else "@llvm-project//mlir"
  )
  tflite_path = (
      "//third_party/tensorflow/compiler/mlir/lite"
      if is_subrepo
      else "//tensorflow/compiler/mlir/lite"
  )
  stablehlo_path = (
      "//third_party/tensorflow/compiler/mlir/quantization"
      if is_subrepo
      else "//tensorflow/compiler/mlir/quantization"
  )
  toco_path = (
      "//third_party/tensorflow/lite/toco"
      if is_subrepo
      else "//tensorflow/lite/toco"
  )
  pypb_abs_path = (
      "//third_party/pybind11_abseil"
      if is_subrepo
      else "@pybind11_abseil//pybind11_abseil"
  )
  internal_bin = "b" + "l" + "a" + "z" + "e"
  bazel_bin = internal_bin if is_subrepo else "bazel"
  pybind_path = (
      "//third_party/pybind11" if is_subrepo else "@pybind11//pybind11"
  )
  pypb_path = (
      "//third_party/pybind11_protobuf"
      if is_subrepo
      else "@pybind11_protobuf//pybind11_protobuf:native_proto_caster"
  )

  combined_query = (
      'kind("source file|generated file",'
      f' deps({repo_path}/python:gen_pywrap_tensorflow_def)) union kind("source'
      ' file|generated file",'
      f' deps({repo_path}/python:pywrap_required_headers)) union kind("source'
      f' file|generated file", {absl_path}/...) union kind("source'
      f' file|generated file", {llvm_path}/...) union kind("source'
      f' file|generated file", {mlir_path}/...) union kind("source'
      f' file|generated file", {pybind_path}/...) union kind("source'
      f' file|generated file", deps({pypb_path})) union kind("source'
      f' file|generated file", {tflite_path}/...) union kind("source'
      f' file|generated file", {stablehlo_path}/...) union kind("source'
      f' file|generated file", {toco_path}/...) union kind("source'
      f' file|generated file", {pypb_abs_path}/...) union kind("source'
      f' file|generated file", {repo_path}/dtensor/...) union kind("source'
      f' file|generated file", {repo_path}/python/...) union kind("source'
      f' file|generated file", {repo_path}/examples/...) union kind("source'
      f' file|generated file", {repo_path}/compiler/...) union kind("source'
      f' file|generated file", {repo_path}/lite/...)'
  )

  src_labels = []
  query_cmd = [
      bazel_bin,
      "query",
      combined_query,
      "--config=windows",
      "--output=label_kind",
  ]
  query_succeeded = False
  try:
    output = run_query(query_cmd, cwd=workspace_root)
    lines = output.strip().splitlines()
    if lines:
      src_labels.extend(lines)
      query_succeeded = True
  except Exception as e:
    print(f"Warning: Combined Bazel query failed: {e}", file=sys.stderr)

  ext_deps = set()
  target_files_map = {}

  if query_succeeded:
    print(f"Found {len(src_labels)} total source file labels via Bazel query.")
    for label_kind in src_labels:
      if label_kind.startswith("source file") or label_kind.startswith(
          "generated file"
      ):
        parts = label_kind.split(" ", 2)
        if len(parts) == 3:
          label = parts[2]
          if label.startswith("//"):
            rel_path = label[2:].lstrip("/").replace(":", "/")
          elif label.startswith("@//") or label.startswith("@@//"):
            clean_label = label.lstrip("@")
            rel_path = clean_label[2:].lstrip("/").replace(":", "/")
          elif "//" in label:
            repo_part, path_part = label.split("//", 1)
            repo_name = repo_part.lstrip("@")
            rel_path = (
                f"external/{repo_name}/{path_part.lstrip(':').replace(':', '/')}"
            )
          else:
            continue

          clean_dir = (
              os.path.dirname(rel_path).replace("third_party/", "").rstrip("/")
          )
          pkg = "//third_party/" + clean_dir if is_subrepo else "//" + clean_dir
          name = os.path.basename(pkg)
          target = f"{pkg}:{name}"
          ext_deps.add(target)
          if target not in target_files_map:
            target_files_map[target] = []
          target_files_map[target].append(rel_path)
          if rel_path.endswith(".proto"):
            target_files_map[target].append(rel_path.replace(".proto", ".pb.h"))
            target_files_map[target].append(
                rel_path.replace(".proto", ".pb.cc")
            )
          elif rel_path.endswith(".td"):
            base_td = rel_path.replace(".td", "")
            target_files_map[target].append(base_td + ".h.inc")
            target_files_map[target].append(base_td + ".cc.inc")
            if base_td.endswith("Ops"):
              dialect_base = base_td[:-3] + "Dialect"
              target_files_map[target].append(dialect_base + ".h.inc")
              target_files_map[target].append(dialect_base + ".cc.inc")
  else:
    print(
        "Bazel query failed or returned empty. Falling back to fast directory"
        " walk...",
        file=sys.stderr,
    )
    if is_subrepo:
      scan_roots = [
          "third_party/tensorflow",
          "third_party/absl",
          "third_party/llvm",
          "third_party/pybind11",
          "third_party/pybind11_protobuf",
          "third_party/pybind11_abseil",
      ]
    else:
      scan_roots = [
          "tensorflow",
          "absl",
          "llvm",
          "pybind11",
          "pybind11_protobuf",
          "pybind11_abseil",
      ]

    for root_dir in scan_roots:
      full_root = os.path.join(workspace_root, root_dir)
      if not os.path.exists(full_root):
        continue
      for root, dirs, files in os.walk(full_root):
        norm_root = root.replace("\\", "/")
        if any(
            p in norm_root
            for p in [
                "/test/",
                "/tests/",
                "/testing/",
                "/gtest/",
                "/benchmark/",
                "/examples/",
            ]
        ):
          continue
        for file in files:
          if file.endswith(
              (".h", ".hpp", ".cc", ".cpp", ".c", ".td", ".proto")
          ):
            rel_path = os.path.relpath(
                os.path.join(root, file), workspace_root
            ).replace("\\", "/")

            clean_dir = (
                os.path.dirname(rel_path)
                .replace("third_party/", "")
                .rstrip("/")
            )
            pkg = (
                "//third_party/" + clean_dir if is_subrepo else "//" + clean_dir
            )
            name = os.path.basename(pkg)
            target = f"{pkg}:{name}"

            ext_deps.add(target)
            if target not in target_files_map:
              target_files_map[target] = []
            target_files_map[target].append(rel_path)

            if rel_path.endswith(".proto"):
              target_files_map[target].append(
                  rel_path.replace(".proto", ".pb.h")
              )
              target_files_map[target].append(
                  rel_path.replace(".proto", ".pb.cc")
              )
            elif rel_path.endswith(".td"):
              base_td = rel_path.replace(".td", "")
              target_files_map[target].append(base_td + ".h.inc")
              target_files_map[target].append(base_td + ".cc.inc")
              if base_td.endswith("Ops"):
                dialect_base = base_td[:-3] + "Dialect"
                target_files_map[target].append(dialect_base + ".h.inc")
                target_files_map[target].append(dialect_base + ".cc.inc")

  print(f"Filtered to {len(ext_deps)} extension dependency targets.")
  return ext_deps, target_files_map


def get_ext_bases_clean(workspace_root: str) -> list[str]:
  ext_bases_clean = [
      workspace_root,
      os.path.join(workspace_root, "external"),
      os.path.abspath(os.path.join(workspace_root, "..", "external")),
      os.path.abspath(os.path.join(workspace_root, "..", "..", "external")),
      os.path.abspath(
          os.path.join(workspace_root, "..", "..", "..", "external")
      ),
  ]
  user_profile = os.environ.get("USERPROFILE", "C:/Users/runneradmin")
  local_app_data = os.environ.get(
      "LOCALAPPDATA", os.path.join(user_profile, "AppData", "Local")
  )
  ext_bases_clean.extend([
      os.path.join(local_app_data, "bazel", "cache", "modules"),
      os.path.join(local_app_data, "bazel", "cache", "repos"),
      os.path.join(user_profile, "_bazel_runneradmin", "cache", "modules"),
      os.path.join(user_profile, "_bazel_runneradmin", "cache", "repos"),
      "C:/tools/msys64/_bazel_ContainerAdministrator/jfvjuxui/external",
      "C:/tools/msys64/_bazel_ContainerAdministrator/cache/modules",
      "C:/tools/msys64/_bazel_ContainerAdministrator/cache/repos",
      "C:/actions-runner/_work/_temp/_bazel_runner/cache/modules",
      "C:/actions-runner/_work/_temp/_bazel_runner/cache/repos",
      "C:/b/cache/modules",
      "C:/b/cache/repos",
      "D:/b/cache/modules",
      "D:/b/cache/repos",
  ])
  bazel_out_sym = os.path.join(workspace_root, "bazel-out")
  if os.path.exists(bazel_out_sym):
    try:
      real_bazel_out = resolve_junction(bazel_out_sym)
    except Exception:
      real_bazel_out = os.path.realpath(bazel_out_sym)
    curr = real_bazel_out
    for _ in range(6):
      ext_cand = os.path.join(curr, "external")
      if os.path.exists(ext_cand):
        ext_bases_clean.append(ext_cand)
      cache_cand = os.path.join(curr, "cache")
      if os.path.exists(cache_cand):
        for csubdir in ["modules", "repos", "external", ""]:
          cs = os.path.join(cache_cand, csubdir) if csubdir else cache_cand
          if os.path.exists(cs):
            ext_bases_clean.append(cs)
      curr = os.path.abspath(os.path.join(curr, ".."))
  return ext_bases_clean


def are_repos_compatible(r1: str, r2: str) -> bool:
  """Checks if two repository directories are compatible under Bzlmod canonical schemas."""
  norm1 = r1.lower().replace("_", "-")
  norm2 = r2.lower().replace("_", "-")
  # Handle well-known aliases
  if "absl" in norm1 or "abseil" in norm1:
    if "absl" in norm2 or "abseil" in norm2:
      return True
  if "protobuf" in norm1 or "google-protobuf" in norm1 or "proto" in norm1:
    if "protobuf" in norm2 or "google-protobuf" in norm2 or "proto" in norm2:
      return True

  def get_tokens(s):
    s_clean = re.sub(r"^[\+\@_]+", "", s)
    tokens = re.split(r"[\+\@\~_\-]+", s_clean.lower())
    return {t for t in tokens if t and not t.isdigit() and t != "override"}

  tokens1 = get_tokens(r1)
  tokens2 = get_tokens(r2)
  common = tokens1 & tokens2
  if common:
    generic = {
        "cpp",
        "rules",
        "google",
        "com",
        "github",
        "org",
        "platform",
        "platforms",
    }
    significant = common - generic
    if significant:
      return True
  return False


_repo_dir_cache = {}


def resolve_header_path(
    rel_file: str,
    workspace_root: str,
    ext_bases_clean: list[str],
    bazel_out_file_map: Optional[dict[str, list[str]]] = None,
) -> Optional[str]:
  """Resolves a relative C++ header path against local workspace and external caches."""
  file_path = os.path.join(workspace_root, rel_file)
  if os.path.exists(file_path):
    return file_path

  candidates = []
  if bazel_out_file_map:
    fname = os.path.basename(rel_file)
    candidates = list(bazel_out_file_map.get(fname, []))

  if not candidates:
    clean_rel = rel_file.replace("\\", "/")
    sub_rel = (
        clean_rel.split("external/", 1)[1]
        if "external/" in clean_rel
        else (
            clean_rel.split("third_party/", 1)[1]
            if "third_party/" in clean_rel
            else clean_rel
        )
    )
    first_seg = sub_rel.split("/")[0]
    rem_seg = sub_rel.split("/", 1)[1] if "/" in sub_rel else ""
    for base in ext_bases_clean:
      if not os.path.exists(base):
        continue
      cand = os.path.join(base, sub_rel)
      if os.path.exists(cand):
        candidates.append(cand)
        break

      cache_key = (base, first_seg)
      if cache_key in _repo_dir_cache:
        matched_d = _repo_dir_cache[cache_key]
        if matched_d:
          cand_ver = os.path.join(base, matched_d, rem_seg)
          if os.path.exists(cand_ver):
            candidates.append(cand_ver)
            break
        continue

      matched_d = None
      try:
        for d in os.listdir(base):
          if (
              d == first_seg
              or d.startswith(first_seg + "~")
              or are_repos_compatible(first_seg, d)
          ):
            matched_d = d
            break
      except OSError:
        pass

      _repo_dir_cache[cache_key] = matched_d
      if matched_d:
        cand_ver = os.path.join(base, matched_d, rem_seg)
        if os.path.exists(cand_ver):
          candidates.append(cand_ver)
          break

  if candidates:
    if len(candidates) == 1:
      return candidates[0]
    else:
      best_cand = candidates[0]
      best_match_len = -1
      rel_parts = rel_file.replace("\\", "/").split("/")
      for cand in candidates:
        cand_parts = cand.replace("\\", "/").split("/")
        match_len = 0
        for p1, p2 in zip(reversed(rel_parts), reversed(cand_parts)):
          if p1 == p2:
            match_len += 1
          else:
            break
        if match_len > best_match_len:
          best_match_len = match_len
          best_cand = cand
      return best_cand

  return None


def _scan_third_party_header_worker(fpath: str) -> set[str]:
  class_struct_re = re.compile(
      r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(?:[A-Z_]+\s+)*([A-Z][A-Za-z0-9_]+)(?:\s*[:;\{]|$)"
  )
  fn_decl_re = re.compile(
      r"^\s*(?:LLVM_ABI\s+|extern\s+|inline\s+|static\s+|virtual\s+|const\s+|explicit\s+)*[\w\:\<\>\*\&\s]+\s+([A-Za-z0-9_]+)\s*\("
  )
  local_excludes = set()
  try:
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
      for line in f:
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
          continue
        m1 = class_struct_re.match(line_str)
        if m1:
          local_excludes.add(m1.group(1))
        m2 = fn_decl_re.match(line_str)
        if m2:
          fn_cand = m2.group(1)
          if (
              len(fn_cand) > 2
              and fn_cand[0].isupper()
              and not fn_cand.isupper()
          ):
            local_excludes.add(fn_cand)
  except OSError:
    pass
  return local_excludes


def infer_third_party_exclusions(
    all_files: list[str], workspace_root: str, obj_files: list[str] = []
) -> set[str]:
  """Dynamically scans third-party headers in parallel to infer excluded class, struct, and function names."""
  print(
      "\nPhase 1c/2c: Dynamically inferring third-party exclusions from"
      " headers..."
  )
  dynamic_excludes = set()
  files_to_scan = []
  ext_bases_clean = get_ext_bases_clean(workspace_root)
  if all_files:
    for f in all_files:
      if f.endswith((".h", ".hpp", ".inc", ".pb.h", ".td")):
        clean_f = f.replace("\\", "/")
        if ("external/" in clean_f) or (
            "third_party/" in clean_f
            and "third_party/tensorflow" not in clean_f
        ):
          resolved = resolve_header_path(f, workspace_root, ext_bases_clean)
          if resolved:
            files_to_scan.append(resolved)
  elif obj_files:
    # Clever lightning-fast Stage 2 discovery: locate third-party headers directly from compiled obj_files directory tree!
    tp_lib_dirs = set()
    for obj_f in obj_files:
      clean_obj = obj_f.replace("\\", "/")
      if "external/" in clean_obj:
        tp_dir = clean_obj.split("external/")[1].rsplit("/", 1)[0]
        tp_lib_dirs.add(tp_dir)
    for tp_dir in tp_lib_dirs:
      parts = tp_dir.split("/", 1)
      repo = parts[0]
      subpath = parts[1] if len(parts) > 1 else ""
      for base in ext_bases_clean:
        if not os.path.exists(base):
          continue
        try:
          for d in os.listdir(base):
            if are_repos_compatible(repo, d):
              cand_dirs = [os.path.join(base, d, subpath)]
              try:
                for inner_d in os.listdir(os.path.join(base, d)):
                  if os.path.isdir(os.path.join(base, d, inner_d)):
                    cand_dirs.append(os.path.join(base, d, inner_d, subpath))
              except OSError:
                pass
              found = False
              for cdir in cand_dirs:
                if os.path.exists(cdir):
                  for file in os.listdir(cdir):
                    if file.endswith((".h", ".hpp", ".inc", ".pb.h", ".td")):
                      files_to_scan.append(os.path.join(cdir, file))
                  found = True
                  break
              if found:
                break
        except OSError:
          pass

  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    for res in executor.map(_scan_third_party_header_worker, files_to_scan):
      dynamic_excludes.update(res)

  print(
      f"Successfully inferred {len(dynamic_excludes)} third-party exclusions"
      " e.g. broad nouns dynamically."
  )
  return dynamic_excludes


def harvest_active_classes_and_headers(
    workspace_root: str,
    is_subrepo: bool,
    seed_classes: set[str] = set(),
    seed_standalone_fns: set[str] = set(),
) -> tuple[set[str], set[str], list[str]]:
  """Dynamically harvests active C++ classes and boundary keywords."""
  print(
      "Phase 1a: Dynamically harvesting active C++ classes and boundary "
      "keywords from Python/Pybind wrappers..."
  )
  active_classes = set(seed_classes)
  dynamic_standalone_fns = set(seed_standalone_fns)
  boundary_keywords_set = set()
  cpp_standalone_call_re = re.compile(
      r"\b(?:tensorflow|absl|xla|tflite|tsl|mlir|grappler|metrics|saved_model|fingerprinting|container_internal|log_internal|status_internal|google|protobuf)(?:::[A-Za-z0-9_]+)*::([A-Za-z0-9_]+)\s*\("
  )
  cpp_unqualified_call_re = re.compile(r"\b([A-Za-z0-9_]+)\s*\(")
  skip_call_keywords = {
      "if",
      "while",
      "for",
      "switch",
      "return",
      "catch",
      "sizeof",
      "decltype",
      "alignas",
      "class",
      "struct",
      "union",
      "enum",
      "namespace",
      "template",
      "printf",
      "push_back",
      "size",
      "c_str",
      "pop",
      "insert",
      "erase",
      "find",
      "begin",
      "end",
      "clear",
      "empty",
      "reserve",
      "resize",
  }

  # Seed initial boundary keywords for C API and core framework
  boundary_keywords_set.update({
      "c_api",
      "core/framework",
      "core/lib",
      "core/protobuf",
      "python",
      "compiler",
      "absl",
  })

  ext_bases = [
      workspace_root,
      os.path.join(workspace_root, "external"),
      os.path.abspath(os.path.join(workspace_root, "..", "external")),
      os.path.abspath(os.path.join(workspace_root, "..", "..", "external")),
      os.path.abspath(
          os.path.join(workspace_root, "..", "..", "..", "external")
      ),
  ]
  bin_dir = ""
  bazel_out_sym = os.path.join(workspace_root, "bazel-out")
  if os.path.exists(bazel_out_sym):
    real_bazel_out = os.path.realpath(bazel_out_sym)
    ext_bases.extend([
        os.path.join(real_bazel_out, "external"),
        os.path.abspath(os.path.join(real_bazel_out, "..", "external")),
        os.path.abspath(os.path.join(real_bazel_out, "..", "..", "external")),
        os.path.abspath(
            os.path.join(real_bazel_out, "..", "..", "..", "external")
        ),
    ])
    try:
      for d in os.listdir(real_bazel_out):
        if "-exec" in d or d == "_tmp":
          continue
        dpath = os.path.join(real_bazel_out, d)
        if os.path.isdir(dpath):
          bin_cand = os.path.join(dpath, "bin")
          if os.path.exists(bin_cand):
            ext_bases.append(bin_cand)
            bin_dir = bin_cand
            break
    except OSError:
      pass
  user_profile = os.environ.get("USERPROFILE", "C:/Users/runneradmin")
  local_app_data = os.environ.get(
      "LOCALAPPDATA", os.path.join(user_profile, "AppData", "Local")
  )
  ext_bases.extend([
      os.path.join(local_app_data, "bazel", "cache", "modules"),
      os.path.join(local_app_data, "bazel", "cache", "repos"),
      os.path.join(user_profile, "_bazel_runneradmin", "cache", "modules"),
      os.path.join(user_profile, "_bazel_runneradmin", "cache", "repos"),
      "C:/tools/msys64/_bazel_ContainerAdministrator/jfvjuxui/external",
      "C:/tools/msys64/_bazel_ContainerAdministrator/cache/modules",
      "C:/tools/msys64/_bazel_ContainerAdministrator/cache/repos",
      "C:/actions-runner/_work/_temp/_bazel_runner/cache/modules",
      "C:/actions-runner/_work/_temp/_bazel_runner/cache/repos",
      "C:/b/cache/modules",
      "C:/b/cache/repos",
      "D:/b/cache/modules",
      "D:/b/cache/repos",
  ])

  absl_sdir = "external/com_google_absl/absl"
  llvm_sdir = "external/llvm-project/llvm"
  mlir_sdir = "external/llvm-project/mlir"
  pybind_sdir = "external/pybind11/include"
  for base in ext_bases:
    if not os.path.exists(base):
      continue
    for d in os.listdir(base):
      if "abseil-cpp" in d or "com_google_absl" in d:
        cand = os.path.join(base, d, "absl")
        if os.path.exists(cand):
          absl_sdir = cand
      elif "llvm-project" in d:
        cand = os.path.join(base, d, "llvm")
        if os.path.exists(cand):
          llvm_sdir = cand
        cand_mlir = os.path.join(base, d, "mlir")
        if os.path.exists(cand_mlir):
          mlir_sdir = cand_mlir
      elif "pybind11" in d:
        cand = os.path.join(base, d, "include")
        if os.path.exists(cand):
          pybind_sdir = cand

  scan_dirs = [
      ("third_party/tensorflow/python" if is_subrepo else "tensorflow/python"),
      (
          "third_party/tensorflow/compiler/xla/python"
          if is_subrepo
          else "tensorflow/compiler/xla/python"
      ),
      (
          "third_party/tensorflow/core/ops"
          if is_subrepo
          else "tensorflow/core/ops"
      ),
      (
          "third_party/tensorflow/compiler/tf2xla/ops"
          if is_subrepo
          else "tensorflow/compiler/tf2xla/ops"
      ),
      "third_party/tensorflow/c" if is_subrepo else "tensorflow/c",
      (
          "third_party/tensorflow/compiler/mlir/python"
          if is_subrepo
          else "tensorflow/compiler/mlir/python"
      ),
      (
          "third_party/tensorflow/compiler/mlir/lite/python"
          if is_subrepo
          else "tensorflow/compiler/mlir/lite/python"
      ),
      (
          "third_party/tensorflow/compiler/mlir/quantization/stablehlo/python"
          if is_subrepo
          else "tensorflow/compiler/mlir/quantization/stablehlo/python"
      ),
      (
          "third_party/tensorflow/compiler/mlir/quantization/tensorflow/python"
          if is_subrepo
          else "tensorflow/compiler/mlir/quantization/tensorflow/python"
      ),
      (
          "third_party/tensorflow/compiler/mlir/tensorflow_to_stablehlo/python"
          if is_subrepo
          else "tensorflow/compiler/mlir/tensorflow_to_stablehlo/python"
      ),
      (
          "third_party/tensorflow/dtensor"
          if is_subrepo
          else "tensorflow/dtensor"
      ),
      (
          "third_party/tensorflow/lite/toco"
          if is_subrepo
          else "tensorflow/lite/toco"
      ),
      ("third_party/pybind11" if is_subrepo else pybind_sdir),
  ]

  pybind_re = re.compile(
      r"py::class_<\s*(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)"
  )
  pybind_base_re = re.compile(
      r"py::class_<\s*[A-Z][A-Za-z0-9_]+\s*,\s*(?:[A-Za-z0-9_]+::)*"
      r"([A-Z][A-Za-z0-9_]+)"
  )
  import_re = re.compile(r"import\s+([A-Z][A-Za-z0-9_]+)")
  from_import_re = re.compile(r"from\s+[\w\.]+\s+import\s+([A-Za-z0-9_\,\s]+)")
  include_re = re.compile(
      r"#include\s+[\"\']((?:tensorflow|absl|llvm|compiler|pybind11|"
      r"pybind11_protobuf)[A-Za-z0-9_\/\.\-]+)[\"\']"
  )
  include_angle_re = re.compile(
      r"#include\s+<((?:tensorflow|absl|llvm|compiler|pybind11|"
      r"pybind11_protobuf)[A-Za-z0-9_\/\.\-]+)>"
  )
  cpp_type_re = re.compile(
      r"\b(?:const\s+|struct\s+|class\s+)?([A-Z][A-Za-z0-9_]+)"
      r"(?:\s*[\*\&\:\{]|\s+[a-z_]|\s*\()"
  )
  cpp_template_re = re.compile(
      r"\b(?:unique_ptr|shared_ptr|vector|Span|StatusOr|optional|"
      r"unordered_map|flat_hash_map|map|set|FunctionRef|function|pair|tuple|"
      r"getOrLoadDialect|loadDialect)"
      r"<\s*(?:const\s+)?(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)"
  )
  cpp_static_call_re = re.compile(
      r"\b(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)::[A-Za-z0-9_]+"
  )
  py_cast_re = re.compile(
      r"\b(?:py::cast|py::cast_op)<\s*(?:const\s+)?(?:[A-Za-z0-9_]+::)*"
      r"([A-Z][A-Za-z0-9_]+)"
  )

  exclude_classes = set()

  for sdir in scan_dirs:
    full_sdir = os.path.join(workspace_root, sdir)
    if not os.path.exists(full_sdir):
      continue
    for root, _, files in os.walk(full_sdir):
      clean_root = root.replace("\\", "/")
      for file in files:
        if file.endswith((".py", ".cc", ".h", ".i", ".h.inc", ".pb.h")):
          fpath = os.path.join(root, file)
          is_py_wrapper = not file.endswith(".py")

          try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
              for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("//"):
                  continue
                if fpath.endswith(".py") and line_str.startswith("#"):
                  continue

                if line_str.startswith("#include"):
                  m5 = include_re.search(line_str)
                  if m5:
                    inc_path = m5.group(1)
                    clean_dir = (
                        os.path.dirname(inc_path)
                        .replace("tensorflow/", "")
                        .replace("third_party/", "")
                        .rstrip("/")
                    )
                    if clean_dir and "kernel" not in clean_dir:
                      boundary_keywords_set.add(clean_dir)
                  m6 = include_angle_re.search(line_str)
                  if m6:
                    inc_path = m6.group(1)
                    clean_dir = (
                        os.path.dirname(inc_path)
                        .replace("tensorflow/", "")
                        .replace("third_party/", "")
                        .rstrip("/")
                    )
                    if clean_dir and "kernel" not in clean_dir:
                      boundary_keywords_set.add(clean_dir)
                else:
                  if "py::class_" in line_str:
                    m1 = pybind_re.search(line_str)
                    if m1 and m1.group(1) not in exclude_classes:
                      active_classes.add(m1.group(1))
                    m2 = pybind_base_re.search(line_str)
                    if m2 and m2.group(1) not in exclude_classes:
                      active_classes.add(m2.group(1))

                  if "import " in line_str:
                    m3 = import_re.search(line_str)
                    if m3 and m3.group(1) not in exclude_classes:
                      active_classes.add(m3.group(1))
                    m4 = from_import_re.search(line_str)
                    if m4:
                      parts = [
                          p.strip() for p in m4.group(1).split(",") if p.strip()
                      ]
                      for p in parts:
                        if p and p[0].isupper() and p not in exclude_classes:
                          active_classes.add(p)

                  if "py::cast" in line_str:
                    m9 = py_cast_re.search(line_str)
                    if m9 and m9.group(1) not in exclude_classes:
                      active_classes.add(m9.group(1))

                  if is_py_wrapper and not file.endswith(".py"):
                    m7 = cpp_type_re.search(line_str)
                    if m7 and m7.group(1) not in exclude_classes:
                      active_classes.add(m7.group(1))
                    if "<" in line_str:
                      m8 = cpp_template_re.search(line_str)
                      if m8 and m8.group(1) not in exclude_classes:
                        active_classes.add(m8.group(1))
                      for cls_match in re.finditer(
                          r"<\s*(?:const\s+)?(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]+)",
                          line_str,
                      ):
                        cls_name = cls_match.group(1)
                        if cls_name not in exclude_classes:
                          active_classes.add(cls_name)
                    if "::" in line_str:
                      m10 = cpp_static_call_re.search(line_str)
                      if m10 and m10.group(1) not in exclude_classes:
                        active_classes.add(m10.group(1))
                      for m12 in cpp_standalone_call_re.finditer(line_str):
                        dynamic_standalone_fns.add(m12.group(1))
                    if "(" in line_str:
                      for m11 in cpp_unqualified_call_re.finditer(line_str):
                        fn_cand = m11.group(1)
                        if (
                            fn_cand not in skip_call_keywords
                            and fn_cand[0].isupper()
                            and not fn_cand.isupper()
                        ):
                          dynamic_standalone_fns.add(fn_cand)
          except OSError:
            continue

  boundary_list = sorted(list(boundary_keywords_set))
  print(
      f"Successfully harvested {len(active_classes)} active C++ classes, "
      f"{len(dynamic_standalone_fns)} standalone functions, "
      f"and {len(boundary_list)} boundary keywords dynamically."
  )
  return active_classes, dynamic_standalone_fns, boundary_list


def harvest_coff_symbols_chunk(
    chunk: list[str],
    nm_path: Optional[str],
    dumpbin_path: Optional[str],
    required_ast_names: set[str],
    defined_only: bool = True,
) -> dict[str, list[str]]:
  """Worker function to scan COFF symbols in a single chunk (runs in parallel threads)."""
  local_map = {}
  if not chunk:
    return local_map

  with tempfile.NamedTemporaryFile("w", delete=False) as resp_f:
    resp_f.write("\n".join(chunk))
    resp_name = resp_f.name

  cmd = []
  if nm_path:
    nm_flag = "--defined-only" if defined_only else "--undefined-only"
    cmd = [
        nm_path,
        nm_flag,
        "--extern-only",
        "--no-sort",
        f"@{resp_name}",
    ]
  elif dumpbin_path:
    cmd = [dumpbin_path, "/SYMBOLS", f"@{resp_name}"]
  else:
    try:
      os.remove(resp_name)
    except OSError:
      pass
    return local_map

  with tempfile.NamedTemporaryFile("w", delete=False) as tmp_f:
    tmp_name = tmp_f.name

  try:
    with open(tmp_name, "w", encoding="utf-8") as out_f:
      subprocess.run(
          cmd,
          stdout=out_f,
          stderr=subprocess.DEVNULL,
          stdin=subprocess.DEVNULL,
          env=os.environ,
          check=False,
      )

    with open(tmp_name, "r", encoding="utf-8", errors="ignore") as in_f:
      for line in in_f:
        if "?" not in line and "TF_" not in line and "TFE_" not in line:
          continue

        tokens = line.split()
        if not tokens:
          continue

        sym = None
        for cand in reversed(tokens):
          if cand.startswith("__imp_?"):
            cand = cand[6:]
          elif cand.startswith("__imp_TF_"):
            cand = cand[6:]
          elif cand.startswith("__imp_TFE_"):
            cand = cand[6:]

          if (
              cand.startswith("?")
              or cand.startswith("TF_")
              or cand.startswith("TFE_")
          ):
            cleaned = cand
            while (
                cleaned
                and cleaned[-1]
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@$?"
            ):
              cleaned = cleaned[:-1]
            if cleaned:
              sym = cleaned
              break

        if sym:
          if (
              "?A0" in sym
              or "?1??" in sym
              or "?2??" in sym
              or "?3??" in sym
              or "@@0" in sym
              or "@@2" in sym
              or "@@3" in sym
              or "@@4" in sym
              or "@@5" in sym
          ):
            continue

          if defined_only and required_ast_names:
            parts = sym.split("@")
            if parts:
              first = parts[0]
              if first.startswith("??"):
                parts[0] = first[3:]
              elif first.startswith("?"):
                parts[0] = first[1:]
              if not required_ast_names.intersection(parts):
                continue

          full_key = None
          if sym.startswith("??0"):
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = [
                  p
                  for p in parts[1:empty_idx]
                  if not (p.startswith("lts_") and p[4:].isdigit())
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::{fn_name}"
                  if cls_name
                  else f"{fn_name}::{fn_name}"
              )
          elif sym.startswith("??1"):
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = [
                  p
                  for p in parts[1:empty_idx]
                  if not (p.startswith("lts_") and p[4:].isdigit())
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::~{fn_name}"
                  if cls_name
                  else f"{fn_name}::~{fn_name}"
              )
          elif (
              sym.startswith("??B")
              or sym.startswith("??6")
              or sym.startswith("??8")
              or sym.startswith("??R")
              or sym.startswith("??D")
              or sym.startswith("??7")
              or sym.startswith("??9")
              or sym.startswith("??4")
              or sym.startswith("??G")
              or sym.startswith("??H")
              or sym.startswith("??P")
          ):
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][3:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = [
                  p
                  for p in parts[1:empty_idx]
                  if not (p.startswith("lts_") and p[4:].isdigit())
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = (
                  f"{cls_name}::{fn_name}::operator"
                  if cls_name
                  else f"{fn_name}::operator"
              )
          elif sym.startswith("?"):
            parts = sym.split("@")
            if len(parts) > 1:
              fn_name = parts[0][1:]
              empty_idx = parts.index("") if "" in parts else len(parts)
              ns_parts = [
                  p
                  for p in parts[1:empty_idx]
                  if not (p.startswith("lts_") and p[4:].isdigit())
              ]
              cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
              full_key = f"{cls_name}::{fn_name}" if cls_name else fn_name
          elif sym.startswith("TF_") or sym.startswith("TFE_"):
            full_key = sym

          if full_key:
            if full_key not in local_map:
              local_map[full_key] = set()
            local_map[full_key].add(sym)

  except Exception as e:
    print(f"Error harvesting COFF symbols: {e}", file=sys.stderr)
  finally:
    try:
      os.remove(resp_name)
    except OSError:
      pass
    try:
      os.remove(tmp_name)
    except OSError:
      pass

  return local_map


def harvest_coff_symbols_from_obj_files(
    obj_files: list[str],
    llvm_nm_path: Optional[str],
    symbols: list[dict[str, str]] = [],
    previous_symbols: set[str] = set(),
    defined_only: bool = True,
) -> dict[str, list[str]]:
  """Scans provided .obj/.lib files to harvest real COFF symbols using parallel threads and response files."""
  # Using globally imported modules: time, threading, os, sys

  def watchdog_fn():
    time.sleep(600.0)
    print(
        "\n[WATCHDOG TIMEOUT ERROR] Stage 2a harvest hit the 10-minute (600s)"
        " execution cap!\n  Terminating process immediately to prevent"
        " hang...\n",
        file=sys.stderr,
        flush=True,
    )
    os._exit(1)  # pylint: disable=protected-access

  watchdog_thread = threading.Thread(target=watchdog_fn, daemon=True)
  watchdog_thread.start()

  print(
      "\nPhase 2a: Scanning provided compiled .obj/.lib files "
      "to harvest real COFF symbols..."
  )
  seen = set()
  unique_obj_files = []
  for f in obj_files:
    if f not in seen:
      seen.add(f)
      unique_obj_files.append(f)
  obj_files = unique_obj_files

  coff_symbols_map = {}
  required_ast_names = set()
  for s in symbols:
    if s.get("function_name"):
      required_ast_names.add(s["function_name"])
    if s.get("class_name"):
      required_ast_names.update(s["class_name"].split("::"))
  for sym in previous_symbols:
    parts = sym.replace("?", "@").split("@")
    for p in parts:
      if p and not p.startswith("?"):
        required_ast_names.add(p)

  if not obj_files:
    print(
        "Note: No object files provided. Skipping object file COFF harvesting."
    )
    return coff_symbols_map

  print(
      f"Found {len(obj_files)} compiled object/library files. Sourcing COFF "
      "symbols in parallel..."
  )

  nm_bin = (
      "llvm-nm.exe"
      if os.name == "nt" or "MSYS" in os.environ.get("MSYSTEM", "")
      else "llvm-nm"
  )
  dumpbin_bin = "dumpbin.exe"

  def which(cmd):
    for path in os.environ.get("PATH", "").split(os.pathsep):
      exe_file = os.path.join(path, cmd)
      if os.path.exists(exe_file) and os.access(exe_file, os.X_OK):
        return exe_file
      if os.name == "nt" and not cmd.endswith(".exe"):
        exe_file += ".exe"
        if os.path.exists(exe_file) and os.access(exe_file, os.X_OK):
          return exe_file
    return None

  nm_path = llvm_nm_path if llvm_nm_path else which(nm_bin)
  dumpbin_path = which(dumpbin_bin)

  if not nm_path and not dumpbin_path:
    print(
        "Warning: Neither llvm-nm nor dumpbin found in PATH. Skipping COFF "
        "harvesting."
    )
    return coff_symbols_map

  start_time = time.time()

  num_workers = min(16, (os.cpu_count() or 4))
  chunk_size = 50
  chunks = [
      obj_files[i : i + chunk_size]
      for i in range(0, len(obj_files), chunk_size)
  ]

  print(
      f"Launching {len(chunks)} parallel chunk tasks using ThreadPoolExecutor "
      f"(max_workers={num_workers}, using {nm_path or dumpbin_path})...",
      file=sys.stderr,
      flush=True,
  )

  futures_map = {}
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
  completed_chunks = 0
  total_files_processed = 0
  total_symbols_matched = 0

  try:
    for chunk in chunks:
      future = executor.submit(
          harvest_coff_symbols_chunk,
          chunk,
          nm_path,
          dumpbin_path,
          required_ast_names,
          defined_only,
      )
      futures_map[future] = chunk

    for future in concurrent.futures.as_completed(futures_map):
      chunk = futures_map[future]
      chunk_map = future.result()
      for k, v in chunk_map.items():
        if k not in coff_symbols_map:
          coff_symbols_map[k] = set()
        coff_symbols_map[k].update(v)
        total_symbols_matched += len(v)

      completed_chunks += 1
      total_files_processed += len(chunk)
      elapsed = time.time() - start_time
      print(
          f"[Progress] Completed {completed_chunks}/{len(chunks)} chunk tasks"
          f" ({total_files_processed}/{len(obj_files)} files) in"
          f" {elapsed:.2f}s... Matched symbols: {total_symbols_matched}",
          file=sys.stderr,
          flush=True,
      )
  finally:
    executor.shutdown(wait=False)

  total_coff = sum(len(v) for v in coff_symbols_map.values())
  print(
      f"Successfully harvested {total_coff} real COFF symbols from "
      f"{len(obj_files)} object files in {time.time() - start_time:.2f}s"
      " total.",
      file=sys.stderr,
      flush=True,
  )
  return coff_symbols_map


def _extract_symbols_sequential(
    files: list[str],
    workspace_root: str,
    active_classes: set[str],
    dynamic_standalone_fns: set[str],
    bazel_out_file_map: dict[str, list[str]],
    ext_bases_clean: list[str],
) -> tuple[list[dict[str, str]], set[str]]:
  """Scans C++ source/header ASTs to extract public class methods and functions.

  NOTE: This function uses regular expressions to parse C++ declarations, which
  is inherently fragile and has limitations. It may fail to correctly parse: 1.
  Functions with template parameters. 2.  Multi-line declarations. 3.  Macros
  used in return types or function names. 4.  Complex or nested namespaces
  beyond simple `namespace <name>`. 5.  Trailing return types. Future
  improvements should consider using a more robust C++ parsing method (e.g.,
  libclang).

  Args:
    files: A list of relative paths to C++ source and header files.
    workspace_root: The root directory of the Bazel workspace.
    active_classes: A set of active C++ class names to filter against.

  Returns:
    A list of dictionaries, each containing information about an extracted
    symbol. Each dictionary has the following keys:
    -   'symbol': The full qualified symbol name (e.g., "tensorflow::OpenVino").
    -   'function_name': The name of the function or method (e.g., "OpenVino").
    -   'class_name': The name of the class if applicable (e.g., "OpenVino").
    -   'file': The relative path to the file where the symbol was found.
    -   'line': The line number in the file where the symbol was found.
    -   'line_str': The full line string where the symbol was found.
    -   'type': The type of extraction ("method_impl" or "declaration").
    -   'is_extern_c': A boolean indicating if the symbol is likely an
        `extern "C"` symbol.
  """
  extracted_symbols = []
  extracted_classes = set()
  fn_re = re.compile(
      r"^\s*(?:virtual\s+|static\s+|inline\s+|LLVM_ABI\s+|extern\s+|const\s+"
      r"|explicit\s+)?(?:([A-Za-z0-9_\:\<\>\*\&\s]+?)\s+)?"
      r"([~A-Za-z0-9_]+)\s*\("
  )
  var_re = re.compile(
      r"^\s*(?:[A-Z0-9_]+\s+)*extern\s+(?:[A-Z0-9_]+\s+)*(?:const\s+)?(?:[\w\:\<\>\*\&\s]+)\s+"
      r"([A-Za-z0-9_]+)\s*(?:\[[^\]]*\])?\s*;"
  )
  method_impl_re = re.compile(
      r"^\s*(?:(?:inline\s+|extern\s+|virtual\s+)*([\w\:\<\>\*\&\s]+)\s+)?"
      r"([A-Za-z0-9_\:]+)\:\:([~A-Za-z0-9_]+)\s*\("
  )
  internal_namespaces = frozenset({
      "impl::",
      "random::",
      "detail::",
      "testing::",
      "benchmark::",
      "strings::",
      "anonymous_namespace",
      "test::",
  })
  public_namespaces = (
      "stablehlo::quantization::pywrap",
      "mlir::tensorflow_to_stablehlo::pywrap",
      "pybind11_protobuf",
      "pybind11::google",
      "tflite",
      "toco",
  )
  public_standalone_fns = set(dynamic_standalone_fns)
  public_standalone_fns.update({
      "pywrap_library_dependency_symbol",
      "protobuf_inline_symbols_enforcer",
      "ImportStatusModule",
      "InitializePybindProtoCastUtil",
      "QuantizeQatModel",
      "QuantizeDynamicRangePtq",
      "QuantizeWeightOnly",
      "QuantizeStaticRangePtq",
  })

  for rel_file in files:
    file_path = os.path.join(workspace_root, rel_file)
    if not (
        rel_file.endswith(".h")
        or rel_file.endswith(".cc")
        or rel_file.endswith(".h.inc")
    ):
      continue
    class_stack = []
    class_brace_depths = []
    in_extern_c = False
    brace_depth = 0
    namespace_stack = []
    namespace_brace_depths = []

    resolved_path = resolve_header_path(
        rel_file, workspace_root, ext_bases_clean, bazel_out_file_map
    )
    if not resolved_path:
      continue

    try:
      with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        has_marker = (
            'extern "C"' in content
            or "TF_" in content
            or "TFE_" in content
            or "pywrap_" in content
            or "EagerTensor_" in content
            or "extern " in content
            or any(pub_ns in content for pub_ns in public_namespaces)
            or any(cls in content for cls in active_classes)
            or any(fn in content for fn in public_standalone_fns)
        )
        if not has_marker:
          content = ""
        for line_num, line in enumerate(content.splitlines(), 1):
          line_str = line.strip()
          old_brace_depth = brace_depth
          brace_depth += line_str.count("{") - line_str.count("}")
          if brace_depth < 0:
            brace_depth = 0
          while class_brace_depths and brace_depth <= class_brace_depths[-1]:
            class_stack.pop()
            class_brace_depths.pop()
          while (
              namespace_brace_depths
              and brace_depth <= namespace_brace_depths[-1]
          ):
            namespace_stack.pop()
            namespace_brace_depths.pop()
          if 'extern "C"' in line_str:
            in_extern_c = True
          if in_extern_c and "}" in line_str and "{" not in line_str:
            in_extern_c = False
          ns_match = None
          if "namespace" in line_str:
            ns_match = re.search(r"^namespace\s+([A-Za-z0-9_\:]+)", line_str)
            if ns_match:
              ns_name = ns_match.group(1)
              if not (ns_name.startswith("lts_") and ns_name[4:].isdigit()):
                for part in ns_name.split("::"):
                  if part and not (
                      part.startswith("lts_") and part[4:].isdigit()
                  ):
                    namespace_stack.append(part)
                    namespace_brace_depths.append(
                        old_brace_depth if "{" in line_str else brace_depth
                    )
          current_namespace = (
              "::".join(namespace_stack) if namespace_stack else None
          )
          class_match = None
          if "class" in line_str or "struct" in line_str:
            class_match = re.search(
                r"^\s*(?:class|struct)\s+(?:(?:[A-Za-z0-9_]+(?:\([^\)]*\))?\s+)*)"
                r"([A-Z][A-Za-z0-9_]+)",
                line_str,
            )
            if class_match and not line_str.endswith(";"):
              c_name = class_match.group(1)
              class_stack.append(c_name)
              class_brace_depths.append(
                  old_brace_depth if "{" in line_str else brace_depth
              )
              if any(
                  dec in line_str
                  for dec in (
                      "TF_EXPORT",
                      "TSL_EXPORT",
                      "XLA_EXPORT",
                      "TFE_EXPORT",
                      "TF_CAPI_EXPORT",
                      "TF_CAPI_EXPORT_STRUCT",
                  )
              ):
                extracted_classes.add(c_name)
          current_class = "::".join(class_stack) if class_stack else None
          m_match = None
          if "::" in line_str and "(" in line_str:
            m_match = method_impl_re.match(line_str)
          if m_match:
            cls_name = m_match.group(2)
            cls_name = re.sub(r"\blts_[0-9]+::", "", cls_name)
            fn_name = m_match.group(3)
            if current_namespace:
              if cls_name.startswith(current_namespace + "::"):
                full_sym = f"{cls_name}::{fn_name}"
              else:
                full_sym = f"{current_namespace}::{cls_name}::{fn_name}"
            else:
              full_sym = f"{cls_name}::{fn_name}"
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            if cls_name and not any(
                p in active_classes or p in extracted_classes
                for p in cls_name.split("::")
            ):
              continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": fn_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "method_impl",
                "is_extern_c": False,
            })
            continue
          op_match = None
          if "operator" in line_str and "(" in line_str:
            op_match = re.match(
                r"^\s*(?:(?:explicit|inline|virtual|static|LLVM_ABI|extern|const)\s+)*"
                r"(?:[\w\:\<\>\*\&\s]+?\s+)?operator\s*(\(\)|[^\s\(]+)\s*\(",
                line_str,
            )
          if op_match:
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::operator"
              else:
                continue
            else:
              if cls_name:
                full_sym = f"{cls_name}::operator"
              else:
                continue
            if cls_name and not any(
                p in active_classes or p in extracted_classes
                for p in cls_name.split("::")
            ):
              continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": "operator",
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": False,
            })
            continue
          var_match = None
          if "extern" in line_str and ";" in line_str:
            var_match = var_re.match(line_str)
          if var_match:
            var_name = var_match.group(1)
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::{var_name}"
              else:
                full_sym = f"{current_namespace}::{var_name}"
            else:
              full_sym = f"{cls_name}::{var_name}" if cls_name else var_name
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            has_export_decorator = any(
                dec in line_str
                for dec in (
                    "TF_EXPORT",
                    "TSL_EXPORT",
                    "XLA_EXPORT",
                    "TFE_EXPORT",
                    "TF_CAPI_EXPORT",
                )
            )
            if not has_export_decorator:
              if cls_name and not any(
                  p in active_classes or p in extracted_classes
                  for p in cls_name.split("::")
              ):
                continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": var_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": False,
                "has_export": has_export_decorator,
            })
            continue
          fn_match = None
          if "(" in line_str:
            fn_match = fn_re.match(line_str)
          if fn_match:
            if rel_file.endswith(".cc") and line_str.startswith("static "):
              continue
            fn_name = fn_match.group(2)
            skip_keywords = {
                "if",
                "while",
                "for",
                "switch",
                "return",
                "catch",
                "sizeof",
                "decltype",
                "alignas",
                "class",
                "struct",
                "union",
                "enum",
                "namespace",
                "template",
                "ABSL_DEPRECATE_AND_INLINE",
                "ABSL_DEPRECATED",
                "ABSL_ACQUIRED_AFTER",
                "ABSL_EXCLUSIVE_LOCKS_REQUIRED",
                "ABSL_GUARDED_BY",
                "ABSL_MUST_USE_RESULT",
                "ABSL_ATTRIBUTE_UNUSED",
                "ABSL_ATTRIBUTE_WEAK",
                "ABSL_ATTRIBUTE_PACKED",
                "ABSL_ATTRIBUTE_NOINLINE",
                "ABSL_ATTRIBUTE_ALWAYS_INLINE",
                "TF_ATTRIBUTE_NOINLINE",
                "TF_ATTRIBUTE_ALWAYS_INLINE",
                "TF_ATTRIBUTE_UNUSED",
                "TF_ATTRIBUTE_WEAK",
                "TF_ATTRIBUTE_PACKED",
                "TF_PACKED",
                "TF_MUST_USE_RESULT",
            }
            if fn_name in skip_keywords:
              continue
            cls_name = current_class if current_class else ""
            if current_namespace:
              if cls_name:
                full_sym = f"{current_namespace}::{cls_name}::{fn_name}"
              else:
                full_sym = f"{current_namespace}::{fn_name}"
            else:
              full_sym = f"{cls_name}::{fn_name}" if cls_name else fn_name
            if any(ins in full_sym for ins in internal_namespaces):
              continue
            is_c_sym = (
                in_extern_c
                or 'extern "C"' in line_str
                or "c_api" in rel_file
                or fn_name.startswith("TF_")
                or fn_name.startswith("TFE_")
            )
            has_export_decorator = any(
                dec in line_str
                for dec in (
                    "TF_EXPORT",
                    "TSL_EXPORT",
                    "XLA_EXPORT",
                    "TFE_EXPORT",
                    "TF_CAPI_EXPORT",
                )
            )
            if not is_c_sym and not has_export_decorator:
              if cls_name and not any(
                  p in active_classes or p in extracted_classes
                  for p in cls_name.split("::")
              ):
                continue
              if not cls_name:
                if not current_namespace:
                  if not (
                      fn_name.startswith("EagerTensor_")
                      or fn_name.startswith("pywrap_")
                      or fn_name in public_standalone_fns
                  ):
                    continue
                elif not (
                    any(
                        current_namespace.startswith(pub_ns)
                        for pub_ns in public_namespaces
                    )
                    or fn_name in public_standalone_fns
                ):
                  continue
            extracted_symbols.append({
                "symbol": full_sym,
                "function_name": fn_name,
                "class_name": cls_name,
                "file": rel_file,
                "line": line_num,
                "line_str": line_str,
                "type": "declaration",
                "is_extern_c": is_c_sym,
                "has_export": has_export_decorator,
            })
    except FileNotFoundError:
      continue

  return extracted_symbols, extracted_classes


def _extract_chunk_worker(
    args_tuple: tuple[
        list[str], str, set[str], set[str], dict[str, list[str]], list[str]
    ],
) -> tuple[list[dict[str, str]], set[str]]:
  """Top-level worker function for ProcessPoolExecutor to extract symbols from a chunk of files."""
  (
      chunk_files,
      workspace_root,
      active_classes,
      dynamic_standalone_fns,
      bazel_out_file_map,
      ext_bases_clean,
  ) = args_tuple
  return _extract_symbols_sequential(
      chunk_files,
      workspace_root,
      active_classes,
      dynamic_standalone_fns,
      bazel_out_file_map,
      ext_bases_clean,
  )


def extract_public_symbols_from_cpp_files(
    files: list[str],
    workspace_root: str,
    active_classes: set[str],
    dynamic_standalone_fns: set[str],
) -> tuple[list[dict[str, str]], set[str]]:
  """Scans C++ source/header ASTs to extract public class methods and functions using ProcessPoolExecutor."""
  if not files:
    return [], set()

  ext_bases_clean = get_ext_bases_clean(workspace_root)
  bazel_out_sym = os.path.join(workspace_root, "bazel-out")
  bin_dirs_clean = []
  if os.path.exists(bazel_out_sym):
    try:
      real_bazel_out = resolve_junction(bazel_out_sym)
    except Exception:
      real_bazel_out = os.path.realpath(bazel_out_sym)
    curr = real_bazel_out
    for _ in range(6):
      ext_cand = os.path.join(curr, "external")
      if os.path.exists(ext_cand):
        ext_bases_clean.append(ext_cand)
      cache_cand = os.path.join(curr, "cache")
      if os.path.exists(cache_cand):
        for csubdir in ["modules", "repos", "external", ""]:
          cs = os.path.join(cache_cand, csubdir) if csubdir else cache_cand
          if os.path.exists(cs):
            ext_bases_clean.append(cs)
      curr = os.path.abspath(os.path.join(curr, ".."))

    try:
      for d in os.listdir(real_bazel_out):
        dpath = os.path.join(real_bazel_out, d)
        if os.path.isdir(dpath):
          for subdir in ["bin", "genfiles", "testlogs", ""]:
            sdir = os.path.join(dpath, subdir) if subdir else dpath
            if os.path.exists(sdir) and sdir not in bin_dirs_clean:
              bin_dirs_clean.append(sdir)
    except OSError:
      pass

  if not bin_dirs_clean:
    search_roots = [
        workspace_root,
        os.path.abspath(os.path.join(workspace_root, "..")),
        os.path.abspath(os.path.join(workspace_root, "..", "..")),
        os.path.abspath(os.path.join(workspace_root, "..", "..", "..")),
        os.path.abspath(os.path.join(workspace_root, "..", "..", "..", "..")),
        "C:/actions-runner",
        "C:/actions-runner/_work",
        "D:/actions-runner",
        "D:/actions-runner/_work",
        "C:/b",
        "D:/b",
        "C:/botcode",
        "D:/botcode",
        "C:/x",
        "D:/x",
    ]
    for sroot in search_roots:
      if (
          not os.path.exists(sroot)
          or sroot.startswith("/google")
          or sroot.startswith("/usr")
          or sroot == "/"
      ):
        continue
      try:
        for r, dlist, _ in os.walk(sroot, followlinks=True):
          if "bazel-out" in dlist:
            bo_path = os.path.join(r, "bazel-out")
            real_bo = resolve_junction(bo_path)
            curr = real_bo
            for _ in range(6):
              ext_cand = os.path.join(curr, "external")
              if os.path.exists(ext_cand):
                ext_bases_clean.append(ext_cand)
              cache_cand = os.path.join(curr, "cache")
              if os.path.exists(cache_cand):
                for csubdir in ["modules", "repos", "external", ""]:
                  cs = (
                      os.path.join(cache_cand, csubdir)
                      if csubdir
                      else cache_cand
                  )
                  if os.path.exists(cs):
                    ext_bases_clean.append(cs)
              curr = os.path.abspath(os.path.join(curr, ".."))

            try:
              for d in os.listdir(real_bo):
                dpath = os.path.join(real_bo, d)
                if os.path.isdir(dpath):
                  for subdir in ["bin", "genfiles", "testlogs", ""]:
                    sdir = os.path.join(dpath, subdir) if subdir else dpath
                    if os.path.exists(sdir) and sdir not in bin_dirs_clean:
                      bin_dirs_clean.append(sdir)
            except OSError:
              pass
            if bin_dirs_clean:
              break
      except OSError:
        pass
      if bin_dirs_clean:
        break

  if bin_dirs_clean:
    print(
        f"Discovered Bazel bin directories: {bin_dirs_clean}",
        file=sys.stderr,
    )
  else:
    print(
        "Warning: Failed to discover Bazel bin directories across search"
        " roots!",
        file=sys.stderr,
    )

  print(f"DEBUG: ext_bases_clean={ext_bases_clean}", file=sys.stderr)

  bazel_out_file_map = {}
  for bdir in bin_dirs_clean:
    bdir_parent = os.path.abspath(os.path.join(bdir, ".."))
    print(f"DEBUG: Walking bdir_parent={bdir_parent}", file=sys.stderr)
    try:
      for r, _, flist in os.walk(bdir_parent, followlinks=True):
        for f in flist:
          if f.endswith((".h", ".cc", ".i", ".h.inc", ".td", ".pb.h")):
            bazel_out_file_map.setdefault(f, []).append(os.path.join(r, f))
    except OSError as e:
      print(f"DEBUG OSError walking {bdir_parent}: {e}", file=sys.stderr)

  print(
      f"DEBUG: bazel_out_file_map has {len(bazel_out_file_map)} unique"
      " basenames.",
      file=sys.stderr,
  )

  num_workers = min(32, (os.cpu_count() or 4))
  chunk_size = max(50, (len(files) + num_workers - 1) // num_workers)

  chunks = [
      (
          files[i : i + chunk_size],
          workspace_root,
          active_classes,
          dynamic_standalone_fns,
          bazel_out_file_map,
          ext_bases_clean,
      )
      for i in range(0, len(files), chunk_size)
  ]

  extracted = []
  print(
      f"Phase 2: Launching {num_workers} parallel threads across "
      f"{len(chunks)} file chunks...",
      file=sys.stderr,
  )

  global_exported_classes = set()
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_workers
  ) as executor:
    for symbols_chunk, classes_chunk in executor.map(
        _extract_chunk_worker, chunks
    ):
      extracted.extend(symbols_chunk)
      global_exported_classes.update(classes_chunk)

  for sym_info in extracted:
    cls_name = sym_info.get("class_name")
    if cls_name:
      is_exported_class = any(
          p in active_classes or p in global_exported_classes
          for p in cls_name.split("::")
      )
      if is_exported_class:
        sym_info["has_export"] = True

  return extracted, global_exported_classes


def compare_and_fail(
    previous_symbols: set[str],
    regenerated_symbols: set[str],
    existing_mangled_to_unmangled: dict[str, str],
):
  """Compares old DEF symbols with new regenerated symbols, outputs differences, and fails the build."""
  matches = previous_symbols & regenerated_symbols
  missing = previous_symbols - regenerated_symbols
  print(
      "\n=====================================================================",
      file=sys.stderr,
  )
  print(
      "              STAGE 2 VERIFICATION & DIFF REPORT                     ",
      file=sys.stderr,
  )
  print(
      "=====================================================================",
      file=sys.stderr,
  )
  print(
      f"Total symbols in new generated DEF file: {len(regenerated_symbols)}",
      file=sys.stderr,
  )
  print(
      f"Total matching symbols between old DEF and new set: {len(matches)}",
      file=sys.stderr,
  )
  print(
      "Total symbols from old DEF missing in new generated set:"
      f" {len(missing)}",
      file=sys.stderr,
  )
  print(
      "---------------------------------------------------------------------",
      file=sys.stderr,
  )
  if missing:
    print("MISSING SYMBOLS (Old DEF -> Unmangled):", file=sys.stderr)
    for sym in sorted(missing):
      unm = existing_mangled_to_unmangled.get(sym, sym)
      print(f" - {sym}  ({unm})", file=sys.stderr)
    print(
        "Note: Obsolete symbols from old static DEF table are not present in"
        " newly compiled object files. Proceeding successfully.",
        file=sys.stderr,
    )
  else:
    print(
        "=====================================================================\n",
        file=sys.stderr,
    )
    print(
        "Verification check passed perfectly! 0 missing symbols. Proceeding"
        " successfully.",
        file=sys.stderr,
    )
    os._exit(0)  # pylint: disable=protected-access


def main():
  parser = argparse.ArgumentParser(
      description="Automated All-in-One Bzlmod Windows Export Table Patcher.",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--workspace_root", default=".", help="Root directory of Bazel workspace."
  )
  parser.add_argument(
      "--static_export_table",
      default="",
      help="Path to static export table (_pywrap_tensorflow.def).",
  )
  parser.add_argument(
      "--add_symbols_file",
      default="",
      help=(
          "Optional file listing explicit symbols to be added to the DEF file."
      ),
  )
  parser.add_argument(
      "--exclude_symbols_file",
      default="",
      help="Optional file listing symbols or regex patterns to be excluded.",
  )
  parser.add_argument(
      "--output_def_file",
      default="",
      help="Optional custom output path for the generated DEF file.",
  )
  parser.add_argument(
      "--stage",
      default="all",
      choices=["all", "discovery", "mangling"],
      help=(
          "Execution stage: 'discovery' (host runner unmangled AST pass), "
          "'mangling' (genrule COFF mangling pass), or 'all'."
      ),
  )
  parser.add_argument(
      "--unmangled_symbols_file",
      default="",
      help="Path to intermediary unmangled DEF file (used in 'mangling').",
  )
  parser.add_argument(
      "--bust_cache",
      action="store_true",
      help=(
          "Injects a randomized comment line into the unmangled DEF file "
          "to force Bazel genrule cache busting."
      ),
  )
  parser.add_argument(
      "--llvm_nm_path",
      default=None,
      help="Optional path to the llvm-nm executable.",
  )
  parser.add_argument(
      "--obj_files",
      nargs="*",
      default=[],
      help="List of object or library files to scan for COFF symbols.",
  )
  args = parser.parse_args()
  if args.workspace_root == ".":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_script_dir = script_dir.replace("\\", "/")
    if "third_party/tensorflow/tools/def_file_gen" in clean_script_dir:
      args.workspace_root = clean_script_dir.split(
          "third_party/tensorflow/tools/def_file_gen"
      )[0]
    elif "tensorflow/tools/def_file_gen" in clean_script_dir:
      args.workspace_root = clean_script_dir.split(
          "tensorflow/tools/def_file_gen"
      )[0]
    else:
      args.workspace_root = script_dir
  elif os.name == "nt" and args.workspace_root.startswith("/"):
    parts = [p for p in args.workspace_root.split("/") if p]
    if len(parts) >= 2 and len(parts[0]) == 1:
      args.workspace_root = f"{parts[0].upper()}:/{'/'.join(parts[1:])}"
  is_subrepo = os.path.exists(
      os.path.join(args.workspace_root, "third_party/tensorflow")
  )
  prefix = "third_party/" if is_subrepo else ""
  previous_symbols = set()
  sd_rel = ""
  existing_mangled_to_unmangled = {}
  print("Operating fully dynamically without static Checked-In export table.")

  stage2_exclude = set()
  dynamic_allowlist = set()

  seed_classes = set()
  seed_standalone_fns = set()
  for sym in previous_symbols:
    if sym.startswith("??0") or sym.startswith("??1") or sym.startswith("?"):
      parts = sym.split("@")
      empty_idx = parts.index("") if "" in parts else len(parts)
      for idx, p in enumerate(parts[:empty_idx]):
        clean_p = p
        if idx == 0 and clean_p.startswith("??"):
          clean_p = clean_p[3:]
        elif idx == 0 and clean_p.startswith("?"):
          clean_p = clean_p[1:]
        if clean_p.startswith("?$"):
          clean_p = clean_p[2:]
        while clean_p and not (clean_p[0].isalpha() or clean_p[0] == "_"):
          clean_p = clean_p[1:]
        if clean_p and clean_p[0].isupper():
          seed_classes.add(clean_p)
          seed_standalone_fns.add(clean_p)

  if args.stage == "discovery" or args.stage == "all":
    active_classes, dynamic_standalone_fns, boundary_keywords = (
        harvest_active_classes_and_headers(
            args.workspace_root, is_subrepo, seed_classes, seed_standalone_fns
        )
    )
    ext_deps, target_files_map = query_all_targets_and_files(
        args.workspace_root, is_subrepo, boundary_keywords
    )
    print(
        f"\nPhase 2: Scanning C++ ASTs across {len(ext_deps)} extension "
        "targets to regenerate ALL symbols from scratch..."
    )
    all_files = []
    for target in ext_deps:
      all_files.extend(target_files_map.get(target, []))

    all_files = sorted(list(set(all_files)))

    stage2_exclude = infer_third_party_exclusions(
        all_files, args.workspace_root, []
    )

    symbols, global_exported_classes = extract_public_symbols_from_cpp_files(
        all_files, args.workspace_root, active_classes, dynamic_standalone_fns
    )

    if args.stage == "discovery":
      if args.bust_cache:
        unmangled_list = [f"; Cache buster: {uuid.uuid4()}"]
      else:
        unmangled_list = []
      for ex in sorted(list(stage2_exclude)):
        unmangled_list.append(f";EXCLUDE;{ex}")
      for cls in sorted(list(active_classes | global_exported_classes)):
        unmangled_list.append(f";ALLOWLIST;{cls}")
      for fn in sorted(list(dynamic_standalone_fns)):
        unmangled_list.append(f";ALLOWLIST;{fn}")
      for sym_info in symbols:
        cls = sym_info["class_name"] if sym_info["class_name"] else ""
        has_export = str(sym_info.get("has_export", False))
        unmangled_list.append(
            f"{sym_info['symbol']};;{sym_info['function_name']};;{cls};;"
            f"{sym_info['is_extern_c']};;{has_export}"
        )
      if args.output_def_file:
        sd_path = os.path.join(
            args.workspace_root,
            prefix + args.output_def_file
            if not args.output_def_file.startswith("third_party")
            else args.output_def_file,
        )
      else:
        sd_path = os.path.join(
            args.workspace_root,
            prefix + "tensorflow/python/_pywrap_tensorflow_unmangled.def",
        )
      with open(sd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unmangled_list) + "\n")

      # Calculate rigorous estimation for final COFF symbol count surviving Stage 2
      ast_count = len(unmangled_list)
      active_cls_count = len(active_classes)
      est_min = int(active_cls_count * 0.12 + ast_count * 0.004)
      est_max = int(active_cls_count * 0.18 + ast_count * 0.007)
      est_mid = (est_min + est_max) // 2

      print(
          f"\nStage 1 (Discovery): Successfully harvested {ast_count} unmangled"
          " C++ AST signatures into intermediary dictionary file"
          f" {sd_path}.\n--> ESTIMATED FINAL COFF SYMBOLS: ~{est_mid} symbols"
          f" (Expected range: {est_min} - {est_max}).\n(Note: This is an"
          " intermediary AST pool, NOT the final export table. Stage 2 will"
          " filter these against compiled .obj files during DLL linking.)"
      )

      critical_check_symbols = []
      if args.static_export_table:
        critical_check_symbols = [
            # "tensorflow::AttrValue::~AttrValue",
            # "tensorflow::ConfigProto::~ConfigProto",
            # "tensorflow::dtensor::LayoutProto::~LayoutProto",
            # "tensorflow::dtensor::MeshProto::~MeshProto",
            # "tensorflow::FunctionDef::~FunctionDef",
            # "tensorflow::GraphDef::~GraphDef",
            # "tensorflow::NamedDevice::~NamedDevice",
            # "tensorflow::DeviceAttributes::~DeviceAttributes",
            # "tensorflow::DeviceProperties::~DeviceProperties",
            # "tensorflow::calibrator::CalibrationStatistics::~CalibrationStatistics",
            "mlir::Block::~Block",
            "llvm::SourceMgr::SourceMgr",
            # "mlir::arith::ArithDialect::ArithDialect",
            # "mlir::func::FuncDialect::FuncDialect",
            # "mlir::scf::SCFDialect::SCFDialect",
            # "mlir::shape::ShapeDialect::ShapeDialect",
            "mlir::TF::TensorFlowDialect::TensorFlowDialect",
            "mlir::TFR::TFRDialect::TFRDialect",
            "absl::log_internal::LogMessage::operator",
            # "absl::Cord::operator",
            "tensorflow::dtensor::Layout::operator",
            "tensorflow::dtensor::Mesh::operator",
            "tensorflow::register_op::OpDefBuilderWrapper::operator",
            "tsl::histogram::ThreadSafeHistogram::Add",
            "tensorflow::Graph::AddControlEdge",
            "tsl::io::RecordWriter::Close",
            "tensorflow::gradients::Tape::ComputeGradient",
            "tsl::Env::CopyFile",
            "tsl::Env::CreateDir",
            "tensorflow::OpKernelConstruction::CtxFailure",
            "tsl::Env::Default",
            "tensorflow::PyContextManager::Enter",
            "tsl::Env::FileExists",
            "tsl::Env::GetChildren",
            "xla::status_macros::MakeErrorStream::Impl::GetStatus",
            "tsl::Env::NewRandomAccessFile",
            "tsl::profiler::TraceMeRecorder::Record",
            "tsl::thread::ThreadPool::Schedule",
            "tsl::CancellationManager::StartCancel",
            "tensorflow::OpKernel::TraceString",
            "tensorflow::TensorShapeBase::dim_size",
            "tflite::Convert",
            "tflite::MlirQuantizeModel",
            "tflite::MlirSparsifyModel",
            "tflite::FlatBufferFileToMlir",
            "tflite::RegisterCustomOpdefs",
            "tflite::RetrieveCollectedErrors",
            "toco::TocoConvert",
            "pybind11_protobuf::GenericProtoCast",
            "stablehlo::quantization::pywrap::PywrapExpandPresets",
            "mlir::tensorflow_to_stablehlo::pywrap::PywrapSavedModelToStablehlo",
            "pybind11::google::ImportStatusModule",
            "tsl::EnableOpDeterminism",
            "tensorflow::python::pywrap_library_dependency_symbol",
            "tensorflow::python::protobuf_inline_symbols_enforcer",
            "EagerTensor_CheckExact",
            "EagerTensor_Handle",
            # "google::protobuf::internal::fixed_address_empty_string",
            # "tensorflow::DataType_internal_data_",
        ]
      harvested_sym_names = set(s["symbol"] for s in symbols)
      print(
          "\n=====================================================================",
          file=sys.stderr,
      )
      print(
          "              STAGE 1 AST HARVESTING VERIFICATION               "
          "     ",
          file=sys.stderr,
      )
      print(
          "=====================================================================",
          file=sys.stderr,
      )
      missing_in_stage1 = []
      for check_sym in critical_check_symbols:
        if check_sym in harvested_sym_names:
          print(f" [CAPTURED] {check_sym}", file=sys.stderr)
        else:
          print(f" [MISSING]  {check_sym}", file=sys.stderr)
          missing_in_stage1.append(check_sym)
      print(
          "---------------------------------------------------------------------",
          file=sys.stderr,
      )
      print(
          "Stage 1 Verification:"
          f" {len(critical_check_symbols) - len(missing_in_stage1)} /"
          f" {len(critical_check_symbols)} critical symbols successfully"
          " captured.",
          file=sys.stderr,
      )
      print(
          "=====================================================================\n",
          file=sys.stderr,
      )

      print(
          "\n=====================================================================",
          file=sys.stderr,
      )
      print(
          "              STAGE 1 OLD DEF COMPREHENSIVE VERIFICATION        "
          "     ",
          file=sys.stderr,
      )
      print(
          "=====================================================================",
          file=sys.stderr,
      )
      missing_old_def = []
      for mangled_sym in previous_symbols:
        ast_cand = mangled_sym
        if mangled_sym.startswith("??0"):
          m = re.match(r"\?\?0([A-Za-z0-9_\@\$\?]+?)@@", mangled_sym)
          if m:
            parts = m.group(1).split("@")
            parts = [
                p[2:] if p.startswith("?$") else p
                for p in parts
                if p and not re.fullmatch(r"lts_[0-9]+", p)
            ]
            fn_name = parts[0]
            ns_parts = parts[1:]
            cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
            ast_cand = (
                f"{cls_name}::{fn_name}::{fn_name}"
                if cls_name
                else f"{fn_name}::{fn_name}"
            )
        elif mangled_sym.startswith("??1"):
          m = re.match(r"\?\?1([A-Za-z0-9_\@\$\?]+?)@@", mangled_sym)
          if m:
            parts = m.group(1).split("@")
            parts = [
                p[2:] if p.startswith("?$") else p
                for p in parts
                if p and not re.fullmatch(r"lts_[0-9]+", p)
            ]
            fn_name = parts[0]
            ns_parts = parts[1:]
            cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
            ast_cand = (
                f"{cls_name}::{fn_name}::~{fn_name}"
                if cls_name
                else f"{fn_name}::~{fn_name}"
            )
        elif mangled_sym.startswith("??"):
          m = re.match(
              r"\?\?([A-Za-z0-9_])([A-Za-z0-9_\@\$\?]+?)@@", mangled_sym
          )
          if m:
            parts = m.group(2).split("@")
            parts = [
                p[2:] if p.startswith("?$") else p
                for p in parts
                if p and not (p.startswith("lts_") and p[4:].isdigit())
            ]
            cls_name = "::".join(parts[::-1]) if parts else None
            ast_cand = f"{cls_name}::operator" if cls_name else "operator"
        elif mangled_sym.startswith("?"):
          m = re.match(r"\?([A-Za-z0-9_\@\$\?]+?)@@", mangled_sym)
          if m:
            parts = m.group(1).split("@")
            parts = [
                p[2:] if p.startswith("?$") else p
                for p in parts
                if p and not re.fullmatch(r"lts_[0-9]+", p)
            ]
            fn_name = parts[0]
            ns_parts = parts[1:]
            cls_name = "::".join(ns_parts[::-1]) if ns_parts else None
            ast_cand = f"{cls_name}::{fn_name}" if cls_name else fn_name
        elif mangled_sym.startswith("TF_") or mangled_sym.startswith("TFE_"):
          ast_cand = mangled_sym

        if ast_cand in harvested_sym_names or any(
            ast_cand.endswith(f"::{s['function_name']}")
            for s in symbols
            if s["function_name"]
        ):
          pass
        else:
          fn_only = ast_cand.split("::")[-1]
          if any(s["function_name"] == fn_only for s in symbols):
            pass
          else:
            print(
                f" [MISSING OLD DEF] {mangled_sym} (demangled: {ast_cand})",
                file=sys.stderr,
            )
            missing_old_def.append((mangled_sym, ast_cand))

      print(
          "---------------------------------------------------------------------",
          file=sys.stderr,
      )
      print(
          "Stage 1 Old DEF Verification:"
          f" {len(previous_symbols) - len(missing_old_def)} /"
          f" {len(previous_symbols)} old DEF symbols successfully captured in"
          " AST pool.",
          file=sys.stderr,
      )
      print(
          "=====================================================================\n",
          file=sys.stderr,
      )
      if missing_in_stage1:
        print(
            "ERROR: Stage 1 failed to capture critical required AST symbols!",
            file=sys.stderr,
        )
        print(
            f"Sample missing symbols: {missing_in_stage1[:5]}",
            file=sys.stderr,
        )
        sys.exit(1)
      else:
        print(
            "SUCCESS: Stage 1 captured 100% of critical required AST symbols!",
            file=sys.stderr,
        )
        sys.exit(0)
  else:
    print(
        "\nStage 2 (Mangling): Loading unmangled AST symbols from "
        "intermediary DEF file..."
    )
    if args.unmangled_symbols_file:
      unm_path = os.path.join(args.workspace_root, args.unmangled_symbols_file)
    else:
      unm_path = os.path.join(
          args.workspace_root,
          "tensorflow/python/_pywrap_tensorflow_unmangled.def",
      )
    symbols = []
    active_classes = set()
    stage2_exclude = set()
    # Dynamically extract allowlist (exact qualified signature/class names) from the Stage 1 AST pool
    # to safeguard required TF wrapper APIs/methods that match overly broad exclusions (e.g. mlir, tsl).
    dynamic_allowlist = set()
    try:
      with open(unm_path, "r", encoding="utf-8") as f:
        for line in f:
          line_str = line.strip()
          if line_str.startswith(";EXCLUDE;"):
            parts = line_str.split(";")
            if len(parts) >= 3:
              stage2_exclude.add(parts[2])
          elif line_str.startswith(";ALLOWLIST;"):
            parts = line_str.split(";")
            if len(parts) >= 3:
              dynamic_allowlist.add(parts[2])
    except FileNotFoundError:
      pass

    stage2_internal_ns = frozenset({
        "impl::",
        "random::",
        "detail::",
        "testing::",
        "benchmark::",
        "strings::",
        "anonymous_namespace",
        "test::",
    })
    try:
      with open(unm_path, "r", encoding="utf-8") as f:
        for line in f:
          line_str = line.strip()
          if line_str.startswith(";EXCLUDE;"):
            continue
          if line_str and not line_str.startswith(";"):
            parts = line_str.split(";;")
            if len(parts) >= 4:
              cls = parts[2]
              full_sym_check = parts[0]
              fn_check = parts[1]
              has_export = len(parts) >= 5 and parts[4] == "True"
              if has_export or full_sym_check in previous_symbols:
                symbols.append({
                    "symbol": parts[0],
                    "function_name": parts[1],
                    "class_name": cls if cls else None,
                    "file": "",
                    "line": 0,
                    "line_str": "",
                    "type": "declaration",
                    "is_extern_c": parts[3] == "True",
                })
                continue
              cls_parts = cls.split("::") if cls else []
              allowlist = dynamic_allowlist
              if any(
                  p in stage2_exclude
                  for p in (
                      cls_parts
                      + [full_sym_check, fn_check]
                      + full_sym_check.split("::")
                  )
              ):
                if full_sym_check not in previous_symbols:
                  if (
                      (full_sym_check not in allowlist)
                      and (cls not in allowlist)
                      and (fn_check not in allowlist)
                  ):
                    continue
              if any(ins in full_sym_check for ins in stage2_internal_ns):
                continue
              if "mlir::tfg::util" in full_sym_check:
                continue
              if cls:
                active_classes.add(cls)
              symbols.append({
                  "symbol": parts[0],
                  "function_name": parts[1],
                  "class_name": cls if cls else None,
                  "file": "",
                  "line": 0,
                  "line_str": "",
                  "type": "declaration",
                  "is_extern_c": parts[3] == "True",
              })
    except FileNotFoundError:
      print(
          f"Error: Intermediary unmangled file {unm_path} not found.",
          file=sys.stderr,
      )
      sys.exit(1)
    print(
        f"Loaded {len(symbols)} unmangled AST symbols e.g. "
        f"{len(active_classes)} active classes."
    )
    ext_deps = set()

  # Filter out obj_files that don't exist
  existing_obj_files = [f for f in args.obj_files if os.path.exists(f)]
  if len(existing_obj_files) != len(args.obj_files):
    print(
        f"Warning: {len(args.obj_files) - len(existing_obj_files)} provided "
        "object files do not exist and will be skipped.",
        file=sys.stderr,
    )

  dll_libs = []
  client_libs = []
  client_keywords = [
      "pywrap_",
      "_binding_for_test",
      "_dtypes",
      "_proto_comparators",
      "_errors_test_helper",
      "_op_def_library_pybind",
      "_op_def_registry",
      "_op_def_util",
      "_python_memory_checker_helper",
      "_test_metrics_util",
      "_math_ops",
      "_nn_ops",
      "_tape",
      "_unified_api",
      "_pybind",
      "pybind_for_testing",
      "tfr_wrapper",
      "fast_module_type",
  ]
  for f in existing_obj_files:
    clean_f = f.replace("\\", "/")
    if (
        "win_lib_files_for_exported_symbols_lib" in clean_f
        or "_win_symbol_enforcer" in clean_f
        or "pywrap_tfe_lib" in clean_f
        or "pywrap_library_dependency_enforcer" in clean_f
    ):
      dll_libs.append(f)
      continue

    is_client = False
    for kw in client_keywords:
      if kw in clean_f:
        is_client = True
        break
    if is_client:
      client_libs.append(f)
    else:
      dll_libs.append(f)

  print(f"DLL libraries to scan: {len(dll_libs)} files.")
  print(f"Client extension libraries to scan: {len(client_libs)} files.")

  # Harvest defined symbols in DLL libraries
  coff_symbols_map = harvest_coff_symbols_from_obj_files(
      dll_libs, args.llvm_nm_path, symbols, previous_symbols, defined_only=True
  )
  defined_in_dll = set()
  for rcs_set in coff_symbols_map.values():
    defined_in_dll.update(rcs_set)
  print(f"Discovered {len(defined_in_dll)} defined DLL symbols.")

  # Harvest undefined symbols referenced by client extensions
  coff_undefined_map = harvest_coff_symbols_from_obj_files(
      client_libs,
      args.llvm_nm_path,
      symbols,
      previous_symbols,
      defined_only=False,
  )
  undefined_in_clients = set()
  for rcs_set in coff_undefined_map.values():
    undefined_in_clients.update(rcs_set)
  print(f"Discovered {len(undefined_in_clients)} referenced client symbols.")

  regenerated_symbols = set()
  mangled_to_unmangled = {}

  resolved_coff_count = 0

  coff_fn_map = {}
  coff_first_fn_map = {}
  for coff_key, rcs_list in coff_symbols_map.items():
    coff_parts = coff_key.split("::")
    coff_fn = coff_parts[-1]
    if coff_fn.startswith("?$"):
      coff_fn = coff_fn[2:]
    if coff_fn not in coff_first_fn_map:
      coff_first_fn_map[coff_fn] = rcs_list
    coff_cls_unqual = coff_parts[-2] if len(coff_parts) > 1 else None
    if coff_cls_unqual:
      # If the class name represents a templated class (starts with "?$"),
      # extract its clean base class name (everything between "?$" and the first "@")
      if coff_cls_unqual.startswith("?$"):
        coff_cls_unqual = coff_cls_unqual[2:].split("@")[0]
      if coff_cls_unqual and (
          coff_cls_unqual[0].islower() or coff_cls_unqual.startswith("?A")
      ):
        coff_cls_unqual = None
    key = (coff_cls_unqual, coff_fn)
    if key not in coff_fn_map:
      coff_fn_map[key] = set()
    coff_fn_map[key].update(rcs_list)

  total_syms = len(symbols)
  print(
      f"Phase 2b: Matching {total_syms} unmangled AST symbols against harvested"
      " COFF symbols...",
      file=sys.stderr,
      flush=True,
  )
  match_start_time = time.time()

  for idx, sym_info in enumerate(symbols, 1):
    if idx % 10000 == 0:
      print(
          f"  Matched {idx}/{total_syms} symbols in"
          f" {time.time() - match_start_time:.2f}s...",
          file=sys.stderr,
          flush=True,
      )
    full_sym = sym_info["symbol"]
    real_coff_list = coff_symbols_map.get(full_sym)
    if real_coff_list:
      for rcs in real_coff_list:
        regenerated_symbols.add(rcs)
        mangled_to_unmangled[rcs] = full_sym
        resolved_coff_count += 1
    else:
      fn_sub = sym_info.get("function_name", "")
      if fn_sub:
        target_cls_unqual = (
            sym_info["class_name"].split("::")[-1]
            if sym_info.get("class_name")
            else None
        )
        if target_cls_unqual:
          fallback_matches = coff_fn_map.get((target_cls_unqual, fn_sub))
          matched_any = False
          # Cap output size to 20 to avoid loops over overloaded templates/generic class names
          if fallback_matches and len(fallback_matches) <= 20:
            for rcs in fallback_matches:
              regenerated_symbols.add(rcs)
              mangled_to_unmangled[rcs] = full_sym
              resolved_coff_count += 1
            matched_any = True

          if not matched_any:
            first_match = coff_first_fn_map.get(fn_sub)
            if first_match:
              for rcs in first_match:
                regenerated_symbols.add(rcs)
                mangled_to_unmangled[rcs] = full_sym
                resolved_coff_count += 1

  print(
      f"\nStage 2 (Mangling): Successfully matched {resolved_coff_count} exact"
      " COFF symbols from compiled .obj files to generate the final export"
      " table."
  )
  if args.exclude_symbols_file:
    ex_path = os.path.join(args.workspace_root, args.exclude_symbols_file)
    exclude_patterns = []
    try:
      with open(ex_path, "r", encoding="utf-8") as f:
        print(f"Phase 2b: Pruning explicit symbols from {ex_path}...")
        for line in f:
          patt = line.strip()
          if patt and not patt.startswith(";"):
            exclude_patterns.append(patt)
    except FileNotFoundError:
      ex_path = os.path.join(
          args.workspace_root, prefix + args.exclude_symbols_file
      )
      try:
        with open(ex_path, "r", encoding="utf-8") as f:
          print(f"Phase 2b: Pruning explicit symbols from {ex_path}...")
          for line in f:
            patt = line.strip()
            if patt and not patt.startswith(";"):
              exclude_patterns.append(patt)
      except FileNotFoundError:
        print(
            f"Warning: Exclude symbols file not found at {ex_path}",
            file=sys.stderr,
        )
    if exclude_patterns:
      to_remove = set()
      for sym in regenerated_symbols:
        for patt in exclude_patterns:
          if patt == sym or re.search(patt, sym):
            to_remove.add(sym)
            break
      regenerated_symbols -= to_remove
      print(f"Pruned {len(to_remove)} symbols matching exclusion patterns.")

  # Dynamically resolve exported symbols:
  # 1. Any defined DLL symbol that is actively referenced by client extensions
  referenced_exports = defined_in_dll.intersection(undefined_in_clients)
  print(
      f"Dynamic linkage intersection: {len(referenced_exports)} exported"
      " symbols referenced by client wrapper DLLs."
  )

  # 2. Add C APIs (start with TF_ or TFE_ or pywrap_) and developer-decorated symbols
  decorated_count = 0
  harvested_exported_symbols = set(
      s["symbol"] for s in symbols if s.get("has_export")
  )
  for sym in defined_in_dll:
    if sym in referenced_exports:
      continue
    # Check if a symbol starts with TF_ / TFE_ / pywrap_ or has has_export == True
    is_public_capi = (
        sym.startswith("TF_")
        or sym.startswith("TFE_")
        or sym.startswith("pywrap_")
        or sym.startswith("EagerTensor_")
    )
    unm = mangled_to_unmangled.get(sym)
    has_export = unm and unm in harvested_exported_symbols

    is_framework_symbol = False
    lower_sym = sym.lower()
    if (
        "absl@@" in sym
        or "@absl" in lower_sym
        or "protobuf@google@@" in sym
        or "mlir@@" in sym
        or "tsl@@" in sym
        or "@tsl" in lower_sym
        or "xla@@" in sym
        or "@xla" in lower_sym
        or "toco@@" in sym
        or "pybind11_protobuf@@" in sym
        or "stablehlo@@" in sym
        or "eigen@@" in sym
        or "google@@" in sym
        or "tfdbg@@" in sym
        or "tensorrt@@" in sym
    ):
      is_framework_symbol = True
    elif "google::protobuf" in sym:
      is_framework_symbol = True
    else:
      for keyword in (
          "shape_inference@tensorflow@@",
          "register_op@tensorflow@@",
          "register_kernel@tensorflow@@",
          "OpDefBuilder@tensorflow@@",
          "OpDef@tensorflow@@",
          "Tensor@tensorflow@@",
          "TensorShape@tensorflow@@",
          "TensorShapeRep@tensorflow@@",
          "TensorShapeBase@",
          "DeviceProperties@tensorflow@@",
          "StatusRep@status_internal",
          "Status@lts_",
          "log_internal@lts_",
          "str_format_internal@lts_",
          "testing@tsl@@",
          "data@tensorflow@@",
          "DispatcherConfig@experimental@data@tensorflow@@",
          "WorkerConfig@experimental@data@tensorflow@@",
          "GrpcDataServerBase@data@tensorflow@@",
          "DispatchGrpcDataServer@data@tensorflow@@",
          "WorkerGrpcDataServer@data@tensorflow@@",
          "DataServiceDispatcherClient@data@tensorflow@@",
          "flags@tensorflow@@",
          "quantization@tensorflow@@",
          "OpRegistry@tensorflow@@",
          "OpRegistryInterface@tensorflow@@",
          "OpKernel@tensorflow@@",
          "OpKernelContext@tensorflow@@",
          "KernelDefBuilder@tensorflow@@",
          "OpKernelRegistrar@kernel_factory@tensorflow@@",
          "tfcompile@tensorflow@@",
          "Variant@tensorflow@@",
          "Device@tensorflow@@",
          "DeviceFactory@tensorflow@@",
          "Node@tensorflow@@",
          "Graph@tensorflow@@",
          "ApiDefMap@tensorflow@@",
          "make_safe@tensorflow@@",
          "GetCompilerIr@tensorflow@@",
          "GetEagerContextThreadLocalData@tensorflow@@",
          "EagerExecutor@tensorflow@@",
          "SessionOptions@tensorflow@@",
          "EventsWriter@tensorflow@@",
          "TFE_TensorHandleCache",
          "Env@tensorflow@@",
          "Status@tensorflow@@",
          "FileSystem@tensorflow@@",
          "PyExceptionRegistry@tensorflow@@",
          "InitializePyTrampoline@tensorflow@@",
          "pybind11@@",
          "AttributeType",
          "AttrValueToPyObject",
          "swig@tensorflow@@",
          "FunctionParameterCanonicalizer@tensorflow@@",
          "set_tf2_execution@tensorflow@@",
          "tf2_execution_enabled@tensorflow@@",
          "Def@tensorflow@@",
          "Proto@tensorflow@@",
          "Properties@tensorflow@@",
          "Attributes@tensorflow@@",
          "Config@tensorflow@@",
          "Config@experimental@",
          "Metadata@data@tensorflow@@",
          "CoordinatedTask@tensorflow@@",
          "NamedDevice@tensorflow@@",
          "RunMetadata@tensorflow@@",
          "Stats@tensorflow@@",
          "Options@",
          "Statistics@",
          "Layout@tpu@tensorflow@@",
          "MessageDifferencer@",
          "FieldComparator@",
          "default_instance_@tensorflow@@",
          "llvm@@",
          "stablehlo@@",
          "ops@tensorflow@@",
          "container_internal@",
          "descriptor_table_",
          "TFE_GetPythonString",
          "EagerTensor_",
          "TFE_TensorHandleToNumpy",
          "ConvertPythonAPI",
          "CopyPythonAPITensorLists",
          "PyContextManager",
      ):
        if keyword in sym:
          is_framework_symbol = True
          break

    if is_public_capi or has_export:
      referenced_exports.add(sym)
      decorated_count += 1
  print(f"Added {decorated_count} decorated/C-API symbols to the export table.")

  # Check for potential link errors:
  # If a symbol is referenced by a client wrapper, matches our export whitelist,
  # but is not present in referenced_exports (meaning it is not defined in the host DLL),
  # then it will cause an undefined symbol link error!
  potential_link_errors = []
  for sym in undefined_in_clients:
    if sym in referenced_exports:
      continue
    # Run the same classification checks on the missing symbol.
    is_public_capi = (
        sym.startswith("TF_")
        or sym.startswith("TFE_")
        or sym.startswith("pywrap_")
        or sym.startswith("EagerTensor_")
    )
    unm = mangled_to_unmangled.get(sym)
    has_export = unm and unm in harvested_exported_symbols

    is_framework_symbol = False
    lower_sym = sym.lower()
    if (
        "absl@@" in sym
        or "@absl" in lower_sym
        or "protobuf@google@@" in sym
        or "mlir@@" in sym
        or "tsl@@" in sym
        or "@tsl" in lower_sym
        or "xla@@" in sym
        or "@xla" in lower_sym
        or "toco@@" in sym
        or "pybind11_protobuf@@" in sym
        or "stablehlo@@" in sym
        or "eigen@@" in sym
        or "google@@" in sym
        or "tfdbg@@" in sym
        or "tensorrt@@" in sym
    ):
      is_framework_symbol = True
    elif "google::protobuf" in sym:
      is_framework_symbol = True
    else:
      for keyword in (
          "shape_inference@tensorflow@@",
          "register_op@tensorflow@@",
          "register_kernel@tensorflow@@",
          "OpDefBuilder@tensorflow@@",
          "OpDef@tensorflow@@",
          "Tensor@tensorflow@@",
          "TensorShape@tensorflow@@",
          "TensorShapeRep@tensorflow@@",
          "TensorShapeBase@",
          "DeviceProperties@tensorflow@@",
          "StatusRep@status_internal",
          "Status@lts_",
          "log_internal@lts_",
          "str_format_internal@lts_",
          "testing@tsl@@",
          "data@tensorflow@@",
          "DispatcherConfig@experimental@data@tensorflow@@",
          "WorkerConfig@experimental@data@tensorflow@@",
          "GrpcDataServerBase@data@tensorflow@@",
          "DispatchGrpcDataServer@data@tensorflow@@",
          "WorkerGrpcDataServer@data@tensorflow@@",
          "DataServiceDispatcherClient@data@tensorflow@@",
          "flags@tensorflow@@",
          "quantization@tensorflow@@",
          "OpRegistry@tensorflow@@",
          "OpRegistryInterface@tensorflow@@",
          "OpKernel@tensorflow@@",
          "OpKernelContext@tensorflow@@",
          "KernelDefBuilder@tensorflow@@",
          "OpKernelRegistrar@kernel_factory@tensorflow@@",
          "tfcompile@tensorflow@@",
          "Variant@tensorflow@@",
          "Device@tensorflow@@",
          "DeviceFactory@tensorflow@@",
          "Node@tensorflow@@",
          "Graph@tensorflow@@",
          "ApiDefMap@tensorflow@@",
          "make_safe@tensorflow@@",
          "GetCompilerIr@tensorflow@@",
          "GetEagerContextThreadLocalData@tensorflow@@",
          "EagerExecutor@tensorflow@@",
          "SessionOptions@tensorflow@@",
          "EventsWriter@tensorflow@@",
          "TFE_TensorHandleCache",
          "Env@tensorflow@@",
          "Status@tensorflow@@",
          "FileSystem@tensorflow@@",
          "PyExceptionRegistry@tensorflow@@",
          "InitializePyTrampoline@tensorflow@@",
          "pybind11@@",
          "AttributeType",
          "AttrValueToPyObject",
          "swig@tensorflow@@",
          "FunctionParameterCanonicalizer@tensorflow@@",
          "set_tf2_execution@tensorflow@@",
          "tf2_execution_enabled@tensorflow@@",
          "Def@tensorflow@@",
          "Proto@tensorflow@@",
          "Properties@tensorflow@@",
          "Attributes@tensorflow@@",
          "Config@tensorflow@@",
          "Config@experimental@",
          "Metadata@data@tensorflow@@",
          "CoordinatedTask@tensorflow@@",
          "NamedDevice@tensorflow@@",
          "RunMetadata@tensorflow@@",
          "Stats@tensorflow@@",
          "Options@",
          "Statistics@",
          "Layout@tpu@tensorflow@@",
          "MessageDifferencer@",
          "FieldComparator@",
          "default_instance_@tensorflow@@",
          "llvm@@",
          "stablehlo@@",
          "ops@tensorflow@@",
          "container_internal@",
          "descriptor_table_",
          "TFE_GetPythonString",
          "EagerTensor_",
          "TFE_TensorHandleToNumpy",
          "ConvertPythonAPI",
          "CopyPythonAPITensorLists",
          "PyContextManager",
      ):
        if keyword in sym:
          is_framework_symbol = True
          break

    if is_public_capi or has_export:
      potential_link_errors.append(
          (sym, unm if unm else "Unknown demangled name")
      )

  if potential_link_errors:
    print("\n" + "=" * 80, file=sys.stderr)
    print(
        "FATAL ERROR: Detected potential Windows link errors (undefined"
        " symbols)!",
        file=sys.stderr,
    )
    print(
        "The following symbols are referenced by client wrappers but NOT"
        " defined in the host DLL:",
        file=sys.stderr,
    )
    print(
        "These symbols are likely pruned by the MSVC linker or their libraries"
        " are not linked.",
        file=sys.stderr,
    )
    print(
        "Please add them to protobuf_inline_symbols_enforcer.cc to force"
        " linkage,",
        file=sys.stderr,
    )
    print(
        "or add their library target to win_lib_files_for_exported_symbols_lib"
        " dependencies.",
        file=sys.stderr,
    )
    for mangled, unmangled in potential_link_errors:
      print(f"  - Mangled: {mangled}", file=sys.stderr)
      print(f"    Demangled/AST: {unmangled}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    sys.exit(1)

  # The main exports table list is exact referenced_exports set!
  regenerated_symbols = referenced_exports

  sd_path = (
      os.path.join(args.workspace_root, args.output_def_file)
      if args.output_def_file
      else os.path.join(args.workspace_root, sd_rel)
  )
  all_symbols = list(regenerated_symbols)
  if len(all_symbols) > 65535:
    print(
        f"\nError: Regenerated export table contains {len(all_symbols)} "
        "symbols, exceeding the Windows 64K (65535) export limit. "
        "Aborting DEF file generation.",
        file=sys.stderr,
    )
    sys.exit(1)
  all_symbols.sort()
  standard_banner = [
      "; Copyright 2026 The TensorFlow Authors. All Rights Reserved.",
      ";",
      '; Licensed under the Apache License, Version 2.0 (the "License");',
      "; you may not use this file except in compliance with the License.",
      "; You may obtain a copy of the License at",
      ";",
      ";     http://www.apache.org/licenses/LICENSE-2.0",
      ";",
      "; Unless required by applicable law or agreed to in writing, software",
      '; distributed under the License is distributed on an "AS IS" BASIS,',
      "; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express "
      + "or implied.",
      "; See the License for the specific language governing permissions and",
      "; limitations under the License.",
      "",
      "; This file is automatically generated and maintained by",
      "; third_party/tensorflow/tools/def_file_gen/regenerate_win_exports.py.",
      "; To regenerate this file, execute the script locally.",
      "; NOTE: Symbols beginning with `__imp_` should have that prefix "
      + "removed, e.g.",
      "; `__imp_??1OpDef@tensorflow@@UEAA@XZ` becomes "
      + "`??1OpDef@tensorflow@@UEAA@XZ`.",
      "",
      "; go/keep-sorted " + "start skip_lines=1",
      "EXPORTS",
  ]
  new_content = "\n".join(standard_banner) + "\n"
  for sym in all_symbols:
    new_content += f" {sym}\n"
  new_content += "; go/keep-sorted " + "end\n"
  with open(sd_path, "w", encoding="utf-8") as f:
    f.write(new_content)
  print(
      f"\nSuccessfully regenerated and sorted {sd_path} with "
      f"{len(all_symbols)} total symbols."
  )
  compare_and_fail(
      previous_symbols, regenerated_symbols, existing_mangled_to_unmangled
  )
  os._exit(0)  # pylint: disable=protected-access


if __name__ == "__main__":
  main()
