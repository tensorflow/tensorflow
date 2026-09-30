#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# Bazel --run_under wrapper for ASAN-instrumented C++ and Python tests.
#
# Resolves libclang_rt.asan.so and llvm-symbolizer from the test's runfiles tree
# so that both local executions and remote RBE test workers have access to the
# hermetic LLVM ASAN shared runtime and symbolizer.

set -euo pipefail

asan_rt=""
symbolizer=""
shopt -s nullglob
for candidate_dir in "${RUNFILES_DIR:-}" "${TEST_SRCDIR:-}" "$0.runfiles"; do
  if [[ -n "${candidate_dir}" && -d "${candidate_dir}" ]]; then
    asan_candidates=("${candidate_dir}"/*llvm*/lib/clang/*/lib/*linux*/libclang_rt.asan.so)
    if [[ ${#asan_candidates[@]} -gt 0 ]]; then
      asan_rt="${asan_candidates[0]}"
      sym_candidates=("${candidate_dir}"/*llvm*/bin/llvm-symbolizer)
      if [[ ${#sym_candidates[@]} -gt 0 ]]; then
        symbolizer="${sym_candidates[0]}"
      fi
      break
    fi
  fi
done
shopt -u nullglob

if [[ -n "${asan_rt}" ]]; then
  asan_rt="$(readlink -f "${asan_rt}")"
  asan_rt_dir="$(dirname "${asan_rt}")"
  export LD_PRELOAD="${asan_rt}${LD_PRELOAD:+:${LD_PRELOAD}}"
  export LD_LIBRARY_PATH="${asan_rt_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "Warning: libclang_rt.asan.so not found in runfiles (RUNFILES_DIR=${RUNFILES_DIR:-}, TEST_SRCDIR=${TEST_SRCDIR:-})" >&2
fi

asan_opts="detect_leaks=0:allocator_may_return_null=1:verify_asan_link_order=0:color=always"
if [[ -n "${symbolizer}" ]]; then
  symbolizer="$(readlink -f "${symbolizer}")"
  asan_opts="${asan_opts}:external_symbolizer_path=${symbolizer}"
fi
export ASAN_OPTIONS="${ASAN_OPTIONS:+${ASAN_OPTIONS}:}${asan_opts}"

exec "$@"
