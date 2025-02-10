/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/cpu/metrics.h"

#include <deque>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/monitoring/counter.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace cpu {

namespace {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

}  // namespace

auto* cpu_compiler_stacktrace_count = tsl::monitoring::Counter<1>::New(
    "/xla/service/cpu/compiler_stacktrace_count",
    "The number of times a compiler stacktrace was called.", "stacktrace");

void RecordCpuCompilerStacktrace() {
  TraceMe trace(
      [&] { return TraceMeEncode("RecordCpuCompilerStacktrace", {}); });
  std::string tsl_stacktrace = tsl::CurrentStackTrace();

  // tsl::CurrentStackTrace() adds a prefix and postfix lines, so remove them.
  std::deque<std::string> stack = absl::StrSplit(tsl_stacktrace, '\n');
  stack.pop_front();
  stack.pop_back();

  const int kMaxStackDepth = 10;
  if (stack.size() > kMaxStackDepth) {
    stack.resize(kMaxStackDepth);
  }

  // Stack traces with addresses would make too many unique streamz cells.
  // We only care about the actual call stack.
  // Format chars added by tsl::CurrentStackTrace().
  constexpr unsigned kFormatChars = 8;
  constexpr unsigned kAddressFormat = kFormatChars + 2 * sizeof(void*);
  for (int i = 0; i < stack.size(); ++i) {
    stack[i] = std::string(absl::StripAsciiWhitespace(
        absl::ClippedSubstr(stack[i], kAddressFormat)));
  }

  std::string stacktrace = absl::StrJoin(stack, ";\n");
  cpu_compiler_stacktrace_count->GetCell(stacktrace)->IncrementBy(1);
}

int GetCpuCompilerStacktraceCount(absl::string_view stacktrace) {
  return cpu_compiler_stacktrace_count->GetCell(std::string(stacktrace))
      ->value();
}

}  // namespace cpu
}  // namespace xla
