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

#include "xla/tests/exhaustive/platform.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/hlo_runner_interface.h"

namespace xla {
namespace exhaustive_op_test {
namespace {

Platform::Value GetPlatformValue(const HloRunnerInterface& runner) {
  if (runner.HasProperty(HloRunnerPropertyTag::kCpu)) {
// We process these copts in a library instead of the final exhaustive_xla_test
// target because we assume the final target will use the same target CPU arch
// as this target.
#ifdef __x86_64__
    return Platform::Value::kX86_64;
#endif
#ifdef __aarch64__
    return Platform::Value::kAarch64;
#endif
  } else if (runner.HasProperty(HloRunnerPropertyTag::kUsingGpuCuda)) {
    return Platform::Value::kCuda;
  } else if (runner.HasProperty(HloRunnerPropertyTag::kUsingGpuRocm)) {
    return Platform::Value::kRocm;
  }
  LOG(FATAL) << "Unhandled platform: " << runner.Name()
             << ". Please add support to " __FILE__ ".";
}
}  // namespace

Platform::Platform(const HloRunnerInterface& runner)
    : value_(GetPlatformValue(runner)) {}

}  // namespace exhaustive_op_test
}  // namespace xla
