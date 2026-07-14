/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_TEST_MATCHERS_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_TEST_MATCHERS_H_

#include <gmock/gmock.h>

namespace xla::gpu {

MATCHER_P(NanCountIs, value, "nan_count") {
  return ExplainMatchResult(value, arg.result.nan_count, result_listener);
}

MATCHER_P(InfCountIs, value, "inf_count") {
  return ExplainMatchResult(value, arg.result.inf_count, result_listener);
}

MATCHER_P(ZeroCountIs, value, "zero_count") {
  return ExplainMatchResult(value, arg.result.zero_count, result_listener);
}

MATCHER_P(MinValueIs, value, "min_value") {
  return ExplainMatchResult(static_cast<double>(value), arg.result.min_value,
                            result_listener);
}

MATCHER_P(MaxValueIs, value, "max_value") {
  return ExplainMatchResult(static_cast<double>(value), arg.result.max_value,
                            result_listener);
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_TEST_MATCHERS_H_
