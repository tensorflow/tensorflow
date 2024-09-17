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

#include "xla/service/gpu/stream_executor_util.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/autotuning.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/util/proto/proto_utils.h"

namespace xla::gpu {
namespace {

struct Result {
  int64_t run_time_ns;
  int64_t scratch_bytes;

  bool operator==(const Result& other) const {
    return other.run_time_ns == run_time_ns &&
           other.scratch_bytes == scratch_bytes;
  };

  explicit operator AutotuneResult() const {
    AutotuneResult result;
    *result.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(absl::Nanoseconds(run_time_ns));
    result.set_scratch_bytes(scratch_bytes);
    return result;
  }
};

static Result ATRToResult(AutotuneResult atr) {
  return Result{.run_time_ns = absl::ToInt64Nanoseconds(
                    tsl::proto_utils::FromDurationProto(atr.run_time())),
                .scratch_bytes = atr.scratch_bytes()};
}

std::vector<AutotuneResult> Results(const std::vector<Result>& stats) {
  std::vector<AutotuneResult> results;
  for (const auto& s : stats) results.push_back(AutotuneResult(s));
  return results;
}

TEST(StreamExecutorTest, PickBestResult) {
  absl::StatusOr<AutotuneResult> atr;

  atr = PickBestResult(Results({{9000, 0}, {1000, 0}, {16000, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({1000, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 0}, {4500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4500, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 2}, {4500, 1}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4700, 0}));

  atr = PickBestResult(Results({{5000, 1}, {6000, 0}, {7500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({6000, 0}));
}

}  // namespace

}  // namespace xla::gpu
