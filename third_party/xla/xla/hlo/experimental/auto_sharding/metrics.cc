/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/metrics.h"

#include <cstdint>
#include <string>

#include "xla/tsl/lib/monitoring/counter.h"

namespace xla {
namespace metrics {
namespace {

auto* xla_auto_sharding_invocations = tsl::monitoring::Counter<0>::New(
    "/tensorflow/compiler/xla/hlo/xla_auto_sharding_invocations",
    "The number of XLA auto sharding invocations used to collect "
    "/tensorflow/compiler/xla/hlo/xla_compilation_time_in_auto_sharding_usecs");

auto* auto_sharding_compilation_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/compiler/xla/hlo/xla_compilation_time_in_auto_sharding_usecs",
    "The total time spent on compiling XLA graphs in auto sharding pass in in "
    "microseconds.");

}  // namespace

void RecordAutoShardingInvocations() {
  xla_auto_sharding_invocations->GetCell()->IncrementBy(1);
}

void RecordAutoShardingCompilationTime(const uint64_t time_usecs) {
  auto_sharding_compilation_time_usecs->GetCell()->IncrementBy(time_usecs);
}

}  // namespace metrics
}  // namespace xla
