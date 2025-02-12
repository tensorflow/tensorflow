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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_METRICS_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_METRICS_H_

#include <cstdint>

namespace xla {
namespace metrics {

void RecordAutoShardingInvocations();

void RecordAutoShardingCompilationTime(uint64_t time_usecs);

}  // namespace metrics
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_METRICS_H_
