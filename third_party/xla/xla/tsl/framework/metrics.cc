/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/metrics.h"

#include <cstdint>

#include "xla/tsl/lib/monitoring/counter.h"

namespace tsl {
namespace metrics {
namespace {

auto* bfc_allocator_delay =
    monitoring::Counter<0>::New("/tensorflow/core/bfc_allocator_delay",
                                "The total time spent running each graph "
                                "optimization pass in microseconds.");

}  // namespace

void UpdateBfcAllocatorDelayTime(const uint64_t delay_usecs) {
  static auto* bfc_allocator_delay_cell = bfc_allocator_delay->GetCell();
  if (delay_usecs > 0) {
    bfc_allocator_delay_cell->IncrementBy(delay_usecs);
  }
}

}  // namespace metrics
}  // namespace tsl
