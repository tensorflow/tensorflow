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

#include "xla/pjrt/gpu/vmm_allocator_gate.h"

#include <atomic>

#include "tsl/platform/platform.h"

namespace xla {
namespace {

// Number of live ScopedAllowVmmAllocatorForTesting instances.
std::atomic<int> allow_for_testing_count{0};

}  // namespace

bool IsVmmAllocatorAllowed() {
  return tsl::kIsOpenSource ||
         allow_for_testing_count.load(std::memory_order_acquire) > 0;
}

ScopedAllowVmmAllocatorForTesting::ScopedAllowVmmAllocatorForTesting() {
  allow_for_testing_count.fetch_add(1, std::memory_order_acq_rel);
}

ScopedAllowVmmAllocatorForTesting::~ScopedAllowVmmAllocatorForTesting() {
  allow_for_testing_count.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace xla
