/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_STEP_ID_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_STEP_ID_H_

#include <atomic>
#include <cstdint>

#include "absl/strings/str_format.h"
#include "tensorflow/core/tfrt/kernels/stream_ops_util_constants.h"

namespace tensorflow {
namespace tfrt_stub {

// A base template for common utilities for a type safe id.
template <typename Derived>
struct SafeId {
  SafeId() : id(0) {}
  explicit constexpr SafeId(int64_t id) : id(id) {}

  using Base = SafeId;

  int64_t id;

  friend bool operator==(const Derived& x, const Derived& y) {
    return x.id == y.id;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Derived& x) {
    absl::Format(&sink, "%d", x.id);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Derived& x) {
    return H::combine(std::move(h), x.id);
  }
};

// A type-safe step id.
struct StepId : SafeId<StepId> {
  using Base::Base;

  bool valid() const { return id != 0; }
  static constexpr StepId GetInvalidStepId() { return StepId(0); }
};

// The initial value of the step id.
std::atomic<uint64_t>& GetGlobalInitialStepId();

// StepIdGenerator provides the utility to generate a monotonically increasing
// step id. And the number of bits can be configured at compile time. The step
// id is positive and the maximum value is 2^(kStepIdBitSize)-1.
class StepIdGenerator {
 public:
  StepIdGenerator() : next_id_(GetGlobalInitialStepId().load()) {}

  StepIdGenerator(const StepIdGenerator&) = delete;
  StepIdGenerator& operator=(const StepIdGenerator&) = delete;

  // Generates a positive step id that is within the bit-range specified by
  // `kStepIdBitSize`.
  StepId GetNextStepId() {
    uint64_t next_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    // Use kStepIdBitSize bits because we need to pack it with batch id if batch
    // function is used.
    static_assert(kStepIdBitSize <= 32);
    next_id = (next_id & ((1ull << kStepIdBitSize) - 1));

    if (next_id == 0) {
      return GetNextStepId();
    }

    return StepId(static_cast<int64_t>(next_id));
  }

 private:
  std::atomic<uint64_t> next_id_{0};
};

// Set up the initial step_id used by StepIdGenerator. This class is
// test-only.
class TEST_ScopedInitialStepId {
 public:
  explicit TEST_ScopedInitialStepId(uint64_t step_id);
  ~TEST_ScopedInitialStepId();

  TEST_ScopedInitialStepId(const TEST_ScopedInitialStepId&) = delete;
  TEST_ScopedInitialStepId& operator=(const TEST_ScopedInitialStepId&) = delete;

 private:
  uint64_t step_id_ = 0;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_STEP_ID_H_
