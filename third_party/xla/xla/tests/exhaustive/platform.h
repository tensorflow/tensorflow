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

#ifndef XLA_TESTS_EXHAUSTIVE_PLATFORM_H_
#define XLA_TESTS_EXHAUSTIVE_PLATFORM_H_

#include <variant>

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"

namespace xla {
namespace exhaustive_op_test {

// Represents an enum class of all possible openXLA execution platforms along
// with helper functions to categorically handle them.
class Platform {
 public:
  enum class CpuValue {
    AARCH64,
    X86_64,
  };

  using Value = std::variant<CpuValue, stream_executor::CudaComputeCapability,
                             stream_executor::RocmComputeCapability>;

  explicit Platform(const stream_executor::Platform& platform);

  bool IsCpu() const { return std::holds_alternative<CpuValue>(value_); }

  bool IsGpu() const {
    return std::holds_alternative<stream_executor::CudaComputeCapability>(
               value_) ||
           std::holds_alternative<stream_executor::RocmComputeCapability>(
               value_);
  }

  bool IsNvidiaGpu() const {
    return std::holds_alternative<stream_executor::CudaComputeCapability>(
        value_);
  }

  bool IsNvidiaP100() const;

  bool IsNvidiaV100() const;

  bool IsNvidiaA100() const;

  bool IsNvidiaH100() const;

  bool IsAmdGpu() const {
    return std::holds_alternative<stream_executor::RocmComputeCapability>(
        value_);
  }

  const Value& value() const { return value_; }

 private:
  const Value value_;
};

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_PLATFORM_H_
