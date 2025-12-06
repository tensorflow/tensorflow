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

#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/service/hlo_runner_interface.h"

namespace xla {
namespace exhaustive_op_test {

// Represents an enum class of all possible openXLA execution platforms along
// with helper functions to categorically handle them.
class Platform {
 public:
  enum class Value {
    kAarch64,
    kX86_64,
    kCuda,
    kRocm,
  };

  explicit Platform(const HloRunnerInterface& runner);

  bool IsCpu() const { return IsIntelCpu() || IsArmCpu(); }

  bool IsIntelCpu() const { return value_ == Value::kX86_64; }

  bool IsArmCpu() const { return value_ == Value::kAarch64; }

  bool IsGpu() const { return IsAmdGpu() || IsNvidiaGpu(); }

  bool IsAmdGpu() const { return value_ == Value::kRocm; }

  bool IsNvidiaGpu() const { return value_ == Value::kCuda; }

  bool IsNvidiaP100() const { return test::DeviceIs(test::kP100); }

  bool IsNvidiaV100() const { return test::DeviceIs(test::kV100); }

  bool IsNvidiaA100() const { return test::DeviceIs(test::kA100); }

  bool IsNvidiaH100() const { return test::DeviceIs(test::kH100); }

 private:
  const Value value_;
};

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_PLATFORM_H_
