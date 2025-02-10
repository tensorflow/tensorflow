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

#ifndef XLA_TESTS_EXHAUSTIVE_TEST_OP_H_
#define XLA_TESTS_EXHAUSTIVE_TEST_OP_H_

#include <cstddef>
#include <type_traits>

#include "xla/tests/exhaustive/exhaustive_op_test.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/exhaustive/platform.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace exhaustive_op_test {

// Declares a single exhaustive test operation.
//
// This class is intended to be subclassed by an actual operation implementation
// that configures EnqueueOp() and EvaluateOp() as necessary.
//
// The exhaustive test can be run using the Run() function defined here.
//
// Pure virtual functions:
// - EnqueueOp
// - EvaluateOp
template <PrimitiveType T, size_t N>
class TestOp {
 public:
  using Traits = ExhaustiveOpTestTraits<T, N>;
  using Test = std::conditional_t<
      N == 1, ExhaustiveUnaryTest<T>,
      std::conditional_t<N == 2, ExhaustiveBinaryTest<T>,
                         std::enable_if_t<N == 1 || N == 2, void>>>;

  explicit TestOp(Test* test) : test_(test) {}

  virtual ~TestOp() = default;

  virtual Traits::EnqueueOp EnqueueOp() const = 0;
  virtual Traits::EvaluateOp EvaluateOp() const = 0;

  // Establish a verification check that each EnqueueOp() value is within range.
  TestOp& OutputRangeCheck(Traits::OutputRangeCheck output_range_check) & {
    output_range_check_ = output_range_check;
    return *this;
  }
  TestOp&& OutputRangeCheck(Traits::OutputRangeCheck output_range_check) && {
    output_range_check_ = output_range_check;
    return std::move(*this);
  }

  // The following methods set ErrorSpecGen for associated platforms. There is a
  // precedence hierarchy to allow for easily setting fallbacks and overriding
  // for certain platforms.
  //
  // CPU Precedence:
  // CPU Make (x86, ARM, etc) Error -> CPU Error -> Error
  //
  // GPU Precedence:
  // GPU Model (P100, V100, etc) Error -> GPU Make (Nvidia) Error -> GPU Error
  // -> Error

  TestOp& Error(Traits::ErrorSpecGen error_spec_gen) & {
    error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& Error(Traits::ErrorSpecGen error_spec_gen) && {
    error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& CpuError(Traits::ErrorSpecGen error_spec_gen) & {
    cpu_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& CpuError(Traits::ErrorSpecGen error_spec_gen) && {
    cpu_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& CpuX86Error(Traits::ErrorSpecGen error_spec_gen) & {
    cpu_x86_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& CpuX86Error(Traits::ErrorSpecGen error_spec_gen) && {
    cpu_x86_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& CpuArmError(Traits::ErrorSpecGen error_spec_gen) & {
    cpu_arm_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& CpuArmError(Traits::ErrorSpecGen error_spec_gen) && {
    cpu_arm_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuError(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuError(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuNvidiaError(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_nv_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuNvidiaError(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_nv_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuP100Error(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_nv_p100_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuP100Error(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_nv_p100_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuV100Error(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_nv_v100_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuV100Error(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_nv_v100_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuA100Error(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_nv_a100_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuA100Error(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_nv_a100_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  TestOp& GpuH100Error(Traits::ErrorSpecGen error_spec_gen) & {
    gpu_nv_h100_error_spec_gen_ = error_spec_gen;
    return *this;
  }
  TestOp&& GpuH100Error(Traits::ErrorSpecGen error_spec_gen) && {
    gpu_nv_h100_error_spec_gen_ = std::move(error_spec_gen);
    return std::move(*this);
  }

  // Execute the TestCase as configured.
  //
  // Requires invoking on a TestCase&& to ensure the TestCase is not used
  // afterwards.
  void Run() && {
    typename Traits::ErrorSpecGen error_spec_gen;
    if (test_->Platform().IsCpu()) {
      switch (std::get<Platform::CpuValue>(test_->Platform().value())) {
        case Platform::CpuValue::X86_64: {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {cpu_x86_error_spec_gen_, cpu_error_spec_gen_, error_spec_gen_});
          break;
        }
        case Platform::CpuValue::AARCH64: {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {cpu_arm_error_spec_gen_, cpu_error_spec_gen_, error_spec_gen_});
          break;
        }
        default: {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {cpu_error_spec_gen_, error_spec_gen_});
          break;
        }
      }
    } else if (test_->Platform().IsGpu()) {
      if (test_->Platform().IsNvidiaGpu()) {
        if (test_->Platform().IsNvidiaP100()) {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {gpu_nv_p100_error_spec_gen_, gpu_nv_error_spec_gen_,
               gpu_error_spec_gen_, error_spec_gen_});
        } else if (test_->Platform().IsNvidiaV100()) {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {gpu_nv_v100_error_spec_gen_, gpu_nv_error_spec_gen_,
               gpu_error_spec_gen_, error_spec_gen_});
        } else if (test_->Platform().IsNvidiaA100()) {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {gpu_nv_a100_error_spec_gen_, gpu_nv_error_spec_gen_,
               gpu_error_spec_gen_, error_spec_gen_});
        } else if (test_->Platform().IsNvidiaH100()) {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {gpu_nv_h100_error_spec_gen_, gpu_nv_error_spec_gen_,
               gpu_error_spec_gen_, error_spec_gen_});
        } else {
          error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
              {gpu_nv_error_spec_gen_, gpu_error_spec_gen_, error_spec_gen_});
        }
      } else {
        error_spec_gen = PickFirstErrorSpecGenPresent<Traits>(
            {gpu_error_spec_gen_, error_spec_gen_});
      }
    } else {
      error_spec_gen = PickFirstErrorSpecGenPresent<Traits>({error_spec_gen_});
    }
    test_->Run(EnqueueOp(), EvaluateOp(), error_spec_gen, output_range_check_);
  }

 private:
  Test* test_ = nullptr;
  Traits::OutputRangeCheck output_range_check_ = nullptr;
  Traits::ErrorSpecGen error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen cpu_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen cpu_x86_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen cpu_arm_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_nv_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_nv_p100_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_nv_v100_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_nv_a100_error_spec_gen_ = nullptr;
  Traits::ErrorSpecGen gpu_nv_h100_error_spec_gen_ = nullptr;
};

template <PrimitiveType T>
using UnaryTestOp = TestOp<T, 1>;

template <PrimitiveType T>
using BinaryTestOp = TestOp<T, 2>;

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_TEST_OP_H_
