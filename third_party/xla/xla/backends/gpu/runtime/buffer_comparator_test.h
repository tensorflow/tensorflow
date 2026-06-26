/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_TEST_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_TEST_H_

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/primitive_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

constexpr double kDefaultTolerance = 0.1;

class BufferComparatorTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(std::string name_str,
                         xla::PlatformUtil::CanonicalPlatformName("gpu"));
    std::string name = absl::AsciiStrToUpper(name_str);
    ASSERT_OK_AND_ASSIGN(platform_,
                         se::PlatformManager::PlatformWithName(name));
    ASSERT_OK_AND_ASSIGN(stream_exec_, platform_->ExecutorForDevice(0));
  }

  template <typename ElementType>
  absl::StatusOr<bool> CompareEqualBuffersInternal(
      absl::Span<const ElementType> current,
      absl::Span<const ElementType> expected, double tolerance) {
    ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                     stream_exec_->CreateStream());

    se::DeviceAddressHandle current_buffer(
        stream_exec_, stream_exec_->AllocateArray<ElementType>(current.size()));
    se::DeviceAddressHandle expected_buffer(
        stream_exec_,
        stream_exec_->AllocateArray<ElementType>(expected.size()));

    RETURN_IF_ERROR(stream->Memcpy(current_buffer.address_ptr(), current.data(),
                                   current_buffer.address().size()));
    RETURN_IF_ERROR(stream->Memcpy(expected_buffer.address_ptr(),
                                   expected.data(),
                                   expected_buffer.address().size()));
    RETURN_IF_ERROR(stream->BlockHostUntilDone());

    BufferComparator comparator(
        ShapeUtil::MakeShape(
            primitive_util::NativeToPrimitiveType<ElementType>(),
            {static_cast<int64_t>(current.size())}),
        tolerance);
    return comparator.CompareEqual(stream.get(), current_buffer.address(),
                                   expected_buffer.address());
  }

  template <typename ElementType>
  bool CompareEqualBuffers(absl::Span<const ElementType> current,
                           absl::Span<const ElementType> expected,
                           double tolerance) {
    absl::StatusOr<bool> res_or =
        CompareEqualBuffersInternal(current, expected, tolerance);
    EXPECT_OK(res_or.status());
    return res_or.ok() && res_or.value();
  }

  // Take floats only for convenience. Still uses ElementType internally.
  template <typename ElementType>
  bool CompareEqualFloatBuffers(const std::vector<float>& lhs_float,
                                const std::vector<float>& rhs_float,
                                double tolerance = kDefaultTolerance) {
    std::vector<ElementType> lhs(lhs_float.begin(), lhs_float.end());
    std::vector<ElementType> rhs(rhs_float.begin(), rhs_float.end());
    return CompareEqualBuffers<ElementType>(lhs, rhs, tolerance);
  }

  template <typename ElementType>
  bool CompareEqualComplex(const std::vector<std::complex<ElementType>>& lhs,
                           const std::vector<std::complex<ElementType>>& rhs) {
    return CompareEqualBuffers<std::complex<ElementType>>(lhs, rhs,
                                                          kDefaultTolerance);
  }

  template <typename ElementType>
  absl::StatusOr<bool> CompareEqualScalarInternal(
      const ElementType& current, const ElementType& expected,
      double tolerance = kDefaultTolerance) {
    ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                     stream_exec_->CreateStream());
    se::DeviceAddressHandle current_buffer(
        stream_exec_, stream_exec_->AllocateScalar<ElementType>());
    se::DeviceAddressHandle expected_buffer(
        stream_exec_, stream_exec_->AllocateScalar<ElementType>());

    RETURN_IF_ERROR(stream->Memcpy(current_buffer.address_ptr(), &current,
                                   current_buffer.address().size()));
    RETURN_IF_ERROR(stream->Memcpy(expected_buffer.address_ptr(), &expected,
                                   expected_buffer.address().size()));
    RETURN_IF_ERROR(stream->BlockHostUntilDone());

    BufferComparator comparator(
        ShapeUtil::MakeShape(
            primitive_util::NativeToPrimitiveType<ElementType>(), {}),
        kDefaultTolerance);
    return comparator.CompareEqual(stream.get(), current_buffer.address(),
                                   expected_buffer.address());
  }

  template <typename ElementType>
  bool CompareEqualScalar(const ElementType& current,
                          const ElementType& expected,
                          double tolerance = kDefaultTolerance) {
    absl::StatusOr<bool> res_or =
        CompareEqualScalarInternal(current, expected, tolerance);
    EXPECT_OK(res_or.status());
    return res_or.ok() && res_or.value();
  }

  se::Platform* platform_;
  se::StreamExecutor* stream_exec_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_TEST_H_
