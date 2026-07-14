/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/encoded_buffer_allocation_info.h"

#include <gtest/gtest.h>
#include "xla/backends/cpu/buffer_allocation_info.h"

namespace xla::cpu {
namespace {

TEST(EncodedBufferAllocationInfoTest, RoundTrip) {
  auto round_trip = [](const BufferAllocationInfo& buffer_info) {
    EncodedBufferAllocationInfo encoded(buffer_info);
    BufferAllocationInfo round_trip(encoded);
    ASSERT_EQ(round_trip, buffer_info);
  };

  round_trip(BufferAllocationInfo::Temp(0));
  round_trip(BufferAllocationInfo::Temp(4));
  round_trip(BufferAllocationInfo::ThreadLocal(0));
  round_trip(BufferAllocationInfo::ThreadLocal(4));
  round_trip(BufferAllocationInfo::Constant(0));
  round_trip(BufferAllocationInfo::Constant(4));
  round_trip(BufferAllocationInfo::EntryParameter(0, 4));
  round_trip(BufferAllocationInfo::EntryParameter(4, 0));
}

}  // namespace
}  // namespace xla::cpu
