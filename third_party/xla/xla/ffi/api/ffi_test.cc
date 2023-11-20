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

#include "xla/ffi/api/ffi.h"

#include <cstdint>
#include <vector>

#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla::ffi {

TEST(FfiTest, DataTypeEnumValue) {
  // Verify that xla::PrimitiveType and xla::ffi::DataType use the same
  // integer value for encoding data types.
  auto encoded = [](auto value) { return static_cast<uint8_t>(value); };

  EXPECT_EQ(encoded(PrimitiveType::PRED), encoded(DataType::PRED));

  EXPECT_EQ(encoded(PrimitiveType::S8), encoded(DataType::S8));
  EXPECT_EQ(encoded(PrimitiveType::S16), encoded(DataType::S16));
  EXPECT_EQ(encoded(PrimitiveType::S32), encoded(DataType::S32));
  EXPECT_EQ(encoded(PrimitiveType::S64), encoded(DataType::S64));

  EXPECT_EQ(encoded(PrimitiveType::U8), encoded(DataType::U8));
  EXPECT_EQ(encoded(PrimitiveType::U16), encoded(DataType::U16));
  EXPECT_EQ(encoded(PrimitiveType::U32), encoded(DataType::U32));
  EXPECT_EQ(encoded(PrimitiveType::U64), encoded(DataType::U64));

  EXPECT_EQ(encoded(PrimitiveType::F16), encoded(DataType::F16));
  EXPECT_EQ(encoded(PrimitiveType::F32), encoded(DataType::F32));
  EXPECT_EQ(encoded(PrimitiveType::F64), encoded(DataType::F64));

  EXPECT_EQ(encoded(PrimitiveType::BF16), encoded(DataType::BF16));
}

TEST(FfiTest, BufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](BufferBase buffer) {
    EXPECT_EQ(buffer.data, storage.data());
    EXPECT_EQ(buffer.dtype, DataType::F32);
    EXPECT_EQ(buffer.dimensions.size(), 2);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Arg<BufferBase>().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

}  // namespace xla::ffi
