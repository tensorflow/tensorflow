/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/byte_size.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::Eq;
using ::testing::Not;

TEST(ByteSizeTest, Constructors) {
  EXPECT_EQ(ByteSize::Bytes(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::Bytes(1), ByteSize::Bytes(1));
  EXPECT_EQ(ByteSize::Bytes(1024), ByteSize::Bytes(1024));
  EXPECT_EQ(ByteSize::Bytes(1024), ByteSize::KB(1));
  EXPECT_EQ(ByteSize::Bytes(size_t{1} << 63), ByteSize::TB(size_t{1} << 23));

  EXPECT_EQ(ByteSize::KB(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::KB(1), ByteSize::Bytes(size_t{1} << 10));
  EXPECT_EQ(ByteSize::KB(0.9), ByteSize::Bytes(1024 * 0.9));
  EXPECT_EQ(ByteSize::KB(1.5), ByteSize::Bytes(1024 * 1.5));
  EXPECT_EQ(ByteSize::KB(1.5), ByteSize::KB(1.5));
  EXPECT_EQ(ByteSize::KB(1024), ByteSize::MB(1));

  EXPECT_EQ(ByteSize::MB(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::MB(1), ByteSize::Bytes(size_t{1} << 20));
  EXPECT_EQ(ByteSize::MB(0.9), ByteSize::Bytes(size_t{1} << 20) * 0.9);
  EXPECT_EQ(ByteSize::MB(1.5), ByteSize::Bytes(size_t{1} << 20) * 1.5);
  EXPECT_EQ(ByteSize::MB(1.5), ByteSize::MB(1.5));
  EXPECT_EQ(ByteSize::MB(1024), ByteSize::GB(1));

  EXPECT_EQ(ByteSize::GB(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::GB(1), ByteSize::Bytes(size_t{1} << 30));
  EXPECT_EQ(ByteSize::GB(0.9), ByteSize::Bytes(size_t{1} << 30) * 0.9);
  EXPECT_EQ(ByteSize::GB(1.5), ByteSize::Bytes(size_t{1} << 30) * 1.5);
  EXPECT_EQ(ByteSize::GB(1.5), ByteSize::GB(1.5));
  EXPECT_EQ(ByteSize::GB(1024), ByteSize::TB(1));

  EXPECT_EQ(ByteSize::TB(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::TB(1), ByteSize::Bytes(size_t{1} << 40));
  EXPECT_EQ(ByteSize::TB(0.9), ByteSize::Bytes(size_t{1} << 40) * 0.9);
  EXPECT_EQ(ByteSize::TB(1.5), ByteSize::Bytes(size_t{1} << 40) * 1.5);
  EXPECT_EQ(ByteSize::TB(1.5), ByteSize::TB(1.5));
  EXPECT_EQ(ByteSize::TB(1024), ByteSize::TB(1024));
  EXPECT_EQ(ByteSize::TB(size_t{1} << 23), ByteSize::TB(size_t{1} << 23));

  EXPECT_THAT(ByteSize::Bytes(0), Not(Eq(ByteSize::Bytes(1))));
  EXPECT_THAT(ByteSize::Bytes(1025), Not(Eq(ByteSize::KB(1))));
  EXPECT_THAT(ByteSize::KB(1), Not(Eq(ByteSize::MB(1))));
  EXPECT_THAT(ByteSize::MB(1), Not(Eq(ByteSize::GB(1))));
  EXPECT_THAT(ByteSize::GB(1), Not(Eq(ByteSize::TB(1))));
  EXPECT_THAT(ByteSize::TB(1), Not(Eq(ByteSize::TB(2))));
}

TEST(ByteSizeTest, ConstexprConstruction) {
  constexpr ByteSize default_byte_size;
  EXPECT_EQ(default_byte_size, ByteSize::Bytes(0));

  constexpr ByteSize bytes = ByteSize::Bytes(1);
  EXPECT_EQ(bytes, ByteSize::Bytes(1));

  constexpr ByteSize kb = ByteSize::KB(1);
  EXPECT_EQ(kb, ByteSize::KB(1));

  constexpr ByteSize mb = ByteSize::MB(1);
  EXPECT_EQ(mb, ByteSize::MB(1));

  constexpr ByteSize gb = ByteSize::GB(1);
  EXPECT_EQ(gb, ByteSize::GB(1));

  constexpr ByteSize tb = ByteSize::TB(1);
  EXPECT_EQ(tb, ByteSize::TB(1));

  constexpr ByteSize tb_copy(tb);
  EXPECT_EQ(tb_copy, tb);
}

TEST(ByteSizeTest, ConvertToBytes) {
  EXPECT_EQ(ByteSize::Bytes(0).ToUnsignedBytes(), 0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0).ToDoubleBytes(), 0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0).ToDoubleKB(), 0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0).ToDoubleMB(), 0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0).ToDoubleGB(), 0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0).ToDoubleTB(), 0);

  EXPECT_EQ(ByteSize::Bytes(1).ToUnsignedBytes(), 1);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1).ToDoubleBytes(), 1.0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1).ToDoubleKB(), 1.0 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1).ToDoubleMB(), 1.0 / 1024 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1).ToDoubleGB(), 1.0 / 1024 / 1024 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1).ToDoubleTB(),
                   1.0 / 1024 / 1024 / 1024 / 1024);

  EXPECT_EQ(ByteSize::KB(0.25).ToUnsignedBytes(), 0.25 * (size_t{1} << 10));
  EXPECT_DOUBLE_EQ(ByteSize::KB(0.25).ToDoubleBytes(), 0.25 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::KB(0.25).ToDoubleKB(), 0.25);
  EXPECT_DOUBLE_EQ(ByteSize::KB(0.25).ToDoubleMB(), 0.25 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::KB(0.25).ToDoubleGB(), 0.25 / 1024 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::KB(0.25).ToDoubleTB(), 0.25 / 1024 / 1024 / 1024);

  EXPECT_EQ(ByteSize::MB(0.5).ToUnsignedBytes(), 0.5 * (size_t{1} << 20));
  EXPECT_DOUBLE_EQ(ByteSize::MB(0.5).ToDoubleBytes(), 0.5 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::MB(0.5).ToDoubleKB(), 0.5 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::MB(0.5).ToDoubleMB(), 0.5);
  EXPECT_DOUBLE_EQ(ByteSize::MB(0.5).ToDoubleGB(), 0.5 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::MB(0.5).ToDoubleTB(), 0.5 / 1024 / 1024);

  EXPECT_EQ(ByteSize::GB(10).ToUnsignedBytes(), 10.0 * (size_t{1} << 30));
  EXPECT_DOUBLE_EQ(ByteSize::GB(10).ToDoubleBytes(), 10.0 * 1024 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::GB(10).ToDoubleKB(), 10.0 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::GB(10).ToDoubleMB(), 10.0 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::GB(10).ToDoubleGB(), 10.0);
  EXPECT_DOUBLE_EQ(ByteSize::GB(10).ToDoubleTB(), 10.0 / 1024);

  EXPECT_EQ(ByteSize::TB(1024).ToUnsignedBytes(), 1024 * (size_t{1} << 40));
  EXPECT_DOUBLE_EQ(ByteSize::TB(1024).ToDoubleBytes(),
                   1024.0 * 1024 * 1024 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::TB(1024).ToDoubleKB(),
                   1024.0 * 1024 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::TB(1024).ToDoubleMB(), 1024.0 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::TB(1024).ToDoubleGB(), 1024.0 * 1024);
  EXPECT_DOUBLE_EQ(ByteSize::TB(1024).ToDoubleTB(), 1024.0);
}

TEST(ByteSizeTest, Arithmetics) {
  // Add.
  EXPECT_EQ(ByteSize::Bytes(0) + ByteSize::Bytes(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::Bytes(0) + ByteSize::Bytes(1), ByteSize::Bytes(1));
  EXPECT_EQ(ByteSize::Bytes(512) + ByteSize::Bytes(512), ByteSize::KB(1));
  EXPECT_EQ(ByteSize::Bytes(512) + ByteSize::KB(1), ByteSize::KB(1.5));
  EXPECT_EQ(ByteSize::KB(0.5) + ByteSize::KB(1), ByteSize::KB(1.5));
  EXPECT_EQ(ByteSize::MB(1) + ByteSize::KB(512), ByteSize::MB(1.5));
  EXPECT_EQ(ByteSize::MB(1) + ByteSize::Bytes(512), ByteSize::Bytes(1049088));
  EXPECT_EQ(ByteSize::GB(0.5) + ByteSize::MB(256) + ByteSize::MB(256),
            ByteSize::GB(1));
  std::vector<ByteSize> GBs(1024, ByteSize::GB(1));
  EXPECT_EQ(absl::c_accumulate(GBs, ByteSize::Bytes(0)), ByteSize::TB(1));
  EXPECT_EQ(ByteSize::TB(1) + ByteSize::TB(0.5) + ByteSize::GB(512),
            ByteSize::TB(2));

  // Substract.
  EXPECT_EQ(ByteSize::Bytes(0) - ByteSize::Bytes(0), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::KB(1) - ByteSize::Bytes(512), ByteSize::KB(0.5));
  EXPECT_EQ(ByteSize::MB(1) - ByteSize::KB(512) - ByteSize::KB(512),
            ByteSize::MB(0));
  EXPECT_EQ(ByteSize::GB(1) - ByteSize::MB(512), ByteSize::GB(0.5));
  EXPECT_EQ(ByteSize::GB(0.5) - ByteSize::MB(512), ByteSize::GB(0));
  EXPECT_EQ(ByteSize::GB(1) - ByteSize::MB(512) - ByteSize::MB(512),
            ByteSize::GB(0));
  EXPECT_EQ(ByteSize::TB(1) - ByteSize::GB(512) - ByteSize::GB(512),
            ByteSize::GB(0));

  // No negative bytes.
  EXPECT_EQ(ByteSize::Bytes(0) - ByteSize::Bytes(1), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::Bytes(0) - ByteSize::GB(1), ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::MB(1) - ByteSize::GB(1), ByteSize::Bytes(0));

  // Multiply.
  EXPECT_EQ(ByteSize::Bytes(0) * 0, ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::KB(1) * 0, ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::MB(1) * 0, ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::GB(1) * 0, ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::TB(1) * 0, ByteSize::Bytes(0));

  EXPECT_EQ(ByteSize::Bytes(1) * 1024, ByteSize::KB(1));
  EXPECT_EQ(ByteSize::KB(1) * 1024, ByteSize::MB(1));
  EXPECT_EQ(ByteSize::MB(1) * 1024, ByteSize::GB(1));
  EXPECT_EQ(ByteSize::GB(1) * 1024, ByteSize::TB(1));

  EXPECT_EQ(ByteSize::Bytes(1) * 1.1, ByteSize::Bytes(1));
  EXPECT_EQ(ByteSize::KB(1) * 1.2, ByteSize::KB(1.2));
  EXPECT_EQ(ByteSize::MB(1) * 1.3, ByteSize::MB(1.3));
  EXPECT_EQ(ByteSize::GB(1) * 1.4, ByteSize::GB(1.4));
  EXPECT_EQ(ByteSize::TB(1) * 1.5, ByteSize::TB(1.5));

  EXPECT_EQ(ByteSize::KB(1) * 0.5, ByteSize::Bytes(512));
  EXPECT_EQ(ByteSize::MB(1) * 0.5, ByteSize::KB(512));
  EXPECT_EQ(ByteSize::GB(1) * 0.5, ByteSize::MB(512));
  EXPECT_EQ(ByteSize::TB(1) * 0.25, ByteSize::GB(256));

  EXPECT_EQ(1024 * ByteSize::Bytes(1), ByteSize::KB(1));
  EXPECT_EQ(1024 * ByteSize::KB(1), ByteSize::MB(1));
  EXPECT_EQ(1024 * ByteSize::MB(1), ByteSize::GB(1));
  EXPECT_EQ(1024 * ByteSize::GB(1), ByteSize::TB(1));
  EXPECT_EQ(0.9 * ByteSize::TB(1), ByteSize::GB(921.6));
  EXPECT_EQ(0 * ByteSize::TB(1), ByteSize::Bytes(0));

  // Divide.
  EXPECT_EQ(ByteSize::Bytes(0) / 1, ByteSize::Bytes(0));
  EXPECT_EQ(ByteSize::KB(1) / 2, ByteSize::KB(0.5));
  EXPECT_EQ(ByteSize::MB(1) / 2, ByteSize::KB(512));
  EXPECT_EQ(ByteSize::GB(1) / 2, ByteSize::MB(512));
  EXPECT_EQ(ByteSize::TB(1.5) / 2, ByteSize::GB(768));

  EXPECT_EQ(ByteSize::KB(1) / 0.5, ByteSize::KB(2));
  EXPECT_EQ(ByteSize::MB(1) / 0.5, ByteSize::MB(2));
  EXPECT_EQ(ByteSize::GB(1) / 0.5, ByteSize::GB(2));
  EXPECT_EQ(ByteSize::TB(1) / 0.25, ByteSize::TB(4));

  // Ratio.
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(0) / ByteSize::KB(1), 0.0);
  EXPECT_DOUBLE_EQ(ByteSize::Bytes(1) / ByteSize::TB(1),
                   1.0 / 1024 / 1024 / 1024 / 1024);
  EXPECT_DOUBLE_EQ(ByteSize::KB(1) / ByteSize::KB(2), 0.5);
  EXPECT_DOUBLE_EQ(ByteSize::KB(512) / ByteSize::MB(1), 0.5);
  EXPECT_DOUBLE_EQ(ByteSize::KB(1) / ByteSize::MB(1), 1.0 / 1024.0);
  EXPECT_DOUBLE_EQ(ByteSize::MB(1) / ByteSize::GB(1), 1.0 / 1024.0);
  EXPECT_DOUBLE_EQ(ByteSize::GB(1) / ByteSize::TB(1), 1.0 / 1024.0);
}

TEST(ByteSizeTest, Assignments) {
  ByteSize byte_size;
  EXPECT_EQ(byte_size, ByteSize::Bytes(0));

  byte_size = ByteSize::Bytes(1);
  EXPECT_EQ(byte_size, ByteSize::Bytes(1));

  for (size_t i = 0; i < 1023; ++i) {
    byte_size += ByteSize::Bytes(1);
  }
  EXPECT_EQ(byte_size, ByteSize::KB(1));

  for (size_t i = 0; i < 10; ++i) {
    byte_size *= 2;
  }
  EXPECT_EQ(byte_size, ByteSize::MB(1));

  byte_size *= 1024 * 1024;
  EXPECT_EQ(byte_size, ByteSize::TB(1));

  for (size_t i = 0; i < 10; ++i) {
    byte_size /= 2;
  }
  EXPECT_EQ(byte_size, ByteSize::GB(1));

  for (size_t i = 0; i < 4; ++i) {
    byte_size -= ByteSize::MB(256);
  }
  EXPECT_EQ(byte_size, ByteSize::Bytes(0));

  // No negative bytes. The result will be 0 bytes.
  byte_size -= ByteSize::Bytes(1);
  EXPECT_EQ(byte_size, ByteSize::Bytes(0));
}

TEST(ByteSizeTest, Comparisons) {
  EXPECT_LE(ByteSize::Bytes(0), ByteSize::Bytes(0));
  EXPECT_LT(ByteSize::Bytes(0), ByteSize::Bytes(1));
  EXPECT_LE(ByteSize::Bytes(0), ByteSize::Bytes(1));
  EXPECT_LT(ByteSize::Bytes(1), ByteSize::Bytes(1024));
  EXPECT_LE(ByteSize::Bytes(1), ByteSize::Bytes(1024));
  EXPECT_LT(ByteSize::Bytes(1024), ByteSize::Bytes(1024 * 1024));
  EXPECT_LE(ByteSize::Bytes(1024), ByteSize::Bytes(1024 * 1024));
  EXPECT_LT(ByteSize::Bytes(1024), ByteSize::KB(1.1));
  EXPECT_LE(ByteSize::Bytes(1024), ByteSize::KB(1.1));

  EXPECT_LE(ByteSize::KB(0), ByteSize::Bytes(0));
  EXPECT_LE(ByteSize::KB(1), ByteSize::Bytes(1024));
  EXPECT_LT(ByteSize::KB(0), ByteSize::Bytes(1));
  EXPECT_LE(ByteSize::KB(0), ByteSize::Bytes(1));
  EXPECT_LT(ByteSize::KB(0.9), ByteSize::Bytes(1024));
  EXPECT_LE(ByteSize::KB(0.9), ByteSize::Bytes(1024));
  EXPECT_LT(ByteSize::KB(1), ByteSize::KB(1024));
  EXPECT_LE(ByteSize::KB(1), ByteSize::KB(1024));
  EXPECT_LT(ByteSize::KB(1), ByteSize::MB(1));
  EXPECT_LE(ByteSize::KB(1), ByteSize::MB(1));
  EXPECT_LT(ByteSize::KB(1024), ByteSize::MB(1.1));
  EXPECT_LE(ByteSize::KB(1024), ByteSize::MB(1.1));

  EXPECT_LE(ByteSize::MB(0), ByteSize::Bytes(0));
  EXPECT_LT(ByteSize::MB(0), ByteSize::Bytes(1));
  EXPECT_LE(ByteSize::MB(0), ByteSize::Bytes(1));
  EXPECT_LT(ByteSize::MB(0.9), ByteSize::KB(1024));
  EXPECT_LE(ByteSize::MB(0.9), ByteSize::KB(1024));
  EXPECT_LT(ByteSize::MB(1), ByteSize::MB(1024));
  EXPECT_LE(ByteSize::MB(1), ByteSize::MB(1024));
  EXPECT_LT(ByteSize::MB(1), ByteSize::GB(1));
  EXPECT_LE(ByteSize::MB(1), ByteSize::GB(1));
  EXPECT_LT(ByteSize::MB(1024), ByteSize::GB(1.1));
  EXPECT_LE(ByteSize::MB(1024), ByteSize::GB(1.1));

  EXPECT_LE(ByteSize::GB(0), ByteSize::Bytes(0));
  EXPECT_LT(ByteSize::GB(0), ByteSize::Bytes(1));
  EXPECT_LE(ByteSize::GB(0), ByteSize::Bytes(1));
  EXPECT_LT(ByteSize::GB(0.9), ByteSize::MB(1024));
  EXPECT_LE(ByteSize::GB(0.9), ByteSize::MB(1024));
  EXPECT_LT(ByteSize::GB(1), ByteSize::GB(1024));
  EXPECT_LE(ByteSize::GB(1), ByteSize::GB(1024));
  EXPECT_LT(ByteSize::GB(1), ByteSize::TB(1));
  EXPECT_LE(ByteSize::GB(1), ByteSize::TB(1));
  EXPECT_LT(ByteSize::GB(1024), ByteSize::TB(1.1));
  EXPECT_LE(ByteSize::GB(1024), ByteSize::TB(1.1));

  EXPECT_LE(ByteSize::TB(0), ByteSize::Bytes(0));
  EXPECT_LT(ByteSize::TB(0), ByteSize::Bytes(1));
  EXPECT_LE(ByteSize::TB(0), ByteSize::Bytes(1));
  EXPECT_LT(ByteSize::TB(0.9), ByteSize::GB(1024));
  EXPECT_LE(ByteSize::TB(0.9), ByteSize::GB(1024));
  EXPECT_LT(ByteSize::TB(1), ByteSize::TB(1024));
  EXPECT_LE(ByteSize::TB(1), ByteSize::TB(1024));
  EXPECT_LT(ByteSize::TB(1024), ByteSize::TB(1025));
  EXPECT_LE(ByteSize::TB(1024), ByteSize::TB(1025));

  EXPECT_GT(ByteSize::TB(1), ByteSize::GB(1));
  EXPECT_GT(ByteSize::GB(1), ByteSize::MB(1));
  EXPECT_GT(ByteSize::MB(1), ByteSize::KB(1));
  EXPECT_GT(ByteSize::KB(1), ByteSize::Bytes(1));
  EXPECT_GT(ByteSize::Bytes(1), ByteSize::Bytes(0));

  EXPECT_GT(ByteSize::TB(1), ByteSize::GB(1));
  EXPECT_GT(ByteSize::TB(1), ByteSize::GB(1) + ByteSize::MB(1) +
                                 ByteSize::KB(1) + ByteSize::Bytes(1));
  EXPECT_GT(ByteSize::GB(1), 0.0000001 * ByteSize::TB(1));
  EXPECT_GT(ByteSize::MB(1), ByteSize::KB(1) * 1023);
  EXPECT_GT(ByteSize::KB(1), ByteSize::KB(3) / 4);
  EXPECT_GT(ByteSize::Bytes(1), ByteSize::TB(0));

  EXPECT_GE(ByteSize::TB(0.5), ByteSize::GB(0.5));
  EXPECT_GE(ByteSize::GB(0.5), ByteSize::MB(0.5));
  EXPECT_GE(ByteSize::MB(0.5), ByteSize::KB(0.5));
  EXPECT_GE(ByteSize::KB(0.5), ByteSize::Bytes(1));
  EXPECT_GE(ByteSize::Bytes(1), ByteSize::Bytes(0));

  EXPECT_GE(ByteSize::TB(0), ByteSize::Bytes(0));
  EXPECT_GE(ByteSize::GB(0), ByteSize::Bytes(0));
  EXPECT_GE(ByteSize::MB(0), ByteSize::Bytes(0));
  EXPECT_GE(ByteSize::KB(0), ByteSize::Bytes(0));
  EXPECT_GE(ByteSize::Bytes(0), ByteSize::Bytes(0));
}

TEST(ByteSizeTest, DebugString) {
  EXPECT_EQ(ByteSize::Bytes(0).DebugString(), "0B");
  EXPECT_EQ(ByteSize::Bytes(1).DebugString(), "1B");
  EXPECT_EQ(ByteSize::Bytes(size_t{1} << 10).DebugString(), "1KB");
  EXPECT_EQ(ByteSize::Bytes(size_t{1} << 20).DebugString(), "1MB");
  EXPECT_EQ(ByteSize::Bytes(size_t{1} << 30).DebugString(), "1GB");
  EXPECT_EQ(ByteSize::Bytes(size_t{1} << 40).DebugString(), "1TB");

  EXPECT_EQ(ByteSize::KB(0.5).DebugString(), "512B");
  EXPECT_EQ(ByteSize::KB(1).DebugString(), "1KB");
  EXPECT_EQ(ByteSize::KB(1.5).DebugString(), "1.5KB");
  EXPECT_EQ(ByteSize::KB(1024).DebugString(), "1MB");
  EXPECT_EQ(ByteSize::KB(1024 * 1024).DebugString(), "1GB");
  EXPECT_EQ(ByteSize::KB(1024 * 1024 * 1024).DebugString(), "1TB");

  EXPECT_EQ(ByteSize::MB(0.5).DebugString(), "512KB");
  EXPECT_EQ(ByteSize::MB(1).DebugString(), "1MB");
  EXPECT_EQ(ByteSize::MB(1.5).DebugString(), "1.5MB");
  EXPECT_EQ(ByteSize::MB(1024).DebugString(), "1GB");
  EXPECT_EQ(ByteSize::MB(1024 * 1024).DebugString(), "1TB");

  EXPECT_EQ(ByteSize::GB(0.5).DebugString(), "512MB");
  EXPECT_EQ(ByteSize::GB(1).DebugString(), "1GB");
  EXPECT_EQ(ByteSize::GB(1.5).DebugString(), "1.5GB");
  EXPECT_EQ(ByteSize::GB(1024).DebugString(), "1TB");

  EXPECT_EQ(ByteSize::TB(0.5).DebugString(), "512GB");
  EXPECT_EQ(ByteSize::TB(1).DebugString(), "1TB");
  EXPECT_EQ(ByteSize::TB(1.5).DebugString(), "1.5TB");
  EXPECT_EQ(ByteSize::TB(1024).DebugString(), "1024TB");
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
