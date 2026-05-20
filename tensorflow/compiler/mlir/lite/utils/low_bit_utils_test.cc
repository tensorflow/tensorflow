/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/low_bit_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {

namespace {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok());
#endif

class LowBitUtilsTest : public ::testing::Test {
 protected:
  LowBitUtilsTest() = default;

  void SetUp() override {
    context_ = std::make_unique<mlir::MLIRContext>();
    builder_ = std::make_unique<mlir::Builder>(context_.get());
  }

  void TearDown() override { builder_.reset(); }

  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(LowBitUtilsTest, Stream4BitValues) {
  auto type = mlir::RankedTensorType::get({3}, builder_->getIntegerType(32));
  auto raw_values = std::vector<int32_t>{1, 2, 3};
  auto attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef(raw_values));

  std::vector<uint8_t> packed_values;
  auto apply_chunk_fn = [&](absl::string_view chunk) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(chunk.data());
    packed_values.insert(packed_values.end(), data, data + chunk.size());
    return absl::OkStatus();
  };

  EXPECT_OK(tflite::StreamPackLowBitValues</*kBitWidth=*/4>(
      attr.getValues<mlir::APInt>(), apply_chunk_fn));
  EXPECT_EQ(packed_values.size(), 2);
  EXPECT_EQ(packed_values[0], 0x21);
  EXPECT_EQ(packed_values[1], 0x03);
}

TEST_F(LowBitUtilsTest, Stream4BitValues8Bit) {
  auto type = mlir::RankedTensorType::get({3}, builder_->getIntegerType(8));
  auto raw_values = std::vector<uint8_t>{1, 2, 3};
  auto attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<uint8_t>(raw_values));

  std::vector<uint8_t> packed_values;
  auto apply_chunk_fn = [&](absl::string_view chunk) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(chunk.data());
    packed_values.insert(packed_values.end(), data, data + chunk.size());
    return absl::OkStatus();
  };

  EXPECT_EQ(attr.getRawData().size(), 3);
  EXPECT_OK(tflite::StreamPackLowBitValues8Bit</*kBitWidth=*/4>(
      attr.getRawData(), apply_chunk_fn));
  EXPECT_EQ(packed_values.size(), 2);
  EXPECT_EQ(packed_values[0], 0x21);
  EXPECT_EQ(packed_values[1], 0x03);
}

TEST_F(LowBitUtilsTest, Stream2BitValues) {
  auto type = mlir::RankedTensorType::get({7}, builder_->getIntegerType(32));
  auto raw_values = std::vector<int32_t>{1, 1, 2, 3, 3, 2, 1};
  auto attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef(raw_values));

  std::vector<uint8_t> packed_values;
  auto apply_chunk_fn = [&](absl::string_view chunk) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(chunk.data());
    packed_values.insert(packed_values.end(), data, data + chunk.size());
    return absl::OkStatus();
  };

  EXPECT_OK(tflite::StreamPackLowBitValues</*kBitWidth=*/2>(
      attr.getValues<mlir::APInt>(), apply_chunk_fn));
  EXPECT_EQ(packed_values.size(), 2);
  EXPECT_EQ(packed_values[0], 0xE5);  // 1110 0101
  EXPECT_EQ(packed_values[1], 0x1B);  // 0001 1011
}

TEST_F(LowBitUtilsTest, Stream2BitValues8Bit) {
  auto type = mlir::RankedTensorType::get({7}, builder_->getIntegerType(8));
  auto raw_values = std::vector<uint8_t>{1, 1, 2, 3, 3, 2, 1};
  auto attr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<uint8_t>(raw_values));

  std::vector<uint8_t> packed_values;
  auto apply_chunk_fn = [&](absl::string_view chunk) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(chunk.data());
    packed_values.insert(packed_values.end(), data, data + chunk.size());
    return absl::OkStatus();
  };

  EXPECT_EQ(attr.getRawData().size(), 7);
  EXPECT_OK(tflite::StreamPackLowBitValues8Bit</*kBitWidth=*/2>(
      attr.getRawData(), apply_chunk_fn));
  EXPECT_EQ(packed_values.size(), 2);
  EXPECT_EQ(packed_values[0], 0xE5);  // 1110 0101
  EXPECT_EQ(packed_values[1], 0x1B);  // 0001 1011
}

}  // namespace

}  // namespace TFL
}  // namespace mlir
