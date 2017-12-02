/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class TransferManagerTest : public LocalClientTestBase {
 protected:
  TransferManagerTest()
      : shape_size_fn_([this](const Shape& shape) {
          return transfer_manager_->GetByteSizeRequirement(shape);
        }) {}

  ~TransferManagerTest() override = default;

  std::unique_ptr<ScopedShapedBuffer> AllocateDeviceBuffer(const Shape& shape) {
    return ScopedShapedBuffer::Allocate(
               shape, GetOrCreateAllocator(local_client_->platform()),
               /*device_ordinal=*/0, shape_size_fn_)
        .ValueOrDie();
  }

 private:
  std::function<int64(const Shape&)> shape_size_fn_;
};

XLA_TEST_F(TransferManagerTest, TransferR0U32) {
  std::unique_ptr<Literal> literal = Literal::CreateR0<uint32>(42);
  const Shape& shape = literal->shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectR0Equal<uint32>(42, *result);
}

XLA_TEST_F(TransferManagerTest, TransferR1F32) {
  std::unique_ptr<Literal> literal =
      Literal::CreateR1<float>({1.25f, 2.5f, -17.0f, -20.125f});
  const Shape& shape = literal->shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>({1.25f, 2.5f, -17.0f, -20.125f},
                                        *result);
}

XLA_TEST_F(TransferManagerTest, TransferR1LargeF32) {
  std::vector<float> test_vector(1024 * 1024);
  std::iota(test_vector.begin(), test_vector.end(), 0);
  std::unique_ptr<Literal> literal = Literal::CreateR1<float>(test_vector);
  const Shape& shape = literal->shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>(test_vector, *result);
}

XLA_TEST_F(TransferManagerTest, TransferR1U8) {
  const char* test_string = "0123456789abcdef";
  std::unique_ptr<Literal> literal = Literal::CreateR1U8(test_string);
  const Shape& shape = literal->shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  EXPECT_EQ(result->u8s_string(), test_string);
}

XLA_TEST_F(TransferManagerTest, TransferR2F32) {
  std::unique_ptr<Literal> literal =
      Literal::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  const Shape& shape = literal->shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, *result);
}

XLA_TEST_F(TransferManagerTest,
           TransferR2F32AndChangeLayoutTransferringToDevice) {
  std::unique_ptr<Literal> literal = Literal::CreateR2WithLayout<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, LayoutUtil::MakeLayout({0, 1}));
  const Shape ondevice_shape =
      ShapeUtil::MakeShapeWithLayout(F32, {2, 3}, {1, 0});
  auto device_buffer = AllocateDeviceBuffer(ondevice_shape);

  // Round trip literal through device. Set the on-device layout to something
  // different than the literal layout.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  EXPECT_FALSE(
      LayoutUtil::Equal(result->shape().layout(), literal->shape().layout()));
  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, *result);
}

XLA_TEST_F(TransferManagerTest, TransferTuple) {
  std::unique_ptr<Literal> literal = Literal::MakeTuple(
      {Literal::CreateR0<float>(123.0f).get(),
       Literal::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}).get(),
       Literal::CreateR1<float>({44.0f, -10.0f, 3333333.3f}).get()});
  auto device_buffer = AllocateDeviceBuffer(literal->shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectEqual(*literal, *result);
}

XLA_TEST_F(TransferManagerTest, TransferEmptyTuple) {
  std::unique_ptr<Literal> literal = Literal::MakeTuple({});
  auto device_buffer = AllocateDeviceBuffer(literal->shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectEqual(*literal, *result);
}

XLA_TEST_F(TransferManagerTest, TransferNestedTuple) {
  std::unique_ptr<Literal> literal = Literal::MakeTuple(
      {Literal::CreateR0<float>(123.0f).get(),
       Literal::MakeTuple(
           {Literal::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}).get(),
            Literal::CreateR1<float>({44.0f, -10.0f, 3333333.3f}).get()})
           .get(),
       Literal::CreateR1<float>({-10.0f, 123.0f}).get()});
  auto device_buffer = AllocateDeviceBuffer(literal->shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, *literal, *device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> result,
                          transfer_manager_->TransferLiteralFromDevice(
                              stream_executor_, *device_buffer));

  LiteralTestUtil::ExpectEqual(*literal, *result);
}

}  // namespace
}  // namespace xla
