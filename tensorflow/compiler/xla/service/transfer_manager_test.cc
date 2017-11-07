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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

namespace {

class CpuTransferManagerTest : public ::testing::Test {
 protected:
  CpuTransferManagerTest()
      : transfer_manager_(se::host::kHostPlatformId,
                          /*pointer_size=*/sizeof(void*)) {
    se::Platform* platform =
        se::MultiPlatformManager::PlatformWithId(se::host::kHostPlatformId)
            .ValueOrDie();
    stream_exec_ =
        platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
            .ValueOrDie();
  }

  ~CpuTransferManagerTest() override {}

  se::StreamExecutor* stream_exec_;
  GenericTransferManager transfer_manager_;
};

TEST_F(CpuTransferManagerTest, TransferR0U32ToDevice) {
  std::vector<uint8> storage(sizeof(uint32), '\x00');
  se::DeviceMemoryBase memptr(storage.data(), storage.size());
  std::unique_ptr<Literal> literal = Literal::CreateR0<uint32>(42);
  TF_CHECK_OK(transfer_manager_.TransferLiteralToDevice(stream_exec_, *literal,
                                                        &memptr));

  CHECK_EQ(42, *reinterpret_cast<uint32*>(&storage[0]));
}

TEST_F(CpuTransferManagerTest, TransferR1F32ToDevice) {
  std::vector<uint8> storage(4 * sizeof(float), '\x00');
  se::DeviceMemoryBase memptr(storage.data(), storage.size());
  std::unique_ptr<Literal> literal =
      Literal::CreateR1<float>({1.25f, 2.5f, -17.0f, -20.125f});
  TF_CHECK_OK(transfer_manager_.TransferLiteralToDevice(stream_exec_, *literal,
                                                        &memptr));

  CHECK_EQ(1.25f, *reinterpret_cast<float*>(&storage[0]));
  CHECK_EQ(2.5f, *reinterpret_cast<float*>(&storage[sizeof(float)]));
  CHECK_EQ(-17.0f, *reinterpret_cast<float*>(&storage[2 * sizeof(float)]));
  CHECK_EQ(-20.125f, *reinterpret_cast<float*>(&storage[3 * sizeof(float)]));
}

TEST_F(CpuTransferManagerTest, TransferR1U8ToDevice) {
  std::vector<uint8> storage(16, '\x00');
  se::DeviceMemoryBase memptr(storage.data(), storage.size());
  const char* str = "0123456789abcdef";
  std::unique_ptr<Literal> literal = Literal::CreateR1U8(str);
  TF_CHECK_OK(transfer_manager_.TransferLiteralToDevice(stream_exec_, *literal,
                                                        &memptr));

  CHECK_EQ('0', storage[0]);
  CHECK_EQ('8', storage[8]);
  CHECK_EQ('f', storage[15]);
}

TEST_F(CpuTransferManagerTest, TransferR0U32FromDevice) {
  std::vector<uint32> storage(1, 42);
  se::DeviceMemoryBase memptr(storage.data(),
                              storage.size() * sizeof(storage[0]));
  Literal literal;
  const Shape shape = ShapeUtil::MakeShape(U32, {});
  TF_CHECK_OK(transfer_manager_.TransferLiteralFromDevice(
      stream_exec_, memptr, shape, shape, &literal));

  LiteralTestUtil::ExpectR0Equal<uint32>(42, literal);
}

TEST_F(CpuTransferManagerTest, TransferR1F32FromDevice) {
  std::vector<float> storage{1.25f, 2.5f, -17.0f, -20.125f};
  se::DeviceMemoryBase memptr(storage.data(),
                              storage.size() * sizeof(storage[0]));
  Literal literal;
  const Shape shape = ShapeUtil::MakeShape(F32, {4});
  TF_CHECK_OK(transfer_manager_.TransferLiteralFromDevice(
      stream_exec_, memptr, shape, shape, &literal));

  LiteralTestUtil::ExpectR1Equal<float>({1.25, 2.5, -17.0, -20.125}, literal);
}

TEST_F(CpuTransferManagerTest, TransferR1U8FromDevice) {
  std::vector<uint8> storage{'k', 'l', 'm', 'n'};
  se::DeviceMemoryBase memptr(storage.data(),
                              storage.size() * sizeof(storage[0]));
  Literal literal;
  const Shape shape = ShapeUtil::MakeShape(U8, {4});
  TF_CHECK_OK(transfer_manager_.TransferLiteralFromDevice(
      stream_exec_, memptr, shape, shape, &literal));
  CHECK_EQ("klmn", literal.u8s_string());
}

TEST_F(CpuTransferManagerTest, TransferBufferFromDevice) {
  std::vector<uint64> storage{1, 5, 42};
  int64 size = storage.size() * sizeof(storage[0]);
  se::DeviceMemoryBase memptr(storage.data(), size);

  std::vector<uint64> dest(3, 0);
  TF_CHECK_OK(transfer_manager_.TransferBufferFromDevice(stream_exec_, memptr,
                                                         size, dest.data()));
  ASSERT_EQ(1, dest[0]);
  ASSERT_EQ(5, dest[1]);
  ASSERT_EQ(42, dest[2]);
}

TEST_F(CpuTransferManagerTest, TransferBufferToDevice) {
  int64 size = 3 * sizeof(uint64);
  std::vector<uint8> storage(size, 0);
  se::DeviceMemoryBase memptr(storage.data(), size);

  std::vector<uint64> dest{1, 5, 42};
  TF_CHECK_OK(transfer_manager_.TransferBufferToDevice(stream_exec_, size,
                                                       dest.data(), &memptr));
  std::vector<uint64>* storage64 =
      reinterpret_cast<std::vector<uint64>*>(&storage);
  ASSERT_EQ(1, (*storage64)[0]);
  ASSERT_EQ(5, (*storage64)[1]);
  ASSERT_EQ(42, (*storage64)[2]);
}

// TODO(b/24679870): add similar tests for GPUs

}  // namespace

}  // namespace xla
