/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/generic_transfer_manager.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class PackingTransferManager : public GenericTransferManager {
 public:
  using GenericTransferManager::GenericTransferManager;
  bool pack_subbyte_types_ = true;

 private:
  bool PackSubbyteTypes() const override { return pack_subbyte_types_; }
};

class GenericTransferManagerTest : public ::testing::Test {
 public:
  GenericTransferManagerTest()
      : transfer_manager_(se::host::kHostPlatformId,
                          /*pointer_size=*/sizeof(void*)) {}
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        se::Platform * platform,
        se::PlatformManager::PlatformWithId(se::host::kHostPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(stream_executor_, platform->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, stream_executor_->CreateStream());
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor_);
  }

  ScopedShapedBuffer AllocateBuffer(const Shape& shape) {
    auto buffer =
        transfer_manager_.AllocateScopedShapedBuffer(shape, allocator_.get(),
                                                     /*device_ordinal=*/0);
    return std::move(buffer.value());
  }

  PackingTransferManager transfer_manager_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

TEST_F(GenericTransferManagerTest, TransferLiteralToDevice) {
  ScopedShapedBuffer buffer = AllocateBuffer(ShapeUtil::MakeShape(U16, {2, 2}));
  Literal literal = LiteralUtil::CreateR2<uint16_t>({{1, 2}, {3, 4}});
  TF_ASSERT_OK(transfer_manager_.TransferLiteralToDevice(stream_.get(), literal,
                                                         buffer));

  se::DeviceMemoryBase device_mem = buffer.buffers().element({});
  uint16_t* device_ptr = static_cast<uint16_t*>(device_mem.opaque());
  std::vector<uint16_t> expected = {1, 2, 3, 4};
  EXPECT_EQ(absl::Span<uint16_t>(device_ptr, expected.size()), expected);
}

MATCHER_P2(MaskedValuesEqual, mask, expected, "") {
  if (arg.size() != expected.size()) {
    *result_listener << "argument sizes do not match";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto v1 = arg[i] & mask;
    const auto v2 = expected[i] & mask;
    if (v1 != v2) {
      *result_listener << "mismatch at position " << i << ", " << v1 << " vs "
                       << v2;
      return false;
    }
  }
  return true;
}

TEST_F(GenericTransferManagerTest, TransferLiteralToDeviceInt4) {
  Literal literal =
      LiteralUtil::CreateR2<s4>({{s4{1}, s4{-2}}, {s4{-3}, s4{4}}});
  for (bool pack : {false, true}) {
    SCOPED_TRACE(absl::StrCat("pack=", pack));
    transfer_manager_.pack_subbyte_types_ = pack;
    ScopedShapedBuffer buffer =
        AllocateBuffer(ShapeUtil::MakeShape(S4, {2, 2}));
    TF_ASSERT_OK(transfer_manager_.TransferLiteralToDevice(stream_.get(),
                                                           literal, buffer));
    se::DeviceMemoryBase device_mem = buffer.buffers().element({});
    ASSERT_EQ(device_mem.size(), pack ? 2 : 4);
    int8_t* device_ptr = static_cast<int8_t*>(device_mem.opaque());
    std::vector<int8_t> expected =
        pack ? std::vector<int8_t>{static_cast<int8_t>(0xe1),
                                   static_cast<int8_t>(0x4d)}
             : std::vector<int8_t>{1, -2, -3, 4};
    // Ignore high bits in equality comparisons.
    EXPECT_THAT(absl::Span<int8_t>(device_ptr, expected.size()),
                MaskedValuesEqual(pack ? 0xFF : 0x0F, expected));
  }
}

TEST_F(GenericTransferManagerTest, TransferLiteralFromDevice) {
  ScopedShapedBuffer buffer = AllocateBuffer(ShapeUtil::MakeShape(U16, {2, 2}));

  se::DeviceMemoryBase device_mem = buffer.buffers().element({});
  uint16_t* device_ptr = static_cast<uint16_t*>(device_mem.opaque());
  for (int i = 0; i < 4; i++) {
    device_ptr[i] = i + 1;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      Literal literal,
      transfer_manager_.TransferManager::TransferLiteralFromDevice(
          stream_.get(), buffer));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      literal, LiteralUtil::CreateR2<uint16_t>({{1, 2}, {3, 4}})));
}

TEST_F(GenericTransferManagerTest, TransferLiteralFromDeviceInt4) {
  for (bool pack : {false, true}) {
    SCOPED_TRACE(absl::StrCat("pack=", pack));
    transfer_manager_.pack_subbyte_types_ = pack;
    ScopedShapedBuffer buffer =
        AllocateBuffer(ShapeUtil::MakeShape(S4, {2, 2}));

    se::DeviceMemoryBase device_mem = buffer.buffers().element({});
    uint8_t* device_ptr = static_cast<uint8_t*>(device_mem.opaque());
    if (pack) {
      ASSERT_EQ(device_mem.size(), 2);
      device_ptr[0] = 0xe1;  // Packed S4 values {1, -2}
      device_ptr[1] = 0x4d;  // Packed S4 values {-3, 4}
    } else {
      ASSERT_EQ(device_mem.size(), 4);
      device_ptr[0] = 1;
      device_ptr[1] = -2;
      device_ptr[2] = -3;
      device_ptr[3] = 4;
    }

    TF_ASSERT_OK_AND_ASSIGN(
        Literal literal,
        transfer_manager_.TransferManager::TransferLiteralFromDevice(
            stream_.get(), buffer));
    EXPECT_TRUE(LiteralTestUtil::Equal(
        literal,
        LiteralUtil::CreateR2<s4>({{s4{1}, s4{-2}}, {s4{-3}, s4{4}}})));
  }
}

TEST_F(GenericTransferManagerTest, ChooseCompactLayoutForShape) {
  auto shape = ShapeUtil::MakeShape(S4, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto compact_shape,
                          transfer_manager_.ChooseCompactLayoutForShape(shape));
  EXPECT_TRUE(Shape::Equal().IgnoreLayout()(compact_shape, shape));
  EXPECT_EQ(compact_shape.layout().element_size_in_bits(), 4);
}

}  // namespace
}  // namespace xla
