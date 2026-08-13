/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace {

TEST(PjRtCopyToMemorySpaceTest, CopyWithoutDonation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetPjRtCpuClient(CpuClientOptions()));
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  Literal literal = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> src_buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> dst_buffer,
      src_buffer->CopyToMemorySpace(client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> received_literal,
                       dst_buffer->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, *received_literal));
}

TEST(PjRtCopyToMemorySpaceTest, CopyWithDonatedRawBuffer) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetPjRtCpuClient(CpuClientOptions()));
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  Literal src_literal = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal initial_literal = LiteralUtil::CreateR1<int32_t>({0, 0, 0, 0});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> src_buffer,
      client->BufferFromHostLiteral(src_literal, client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> donated_buffer,
                       client->BufferFromHostLiteral(
                           initial_literal, client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(
      PjRtRawBufferRef raw_alias,
      PjRtRawBuffer::CreateRawAliasOfBuffer(donated_buffer.get()));

  auto* common_src = dynamic_cast<CommonPjRtBufferImpl*>(src_buffer.get());
  ASSERT_NE(common_src, nullptr);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> dst_buffer,
      common_src->CopyToMemorySpace(client->memory_spaces()[0], raw_alias));

  // Check that the destination buffer contains the copied data.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> dst_received_literal,
                       dst_buffer->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, *dst_received_literal));

  // Check that dst_buffer reuses the exact raw buffer from donated_buffer.
  ASSERT_OK_AND_ASSIGN(PjRtRawBufferRef dst_raw_alias,
                       PjRtRawBuffer::CreateRawAliasOfBuffer(dst_buffer.get()));
  EXPECT_EQ(dst_raw_alias.get(), raw_alias.get());
}

}  // namespace
}  // namespace xla
