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

#include <unistd.h>

#include <cstdint>
#include <memory>

#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/test_helpers.h"
#include "xla/tests/client_library_test_base.h"
#include "tsl/platform/env.h"

namespace xla {
namespace {

class InfeedTest : public ClientLibraryTestBase {
 protected:
  // Transfers the given literal to the infeed interface of the device, and
  // check if the returned data from Infeed HLO is same as the literal.
  void TestInfeedRoundTrip(const Literal& literal) {
    // TODO(b/30481585) Explicitly reset the Infeed state so that the
    // test is not affected by the state from the previous tests.
    ASSERT_IS_OK(client_->TransferToInfeed(literal));
    XlaBuilder builder(TestName());
    Infeed(&builder, literal.shape());
    if (literal.shape().IsTuple()) {
      // TODO(b/30609564): Use ComputeAndCompareLiteral instead.
      ComputeAndCompareTuple(&builder, literal, {});
    } else {
      ComputeAndCompareLiteral(&builder, literal, {});
    }
  }
};

TEST_F(InfeedTest, SingleInfeedR0Bool) {
  TestInfeedRoundTrip(LiteralUtil::CreateR0<bool>(true));
}

TEST_F(InfeedTest, SingleInfeedR1U32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR1<uint32_t>({1, 2, 3}));
}

TEST_F(InfeedTest, SingleInfeedR2F32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR2F32Linspace(0.0, 1.0, 128, 64));
}

TEST_F(InfeedTest, SingleInfeedR3F32) {
  TestInfeedRoundTrip(
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

TEST_F(InfeedTest, SingleInfeedR3F32DifferentLayout) {
  const Layout r3_dim0minor = LayoutUtil::MakeLayout({0, 1, 2});
  const Layout r3_dim0major = LayoutUtil::MakeLayout({2, 1, 0});

  TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0minor));

  TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0major));
}

TEST_F(InfeedTest, SingleInfeedR4S32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

// Tests that a large infeed can be handled.
TEST_F(InfeedTest, LargeInfeed) {
  Array4D<float> array(80, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedRoundTrip(LiteralUtil::CreateR4FromArray4D<float>(array));
}

TEST_F(InfeedTest, SingleInfeedTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<uint32_t>({1, 2, 3}),
       LiteralUtil::CreateR0<bool>(false)}));
}

TEST_F(InfeedTest, SingleInfeedEmptyTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

// Tests that a large tuple infeed can be handled.
TEST_F(InfeedTest, SingleInfeedLargeTuple) {
  Array4D<float> array(40, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR4FromArray4D<float>(array),
       LiteralUtil::CreateR0<int32_t>(5)}));
}

class BlockingInfeedTest : public ClientLibraryTestBase {};

TEST_F(BlockingInfeedTest, TestNoOoms) {
  Array3D<float> array(1024, 1024, 64);
  array.FillIota(1.0f);
  auto literal = LiteralUtil::CreateR3FromArray3D<float>(array);

  int64_t kMemoryPressure = 32ul * 1024 * 1024 * 1024;
  int64_t infeed_count =
      kMemoryPressure / (array.num_elements() * sizeof(float));

  auto transfer_infeeds = [&] {
    for (int i = 0; i < infeed_count; i++) {
      ASSERT_IS_OK(client_->TransferToInfeed(literal));
    }
  };

  auto* env = tsl::Env::Default();

  std::unique_ptr<tsl::Thread> thread{env->StartThread(
      tsl::ThreadOptions{}, "transfer_infeeds", transfer_infeeds)};

  // Sleep for 30s waiting for the infeed thread to "catch up".
  //
  // Without the fix accompanying this test, transfer_infeeds causes an OOM if
  // the consumer (XLA computation running on the main thread) is unable to keep
  // up with the producer (the transfer_infeeds thread).  When that happens, the
  // GPU buffers from the producer pile up and consume all of GPU memory.
  //
  // To reliably reproduce the issue we need to slow down the consumer, and we
  // do that by inserting a sleep here.
  //
  // The fix is to back TransferToInfeed by a blocking queue that does not allow
  // more than kMaxInfeedsInFlight infeeds in flight.
  env->SleepForMicroseconds(30ul * 1000 * 1000);

  XlaBuilder builder(TestName());
  for (int i = 0; i < infeed_count; i++) {
    Infeed(&builder, literal.shape());
  }

  ComputeAndCompareLiteral(&builder, literal, {});
}

}  // namespace
}  // namespace xla
