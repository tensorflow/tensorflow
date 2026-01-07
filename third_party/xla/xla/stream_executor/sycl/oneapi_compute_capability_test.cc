/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor::sycl {
namespace {

TEST(OneAPIComputeCapabilityTest, ProtoTest1) {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture("bmg");
  auto compute_capability = OneAPIComputeCapability(proto);
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, ToProtoTest1) {
  OneAPIComputeCapabilityProto proto =
      OneAPIComputeCapability(0xc, 0x37).ToProto();
  EXPECT_EQ(proto.architecture(), "DG2");
}

TEST(OneAPIComputeCapabilityTest, ToProtoTest2) {
  OneAPIComputeCapabilityProto proto = OneAPIComputeCapability("bmg").ToProto();
  EXPECT_EQ(proto.architecture(), "BMG");
}

TEST(OneAPIComputeCapabilityTest, ToString) {
  EXPECT_EQ(OneAPIComputeCapability(100, 20).ToString(), "100.20");
}

TEST(OneAPIComputeCapabilityTest, PlatformCCTest) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability::DG2();
  EXPECT_EQ(compute_capability.generation(), 0xc);
  EXPECT_EQ(compute_capability.version(), 0x37);
}

}  // namespace
}  // namespace stream_executor::sycl
