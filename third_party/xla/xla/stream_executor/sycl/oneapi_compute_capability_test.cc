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
  proto.set_architecture("BMG");
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability(proto);
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, ProtoTest2) {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture("DG2");
  proto.set_variant("G11");
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability(proto);
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_TRUE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityFromName) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("pvc", "vg");
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_TRUE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityFromGenAndVer) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability(0x14, 0x2);
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityForUnknownDevice) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("Device", "Variant");
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_EQ(compute_capability.ToString(), "Unknown");
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityForUnknownVariant) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("BMG", "Variant");
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_EQ(compute_capability.ToString(), "Xe2");
}

TEST(OneAPIComputeCapabilityTest, ToString) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("DG2", "G10");
  EXPECT_EQ(compute_capability.ToString(), "DG2_G10");
}

TEST(OneAPIComputeCapabilityTest, ToProto) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("PVC", "VG");
  OneAPIComputeCapabilityProto proto = compute_capability.ToProto();
  EXPECT_EQ(proto.architecture(), "PVC");
  EXPECT_EQ(proto.variant(), "VG");
}

TEST(OneAPIComputeCapabilityTest, ToProtoUnknownDevice) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("ABC", "DEF");
  OneAPIComputeCapabilityProto proto = compute_capability.ToProto();
  EXPECT_EQ(proto.architecture(), "Unknown");
  EXPECT_EQ(proto.variant(), "");
}

TEST(OneAPIComputeCapabilityTest, UtilityFunc) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability::BMG();
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
}

TEST(OneAPIComputeCapabilityTest, GetterFunc) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability::DG2();
  EXPECT_EQ(compute_capability.architecture(), "DG2");
  EXPECT_EQ(compute_capability.variant(), "G10");
}

}  // namespace
}  // namespace stream_executor::sycl
