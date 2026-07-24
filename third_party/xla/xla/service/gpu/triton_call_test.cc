/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/triton_call.h"

#include <string>

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {
namespace {

// Builds a minimal __gpu$xla.gpu.triton backend config dictionary, optionally
// appending an extra attribute (e.g. serialized_metadata).
std::string BackendConfig(const std::string& extra_attrs) {
  std::string config =
      R"({name = "k", ir = "", num_stages = 1 : i32, num_warps = 4 : i32, )"
      R"(grid_x = 1 : i32, grid_y = 1 : i32, grid_z = 1 : i32)";
  if (!extra_attrs.empty()) {
    config += ", " + extra_attrs;
  }
  config += "}";
  return config;
}

TEST(TritonCallTest, WavesPerEuDefaultsToZeroWhenAbsent) {
  mlir::MLIRContext ctx;
  TritonCall call = TritonCall::Parse(BackendConfig(/*extra_attrs=*/""), &ctx);
  EXPECT_EQ(call.waves_per_eu, 0);
}

TEST(TritonCallTest, WavesPerEuParsedFromSerializedMetadataStringValue) {
  mlir::MLIRContext ctx;
  TritonCall call = TritonCall::Parse(
      BackendConfig(R"(serialized_metadata = "{\"waves_per_eu\": \"2\"}")"),
      &ctx);
  EXPECT_EQ(call.waves_per_eu, 2);
}

TEST(TritonCallTest, WavesPerEuParsedFromSerializedMetadataNumberValue) {
  mlir::MLIRContext ctx;
  TritonCall call = TritonCall::Parse(
      BackendConfig(R"(serialized_metadata = "{\"waves_per_eu\": 3}")"), &ctx);
  EXPECT_EQ(call.waves_per_eu, 3);
}

TEST(TritonCallTest, WavesPerEuUnsetWhenMetadataHasNoSuchKey) {
  mlir::MLIRContext ctx;
  TritonCall call = TritonCall::Parse(
      BackendConfig(R"(serialized_metadata = "{\"other\": \"1\"}")"), &ctx);
  EXPECT_EQ(call.waves_per_eu, 0);
}

TEST(TritonCallTest, WavesPerEuUnsetWhenMetadataIsNotValidJson) {
  mlir::MLIRContext ctx;
  TritonCall call = TritonCall::Parse(
      BackendConfig(R"(serialized_metadata = "not-json")"), &ctx);
  EXPECT_EQ(call.waves_per_eu, 0);
}

}  // namespace
}  // namespace xla::gpu
