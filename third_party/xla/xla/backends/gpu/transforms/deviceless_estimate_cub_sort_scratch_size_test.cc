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

#include "xla/backends/gpu/transforms/deviceless_estimate_cub_sort_scratch_size.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::stream_executor::SemanticVersion;

class DevicelessEstimateCubSortScratchSizeTest
    : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<int64_t> RunPassAndExtractScratchSize(
      absl::string_view hlo_text, DevicelessEstimateCubSortScratchSize& pass) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_text));

    TF_ASSIGN_OR_RETURN(bool changed, RunHloPass(&pass, module.get()));
    if (!changed) {
      return absl::InternalError("Pass did not change the module");
    }

    const HloInstruction* custom_call = nullptr;
    for (auto* comp : module->computations()) {
      for (auto* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kCustomCall) {
          custom_call = inst;
          break;
        }
      }
      if (custom_call) {
        break;
      }
    }

    if (!custom_call) {
      return absl::NotFoundError("Custom call not found");
    }

    const Shape& shape = custom_call->shape();
    if (!shape.IsTuple()) {
      return absl::InternalError("Custom call output is not a tuple");
    }

    const Shape& scratch_shape = shape.tuple_shapes().back();
    if (scratch_shape.element_type() != U8) {
      return absl::InternalError("Last element of tuple is not U8");
    }

    return scratch_shape.dimensions(0);
  }
};

TEST_F(DevicelessEstimateCubSortScratchSizeTest, KeyOnly) {
  ASSERT_OK_AND_ASSIGN(const CubScratchSizeDevicelessLookup& lookup_table,
                       CubScratchSizeDevicelessLookup::GetInstance());
  absl::string_view hlo = R"hlo(
    HloModule m
    ENTRY main {
      %keys = f32[8192] parameter(0)
      %custom-call = (f32[8192]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[8192]{0} get-tuple-element(%custom-call), index=0
  })hlo";

  DevicelessEstimateCubSortScratchSize pass("CUDA", "NVIDIA H100 80GB HBM3",
                                            SemanticVersion(3, 1, 2));
  ASSERT_OK_AND_ASSIGN(int64_t pass_scratch_size,
                       RunPassAndExtractScratchSize(hlo, pass));

  std::optional<int64_t> lookup_scratch_size = lookup_table.Lookup(
      SemanticVersion(3, 1, 2),
      /*device_name=*/"NVIDIA H100 80GB HBM3", /*key_type_size=*/4,
      /*value_type_size=*/std::nullopt, /*num_items=*/8192);
  EXPECT_TRUE(lookup_scratch_size.has_value());

  EXPECT_EQ(pass_scratch_size, *lookup_scratch_size);
}

TEST_F(DevicelessEstimateCubSortScratchSizeTest, KeyValue) {
  ASSERT_OK_AND_ASSIGN(const CubScratchSizeDevicelessLookup& lookup_table,
                       CubScratchSizeDevicelessLookup::GetInstance());
  absl::string_view hlo = R"hlo(
    HloModule m
    ENTRY main {
      %keys = f32[8192] parameter(0)
      %values = s32[8192] parameter(1)
      %custom-call = (f32[8192]{0}, s32[8192]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[8192]{0} get-tuple-element(%custom-call), index=0
  })hlo";

  DevicelessEstimateCubSortScratchSize pass("CUDA", "NVIDIA H100 80GB HBM3",
                                            SemanticVersion(3, 1, 2));
  ASSERT_OK_AND_ASSIGN(int64_t pass_scratch_size,
                       RunPassAndExtractScratchSize(hlo, pass));

  std::optional<int64_t> lookup_scratch_size = lookup_table.Lookup(
      SemanticVersion(3, 1, 2),
      /*device_name=*/"NVIDIA H100 80GB HBM3", /*key_type_size=*/4,
      /*value_type_size=*/4, /*num_items=*/8192);
  EXPECT_TRUE(lookup_scratch_size.has_value());

  EXPECT_EQ(pass_scratch_size, *lookup_scratch_size);
}

TEST_F(DevicelessEstimateCubSortScratchSizeTest, Batched) {
  ASSERT_OK_AND_ASSIGN(const CubScratchSizeDevicelessLookup& lookup_table,
                       CubScratchSizeDevicelessLookup::GetInstance());
  absl::string_view hlo = R"hlo(
    HloModule m
    ENTRY main {
      %keys = f32[10,8192] parameter(0)
      %custom-call = (f32[10,8192]{1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[10,8192]{1,0} get-tuple-element(%custom-call), index=0
  })hlo";

  DevicelessEstimateCubSortScratchSize pass("CUDA", "NVIDIA H100 80GB HBM3",
                                            SemanticVersion(3, 1, 2));
  ASSERT_OK_AND_ASSIGN(int64_t pass_scratch_size,
                       RunPassAndExtractScratchSize(hlo, pass));

  std::optional<int64_t> lookup_scratch_size = lookup_table.Lookup(
      SemanticVersion(3, 1, 2),
      /*device_name=*/"NVIDIA H100 80GB HBM3", /*key_type_size=*/4,
      /*value_type_size=*/std::nullopt, /*num_items=*/81920, /*batch_size=*/10);
  EXPECT_TRUE(lookup_scratch_size.has_value());

  EXPECT_EQ(pass_scratch_size, *lookup_scratch_size);
}

}  // namespace
}  // namespace xla::gpu
