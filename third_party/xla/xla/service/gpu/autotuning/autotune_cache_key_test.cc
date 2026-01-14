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

#include "xla/service/gpu/autotuning/autotune_cache_key.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {
namespace {
using testing::HasSubstr;

constexpr absl::string_view kDotFusionHloText = R"hlo(
    HloModule module
    fused_computation {
          tmp_0 = f16[1,16,17,3]{3,2,1,0} parameter(0) 
          tmp_1 = f16[16,51]{1,0} bitcast(f16[1,16,17,3]{3,2,1,0} tmp_0)
          tmp_2 = s8[16,17,3]{2,1,0} parameter(1)
          tmp_3 = s8[51,16]{0,1} bitcast(s8[16,17,3]{2,1,0} tmp_2)
          tmp_4 = f16[51,16]{0,1} convert(s8[51,16]{0,1} tmp_3)
          tmp_5 = f16[16,16]{1,0} dot(f16[16,51]{1,0} tmp_1, f16[51,16]{0,1} tmp_4), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          ROOT tmp_6 = f16[1,16,16]{2,1,0} bitcast(f16[16,16]{1,0} tmp_5)
    }
    
    ENTRY main {
          p0 = f16[1,16,17,3]{3,2,1,0} parameter(0) 
          p1 = s8[16,17,3]{2,1,0} parameter(1)
          ROOT fusion = f16[1,16,16]{2,1,0} fusion(p0, p1), kind=kCustom, calls=fused_computation
    }
  )hlo";

TEST(AutotuneCacheKeyTest, DeviceDescriptionToCacheKey) {
  auto device_description =
      [](absl::string_view spec_file_name) -> se::DeviceDescription {
    se::GpuTargetConfigProto proto;
    std::string spec_string;
    CHECK_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                          "backends/gpu/target_config/specs", spec_file_name),
        &spec_string));
    EXPECT_TRUE(
        tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto));
    absl::StatusOr<se::DeviceDescription> device_description =
        se::DeviceDescription::FromProto(proto.gpu_device_info());
    CHECK_OK(device_description.status());
    return *device_description;
  };

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("a100_sxm_40.txtpb")),
            "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: "
            "1555 GB/s, L2 cache: 40 MB, DNN version: 0.0.0");

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("a100_sxm_80.txtpb")),
            "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: "
            "2039 GB/s, L2 cache: 40 MB, DNN version: 0.0.0");

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("mi200.txtpb")),
            "ROCM: gfx90a, Cores: 110, GPU clock: 1.7 GHz, Memory bandwidth: "
            "1638 GB/s, L2 cache: 8 MB, DNN version: 0.0.0");
}

TEST(AutotuneCacheKeyTest, VersionIsIncludedInCacheKey) {
  stream_executor::DeviceDescription empty_device_description;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kDotFusionHloText));
  AutotuneCacheKey key =
      AutotuneCacheKey(empty_device_description,
                       *module->entry_computation()->root_instruction());
  EXPECT_THAT(key.ToString(),
              HasSubstr(absl::StrFormat("version=%d", key.GetVersion())));
}

TEST(AutotuneCacheKeyTest, VersionChangeInvalidateCacheKey) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kDotFusionHloText));
  stream_executor::DeviceDescription empty_device_description;

  AutotuneCacheKey key0 = AutotuneCacheKey(
      empty_device_description,
      *module->entry_computation()->root_instruction(), /*version=*/0);
  AutotuneCacheKey key1 = AutotuneCacheKey(
      empty_device_description,
      *module->entry_computation()->root_instruction(), /*version=*/1);
  EXPECT_FALSE(key0 == key1);
  EXPECT_NE(key0.ToString(), key1.ToString());
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      key0,
      key1,
  }));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
