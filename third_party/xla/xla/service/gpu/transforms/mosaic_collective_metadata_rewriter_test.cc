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

#include "xla/service/gpu/transforms/mosaic_collective_metadata_rewriter.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/layout_assignment.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class MosaicCollectiveMetadataRewriterTest
    : public HloHardwareIndependentTestBase {
 public:
  MosaicCollectiveMetadataRewriterTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/true,
            /*allow_mixed_precision_in_hlo_verifier=*/true,
            LayoutAssignment::InstructionCanChangeLayout) {}
  void CheckMosaicCollectiveMetadataMemorySpace(
      absl::string_view hlo, std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, MosaicCollectiveMetadataRewriter{},
                              expected);
  }
};

TEST_F(MosaicCollectiveMetadataRewriterTest, MoveSingleResultToUnifiedMemory) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  ROOT %pallas_call = s64[11]{0} custom-call(), custom_call_target="mosaic_gpu_v2", backend_config="uses_xla_collective_metadata"
}
)";

  CheckMosaicCollectiveMetadataMemorySpace(hlo, R"(
// CHECK: ROOT [[pallas_call:%[^ ]+]] = s64[11]{0:S(3)} custom-call()
)");
}

TEST_F(MosaicCollectiveMetadataRewriterTest, MoveLastResultToUnifiedMemory) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  ROOT %pallas_call = (s64[11]{0}, s64[11]{0}) custom-call(), custom_call_target="mosaic_gpu_v2", backend_config="uses_xla_collective_metadata"
}
)";

  CheckMosaicCollectiveMetadataMemorySpace(hlo, R"(
// CHECK: ROOT [[pallas_call:%[^ ]+]] = (s64[11]{0}, s64[11]{0:S(3)}) custom-call()
)");
}

TEST_F(MosaicCollectiveMetadataRewriterTest, SkipNonMosaicCustomCall) {
  const char* kHloString = R"(
HloModule module

ENTRY main {
  ROOT %pallas_call = s64[11]{0} custom-call(), custom_call_target="non_mosaic_gpu_v2", backend_config="uses_xla_collective_metadata"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  MosaicCollectiveMetadataRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed) << "Non-mosaic custom call should not be rewritten";
}

TEST_F(MosaicCollectiveMetadataRewriterTest,
       SkipMosaicCustomCallWithoutBackendConfig) {
  const char* kHloString = R"(
HloModule module

ENTRY main {
  ROOT %pallas_call = (s64[11]{0}, s64[11]{0}) custom-call(), custom_call_target="mosaic_gpu_v2"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  MosaicCollectiveMetadataRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed)
      << "Mosaic custom call without backend config should not be rewritten";
}

TEST_F(MosaicCollectiveMetadataRewriterTest,
       SkipMosaicCustomCallWithoutCollectiveMetadata) {
  const char* kHloString = R"(
HloModule module

ENTRY main {
  ROOT %pallas_call = (s64[11]{0}, s64[11]{0}) custom-call(), custom_call_target="mosaic_gpu_v2", backend_config="some_other_config"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  MosaicCollectiveMetadataRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed)
      << "Mosaic custom call without collective metadata should not be"
         " rewritten";
}

}  // namespace
}  // namespace xla
