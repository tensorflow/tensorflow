/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/experimental/symbolic_tiled_hlo_computation.h"

#include <memory>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/experimental/test_utils.h"
#include "xla/util.h"

namespace xla::gpu::experimental {
namespace {

class SymbolicTileAnalysisTest : public HloHardwareIndependentTestBase {
 public:
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(SymbolicTileAnalysisTest, SimpleNormalizationDiamondIsSupported) {
  HloInstruction* root = ParseAndGetRoot(R"(
    max {
      p1 = f32[] parameter(1)
      p0 = f32[] parameter(0)
      ROOT m = f32[] maximum(p0, p1)
    }

    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      constant = f32[] constant(-inf)
      reduce = f32[2] reduce(p0, constant), dimensions={1}, to_apply=max
      broadcast = f32[2,97]{1,0} broadcast(reduce), dimensions={0}
      ROOT subtract = f32[2,97]{1,0} subtract(p0, broadcast)
    }

    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      ROOT fusion = f32[2,97]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto tiled_computation_or =
      SymbolicTiledComputation::Tile(*fusion_adaptor, &mlir_context_);
  ASSERT_TRUE(
      std::holds_alternative<SymbolicTiledComputation>(tiled_computation_or));
  auto tiled_computation =
      std::get<SymbolicTiledComputation>(std::move(tiled_computation_or));

  EXPECT_THAT(tiled_computation, MatchString(R"(
    Dimensions:
    0 type: parallel size: 2 dim ID:0
      hlo: %subtract = f32[2,97]{1,0} subtract(%p0.1, %broadcast)
    1 type: parallel size: 97 dim ID:1
      hlo: %subtract = f32[2,97]{1,0} subtract(%p0.1, %broadcast)
    2 type: sequential size: 97 dim ID:1
      hlo: %reduce = f32[2]{0} reduce(%p0.1, %constant), dimensions={1},
                     to_apply=%max

    Root tiles:
    0 root tile:  offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
                  strides [1, 1] upper bounds [2, 97]

    Tiled HLO:
      p0.2.tile_0 = parameter() offsets [tid_0 * ts_0, tid_1 * ts_1]
        sizes [ts_0, ts_1] strides [1, 1] upper bounds [2, 97]
      p0.2.tile_1 = parameter() offsets [tid_0 * ts_0, tid_2 * ts_2]
        sizes [ts_0, ts_2] strides [1, 1] upper bounds [2, 97]
      constant.tile_0 = constant() offsets [] sizes [] strides []
        upper bounds []
      reduce.tile_0 = reduce(p0.2.tile_1, constant.tile_0)
        offsets [tid_0 * ts_0] sizes [ts_0] strides [1] upper bounds [2]
      broadcast.tile_0 = broadcast(reduce.tile_0)
        offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
        strides [1, 1] upper bounds [2, 97]
      subtract.tile_0 = subtract(p0.2.tile_0, broadcast.tile_0)
        offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
        strides [1, 1] upper bounds [2, 97]
  )"));
}

// TODO(b/422676780): Port the remaining tests.

}  // namespace
}  // namespace xla::gpu::experimental
