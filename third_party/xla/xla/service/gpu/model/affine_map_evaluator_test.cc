/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/affine_map_evaluator.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::bindDims;
using ::mlir::bindSymbols;
using ::testing::ElementsAre;

class AffineMapEvaluator : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

TEST_F(AffineMapEvaluator, EvaluateMap) {
  AffineExpr d0, d1, s0, s1;
  bindDims(&mlir_context_, d0, d1);
  bindSymbols(&mlir_context_, s0, s1);

  auto affine_map =
      AffineMap::get(2, 2, {d0 + d1.floorDiv(8), s0 + s1 % 16}, &mlir_context_);

  auto res = EvaluateAffineMap(affine_map, /*dim_values=*/{1, 2},
                               /*symbol_values=*/{3, 4});
  EXPECT_THAT(res, ElementsAre(1, 7));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
