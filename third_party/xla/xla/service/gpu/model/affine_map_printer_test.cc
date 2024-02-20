/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/affine_map_printer.h"

#include <gmock/gmock.h>
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::bindDims;
using ::mlir::bindSymbols;
using ::testing::HasSubstr;

class IndexingMapTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter printer_;
};

TEST_F(IndexingMapTest, AffineMapPrinterTest) {
  AffineExpr d0, d1, s0, s1;
  bindDims(&mlir_context_, d0, d1);
  bindSymbols(&mlir_context_, s0, s1);

  // (d0, d1)[s0, s1] -> (d0 + d1 floordiv 8, s0 + s1 mod 16).
  auto map =
      AffineMap::get(2, 2, {d0 + d1.floorDiv(8), s0 + s1 % 16}, &mlir_context_);

  printer_.SetDimensionName(0, "offset");
  printer_.SetSymbolName(1, "linear_index");
  EXPECT_THAT(printer_.ToString(map),
              HasSubstr("(offset, d1)[s0, linear_index] -> "
                        "(offset + d1 floordiv 8, s0 + linear_index mod 16)"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
