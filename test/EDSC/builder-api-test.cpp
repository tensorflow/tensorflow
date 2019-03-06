//===- builder-api-test.cpp - Tests for Declarative Builder APIs
//-----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// RUN: %p/builder-api-test | FileCheck %s

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LoopUtils.h"

#include "Test.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static MLIRContext &globalContext() {
  static thread_local MLIRContext context;
  return context;
}

static std::unique_ptr<Function> makeFunction(StringRef name,
                                              ArrayRef<Type> results = {},
                                              ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = llvm::make_unique<Function>(
      UnknownLoc::get(&ctx), name, FunctionType::get(args, results, &ctx));
  function->addEntryBlock();
  return function;
}

TEST_FUNC(builder_dynamic_for_func_args) {
  using namespace edsc;
  using namespace edsc::op;
  using namespace edsc::intrinsics;
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto f =
      makeFunction("builder_dynamic_for_func_args", {}, {indexType, indexType});

  ScopedContext scope(f.get());
  ValueHandle i(indexType), j(indexType), lb(f->getArgument(0)),
      ub(f->getArgument(1));
  ValueHandle f7(
      ValueHandle::create<ConstantFloatOp>(llvm::APFloat(7.0f), f32Type));
  ValueHandle f13(
      ValueHandle::create<ConstantFloatOp>(llvm::APFloat(13.0f), f32Type));
  ValueHandle i7(ValueHandle::create<ConstantIntOp>(7, 32));
  ValueHandle i13(ValueHandle::create<ConstantIntOp>(13, 32));
  LoopBuilder(&i, lb, ub, 3)({
      lb * index_t(3) + ub,
      lb + index_t(3),
      LoopBuilder(&j, lb, ub, 2)({
          ceilDiv(index_t(31) * floorDiv(i + j * index_t(3), index_t(32)),
                  index_t(32)),
          ((f7 + f13) / f7) % f13 - f7 * f13,
          ((i7 + i13) / i7) % i13 - i7 * i13,
      }),
  });

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for_func_args(%arg0: index, %arg1: index) {
  // CHECK:  for %i0 = (d0) -> (d0)(%arg0) to (d0) -> (d0)(%arg1) step 3 {
  // CHECK:  {{.*}} = affine.apply (d0) -> (d0 * 3)(%arg0)
  // CHECK:  {{.*}} = affine.apply (d0, d1) -> (d0 * 3 + d1)(%arg0, %arg1)
  // CHECK:  {{.*}} = affine.apply (d0) -> (d0 + 3)(%arg0)
  // CHECK:  for %i1 = (d0) -> (d0)(%arg0) to (d0) -> (d0)(%arg1) step 2 {
  // CHECK:    {{.*}} = affine.apply (d0, d1) -> ((d0 + d1 * 3) floordiv 32)(%i0, %i1)
  // CHECK:    {{.*}} = affine.apply (d0, d1) -> (((d0 + d1 * 3) floordiv 32) * 31)(%i0, %i1)
  // CHECK:    {{.*}} = affine.apply (d0, d1) -> ((((d0 + d1 * 3) floordiv 32) * 31) ceildiv 32)(%i0, %i1)
  // CHECK:    [[rf1:%[0-9]+]] = addf {{.*}}, {{.*}} : f32
  // CHECK:    [[rf2:%[0-9]+]] = divf [[rf1]], {{.*}} : f32
  // CHECK:    [[rf3:%[0-9]+]] = remf [[rf2]], {{.*}} : f32
  // CHECK:    [[rf4:%[0-9]+]] = mulf {{.*}}, {{.*}} : f32
  // CHECK:    {{.*}} = subf [[rf3]], [[rf4]] : f32
  // CHECK:    [[ri1:%[0-9]+]] = addi {{.*}}, {{.*}} : i32
  // CHECK:    [[ri2:%[0-9]+]] = divis [[ri1]], {{.*}} : i32
  // CHECK:    [[ri3:%[0-9]+]] = remis [[ri2]], {{.*}} : i32
  // CHECK:    [[ri4:%[0-9]+]] = muli {{.*}}, {{.*}} : i32
  // CHECK:    {{.*}} = subi [[ri3]], [[ri4]] : i32
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_dynamic_for) {
  using namespace edsc;
  using namespace edsc::op;
  using namespace edsc::intrinsics;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_dynamic_for", {},
                        {indexType, indexType, indexType, indexType});

  ScopedContext scope(f.get());
  ValueHandle i(indexType), a(f->getArgument(0)), b(f->getArgument(1)),
      c(f->getArgument(2)), d(f->getArgument(3));
  LoopBuilder(&i, a - b, c + d, 2)({});

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK:        %0 = affine.apply (d0, d1) -> (d0 - d1)(%arg0, %arg1)
  // CHECK-NEXT:   %1 = affine.apply (d0, d1) -> (d0 + d1)(%arg2, %arg3)
  // CHECK-NEXT:   for %i0 = (d0) -> (d0)(%0) to (d0) -> (d0)(%1) step 2 {
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_max_min_for) {
  using namespace edsc;
  using namespace edsc::op;
  using namespace edsc::intrinsics;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_max_min_for", {},
                        {indexType, indexType, indexType, indexType});

  ScopedContext scope(f.get());
  ValueHandle i(indexType), lb1(f->getArgument(0)), lb2(f->getArgument(1)),
      ub1(f->getArgument(2)), ub2(f->getArgument(3));
  LoopBuilder(&i, {lb1, lb2}, {ub1, ub2}, 1)({});
  RETURN({});

  // clang-format off
  // CHECK-LABEL: func @builder_max_min_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK:  for %i0 = max (d0, d1) -> (d0, d1)(%arg0, %arg1) to min (d0, d1) -> (d0, d1)(%arg2, %arg3) {
  // CHECK:  return
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_blocks) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks");

  ScopedContext scope(f.get());
  ValueHandle c1(ValueHandle::create<ConstantIntOp>(42, 32)),
      c2(ValueHandle::create<ConstantIntOp>(1234, 32));
  ValueHandle arg1(c1.getType()), arg2(c1.getType()), arg3(c1.getType()),
      arg4(c1.getType()), r(c1.getType());

  BlockHandle b1, b2, functionBlock(&f->front());
  BlockBuilder(&b1, {&arg1, &arg2})({
      // b2 has not yet been constructed, need to come back later.
      // This is a byproduct of non-structured control-flow.
  });
  BlockBuilder(&b2, {&arg3, &arg4})({
      BR(b1, {arg3, arg4}),
  });
  // The insertion point within the toplevel function is now past b2, we will
  // need to get back the entry block.
  // This is what happens with unstructured control-flow..
  BlockBuilder(b1, Append())({
      r = arg1 + arg2,
      BR(b2, {arg1, r}),
  });
  // Get back to entry block and add a branch into b1
  BlockBuilder(functionBlock, Append())({
      BR(b1, {c1, c2}),
  });

  // clang-format off
  // CHECK-LABEL: @builder_blocks
  // CHECK:        %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   %c1234_i32 = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%c42_i32, %c1234_i32 : i32, i32)
  // CHECK-NEXT: ^bb1(%0: i32, %1: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   %2 = addi %0, %1 : i32
  // CHECK-NEXT:   br ^bb2(%0, %2 : i32, i32)
  // CHECK-NEXT: ^bb2(%3: i32, %4: i32):   // pred: ^bb1
  // CHECK-NEXT:   br ^bb1(%3, %4 : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_blocks_eager) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks_eager");

  ScopedContext scope(f.get());
  ValueHandle c1(ValueHandle::create<ConstantIntOp>(42, 32)),
      c2(ValueHandle::create<ConstantIntOp>(1234, 32));
  ValueHandle arg1(c1.getType()), arg2(c1.getType()), arg3(c1.getType()),
      arg4(c1.getType()), r(c1.getType());

  // clang-format off
  BlockHandle b1, b2;
  { // Toplevel function scope.
    BR(&b1, {&arg1, &arg2}, {c1, c2}); // eagerly builds a new block for b1
    // We cannot construct b2 eagerly with a `BR(&b2, ...)` call from within b1
    // because it would result in b2 being nested under b1 which is not what we
    // want in this test.
    BlockBuilder(&b2, {&arg3, &arg4})({
        // Instead, construct explicitly
        BR(b1, {arg3, arg4}),
    });
    /// And come back to append into b1 once b2 exists.
    BlockBuilder(b1, Append())({
        r = arg1 + arg2,
        BR(b2, {arg1, r}),
    });
  }

  // CHECK-LABEL: @builder_blocks_eager
  // CHECK:        %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   %c1234_i32 = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%c42_i32, %c1234_i32 : i32, i32)
  // CHECK-NEXT: ^bb1(%0: i32, %1: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   %2 = addi %0, %1 : i32
  // CHECK-NEXT:   br ^bb2(%0, %2 : i32, i32)
  // CHECK-NEXT: ^bb2(%3: i32, %4: i32):   // pred: ^bb1
  // CHECK-NEXT:   br ^bb1(%3, %4 : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_cond_branch) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  auto f = makeFunction("builder_cond_branch", {},
                        {IntegerType::get(1, &globalContext())});

  ScopedContext scope(f.get());
  ValueHandle funcArg(f->getArgument(0));
  ValueHandle c32(ValueHandle::create<ConstantIntOp>(32, 32)),
      c64(ValueHandle::create<ConstantIntOp>(64, 64)),
      c42(ValueHandle::create<ConstantIntOp>(42, 32));
  ValueHandle arg1(c32.getType()), arg2(c64.getType()), arg3(c32.getType());

  BlockHandle b1, b2, functionBlock(&f->front());
  BlockBuilder(&b1, {&arg1})({
      RETURN({}),
  });
  BlockBuilder(&b2, {&arg2, &arg3})({
      RETURN({}),
  });
  // Get back to entry block and add a conditional branch
  BlockBuilder(functionBlock, Append())({
      COND_BR(funcArg, b1, {c32}, b2, {c64, c42}),
  });

  // clang-format off
  // CHECK-LABEL: @builder_cond_branch
  // CHECK:   %c32_i32 = constant 32 : i32
  // CHECK-NEXT:   %c64_i64 = constant 64 : i64
  // CHECK-NEXT:   %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   cond_br %arg0, ^bb1(%c32_i32 : i32), ^bb2(%c64_i64, %c42_i32 : i64, i32)
  // CHECK-NEXT: ^bb1(%0: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%1: i64, %2: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_cond_branch_eager) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto f = makeFunction("builder_cond_branch_eager", {},
                        {IntegerType::get(1, &globalContext())});

  ScopedContext scope(f.get());
  ValueHandle funcArg(f->getArgument(0));
  ValueHandle c32(ValueHandle::create<ConstantIntOp>(32, 32)),
      c64(ValueHandle::create<ConstantIntOp>(64, 64)),
      c42(ValueHandle::create<ConstantIntOp>(42, 32));
  ValueHandle arg1(c32.getType()), arg2(c64.getType()), arg3(c32.getType());

  // clang-format off
  BlockHandle b1, b2;
  COND_BR(funcArg, &b1, {&arg1}, {c32}, &b2, {&arg2, &arg3}, {c64, c42});
  BlockBuilder(b1, Append())({
      RETURN({}),
  });
  BlockBuilder(b2, Append())({
      RETURN({}),
  });

  // CHECK-LABEL: @builder_cond_branch_eager
  // CHECK:   %c32_i32 = constant 32 : i32
  // CHECK-NEXT:   %c64_i64 = constant 64 : i64
  // CHECK-NEXT:   %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   cond_br %arg0, ^bb1(%c32_i32 : i32), ^bb2(%c64_i64, %c42_i32 : i64, i32)
  // CHECK-NEXT: ^bb1(%0: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%1: i64, %2: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(builder_helpers) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get({-1, -1, -1}, f32Type, {}, 0);
  auto f =
      makeFunction("builder_helpers", {}, {memrefType, memrefType, memrefType});

  ScopedContext scope(f.get());
  // clang-format off
  ValueHandle f7(
      ValueHandle::create<ConstantFloatOp>(llvm::APFloat(7.0f), f32Type));
  MemRefView vA(f->getArgument(0)), vB(f->getArgument(1)), vC(f->getArgument(2));
  IndexedValue A(vA), B(vB), C(vC);
  IndexHandle i, j, k1, k2, lb0, lb1, lb2, ub0, ub1, ub2;
  int64_t step0, step1, step2;
  std::tie(lb0, ub0, step0) = vA.range(0);
  std::tie(lb1, ub1, step1) = vA.range(1);
  std::tie(lb2, ub2, step2) = vA.range(2);
  LoopNestBuilder({&i, &j}, {lb0, lb1}, {ub0, ub1}, {step0, step1})({
    LoopBuilder(&k1, lb2, ub2, step2)({
      C({i, j, k1}) = f7 + A({i, j, k1}) + B({i, j, k1}),
    }),
    LoopBuilder(&k2, lb2, ub2, step2)({
      C({i, j, k2}) += A({i, j, k2}) + B({i, j, k2}),
    }),
  });

  // CHECK-LABEL: @builder_helpers
  //      CHECK:   for %i0 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:     for %i1 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:       for %i2 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:         [[a:%.*]] = load %arg0[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[b:%.*]] = addf {{.*}}, [[a]] : f32
  // CHECK-NEXT:         [[c:%.*]] = load %arg1[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[d:%.*]] = addf [[b]], [[c]] : f32
  // CHECK-NEXT:         store [[d]], %arg2[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       for %i3 = (d0) -> (d0)(%c0_1) to (d0) -> (d0)(%2) {
  // CHECK-NEXT:         [[a:%.*]] = load %arg1[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[b:%.*]] = load %arg0[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[c:%.*]] = addf [[b]], [[a]] : f32
  // CHECK-NEXT:         [[d:%.*]] = load %arg2[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[e:%.*]] = addf [[d]], [[c]] : f32
  // CHECK-NEXT:         store [[e]], %arg2[%i0, %i1, %i3] : memref<?x?x?xf32>
  // clang-format on
  f->print(llvm::outs());
}

int main() {
  RUN_TESTS();
  return 0;
}
