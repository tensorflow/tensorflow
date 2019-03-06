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
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/EDSC/MLIREmitter.h"
#include "mlir/EDSC/Types.h"
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

int main() {
  RUN_TESTS();
  return 0;
}
