//===- builder-api-test.cpp - Tests for Declarative Builder APIs ----------===//
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
  ValueHandle f7(constant_float(llvm::APFloat(7.0f), f32Type));
  ValueHandle f13(constant_float(llvm::APFloat(13.0f), f32Type));
  ValueHandle i7(constant_int(7, 32));
  ValueHandle i13(constant_int(13, 32));
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
  // CHECK:  affine.for %i0 = (d0) -> (d0)(%arg0) to (d0) -> (d0)(%arg1) step 3 {
  // CHECK:  {{.*}} = affine.apply ()[s0] -> (s0 * 3)()[%arg0]
  // CHECK:  {{.*}} = affine.apply ()[s0, s1] -> (s1 + s0 * 3)()[%arg0, %arg1]
  // CHECK:  {{.*}} = affine.apply ()[s0] -> (s0 + 3)()[%arg0]
  // CHECK:  affine.for %i1 = (d0) -> (d0)(%arg0) to (d0) -> (d0)(%arg1) step 2 {
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
  // CHECK:        %0 = affine.apply ()[s0, s1] -> (s0 - s1)()[%arg0, %arg1]
  // CHECK-NEXT:   %1 = affine.apply ()[s0, s1] -> (s0 + s1)()[%arg2, %arg3]
  // CHECK-NEXT:   affine.for %i0 = (d0) -> (d0)(%0) to (d0) -> (d0)(%1) step 2 {
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
  ret();

  // clang-format off
  // CHECK-LABEL: func @builder_max_min_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK:  affine.for %i0 = max (d0, d1) -> (d0, d1)(%arg0, %arg1) to min (d0, d1) -> (d0, d1)(%arg2, %arg3) {
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
      br(b1, {arg3, arg4}),
  });
  // The insertion point within the toplevel function is now past b2, we will
  // need to get back the entry block.
  // This is what happens with unstructured control-flow..
  BlockBuilder(b1, Append())({
      r = arg1 + arg2,
      br(b2, {arg1, r}),
  });
  // Get back to entry block and add a branch into b1
  BlockBuilder(functionBlock, Append())({
      br(b1, {c1, c2}),
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
    // Build a new block for b1 eagerly.
    br(&b1, {&arg1, &arg2}, {c1, c2});
    // Construct a new block b2 explicitly with a branch into b1.
    BlockBuilder(&b2, {&arg3, &arg4})({
        br(b1, {arg3, arg4}),
    });
    /// And come back to append into b1 once b2 exists.
    BlockBuilder(b1, Append())({
        r = arg1 + arg2,
        br(b2, {arg1, r}),
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
      ret(),
  });
  BlockBuilder(&b2, {&arg2, &arg3})({
      ret(),
  });
  // Get back to entry block and add a conditional branch
  BlockBuilder(functionBlock, Append())({
      cond_br(funcArg, b1, {c32}, b2, {c64, c42}),
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
  cond_br(funcArg, &b1, {&arg1}, {c32}, &b2, {&arg2, &arg3}, {c64, c42});
  BlockBuilder(b1, Append())({
      ret(),
  });
  BlockBuilder(b2, Append())({
      ret(),
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
  MemRefView vA(f->getArgument(0)), vB(f->getArgument(1)),
      vC(f->getArgument(2));
  IndexedValue A(f->getArgument(0)), B(f->getArgument(1)), C(f->getArgument(2));
  IndexHandle i, j, k1, k2, lb0, lb1, lb2, ub0, ub1, ub2;
  int64_t step0, step1, step2;
  std::tie(lb0, ub0, step0) = vA.range(0);
  std::tie(lb1, ub1, step1) = vA.range(1);
  lb2 = vA.lb(2);
  ub2 = vA.ub(2);
  step2 = vA.step(2);
  LoopNestBuilder({&i, &j}, {lb0, lb1}, {ub0, ub1}, {step0, step1})({
    LoopBuilder(&k1, lb2, ub2, step2)({
      C(i, j, k1) = f7 + A(i, j, k1) + B(i, j, k1),
    }),
    LoopBuilder(&k2, lb2, ub2, step2)({
      C(i, j, k2) += A(i, j, k2) + B(i, j, k2),
    }),
  });

  // CHECK-LABEL: @builder_helpers
  //      CHECK:   affine.for %i0 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:     affine.for %i1 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:       affine.for %i2 = (d0) -> (d0)({{.*}}) to (d0) -> (d0)({{.*}}) {
  // CHECK-NEXT:         [[a:%.*]] = load %arg0[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[b:%.*]] = addf {{.*}}, [[a]] : f32
  // CHECK-NEXT:         [[c:%.*]] = load %arg1[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[d:%.*]] = addf [[b]], [[c]] : f32
  // CHECK-NEXT:         store [[d]], %arg2[%i0, %i1, %i2] : memref<?x?x?xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       affine.for %i3 = (d0) -> (d0)(%c0_1) to (d0) -> (d0)(%2) {
  // CHECK-NEXT:         [[a:%.*]] = load %arg1[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[b:%.*]] = load %arg0[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[c:%.*]] = addf [[b]], [[a]] : f32
  // CHECK-NEXT:         [[d:%.*]] = load %arg2[%i0, %i1, %i3] : memref<?x?x?xf32>
  // CHECK-NEXT:         [[e:%.*]] = addf [[d]], [[c]] : f32
  // CHECK-NEXT:         store [[e]], %arg2[%i0, %i1, %i3] : memref<?x?x?xf32>
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(custom_ops) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("custom_ops", {}, {indexType, indexType});

  ScopedContext scope(f.get());
  CustomInstruction<ValueHandle> MY_CUSTOM_OP("my_custom_op");
  CustomInstruction<InstructionHandle> MY_CUSTOM_INST_0("my_custom_inst_0");
  CustomInstruction<InstructionHandle> MY_CUSTOM_INST_2("my_custom_inst_2");

  // clang-format off
  ValueHandle vh(indexType), vh20(indexType), vh21(indexType);
  InstructionHandle ih0, ih2;
  IndexHandle m, n, M(f->getArgument(0)), N(f->getArgument(1));
  IndexHandle ten(index_t(10)), twenty(index_t(20));
  LoopNestBuilder({&m, &n}, {M, N}, {M + ten, N + twenty}, {1, 1})({
    vh = MY_CUSTOM_OP({m, m + n}, {indexType}, {}),
    ih0 = MY_CUSTOM_INST_0({m, m + n}, {}),
    ih2 = MY_CUSTOM_INST_2({m, m + n}, {indexType, indexType}),
    // These captures are verbose for now, can improve when used in practice.
    vh20 = ValueHandle(ih2.getOperation()->getResult(0)),
    vh21 = ValueHandle(ih2.getOperation()->getResult(1)),
    MY_CUSTOM_OP({vh20, vh21}, {indexType}, {}),
  });

  // CHECK-LABEL: @custom_ops
  // CHECK: affine.for %i0 {{.*}}
  // CHECK:   affine.for %i1 {{.*}}
  // CHECK:     {{.*}} = "my_custom_op"{{.*}} : (index, index) -> index
  // CHECK:     "my_custom_inst_0"{{.*}} : (index, index) -> ()
  // CHECK:     [[TWO:%[a-z0-9]+]] = "my_custom_inst_2"{{.*}} : (index, index) -> (index, index)
  // CHECK:     {{.*}} = "my_custom_op"([[TWO]]#0, [[TWO]]#1) : (index, index) -> index
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(insertion_in_block) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("insertion_in_block", {}, {indexType, indexType});
  ScopedContext scope(f.get());
  BlockHandle b1;
  // clang-format off
  ValueHandle::create<ConstantIntOp>(0, 32);
  BlockBuilder(&b1, {})({
    ValueHandle::create<ConstantIntOp>(1, 32)
  });
  ValueHandle::create<ConstantIntOp>(2, 32);
  // CHECK-LABEL: @insertion_in_block
  // CHECK: {{.*}} = constant 0 : i32
  // CHECK: {{.*}} = constant 2 : i32
  // CHECK: ^bb1:   // no predecessors
  // CHECK: {{.*}} = constant 1 : i32
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(select_op) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  using namespace edsc::op;
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get({-1, -1, -1}, f32Type, {}, 0);
  auto f = makeFunction("select_op", {}, {memrefType});

  ScopedContext scope(f.get());
  // clang-format off
  ValueHandle zero = constant_index(0), one = constant_index(1);
  MemRefView vA(f->getArgument(0));
  IndexedValue A(f->getArgument(0));
  IndexHandle i, j;
  LoopNestBuilder({&i, &j}, {zero, zero}, {one, one}, {1, 1})({
    // This test exercises IndexedValue::operator Value*.
    // Without it, one must force conversion to ValueHandle as such:
    //   edsc::intrinsics::select(
    //      i == zero, ValueHandle(A(zero, zero)), ValueHandle(ValueA(i, j)))
    edsc::intrinsics::select(i == zero, A(zero, zero), A(i, j))
  });

  // CHECK-LABEL: @select_op
  //      CHECK: affine.for %i0 = 0 to 1 {
  // CHECK-NEXT:   affine.for %i1 = 0 to 1 {
  // CHECK-NEXT:     {{.*}} = cmpi "eq"
  // CHECK-NEXT:     {{.*}} = load
  // CHECK-NEXT:     {{.*}} = load
  // CHECK-NEXT:     {{.*}} = select
  // clang-format on
  f->print(llvm::outs());
}

int main() {
  RUN_TESTS();
  return 0;
}
