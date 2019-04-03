//===- Conversion.cpp - Linalg to LLVM conversion driver ------------------===//
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

// RUN: %p/conversion | FileCheck %s

#include "TestHarness.h"

#include "linalg1/Common.h"
#include "linalg1/ConvertToLLVMDialect.h"
#include "linalg1/Intrinsics.h"
#include "linalg1/Ops.h"
#include "linalg1/Types.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Function.h"

using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;
using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

TEST_FUNC(rangeConversion) {
  // Define the MLIR context, create a Module in this context, and a Builder to
  // facilitate type construction.
  MLIRContext context;
  Module module(&context);
  Builder builder(&module);

  // Declare a function called "rangeConversion" with type:
  //   (index, index, index) -> ()
  // define it, and add it to the module.
  FunctionType funcType = builder.getFunctionType(
      {builder.getIndexType(), builder.getIndexType(), builder.getIndexType()},
      {});
  Function *f =
      new Function(builder.getUnknownLoc(), "rangeConversion", funcType);
  f->addEntryBlock();
  module.getFunctions().push_back(f);

  // Construct a linalg::RangeOp taking function arguments as operands.
  ScopedContext scope(f);
  ValueHandle arg0(f->getArgument(0)), arg1(f->getArgument(1)),
      arg2(f->getArgument(2));
  {
    range(arg0, arg1, arg2);
    ret();
  }

  // clang-format off
  // CHECK-LABEL: @rangeConversion
  // CHECK-NEXT: %0 = llvm.undef : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm<"{ i64, i64, i64 }">
  // clang-format on
  convertToLLVM(module);
  module.print(llvm::outs());
}

TEST_FUNC(viewRangeConversion) {
  // Define the MLIR context, create a Module in this context, and a Builder to
  // facilitate type construction.
  MLIRContext context;
  Module module(&context);
  Builder builder(&module);

  // Declare a function called "viewRangeConversion" with type:
  //   (memref<?x?xf32>, !linalg.range, !linalg.range) -> ()
  // define it, and add it to the module.
  FunctionType funcType = builder.getFunctionType(
      {builder.getMemRefType({-1, -1}, builder.getF32Type(), {}, 0),
       builder.getType<RangeType>(), builder.getType<RangeType>()},
      {});
  Function *f =
      new Function(builder.getUnknownLoc(), "viewRangeConversion", funcType);
  f->addEntryBlock();
  module.getFunctions().push_back(f);

  // Construct a linalg::ViewOp taking function arguments as operands.
  ScopedContext scope(f);
  ValueHandle memref(f->getArgument(0)), range1(f->getArgument(1)),
      range2(f->getArgument(2));
  {
    view(memref, {range1, range2});
    ret();
  }

  // clang-format off
  // CHECK-LABEL: @viewRangeConversion
  // CHECK-NEXT: %0 = llvm.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %1 = llvm.extractvalue %arg0[0] : !llvm<"{ float*, i64, i64 }">
  // CHECK-NEXT: %2 = llvm.insertvalue %1, %0[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
  // CHECK-NEXT: %4 = llvm.constant(1 : index) : !llvm<"i64">
  // CHECK-NEXT: %5 = llvm.mul %4, %3 : !llvm<"i64">
  // CHECK-NEXT: %6 = llvm.constant(0 : index) : !llvm<"i64">
  // CHECK-NEXT: %7 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %8 = llvm.mul %7, %5 : !llvm<"i64">
  // CHECK-NEXT: %9 = llvm.add %6, %8 : !llvm<"i64">
  // CHECK-NEXT: %10 = llvm.extractvalue %arg2[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %11 = llvm.mul %10, %4 : !llvm<"i64">
  // CHECK-NEXT: %12 = llvm.add %9, %11 : !llvm<"i64">
  // CHECK-NEXT: %13 = llvm.insertvalue %12, %2[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %14 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %15 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %16 = llvm.sub %15, %14 : !llvm<"i64">
  // CHECK-NEXT: %17 = llvm.insertvalue %16, %13[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %18 = llvm.extractvalue %arg2[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %19 = llvm.extractvalue %arg2[1] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %20 = llvm.sub %19, %18 : !llvm<"i64">
  // CHECK-NEXT: %21 = llvm.insertvalue %20, %17[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %22 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %23 = llvm.mul %5, %22 : !llvm<"i64">
  // CHECK-NEXT: %24 = llvm.insertvalue %23, %21[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %25 = llvm.extractvalue %arg2[2] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %26 = llvm.mul %4, %25 : !llvm<"i64">
  // CHECK-NEXT: %27 = llvm.insertvalue %26, %24[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // clang-format on
  convertToLLVM(module);
  module.print(llvm::outs());
}

TEST_FUNC(viewNonRangeConversion) {
  // Define the MLIR context, create a Module in this context, and a Builder to
  // facilitate type construction.
  MLIRContext context;
  Module module(&context);
  Builder builder(&module);

  // Declare a function called "viewNonRangeConversion" with type:
  //   (memref<?x?xf32>, !linalg.range, index) -> ()
  // define it, and add it to the module.
  FunctionType funcType = builder.getFunctionType(
      {builder.getMemRefType({-1, -1}, builder.getF32Type(), {}, 0),
       builder.getType<RangeType>(), builder.getIndexType()},
      {});
  Function *f =
      new Function(builder.getUnknownLoc(), "viewNonRangeConversion", funcType);
  f->addEntryBlock();
  module.getFunctions().push_back(f);

  // Construct a linalg::ViewOp taking function arguments as operands.
  ScopedContext scope(f);
  ValueHandle memref(f->getArgument(0)), range(f->getArgument(1)),
      index(f->getArgument(2));
  {
    view(memref, {range, index});
    ret();
  }

  // clang-format off
  // CHECK-LABEL: @viewNonRangeConversion
  // CHECK-NEXT: %0 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: %1 = llvm.extractvalue %arg0[0] : !llvm<"{ float*, i64, i64 }">
  // CHECK-NEXT: %2 = llvm.insertvalue %1, %0[0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: %3 = llvm.extractvalue %arg0[2] : !llvm<"{ float*, i64, i64 }">
  // CHECK-NEXT: %4 = llvm.constant(1 : index) : !llvm<"i64">
  // CHECK-NEXT: %5 = llvm.mul %4, %3 : !llvm<"i64">
  // CHECK-NEXT: %6 = llvm.constant(0 : index) : !llvm<"i64">
  // CHECK-NEXT: %7 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %8 = llvm.mul %7, %5 : !llvm<"i64">
  // CHECK-NEXT: %9 = llvm.add %6, %8 : !llvm<"i64">
  // CHECK-NEXT: %10 = llvm.mul %arg2, %4 : !llvm<"i64">
  // CHECK-NEXT: %11 = llvm.add %9, %10 : !llvm<"i64">
  // CHECK-NEXT: %12 = llvm.insertvalue %11, %2[1] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: %13 = llvm.extractvalue %arg1[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %14 = llvm.extractvalue %arg1[1] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %15 = llvm.sub %14, %13 : !llvm<"i64">
  // CHECK-NEXT: %16 = llvm.insertvalue %15, %12[2, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // CHECK-NEXT: %17 = llvm.extractvalue %arg1[2] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %18 = llvm.mul %5, %17 : !llvm<"i64">
  // CHECK-NEXT: %19 = llvm.insertvalue %18, %16[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  // clang-format on
  convertToLLVM(module);
  module.print(llvm::outs());
}

TEST_FUNC(sliceRangeConversion) {
  // Define the MLIR context, create a Module in this context, and a Builder to
  // facilitate type construction.
  MLIRContext context;
  Module module(&context);
  Builder builder(&module);

  // Declare a function called "sliceRangeConversion" with type:
  //   (memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.range) -> ()
  // define it, and add it to the module.
  FunctionType funcType = builder.getFunctionType(
      {builder.getMemRefType({-1, -1}, builder.getF32Type(), {}, 0),
       builder.getType<RangeType>(), builder.getType<RangeType>(),
       builder.getType<RangeType>()},
      {});
  Function *f =
      new Function(builder.getUnknownLoc(), "sliceRangeConversion", funcType);
  f->addEntryBlock();
  module.getFunctions().push_back(f);

  // Construct a linalg::SliceOp based on the result of a linalg::ViewOp.
  // Note: SliceOp builder does not support ViewOps that are not defined by
  // a dominating ViewOp.
  ScopedContext scope(f);
  ValueHandle memref(f->getArgument(0)), range1(f->getArgument(1)),
      range2(f->getArgument(2)), range3(f->getArgument(3));
  {
    slice(view(memref, {range1, range2}), range3, 0);
    ret();
  }

  // clang-format off
  // CHECK-LABEL: @sliceRangeConversion
  // CHECK:      %28 = llvm.undef : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %29 = llvm.extractvalue %27[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %30 = llvm.insertvalue %29, %28[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %31 = llvm.extractvalue %27[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %32 = llvm.extractvalue %arg3[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %33 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %34 = llvm.mul %32, %33 : !llvm<"i64">
  // CHECK-NEXT: %35 = llvm.add %31, %34 : !llvm<"i64">
  // CHECK-NEXT: %36 = llvm.insertvalue %35, %30[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %37 = llvm.extractvalue %arg3[1] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %38 = llvm.extractvalue %arg3[0] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %39 = llvm.sub %37, %38 : !llvm<"i64">
  // CHECK-NEXT: %40 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %41 = llvm.extractvalue %arg3[2] : !llvm<"{ i64, i64, i64 }">
  // CHECK-NEXT: %42 = llvm.mul %40, %41 : !llvm<"i64">
  // CHECK-NEXT: %43 = llvm.insertvalue %39, %36[2, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %44 = llvm.insertvalue %42, %43[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %45 = llvm.extractvalue %27[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %46 = llvm.extractvalue %27[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %47 = llvm.insertvalue %45, %44[2, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %48 = llvm.insertvalue %46, %47[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // clang-format on
  convertToLLVM(module);
  module.print(llvm::outs());
}

TEST_FUNC(sliceNonRangeConversion) {
  // Define the MLIR context, create a Module in this context, and a Builder to
  // facilitate type construction.
  MLIRContext context;
  Module module(&context);
  Builder builder(&module);

  // Declare a function called "sliceNonRangeConversion" with type:
  //   (memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.range) -> ()
  // define it, and add it to the module.
  FunctionType funcType = builder.getFunctionType(
      {builder.getMemRefType({-1, -1}, builder.getF32Type(), {}, 0),
       builder.getType<RangeType>(), builder.getType<RangeType>(),
       builder.getIndexType()},
      {});
  Function *f = new Function(builder.getUnknownLoc(), "sliceNonRangeConversion",
                             funcType);
  f->addEntryBlock();
  module.getFunctions().push_back(f);

  // Construct a linalg::SliceOp based on the result of a linalg::ViewOp.
  // Note: SliceOp builder does not support ViewOps that are not defined by
  // a dominating ViewOp.
  ScopedContext scope(f);
  ValueHandle memref(f->getArgument(0)), range1(f->getArgument(1)),
      range2(f->getArgument(2)), index(f->getArgument(3));
  {
    slice(view(memref, {range1, range2}), index, 0);
    ret();
  }

  // CHECK-LABEL: @sliceNonRangeConversion
  // CHECK:      %28 = llvm.undef : !llvm<"{ float*, i64, [1 x i64], [1 x i64]
  // }"> CHECK-NEXT: %29 = llvm.extractvalue %27[0] : !llvm<"{ float*, i64, [2 x
  // i64], [2 x i64] }"> CHECK-NEXT: %30 = llvm.insertvalue %29, %28[0] :
  // !llvm<"{ float*, i64, [1 x i64], [1 x i64] }"> CHECK-NEXT: %31 =
  // llvm.extractvalue %27[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: %32 = llvm.extractvalue %27[3, 0] : !llvm<"{ float*, i64, [2 x
  // i64], [2 x i64] }"> CHECK-NEXT: %33 = llvm.mul %arg3, %32 : !llvm<"i64">
  // CHECK-NEXT: %34 = llvm.add %31, %33 : !llvm<"i64">
  // CHECK-NEXT: %35 = llvm.insertvalue %34, %30[1] : !llvm<"{ float*, i64, [1 x
  // i64], [1 x i64] }"> CHECK-NEXT: %36 = llvm.extractvalue %27[2, 1] :
  // !llvm<"{ float*, i64, [2 x i64], [2 x i64] }"> CHECK-NEXT: %37 =
  // llvm.extractvalue %27[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64]
  // }"> CHECK-NEXT: %38 = llvm.insertvalue %36, %35[2, 0] : !llvm<"{ float*,
  // i64, [1 x i64], [1 x i64] }"> CHECK-NEXT: %39 = llvm.insertvalue %37,
  // %38[3, 0] : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">
  convertToLLVM(module);
  module.print(llvm::outs());
}

int main() {
  RUN_TESTS();
  return 0;
}
