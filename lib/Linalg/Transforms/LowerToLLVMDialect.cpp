//===- LowerToLLVMDialect.cpp - conversion from Linalg to LLVM dialect ----===//
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

#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/LLVMLowering.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::LLVM;

using undef = ValueBuilder<mlir::LLVM::UndefOp>;
using insertvalue = ValueBuilder<mlir::LLVM::InsertValueOp>;
using extractvalue = ValueBuilder<mlir::LLVM::ExtractValueOp>;
using constant = ValueBuilder<mlir::LLVM::ConstantOp>;
using add = ValueBuilder<mlir::LLVM::AddOp>;
using sub = ValueBuilder<mlir::LLVM::SubOp>;
using mul = ValueBuilder<mlir::LLVM::MulOp>;

template <typename T>
static llvm::Type *getPtrToElementType(T containerType,
                                       LLVMLowering &lowering) {
  return lowering.convertType(containerType.getElementType())
      .template cast<LLVMType>()
      .getUnderlyingType()
      ->getPointerTo();
}

// Convert the given type to the LLVM IR Dialect type.  The following
// conversions are supported:
//   - an Index type is converted into an LLVM integer type with pointer
//     bitwidth (analogous to intptr_t in C);
//   - an Integer type is converted into an LLVM integer type of the same width;
//   - an F32 type is converted into an LLVM float type
//   - a Buffer, Range or View is converted into an LLVM structure type
//     containing the respective dynamic values.
static Type convertLinalgType(Type t, LLVMLowering &lowering) {
  auto *context = t.getContext();
  auto *int64Ty = lowering.convertType(IntegerType::get(64, context))
                      .cast<LLVM::LLVMType>()
                      .getUnderlyingType();

  // A buffer descriptor contains the pointer to a flat region of storage and
  // the size of the region.
  //
  // template <typename Elem, size_t Rank>
  // struct {
  //   Elem *ptr;
  //   int64_t size;
  // };
  if (auto bufferTy = t.dyn_cast<BufferType>()) {
    auto *ptrTy = getPtrToElementType(bufferTy, lowering);
    auto *structTy = llvm::StructType::get(ptrTy, int64Ty);
    return LLVMType::get(context, structTy);
  }

  // Range descriptor contains the range bounds and the step as 64-bit integers.
  //
  // struct {
  //   int64_t min;
  //   int64_t max;
  //   int64_t step;
  // };
  if (auto rangeTy = t.dyn_cast<RangeType>()) {
    auto *structTy = llvm::StructType::get(int64Ty, int64Ty, int64Ty);
    return LLVMType::get(context, structTy);
  }

  // View descriptor contains the pointer to the data buffer, followed by a
  // 64-bit integer containing the distance between the beginning of the buffer
  // and the first element to be accessed through the view, followed by two
  // arrays, each containing as many 64-bit integers as the rank of the View.
  // The first array represents the size, in number of original elements, of the
  // view along the given dimension.  When taking the view, the size is the
  // difference between the upper and the lower bound of the range.  The second
  // array represents the "stride" (in tensor abstraction sense), i.e. the
  // number of consecutive elements of the underlying buffer that separate two
  // consecutive elements addressable through the view along the given
  // dimension.  When taking the view, the strides are constructed as products
  // of the original sizes along the trailing dimensions, multiplied by the view
  // step.  For example, a view of a MxN memref with ranges {0:M:1}, {0:N:1},
  // i.e. the view of a complete memref, will have strides N and 1.  A view with
  // ranges {0:M:2}, {0:N:3} will have strides 2*N and 3.
  //
  // template <typename Elem, size_t Rank>
  // struct {
  //   Elem *ptr;
  //   int64_t offset;
  //   int64_t sizes[Rank];
  //   int64_t strides[Rank];
  // };
  if (auto viewTy = t.dyn_cast<ViewType>()) {
    auto *ptrTy = getPtrToElementType(viewTy, lowering);
    auto *arrayTy = llvm::ArrayType::get(int64Ty, viewTy.getRank());
    auto *structTy = llvm::StructType::get(ptrTy, int64Ty, arrayTy, arrayTy);
    return LLVMType::get(context, structTy);
  }

  return Type();
}

// Create an array attribute containing integer attributes with values provided
// in `position`.
static ArrayAttr makePositionAttr(FuncBuilder &builder,
                                  ArrayRef<int> position) {
  SmallVector<Attribute, 4> attrs;
  attrs.reserve(position.size());
  for (auto p : position)
    attrs.push_back(builder.getI64IntegerAttr(p));
  return builder.getArrayAttr(attrs);
}

// BufferSizeOp creates a new `index` value.
class BufferSizeOpConversion : public LLVMOpLowering {
public:
  BufferSizeOpConversion(MLIRContext *context, LLVMLowering &lowering_)
      : LLVMOpLowering(BufferSizeOp::getOperationName(), context, lowering_) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto bufferSizeType = lowering.convertType(operands[0]->getType());
    edsc::ScopedContext context(rewriter, op->getLoc());
    return {extractvalue(bufferSizeType, operands[0],
                         makePositionAttr(rewriter, 1))};
  }
};

// RangeOp creates a new range descriptor.
class RangeOpConversion : public LLVMOpLowering {
public:
  explicit RangeOpConversion(MLIRContext *context, LLVMLowering &lowering_)
      : LLVMOpLowering(RangeOp::getOperationName(), context, lowering_) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto rangeOp = op->cast<RangeOp>();
    auto rangeDescriptorType =
        convertLinalgType(rangeOp.getResult()->getType(), lowering);

    edsc::ScopedContext context(rewriter, op->getLoc());

    // Fill in an aggregate value of the descriptor.
    Value *desc = undef(rangeDescriptorType);
    desc = insertvalue(rangeDescriptorType, desc, operands[0],
                       makePositionAttr(rewriter, 0));
    desc = insertvalue(rangeDescriptorType, desc, operands[1],
                       makePositionAttr(rewriter, 1));
    desc = insertvalue(rangeDescriptorType, desc, operands[2],
                       makePositionAttr(rewriter, 2));

    return {desc};
  }
};

class SliceOpConversion : public LLVMOpLowering {
public:
  explicit SliceOpConversion(MLIRContext *context, LLVMLowering &lowering_)
      : LLVMOpLowering(SliceOp::getOperationName(), context, lowering_) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto sliceOp = op->cast<SliceOp>();
    auto viewDescriptorType =
        convertLinalgType(sliceOp.getViewType(), lowering);
    auto viewType = sliceOp.getBaseViewType();
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));

    // Helper function to create an integer array attribute out of a list of
    // values.
    auto pos = [&rewriter](ArrayRef<int> values) {
      return makePositionAttr(rewriter, values);
    };
    // Helper function to obtain the ptr of the given `view`.
    auto getViewPtr = [pos, &rewriter, this](ViewType type,
                                             Value *view) -> Value * {
      auto elementPtrTy =
          rewriter.getType<LLVMType>(getPtrToElementType(type, lowering));
      return extractvalue(elementPtrTy, view, pos(0));
    };

    edsc::ScopedContext context(rewriter, op->getLoc());
    // Declare the view descriptor and insert data ptr.
    Value *desc = undef(viewDescriptorType);
    desc = insertvalue(viewDescriptorType, desc,
                       getViewPtr(viewType, operands[0]), pos(0));

    // TODO(ntv): extract sizes and emit asserts.
    SmallVector<Value *, 4> strides(viewType.getRank());
    for (int dim = 0, e = viewType.getRank(); dim < e; ++dim) {
      strides[dim] = extractvalue(int64Ty, operands[0], pos({3, dim}));
    }

    // Compute and insert base offset.
    Value *baseOffset = extractvalue(int64Ty, operands[0], pos(1));
    for (int j = 0, e = viewType.getRank(); j < e; ++j) {
      Value *indexing = operands[1 + j];
      Value *min =
          sliceOp.getIndexing(j)->getType().isa<RangeType>()
              ? static_cast<Value *>(extractvalue(int64Ty, indexing, pos(0)))
              : indexing;
      Value *product = mul(min, strides[j]);
      baseOffset = add(baseOffset, product);
    }
    desc = insertvalue(viewDescriptorType, desc, baseOffset, pos(1));

    // Compute and insert view sizes (max - min along the range).  Skip the
    // non-range operands as they will be projected away from the view.
    int i = 0;
    for (Value *index : sliceOp.getIndexings()) {
      if (!index->getType().isa<RangeType>())
        continue;

      Value *rangeDescriptor = operands[1 + i];
      Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
      Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
      Value *size = sub(max, min);

      desc = insertvalue(viewDescriptorType, desc, size, pos({2, i}));
      ++i;
    }

    // Compute and insert view strides.  Step over the strides that correspond
    // to non-range operands as they are projected away from the view.
    i = 0;
    for (int j = 0, e = strides.size(); j < e; ++j) {
      if (!sliceOp.getIndexing(j)->getType().isa<RangeType>())
        continue;
      Value *step = extractvalue(int64Ty, operands[1 + j], pos(2));
      Value *stride = mul(strides[j], step);
      desc = insertvalue(viewDescriptorType, desc, stride, pos({3, i}));
      ++i;
    }

    return {desc};
  }
};

class ViewOpConversion : public LLVMOpLowering {
public:
  explicit ViewOpConversion(MLIRContext *context, LLVMLowering &lowering_)
      : LLVMOpLowering(ViewOp::getOperationName(), context, lowering_) {}

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto viewOp = op->cast<ViewOp>();
    auto viewDescriptorType = convertLinalgType(viewOp.getViewType(), lowering);
    auto elementType = rewriter.getType<LLVMType>(
        getPtrToElementType(viewOp.getViewType(), lowering));
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));

    auto pos = [&rewriter](ArrayRef<int> values) {
      return makePositionAttr(rewriter, values);
    };

    // First operand to `view` is the buffer descriptor.
    Value *bufferDescriptor = operands[0];

    // Declare the descriptor of the view.
    edsc::ScopedContext context(rewriter, op->getLoc());
    Value *desc = undef(viewDescriptorType);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value *buffer = extractvalue(elementType, bufferDescriptor, pos(0));
    desc = insertvalue(viewDescriptorType, desc, buffer, pos(0));

    // Zero base offset.
    auto indexTy = rewriter.getIndexType();
    Value *baseOffset = constant(int64Ty, IntegerAttr::get(indexTy, 0));
    desc = insertvalue(viewDescriptorType, desc, baseOffset, pos(1));

    // Compute and insert view sizes (max - min along the range).
    int numIndexings = llvm::size(viewOp.getIndexings());
    Value *runningStride = constant(int64Ty, IntegerAttr::get(indexTy, 1));
    for (int i = 0; i < numIndexings; ++i) {
      // Update stride.
      Value *rangeDescriptor = operands[1 + i];
      Value *step = extractvalue(int64Ty, rangeDescriptor, pos(2));
      Value *stride = mul(runningStride, step);
      desc = insertvalue(viewDescriptorType, desc, stride, pos({3, i}));
      // Update size.
      Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
      Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
      Value *size = sub(max, min);
      desc = insertvalue(viewDescriptorType, desc, size, pos({2, i}));
      ++i;
      // Update stride for the next dimension.
      if (i < numIndexings - 1)
        runningStride = mul(runningStride, max);
    }

    return {desc};
  }
};

// DotOp creates a new range descriptor.
class DotOpConversion : public LLVMOpLowering {
public:
  explicit DotOpConversion(MLIRContext *context, LLVMLowering &lowering_)
      : LLVMOpLowering(DotOp::getOperationName(), context, lowering_) {}

  static StringRef libraryFunctionName() { return "linalg_dot"; }

  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto *f =
        op->getFunction()->getModule()->getNamedFunction(libraryFunctionName());
    if (!f)
      op->emitError("Could not find function: " + libraryFunctionName() +
                    "in lowering to LLVM ");
    auto fAttr = rewriter.getFunctionAttr(f);
    auto named = rewriter.getNamedAttr("callee", fAttr);
    rewriter.create<LLVM::CallOp>(op->getLoc(), operands, ArrayRef<NamedAttribute>{named});
    return {};
  }
};

namespace {
// The conversion class from Linalg to LLVMIR.
class Lowering : public LLVMLowering {
protected:
  llvm::DenseSet<DialectOpConversion *> initAdditionalConverters() override {
    return ConversionListBuilder<
        BufferSizeOpConversion, DotOpConversion, RangeOpConversion,
        SliceOpConversion, ViewOpConversion>::build(&converterStorage,
                                                    llvmDialect->getContext(),
                                                    *this);
  }

  Type convertAdditionalType(Type t) override {
    return convertLinalgType(t, *this);
  }
};
} // end anonymous namespace

namespace {
struct LowerLinalgToLLVMPass : public ModulePass<LowerLinalgToLLVMPass> {
  void runOnModule();
};
} // namespace

void LowerLinalgToLLVMPass::runOnModule() {
  auto &module = getModule();

  // Convert to the LLVM IR dialect using the converter defined above.
  auto r = Lowering().convert(&module);
  if (failed(r))
    signalPassFailure();
}

ModulePassBase *createLowerLinalgToLLVMPass() {
  return new LowerLinalgToLLVMPass();
}

static PassRegistration<LowerLinalgToLLVMPass>
    pass("linalg-lower-to-llvm-dialect",
         "Lower the operations from the linalg dialect into the LLVM dialect");
