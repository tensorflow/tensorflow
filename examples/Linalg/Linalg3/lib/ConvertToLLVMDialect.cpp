//===- ConvertToLLVMDialect.cpp - conversion from Linalg to LLVM dialect --===//
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
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

#include "linalg1/ConvertToLLVMDialect.h"
#include "linalg1/LLVMIntrinsics.h"

#include "linalg3/ConvertToLLVMDialect.h"
#include "linalg3/Ops.h"

using namespace mlir;

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

namespace {
// Common functionality for Linalg LoadOp and StoreOp conversion to the
// LLVM IR Dialect.
template <typename Op>
class LoadStoreOpConversion : public DialectOpConversion {
public:
  explicit LoadStoreOpConversion(MLIRContext *context)
      : DialectOpConversion(Op::getOperationName(), 1, context) {}
  using Base = LoadStoreOpConversion<Op>;

  // Match the Op specified as template argument.
  PatternMatchResult match(Operation *op) const override {
    if (op->isa<Op>())
      return matchSuccess();
    return matchFailure();
  }

  // Compute the pointer to an element of the buffer underlying the view given
  // current view indices.  Use the base offset and strides stored in the view
  // descriptor to emit IR iteratively computing the actual offset, followed by
  // a getelementptr.
  Value *obtainDataPtr(Operation *op, Value *viewDescriptor,
                       ArrayRef<Value *> indices, FuncBuilder &rewriter) const {
    auto loadOp = cast<Op>(op);
    auto elementType =
        loadOp.getViewType().template cast<linalg::ViewType>().getElementType();
    auto *llvmPtrType = linalg::convertLinalgType(elementType)
                            .template cast<LLVM::LLVMType>()
                            .getUnderlyingType()
                            ->getPointerTo();
    elementType = rewriter.getType<LLVM::LLVMType>(llvmPtrType);
    auto int64Ty = linalg::convertLinalgType(rewriter.getIntegerType(64));

    auto pos = [&rewriter](ArrayRef<int> values) {
      return makePositionAttr(rewriter, values);
    };

    using namespace intrinsics;

    // Linearize subscripts as:
    //   base_offset + SUM_i index_i * stride_i.
    Value *offset = extractvalue(int64Ty, viewDescriptor, pos(1));
    for (int i = 0, e = loadOp.getRank(); i < e; ++i) {
      Value *stride = extractvalue(int64Ty, viewDescriptor, pos({3, i}));
      Value *additionalOffset = mul(indices[i], stride);
      offset = add(offset, additionalOffset);
    }
    Value *base = extractvalue(elementType, viewDescriptor, pos(0));
    return gep(elementType, base, ArrayRef<Value *>{offset});
  }
};

// A load is converted into the actual address computation, getelementptr and
// an LLVM IR load.
class LoadOpConversion : public LoadStoreOpConversion<linalg::LoadOp> {
  using Base::Base;
  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    edsc::ScopedContext edscContext(rewriter, op->getLoc());
    auto elementType = linalg::convertLinalgType(*op->getResultTypes().begin());
    Value *viewDescriptor = operands[0];
    ArrayRef<Value *> indices = operands.drop_front();
    Value *ptr = obtainDataPtr(op, viewDescriptor, indices, rewriter);
    Value *element = intrinsics::load(elementType, ptr);
    return {element};
  }
};

// A store is converted into the actual address computation, getelementptr and
// an LLVM IR store.
class StoreOpConversion : public LoadStoreOpConversion<linalg::StoreOp> {
  using Base::Base;
  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    edsc::ScopedContext edscContext(rewriter, op->getLoc());
    Value *viewDescriptor = operands[1];
    Value *data = operands[0];
    ArrayRef<Value *> indices = operands.drop_front(2);
    Value *ptr = obtainDataPtr(op, viewDescriptor, indices, rewriter);
    intrinsics::store(data, ptr);
    return {};
  }
};

} // end anonymous namespace

// Helper function that allocates the descriptor converters and adds load/store
// coverters to the list.
static llvm::DenseSet<mlir::DialectOpConversion *>
allocateConversions(llvm::BumpPtrAllocator *allocator,
                    mlir::MLIRContext *context) {
  auto conversions = linalg::allocateDescriptorConverters(allocator, context);
  auto additional =
      ConversionListBuilder<LoadOpConversion, StoreOpConversion>::build(
          allocator, context);
  conversions.insert(additional.begin(), additional.end());
  return conversions;
}

void linalg::convertLinalg3ToLLVM(Module &module) {
  // Remove affine constructs if any by using an existing pass.
  PassManager pm;
  pm.addPass(createLowerAffinePass());
  auto rr = pm.run(&module);
  (void)rr;
  assert(succeeded(rr) && "affine loop lowering failed");

  auto lowering = makeLinalgToLLVMLowering(allocateConversions);
  auto r = lowering->convert(&module);
  (void)r;
  assert(succeeded(r) && "conversion failed");

  // Convert the remaining standard MLIR operations to the LLVM IR dialect using
  // the default converter.
  auto converter = createStdToLLVMConverter();
  r = converter->convert(&module);
  (void)r;
  assert(succeeded(r) && "second conversion failed");
}
