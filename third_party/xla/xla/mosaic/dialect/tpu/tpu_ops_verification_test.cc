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

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;

class TpuOpsVerificationTest : public ::testing::Test {
 protected:
  TpuOpsVerificationTest()
      : context_([]() {
          DialectRegistry registry;
          registry.insert<arith::ArithDialect, func::FuncDialect,
                          memref::MemRefDialect, TPUDialect>();
          return registry;
        }()),
        builder_(UnknownLoc::get(&context_), &context_) {
    context_.loadAllAvailableDialects();
    context_.printOpOnDiagnostic(true);
  }
  ~TpuOpsVerificationTest() {
    for (int i = ops_.size() - 1; i >= 0; --i) {
      ops_[i]->erase();
    }
  }

  template <typename OpTy, typename... Args>
  OpTy Create(Args&&... args) {
    OpTy op = OpTy::create(builder_, std::forward<Args>(args)...);
    ops_.push_back(op.getOperation());
    return op;
  }

  template <typename OpTy>
  absl::Status VerifyOp(OpTy op) {
    BaseScopedDiagnosticHandler diag(&context_);
    if (op.verify().succeeded()) {
      return absl::OkStatus();
    }
    return diag.ConsumeStatus();
  }

  Type i32() { return builder_.getI32Type(); }

  MemRefType GetMemRefType(
      ArrayRef<int64_t> shape, Type element_type,
      std::optional<MemorySpace> memory_space = std::nullopt) {
    return MemRefType::get(
        shape, element_type, nullptr,
        memory_space.has_value()
            ? MemorySpaceAttr::get(builder_.getContext(), *memory_space)
            : Attribute());
  }

  Value AllocaI32(ArrayRef<int64_t> shape,
                  std::optional<MemorySpace> memory_space = std::nullopt) {
    return Create<memref::AllocaOp>(GetMemRefType(shape, i32(), memory_space))
        .getMemref();
  }

  Value AllocaSemaphore() {
    return Create<tpu::AllocaSemaphoreOp>(
               GetMemRefType({}, SemaphoreType::get(builder_.getContext()),
                             MemorySpace::kSemaphoreMem))
        .getResult();
  }

  Value ConstantI1Vector(ArrayRef<int64_t> shape, ArrayRef<bool> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, builder().getI1Type()),
               /*value=*/builder().getBoolVectorAttr(values))
        .getResult();
  }

  Value ConstantI8Vector(ArrayRef<int64_t> shape, ArrayRef<int8_t> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, builder().getI8Type()),
               /*value=*/dyn_cast<TypedAttr>(
                   builder().getDenseI8ArrayAttr(values)))
        .getResult();
  }

  Value ConstantIndexVector(ArrayRef<int64_t> shape, ArrayRef<int64_t> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, builder().getIndexType()),
               /*value=*/builder().getIndexVectorAttr(values))
        .getResult();
  }

  Value ConstantI32Vector(ArrayRef<int64_t> shape, ArrayRef<int32_t> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, i32()),
               /*value=*/builder().getI32VectorAttr(values))
        .getResult();
  }

  Value ConstantBF16Vector(ArrayRef<int64_t> shape, float value) {
    VectorType bf16_vector_type =
        VectorType::get(shape, builder().getBF16Type());
    return Create<arith::ConstantOp>(
               /*result=*/bf16_vector_type,
               /*value=*/SplatElementsAttr::get(
                   bf16_vector_type,
                   builder().getFloatAttr(builder().getBF16Type(), value)))
        .getResult();
  }

  Value ConstantF32Vector(ArrayRef<int64_t> shape, ArrayRef<float> values) {
    auto ty = VectorType::get(shape, builder().getF32Type());
    return Create<arith::ConstantOp>(
               /*result=*/ty,
               /*value=*/DenseElementsAttr::get(ty, values))
        .getResult();
  }

  ImplicitLocOpBuilder& builder() { return builder_; }

 private:
  MLIRContext context_;
  ImplicitLocOpBuilder builder_;
  std::vector<Operation*> ops_;
};

class TpuOpsVectorSubcoreVerificationTest : public TpuOpsVerificationTest {
 protected:
  TpuOpsVectorSubcoreVerificationTest() {
    auto func_op = Create<func::FuncOp>("vector_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), CoreType::kScVectorSubcore));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
  }
};

TEST_F(TpuOpsVerificationTest, VectorLoadVerificationWorks) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadRankOfStridesDoesNotMatchBaseMemrefRank) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({1, 1, 1, 1}),
      /*mask=*/nullptr);
  ASSERT_THAT(VerifyOp(vl), StatusIs(_, HasSubstr("Expected 1 strides.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadStridesFeatureNotImplemented) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({1}),
      /*mask=*/nullptr);
  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr("Not implemented: general vector load with strides.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadBaseAndResultTypesDoNotMatch) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, builder().getF32Type()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_,
               HasSubstr("Expected base and result element type to match.")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadRankOfIndicesDoesNotMatchBaseMemrefRank) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(VerifyOp(vl), StatusIs(_, HasSubstr("Expected 1 indices.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadValidMaskSucceeds) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128}, MemorySpace::kVmem);
  Value mask = ConstantI32Vector(/*shape=*/{8, 1},
                                 /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorLoadMaskInvalidResultBitWidth) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  auto memref = Create<memref::AllocaOp>(
      GetMemRefType({8, 128}, builder().getI64Type(), MemorySpace::kVmem));
  Value mask = ConstantI32Vector(/*shape=*/{8, 1},
                                 /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, builder().getI64Type()),
      /*base=*/memref.getMemref(),
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Not implemented: masked load with non-32-bit element type")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadMaskNotBroadcastableToResultShapeInvalidMinor) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128}, MemorySpace::kVmem);
  Value mask = ConstantI32Vector(/*shape=*/{8, 2},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Expected mask shape to be broadcastable to result shape.")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadMaskNotBroadcastableToResultShapeInvalidMajor) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128}, MemorySpace::kVmem);
  Value mask = ConstantI32Vector(/*shape=*/{5, 1},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Expected mask shape to be broadcastable to result shape.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadInvalidMemorySpace) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kHbm);
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(VerifyOp(vl),
              StatusIs(_, HasSubstr("Expected base memref to be in VMEM.")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreInvalidMemorySpace) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8}, MemorySpace::kHbm);
  Value vector_to_store = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  auto vs = Create<VectorStoreOp>(
      /*valueToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(VerifyOp(vs),
              StatusIs(_, HasSubstr("Expected base memref to be in VMEM.")));
}

TEST_F(TpuOpsVerificationTest, UnpackSubelementsValidIndex) {
  Value source = ConstantI8Vector(/*shape=*/{4, 8}, /*values=*/{1});
  auto unpack = Create<UnpackSubelementsOp>(
      /*output=*/VectorType::get({16}, builder().getI16Type()), source,
      /*index=*/builder().getI32IntegerAttr(1),
      /*pack_format=*/
      PackFormatAttr::get(builder().getContext(), PackFormat::kInterleaved));
  ASSERT_OK(VerifyOp(unpack));
}

TEST_F(TpuOpsVerificationTest, UnpackSubelementsInvalidIndex) {
  Value source = ConstantI8Vector(/*shape=*/{4, 8}, /*values=*/{1});
  auto unpack = Create<UnpackSubelementsOp>(
      /*output=*/VectorType::get({16}, builder().getI16Type()), source,
      /*index=*/builder().getI32IntegerAttr(4),
      /*pack_format=*/
      PackFormatAttr::get(builder().getContext(), PackFormat::kInterleaved));
  ASSERT_THAT(
      VerifyOp(unpack),
      StatusIs(
          _, HasSubstr("Index must be between 0 and the packing factor (2)")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxVerificationWorks) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxInvalidMemorySpace) {
  Value memref = AllocaI32({8}, MemorySpace::kHbm);
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr);

  ASSERT_THAT(VerifyOp(vl),
              StatusIs(_, HasSubstr("Expected base memref to be in VMEM.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxInvalidElementType) {
  Value memref =
      Create<memref::AllocaOp>(
          GetMemRefType({8}, builder().getF32Type(), MemorySpace::kVmem))
          .getMemref();
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_,
               HasSubstr("Expected base and result element type to match.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxInvalidIndicesDimension) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value indices = ConstantIndexVector(/*shape=*/{4, 1},
                                      /*values=*/{0});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices, indices},
      /*mask=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_, HasSubstr("Expected one index vector for each dimension of "
                            "the base memref with dimension: 1. Got: 2.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxValidMask) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  Value mask = ConstantI32Vector(/*shape=*/{8},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/mask);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorLoadIdxInvalidMaskShape) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  Value mask = ConstantI32Vector(/*shape=*/{4, 2},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadIdxOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Expected mask shape to be broadcastable to result shape.")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxVerificationWorks) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value vector_to_store =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr,
      /*add=*/builder().getBoolAttr(true));

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxInvalidMemorySpace) {
  Value memref = AllocaI32({8}, MemorySpace::kHbm);
  Value vector_to_store =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr,
      /*add=*/nullptr);

  ASSERT_THAT(VerifyOp(vl),
              StatusIs(_, HasSubstr("Expected base memref to be in VMEM.")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxInvalidElementType) {
  Value memref =
      Create<memref::AllocaOp>(
          GetMemRefType({8}, builder().getF32Type(), MemorySpace::kVmem))
          .getMemref();
  Value vector_to_store = ConstantI32Vector(/*shape=*/{8},
                                            /*values=*/{1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr,
      /*add=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_, HasSubstr(
                      "Expected base and valueToStore element type to match")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxInvalidIndicesDimension) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value vector_to_store = ConstantI32Vector(/*shape=*/{8},
                                            /*values=*/{1});
  Value indices = ConstantIndexVector(/*shape=*/{4, 1},
                                      /*values=*/{0});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices, indices},
      /*mask=*/nullptr,
      /*add=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_, HasSubstr("Expected one index vector for each dimension of "
                            "the base memref with dimension: 1. Got: 2.")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxInvalidValueToStoreDimension) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value vector_to_store = ConstantI32Vector(/*shape=*/{4, 2},
                                            /*values=*/{1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/nullptr,
      /*add=*/nullptr);

  ASSERT_THAT(VerifyOp(vl),
              StatusIs(_, HasSubstr("Expected value to have rank 1. Got: 2.")));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxValidMask) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value vector_to_store = ConstantI32Vector(/*shape=*/{8},
                                            /*values=*/{1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  Value mask = ConstantI32Vector(/*shape=*/{8},
                                 /*values=*/{1});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/mask,
      /*add=*/nullptr);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorStoreIdxInvalidMaskShape) {
  Value memref = AllocaI32({8}, MemorySpace::kVmem);
  Value vector_to_store = ConstantI32Vector(/*shape=*/{8},
                                            /*values=*/{1});
  Value indices = ConstantIndexVector(/*shape=*/{8},
                                      /*values=*/{0});
  Value mask = ConstantI32Vector(/*shape=*/{4, 2},
                                 /*values=*/{1});
  auto vl = Create<VectorStoreIdxOp>(
      /*vectorToStore=*/vector_to_store,
      /*base=*/memref,
      /*indices=*/ValueRange{indices},
      /*mask=*/mask,
      /*add=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _,
          HasSubstr(
              "Expected mask shape to match result shape: (8). Got: (4, 2).")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationWorksI32) {
  Value src = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_OK(VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationWorksBF16) {
  Value src = ConstantBF16Vector(/*shape=*/{2, 8}, /*value=*/1);
  Type dst =
      VectorType::get(/*shape=*/{2, 8}, /*type=*/builder().getBF16Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_OK(VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationWorksI1) {
  Value src = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());

  ASSERT_OK(VerifyOp(
      Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, /*mask=*/nullptr)));
}

TEST_F(TpuOpsVerificationTest, ScanOnUnsupportedCore) {
  auto func_op =
      Create<func::FuncOp>("scalar_kernel", builder().getFunctionType({}, {}));
  func_op->setAttr(TPUDialect::GetCoreTypeKey(),
                   CoreTypeAttr::get(builder().getContext(), CoreType::kTc));
  builder().setInsertionPointToStart(func_op.addEntryBlock());
  Value src = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)),
      StatusIs(_,
               HasSubstr("Scan is supported only on the SC vector subcore")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       ScanVerificationInvalidOutputTypeWithI1Input) {
  Value src = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI1Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kMin, mask)),
      StatusIs(
          _,
          HasSubstr(
              "Output element type must be i32 vector for i1 vector inputs.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       ScanVerificationMismatchElementType) {
  Value src = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getF32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)),
      StatusIs(_, HasSubstr("Input and output element type mismatch.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationMismatchShape) {
  Value src = ConstantI32Vector(/*shape=*/{16}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{16}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)),
      StatusIs(_, HasSubstr("Input and output shape mismatch. Input "
                            "shape: (16). Output shape: (8).")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationInvalidInputRank) {
  Value src = ConstantI32Vector(/*shape=*/{8, 1, 1}, /*values=*/{1});
  Type dst =
      VectorType::get(/*shape=*/{8, 1, 1}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)),
      StatusIs(_, HasSubstr("Input must be a rank 1 or 2 vector.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       ScanVerificationInvalidReductionKind) {
  Value src = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kArgMax, mask)),
      StatusIs(_,
               HasSubstr("Only sum, max and min reductions are supported.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       ScanVerificationInvalidReductionKindWithI1Input) {
  Value src = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kMin, mask)),
      StatusIs(
          _,
          HasSubstr("Only sum reduction is supported for i1 vector inputs.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       ScanVerificationInvalidMaskWithI1Input) {
  Value src = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kSum, mask)),
      StatusIs(_, HasSubstr("Mask is not supported for i1 vector inputs.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationInvalidMaskRank) {
  Value src = ConstantI32Vector(/*shape=*/{1, 8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{1, 8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{1, 8}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kMax, mask)),
      StatusIs(_, HasSubstr("Mask must be a rank 1 vector.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, ScanVerificationInvalidMaskShape) {
  Value src = ConstantI32Vector(/*shape=*/{1, 8}, /*values=*/{1});
  Type dst = VectorType::get(/*shape=*/{1, 8}, /*type=*/builder().getI32Type());
  Value mask = ConstantI1Vector(/*shape=*/{16}, /*values=*/{1});

  ASSERT_THAT(
      VerifyOp(Create<ScanOp>(dst, src, tpu::ReductionKind::kMax, mask)),
      StatusIs(_, HasSubstr("Mask and input mismatch. Expected mask of "
                            "length: 8, but got 16.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, DmaElementTypeMismatch) {
  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/AllocaSemaphore(),
      /*target=*/
      Create<memref::AllocaOp>(GetMemRefType({1024, 256, 128},
                                             builder().getI64Type(),
                                             MemorySpace::kHbm))
          .getMemref(),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_, HasSubstr("DMA source and target element type mismatch")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, DmaDynamicRankMismatch) {
  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({ShapedType::kDynamic, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/AllocaSemaphore(),
      /*target=*/
      AllocaI32({ShapedType::kDynamic, ShapedType::kDynamic, 128},
                MemorySpace::kHbm),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("DMA source and target shape mismatch.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, DmaStrictOrderingSupported) {
  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/nullptr,
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmem),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr,
      /*priority=*/0,
      /*strict_ordering=*/true);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVerificationTest, DmaStrictOrderingNotSupportedOnTc) {
  auto func_op =
      Create<func::FuncOp>("tc_kernel", builder().getFunctionType({}, {}));
  func_op->setAttr(TPUDialect::GetCoreTypeKey(),
                   CoreTypeAttr::get(builder().getContext(), CoreType::kTc));
  builder().setInsertionPointToStart(func_op.addEntryBlock());

  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/nullptr,
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmem),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr,
      /*priority=*/0,
      /*strict_ordering=*/true);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Strict ordering is only supported on the "
                                    "SC scalar and vector subcores")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaHbmChunkGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVmemSharedChunkGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared),
      /*target=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaSublaneGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaElementGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024}, MemorySpace::kHbm),
      /*target=*/AllocaI32({128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({128}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaFilteredGatherVerificationWorks) {
  Value offset_filter =
      Create<arith::ConstantOp>(builder().getIntegerAttr(i32(), -1))
          .getResult();
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/offset_filter,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorGatherVerificationWorks) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({8, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaHbmScatterVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVmemSharedScatterVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 128}, MemorySpace::kVmemShared),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDma2dOffsetsScatterVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1, 1024}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({1, 128}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorScatterVerificationWorks) {
  Value vector_of_offsets = ConstantI32Vector(
      /*shape=*/{16},
      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({16, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterAddVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/true);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVerificationTest, IndirectDmaOnUnsupportedCore) {
  std::vector<CoreType> unsupported_cores = {CoreType::kScScalarSubcore,
                                             CoreType::kTc};
  for (CoreType unsupported_core : unsupported_cores) {
    auto func_op = Create<func::FuncOp>("scalar_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), unsupported_core));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
    auto dma = Create<EnqueueIndirectDMAOp>(
        /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
        /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
        /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
        /*semaphore=*/AllocaSemaphore(),
        /*offset_filter=*/nullptr,
        /*add=*/false);

    ASSERT_THAT(
        VerifyOp(dma),
        StatusIs(_, HasSubstr("Enqueue indirect DMA is supported only on "
                              "the SC vector subcore")));
  }
}

TEST_F(TpuOpsVerificationTest, IndirectDmaOnUnsupportedTc) {
  auto func_op =
      Create<func::FuncOp>("tc_kernel", builder().getFunctionType({}, {}));
  func_op->setAttr(TPUDialect::GetCoreTypeKey(),
                   CoreTypeAttr::get(builder().getContext(), CoreType::kTc));
  builder().setInsertionPointToStart(func_op.addEntryBlock());
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Enqueue indirect DMA is supported only on "
                                    "the SC vector subcore")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherSourceAndTargetTypeMismatch) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/
      Create<memref::AllocaOp>(GetMemRefType({64, 32, 128},
                                             builder().getI64Type(),
                                             MemorySpace::kVmem))
          .getMemref(),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_, HasSubstr("Source and target element type mismatch")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDmaWithoutLocalMem) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_, HasSubstr("The transfer must be between HBM and VMEM, "
                            "VMEM_SHARED and VMEM, or TC VMEM and VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDmaOffsetsNotInVmem) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kHbm),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Offsets memref must be in VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDma1DSemaphore) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/
      Create<tpu::AllocaSemaphoreOp>(
          GetMemRefType({1}, SemaphoreType::get(builder().getContext()),
                        MemorySpace::kSemaphoreMem))
          .getResult(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Semaphore must be rank 0")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherOffsetsShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr("Offsets shape must be 1D or (1, N), got (64, 32)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherOffsetsAndSourceSampleRankMismatch) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({1, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({1, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr("Source (gather operand) sample rank must match "
                         "offsets rank, got 1 vs 2")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherTargetShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({512, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "Offsets shape (64) must match the majormost dimensions "
                   "of the target (gather result) shape (512, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorGatherTargetShapeInvalid) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({512, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr("Offsets shape (8) must match the majormost dimensions "
                       "of the target (gather result) shape (512, 32, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherOperandShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 512}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "1 minormost dimension of the source (gather operand) shape "
                   "(1024, 512) must match the minormost dimension of "
                   "the target (gather result) shape (64, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterUpdatesShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({512, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "Offsets shape (64) must match the majormost dimensions "
                   "of the source (scatter updates) shape (512, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterOperandShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 512}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr(
                 "1 minormost dimension of the source (scatter updates) shape "
                 "(64, 128) must match the minormost dimension of the "
                 "target (scatter operand) shape (1024, 512)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorScatterOperandShapeInvalid) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({8, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 96, 512}, MemorySpace::kHbm),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr(
                 "2 minormost dimensions of the source (scatter updates) shape "
                 "(8, 32, 128) must match the minormost dimensions of the "
                 "target (scatter operand) shape (1024, 96, 512)")));
}

TEST_F(TpuOpsVerificationTest, IndirectDmaWaitOnUnsupportedCoreInvalid) {
  static constexpr std::array<CoreType, 2> unsupported_cores = {
      CoreType::kScScalarSubcore, CoreType::kTc};
  for (CoreType unsupported_core : unsupported_cores) {
    SCOPED_TRACE(testing::Message()
                 << "Testing unsupported core type: "
                 << stringifyCoreType(unsupported_core).str());
    auto func_op = Create<func::FuncOp>("scalar_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), unsupported_core));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
    auto wait = Create<WaitIndirectDMAOp>(
        /*semaphore=*/AllocaSemaphore(),
        /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
        /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem));

    ASSERT_THAT(VerifyOp(wait),
                StatusIs(_, HasSubstr("Wait indirect DMA is supported only on "
                                      "the SC vector subcore")));
  }
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitGatherVerificationWorks) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem));

  ASSERT_OK(VerifyOp(wait));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, SortVerificationWorks) {
  Value keys = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Value values = ConstantI32Vector(/*shape=*/{8}, /*values=*/{2});
  Type mask_ty = VectorType::get({8}, builder().getI1Type());
  Type keys_ty = keys.getType();
  Type values_ty = values.getType();
  auto sort =
      Create<SortOp>(/*result_types=*/TypeRange{mask_ty, keys_ty, values_ty},
                     /*keys=*/keys, /*values=*/values,
                     /*mask=*/nullptr,
                     /*descending=*/builder().getBoolAttr(false));
  ASSERT_OK(VerifyOp(sort));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, SortF32KeysVerificationWorks) {
  Value keys = ConstantF32Vector(/*shape=*/{8}, /*values=*/{1.0f});
  Value values = ConstantI32Vector(/*shape=*/{8}, /*values=*/{2});
  Type mask_ty = VectorType::get({8}, builder().getI1Type());
  Type keys_ty = keys.getType();
  Type values_ty = values.getType();
  auto sort =
      Create<SortOp>(/*result_types=*/TypeRange{mask_ty, keys_ty, values_ty},
                     /*keys=*/keys, /*values=*/values,
                     /*mask=*/nullptr,
                     /*descending=*/builder().getBoolAttr(false));
  ASSERT_OK(VerifyOp(sort));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, SortKeyValShapeMismatch) {
  Value keys = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Value values = ConstantI32Vector(/*shape=*/{16}, /*values=*/{2});
  Type mask_ty = VectorType::get({8}, builder().getI1Type());
  Type keys_ty = keys.getType();
  Type values_ty = values.getType();
  auto sort =
      Create<SortOp>(/*result_types=*/TypeRange{mask_ty, keys_ty, values_ty},
                     /*keys=*/keys, /*values=*/values,
                     /*mask=*/nullptr,
                     /*descending=*/builder().getBoolAttr(false));
  ASSERT_THAT(VerifyOp(sort),
              StatusIs(_, HasSubstr("Key and value shapes must match")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, SortResultTypeMismatch) {
  Value keys = ConstantI32Vector(/*shape=*/{8}, /*values=*/{1});
  Value values = ConstantI32Vector(/*shape=*/{8}, /*values=*/{2});
  Type mask_ty = VectorType::get({8}, builder().getI1Type());
  Type keys_ty = keys.getType();
  Type f32_ty = VectorType::get({8}, builder().getF32Type());
  auto sort =
      Create<SortOp>(/*result_types=*/TypeRange{mask_ty, keys_ty, f32_ty},
                     /*keys=*/keys, /*values=*/values,
                     /*mask=*/nullptr,
                     /*descending=*/builder().getBoolAttr(false));
  ASSERT_THAT(
      VerifyOp(sort),
      StatusIs(_, HasSubstr("Value and sorted_value types must match")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitScatterVerificationWorks) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared));

  ASSERT_OK(VerifyOp(wait));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitWithoutLocalMemInvalid) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kHbm));

  ASSERT_THAT(
      VerifyOp(wait),
      StatusIs(_, HasSubstr("The transfer must be between HBM and VMEM, "
                            "VMEM_SHARED and VMEM, or TC VMEM and VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitInvalidSemaphoreRank) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/Create<tpu::AllocaSemaphoreOp>(
          GetMemRefType({8}, SemaphoreType::get(builder().getContext()),
                        MemorySpace::kSemaphoreMem))
          .getResult(),
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared));

  ASSERT_THAT(
      VerifyOp(wait),
      StatusIs(_, HasSubstr("Indirect DMA wait semaphore must be rank 0")));
}
}  // namespace
}  // namespace mlir::tpu
