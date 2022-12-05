/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>
#include <iostream>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

namespace TF {

namespace {

// Note: This implements the fusions performed in the old Remapper Grappler
// pass. That pass has specific cases for GPU and based on different
// target configurations on both CPU and GPU (Intel MKL, ROCm, etc.). This MLIR
// pass covers (some of) the general CPU case and at the moment does not account
// for any target-specific configurations.

// This pass is being ported over from the Grappler Remapper pass based on
// need/usage. File a bug to request porting over additional fusions.

// TODO(b/158265178): Support GPU-specific fusions.
// TODO(b/158266710): Support CPU MKL configurations.

#define GEN_PASS_DEF_FUSEDKERNELMATCHERPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Optimizes TF computations by fusing subgraphs/nodes onto more efficient
// implementations to decrease the number of operations needed to perform a
// computation.
struct FusedKernelMatcherPass
    : public impl::FusedKernelMatcherPassBase<FusedKernelMatcherPass> {
  void runOnOperation() override;
};

bool IsActivationFunction(Operation *op) {
  return isa<EluOp, ReluOp, Relu6Op>(op);
}

// Finds and returns an activation op that uses the result of `op`. If there are
// multiple such activations, one is returned (with no guarantee as to which
// one). If there are no activation functions that use the output, returns
// nullptr.
Operation *GetActivation(Value op) {
  for (auto &use : op.getUses()) {
    if (IsActivationFunction(use.getOwner())) return use.getOwner();
  }
  return nullptr;
}

// Finds and returns a BiasAdd that uses the result of `op` as the `value`
// input. If there are multiple such BiasAdds, one is returned (with no
// guarantee as to which one). If there are no BiasAdds that use the output,
// returns a null BiasAddOp.
BiasAddOp GetBiasAdd(Value op) {
  for (auto &use : op.getUses()) {
    auto bias_add = dyn_cast_or_null<BiasAddOp>(use.getOwner());
    // If it's a BiasAdd, check that the conv op is the first input.
    if (bias_add && bias_add.getValue() == op) return bias_add;
  }
  // No BiasAddOps found among uses.
  return BiasAddOp();
}

// Performs a fusion of the following pattern(s), if possible:
//   <Contraction> + BiasAdd + <Activation> -> <FusedContraction>
//
// Note that fusion with activation is preferred, but a contraction and BiasAdd
// can also be replaced by a _FusedConv2D if there is no other activation
// function.
// i.e., this class also supports the following fusion:
//   <Contraction> + BiasAdd -> <FusedContraction>
//
// TODO(b/158266331): Support fusing activation chains of arbitrary length.
template <typename SrcOpT, typename FusedOpT>
class FuseContractionWithBiasAdd : public OpRewritePattern<SrcOpT> {
 public:
  using OpRewritePattern<SrcOpT>::OpRewritePattern;
  // Class users should override this method if there are any op-specific
  // compatibility requirements between the contraction op and the BiasAdd op.
  virtual bool AreFuseCompatible(SrcOpT contraction_op, BiasAddOp bias_add,
                                 PatternRewriter &rewriter) const {
    return true;
  }

  // Class users should override this method if there are any op-specific
  // compatibility requirements for devices.
  virtual bool IsDeviceCompatible(SrcOpT contraction_op, BiasAddOp bias_add,
                                  PatternRewriter &rewriter) const {
    return true;
  }

  LogicalResult matchAndRewrite(SrcOpT contraction,
                                PatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();

    // We do support fusion only if the contraction operation is inside one of
    // the expected operations with regions. Other operations can have semantics
    // that is not compatible with fusion (e.g. region compilation).
    if (!isa<func::FuncOp, IfOp, WhileOp>(contraction->getParentOp())) {
      return rewriter.notifyMatchFailure(
          contraction,
          "fused operation must be nested inside a function, If or While");
    }

    // If the contraction is used in multiple places, fusing it will only create
    // more contraction nodes, which is slower.
    if (!contraction.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(contraction,
                                         "result is used by multiple ops");

    BiasAddOp bias_add = GetBiasAdd(contraction.getResult());
    if (!bias_add) {
      return rewriter.notifyMatchFailure(
          contraction, "does not feed into a tf.BiasAdd/tf.BiasAddV1 op");
    }

    if (!AreFuseCompatible(contraction, bias_add, rewriter)) {
      return rewriter.notifyMatchFailure(
          contraction, "cannot fuse with the subsequent BiasAdd op");
    }

    if (!IsDeviceCompatible(contraction, bias_add, rewriter)) {
      return rewriter.notifyMatchFailure(
          contraction,
          "cannot fuse with the subsequent op as it's not supported by the "
          "target device.");
    }

    SmallVector<Location, 3> locations{contraction.getLoc(), bias_add.getLoc()};
    SmallVector<Attribute, 2> fused_ops{StringAttr::get(
        context, bias_add.getOperation()->getName().stripDialect())};

    // BiasAdd may or may not feed into an activation function.
    auto activation = GetActivation(bias_add);

    // If there is an activation, only fuse it if this is the only op to use the
    // result of the BiasAdd.
    bool fuse_activation = activation && bias_add.getOutput().hasOneUse();
    Type result_type;

    // Include info about the activation function if applicable.
    if (fuse_activation) {
      locations.push_back(activation->getLoc());
      fused_ops.push_back(
          StringAttr::get(context, activation->getName().stripDialect()));
      result_type = activation->getResultTypes().front();
    } else {
      result_type = bias_add.getResult().getType();
    }

    auto fused_loc = rewriter.getFusedLoc(locations);

    // The fused contraction has the same operands as the original contraction
    // with `bias` from the BiasAddOp appended.
    SmallVector<Value, 4> operands(contraction.operand_begin(),
                                   contraction.operand_end());
    operands.push_back(bias_add.getBias());

    // The fused contraction has the same attributes as the original
    // contraction, with two additions: the list of ops which have been fused
    // together; epsilon (only with FusedBatchNorm).
    std::vector<NamedAttribute> attrs = contraction->getAttrs();
    ArrayAttr fused_ops_attr = ArrayAttr::get(context, fused_ops);
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "fused_ops"), fused_ops_attr));
    // Epsilon is used only in fusions with the FusedBatchNorm op, so we zero it
    // here.
    Attribute epsilon = rewriter.getF32FloatAttr(0);
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "epsilon"), epsilon));

    if (std::is_same<FusedOpT, _FusedConv2DOp>::value) {
      // Here TArgs types do not include types of the first two parameters,
      // i.e. the convolution input and the filter. TArgs are parameters for
      // the extras like the bias etc.
      auto attr = TypeAttr::get(getElementTypeOrSelf(contraction.getType()));
      SmallVector<Attribute, 4> targs_values(operands.size() - 2, attr);

      ArrayAttr targs_attr = ArrayAttr::get(context, targs_values);
      attrs.push_back(
          NamedAttribute(StringAttr::get(context, "TArgs"), targs_attr));

      auto num_args_attr = IntegerAttr::get(IntegerType::get(context, 64), 1);
      attrs.push_back(
          NamedAttribute(StringAttr::get(context, "num_args"), num_args_attr));

      // Fused conv operands are input, filter, args and host args. Here, bias
      // input of the BiasAdd op. Host args corresponds to conv_input_scale and
      // side_input_scale and not relevant in this case.
      auto sizes = mlir::DenseI32ArrayAttr::get(context, {1, 1, 1, 0});
      auto attr_name =
          StringAttr::get(context, mlir::OpTrait::AttrSizedOperandSegments<
                                       void>::getOperandSegmentSizeAttr());
      attrs.push_back(NamedAttribute(attr_name, sizes));
    }

    // Insert fused operation right before the BiasAdd operation to guarantee
    // that bias value dominates the fused operation. We already verified that
    // original operation has a single use, so this is safe to do.
    auto *bias_add_op = bias_add.getOperation();
    if (bias_add_op) rewriter.setInsertionPoint(bias_add_op);

    Value fused_op = rewriter.create<FusedOpT>(fused_loc, result_type,
                                               ValueRange(operands), attrs);
    auto op_to_replace = fuse_activation ? activation : bias_add;
    rewriter.replaceOp(op_to_replace, ValueRange({fused_op}));
    return success();
  }
};

const char kDeviceAttr[] = "device";
const char kDeviceGpu[] = "GPU";

llvm::Optional<std::string> GetDevice(mlir::Operation *op) {
  mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
  if (!device || device.getValue().empty()) {
    return llvm::None;
  }
  const std::string device_name = device.str();
  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return llvm::None;
  }
  if (!parsed_name.has_type) {
    return llvm::None;
  }
  return parsed_name.type;
}

bool IsGpuDevice(mlir::Operation *op) {
  llvm::Optional<std::string> device = GetDevice(op);
  if (!device) return false;
  return *device == kDeviceGpu;
}

// Performs a fusion of the following pattern(s), if possible:
//   Conv2D + BiasAdd + <Activation> -> _FusedConv2D
class FuseConv2DBiasAdd
    : public FuseContractionWithBiasAdd<Conv2DOp, _FusedConv2DOp> {
 public:
  using FuseContractionWithBiasAdd<Conv2DOp,
                                   _FusedConv2DOp>::FuseContractionWithBiasAdd;
  // Verify that the Conv2D and BiasAdd data formats match. This is necessary
  // for the ops to fuse correctly, the fused Conv2D op has one data format
  // attribute which is shared.
  bool AreFuseCompatible(Conv2DOp conv, BiasAddOp bias_add,
                         PatternRewriter &rewriter) const override {
    // Verify that the data formats match and are valid for fusion.
    if (conv.getDataFormat() != bias_add.getDataFormat()) {
      (void)rewriter.notifyMatchFailure(conv, [&](Diagnostic &diag) {
        diag << "data format does not match Conv2D data format ("
             << bias_add.getDataFormat() << " vs " << conv.getDataFormat()
             << ")";
      });
      return false;
    }

    // Verify the data type is supported.
    Type element_ty = getElementTypeOrSelf(conv.getType());
    if (!element_ty.isF32() && !element_ty.isF64()) {
      (void)rewriter.notifyMatchFailure(conv, [&](Diagnostic &diag) {
        diag << "supported data types for _FusedConv2D are float and double, "
             << " but got " << element_ty;
      });
      return false;
    }
    return true;
  }

  bool IsDeviceCompatible(Conv2DOp conv, BiasAddOp bias_add,
                          PatternRewriter &rewriter) const override {
    // Currently, GPU only supports Conv2D+BiasAdd+Relu fusion.
    if (IsGpuDevice(conv)) {
      auto activation = GetActivation(bias_add);
      if (!activation || activation->getName().stripDialect() != "Relu" ||
          !bias_add.getOutput().hasOneUse()) {
        (void)rewriter.notifyMatchFailure(conv, [&](Diagnostic &diag) {
          diag << "GPU only supports Conv2D+BiasAdd+Relu fusion";
        });
        return false;
      }
    }
    return true;
  }
};

// Performs a fusion of the following pattern(s), if possible:
//   MatMulOp + BiasAdd + <Activation> -> _FusedMatMulOp
class FuseMatMulBiasAdd
    : public FuseContractionWithBiasAdd<MatMulOp, _FusedMatMulOp> {
  using FuseContractionWithBiasAdd<MatMulOp,
                                   _FusedMatMulOp>::FuseContractionWithBiasAdd;

  bool AreFuseCompatible(MatMulOp matmul, BiasAddOp bias_add,
                         PatternRewriter &rewriter) const override {
    // FusedMatMul kernel supports limited set of data types.
    Type element_ty = getElementTypeOrSelf(matmul.getType());
    if (!element_ty.isF32() && !element_ty.isBF16()) {
      (void)rewriter.notifyMatchFailure(matmul, [&](Diagnostic &diag) {
        diag << "supported data types for _FusedMatMul are float and bfloat16, "
             << " but got " << element_ty;
      });
      return false;
    }
    return true;
  }

  bool IsDeviceCompatible(MatMulOp matmul, BiasAddOp bias_add,
                          PatternRewriter &rewriter) const override {
    if (IsGpuDevice(matmul)) {
      (void)rewriter.notifyMatchFailure(matmul, [&](Diagnostic &diag) {
        diag << "_FusedMatMul is not supported by GPU";
      });
      return false;
    }
    return true;
  }
};

void FusedKernelMatcherPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  patterns.add<FuseConv2DBiasAdd, FuseMatMulBiasAdd>(&getContext());

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateFusedKernelMatcherPass() {
  return std::make_unique<FusedKernelMatcherPass>();
}

}  // namespace TF

}  // namespace mlir
