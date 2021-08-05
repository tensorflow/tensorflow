/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/tf_cpurt_passes.h"

#include <functional>
#include <memory>
#include <string>

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_cpurt_clustering.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace tensorflow {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/tf_cpurt_passes.h.inc"

// -------------------------------------------------------------------------- //
// Helper functions used by the passes implemented below.
// -------------------------------------------------------------------------- //

// Returns true if the `value` type is a memref that is contiguous in memory.
static bool IsContiguousMemref(mlir::Value value) {
  mlir::MemRefType memref_type = value.getType().dyn_cast<mlir::MemRefType>();
  if (!memref_type) return false;
  mlir::MemRefType canonical_type = canonicalizeStridedLayout(memref_type);
  return canonical_type.getAffineMaps().empty();
}

// -------------------------------------------------------------------------- //
// Trivial buffer forwarding for the linalg.generic operations.
// -------------------------------------------------------------------------- //

namespace {

struct ForwardingCandidate {
  mlir::Value memref;
  mlir::AffineMap indexing_map;
};

struct LinalgTrivialBufferForwardingPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

struct LinalgTrivialBufferForwardingPass
    : public LinalgTrivialBufferForwardingBase<
          LinalgTrivialBufferForwardingPass> {
  void runOnFunction() override;
};
}  // namespace

// Returns true if all linalg.generic operation iterators are "parallel".
static bool AllIteratorsAreParallel(mlir::linalg::GenericOp op) {
  return llvm::all_of(op.iterator_types(), [](mlir::Attribute attr) -> bool {
    auto str_attr = attr.dyn_cast<mlir::StringAttr>();
    return str_attr && str_attr.getValue() == "parallel";
  });
}

// Returns buffer inputs that can be safely used as buffer outputs.
static llvm::SmallVector<mlir::OpOperand*> FindBufferForwardingCandidates(
    mlir::linalg::GenericOp op) {
  llvm::SmallVector<mlir::OpOperand*> candidates;

  for (mlir::OpOperand* input_buffer : op.getInputOperands()) {
    // Input must be a contiguous memref ...
    if (!IsContiguousMemref(input_buffer->get())) continue;

    // ... allocated in the same function.
    auto* alloc = input_buffer->get().getDefiningOp();
    if (!alloc || !mlir::isa<mlir::memref::AllocOp>(alloc)) continue;

    // Find input users that are after linalg.generic operation in the block.
    auto users = llvm::make_filter_range(alloc->getUsers(),
                                         [&](mlir::Operation* user) -> bool {
                                           return op->isBeforeInBlock(user);
                                         });

    // Input buffer must have exactly one user after linalg.generic.
    llvm::SmallVector<mlir::Operation*> input_users(users.begin(), users.end());
    if (input_users.size() > 1) continue;

    // And it must be a memref.dealloc operation.
    if (!mlir::isa<mlir::memref::DeallocOp>(input_users[0])) continue;

    // This input memref can be safely reused for the output.
    candidates.push_back(input_buffer);
  }

  return candidates;
}

mlir::LogicalResult LinalgTrivialBufferForwardingPattern::matchAndRewrite(
    mlir::linalg::GenericOp op, mlir::PatternRewriter& rewriter) const {
  // With parallel iterators it is guaranteed that every value used once, and it
  // is safe to forward input to output.
  if (!AllIteratorsAreParallel(op))
    return rewriter.notifyMatchFailure(op, "all iterators must be parallel");

  // Find memrefs that potentially could be forwarded.
  llvm::SmallVector<mlir::OpOperand*> forwarding_candidates =
      FindBufferForwardingCandidates(op);
  if (forwarding_candidates.empty())
    return rewriter.notifyMatchFailure(
        op, "did not find any candidates for input forwarding");

  // Inputs that were reused.
  llvm::DenseSet<mlir::OpOperand*> reused_inputs;

  // Try to match output buffers to forwarding candidates.
  for (mlir::OpOperand* output_buffer : op.getOutputOperands()) {
    // Output must be allocated in the same function.
    auto* alloc = output_buffer->get().getDefiningOp();
    if (!alloc || !mlir::isa<mlir::memref::AllocOp>(alloc)) continue;

    // Find compatible input buffer.
    for (mlir::OpOperand* input_buffer : forwarding_candidates) {
      if (reused_inputs.contains(input_buffer)) continue;

      // Memref types must match (dimensions and affine maps).
      if (input_buffer->get().getType() != output_buffer->get().getType())
        continue;

      mlir::AffineMap src_map = op.getTiedIndexingMap(input_buffer);
      mlir::AffineMap dst_map = op.getTiedIndexingMap(output_buffer);

      // Only support identity maps for the output for now.
      if (!dst_map.isIdentity()) continue;

      auto is_projection = [](mlir::AffineMap map) {
        // Allow adding/dropping dimensions but no permutations.
        int64_t i = -1;
        for (mlir::AffineExpr expr : map.getResults()) {
          auto constant = expr.dyn_cast<mlir::AffineConstantExpr>();
          if (constant && constant.getValue() == 0) continue;
          auto dim_expr = expr.dyn_cast<mlir::AffineDimExpr>();
          if (!dim_expr || dim_expr.getPosition() <= i) return false;
          i = dim_expr.getPosition();
        }
        return true;
      };

      auto same_shape = [](mlir::Value src, mlir::Value dst) {
        auto src_type = src.getType().cast<mlir::ShapedType>();
        auto dst_type = dst.getType().cast<mlir::ShapedType>();
        mlir::OperandRange src_operands =
            src.getDefiningOp<mlir::memref::AllocOp>().getDynamicSizes();
        mlir::OperandRange dst_operands =
            dst.getDefiningOp<mlir::memref::AllocOp>().getDynamicSizes();
        return src_type.getShape().equals(dst_type.getShape()) &&
               std::equal(src_operands.begin(), src_operands.end(),
                          dst_operands.begin());
      };

      // A reuse is valid if the maps are the same or if the shape is the same
      // and the source is a projection map (in which case the ignored
      // dimensions must be 1 assuming that the operation reads the entire
      // input). Note that we already know that the destination map is an
      // identity map.
      if (src_map != dst_map &&
          !(is_projection(src_map) &&
            same_shape(input_buffer->get(), output_buffer->get()))) {
        continue;
      }

      // Find the input buffer dealloc operation.
      mlir::Operation* input_dealloc = *llvm::find_if(
          input_buffer->get().getUsers(), [](mlir::Operation* user) -> bool {
            return mlir::isa<mlir::memref::DeallocOp>(user);
          });

      // Deallocate output buffer instead of the input buffer.
      input_buffer->get().replaceUsesWithIf(
          output_buffer->get(), [&](mlir::OpOperand& operand) -> bool {
            return operand.getOwner() == input_dealloc;
          });

      // Forward users of output buffer to the input buffer, if they are after
      // linalg.generic operation in the block (or linalg.generic itself).
      output_buffer->get().replaceUsesWithIf(
          input_buffer->get(), [&](mlir::OpOperand& operand) -> bool {
            return operand.getOwner() != input_dealloc &&
                   !operand.getOwner()->isBeforeInBlock(op);
          });

      reused_inputs.insert(input_buffer);
    }
  }

  return mlir::success(!reused_inputs.empty());
}

void LinalgTrivialBufferForwardingPass::runOnFunction() {
  mlir::FuncOp function = getFunction();
  mlir::MLIRContext* ctx = function.getContext();

  mlir::RewritePatternSet patterns(ctx);
  patterns.insert<LinalgTrivialBufferForwardingPattern>(ctx);

  (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
}

std::unique_ptr<mlir::FunctionPass> CreateLinalgTrivialBufferForwardingPass() {
  return std::make_unique<LinalgTrivialBufferForwardingPass>();
}

// -------------------------------------------------------------------------- //
// Trivial buffer forwarding for the linalg.generic operations.
// -------------------------------------------------------------------------- //

namespace {

struct LinalgTrivialCopyRemovalPass
    : public LinalgTrivialCopyRemovalBase<LinalgTrivialCopyRemovalPass> {
  void runOnFunction() override;
};

}  // namespace

void LinalgTrivialCopyRemovalPass::runOnFunction() {
  mlir::FuncOp function = getFunction();

  mlir::SmallVector<mlir::Operation*> to_erase;
  function.walk([&to_erase](mlir::linalg::CopyOp copy) {
    // Only match precise alloc/copy/dealloc triples.
    auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(copy->getPrevNode());
    auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(copy->getNextNode());

    if (!alloc || !dealloc) return;

    // Make sure the alloc and dealloc handle the operands of the copy.
    if (alloc.getResult() != copy.getTarget() ||
        dealloc.memref() != copy.getSource()) {
      return;
    }

    // Remember the operations to delete.
    to_erase.push_back(alloc);
    to_erase.push_back(dealloc);
    to_erase.push_back(copy);
    copy.getTarget().replaceAllUsesWith(copy.getSource());
  });

  for (auto op : to_erase) {
    op->erase();
  }
}

std::unique_ptr<mlir::FunctionPass> CreateLinalgTrivialCopyRemovalPass() {
  return std::make_unique<LinalgTrivialCopyRemovalPass>();
}

// -------------------------------------------------------------------------- //
// Dispatch linalg.matmul to one of the more specialized operations at runtime.
// -------------------------------------------------------------------------- //

namespace {

struct LinalgMatmulSpecializationPattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::MatmulOp matmul,
      mlir::PatternRewriter& rewriter) const override;
};

struct LinalgMatmulSpecializationPass
    : public LinalgMatmulSpecializationBase<LinalgMatmulSpecializationPass> {
  void runOnFunction() override;
};
}  // namespace

// Convert 2D memref into a 1D memref (vector).
static mlir::Value MemrefToVector(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value memref, mlir::Value size,
                                  int64_t static_size) {
  assert(static_size >= 0 || static_size == mlir::ShapedType::kDynamicSize);
  auto memref_type = memref.getType().cast<mlir::MemRefType>();
  auto vec_type =
      mlir::MemRefType::get({static_size}, memref_type.getElementType());

  auto static_offsets = builder.getI64ArrayAttr({0});
  auto static_sizes = builder.getI64ArrayAttr({static_size});
  auto static_strided = builder.getI64ArrayAttr({1});

  auto empty = mlir::ValueRange();
  auto sizes = static_size == mlir::ShapedType::kDynamicSize
                   ? mlir::ValueRange(size)
                   : mlir::ValueRange();

  return builder.create<mlir::memref::ReinterpretCastOp>(
      loc, vec_type, memref, /*offsets=*/empty,
      /*sizes=*/sizes, /*strides=*/empty, static_offsets, static_sizes,
      static_strided);
}

// Convert 2D memref into a 0D memref (scalar).
static mlir::Value MemrefToScalar(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value memref) {
  auto memref_type = memref.getType().cast<mlir::MemRefType>();
  auto scalar_type = mlir::MemRefType::get({}, memref_type.getElementType());

  std::array<int64_t, 0> empty;
  return builder.create<mlir::memref::ReinterpretCastOp>(
      loc, scalar_type, memref, /*offset=*/0,
      /*sizes=*/empty, /*strides=*/empty);
}

mlir::LogicalResult LinalgMatmulSpecializationPattern::matchAndRewrite(
    mlir::linalg::MatmulOp matmul, mlir::PatternRewriter& rewriter) const {
  if (matmul->hasAttr("__tf_cpurt_specialized"))
    return rewriter.notifyMatchFailure(matmul,
                                       "operation was already specialized");

  auto rhs = matmul.getInputOperand(1)->get();
  auto lhs = matmul.getInputOperand(0)->get();
  auto out = matmul.getOutputOperand(0)->get();

  // We do not support inputs or outputs that are not contiguous in memory.
  if (!IsContiguousMemref(lhs) || !IsContiguousMemref(rhs) ||
      !IsContiguousMemref(out))
    return rewriter.notifyMatchFailure(
        matmul, "inputs and output must be contiguous memrefs");

  auto loc = matmul.getLoc();

  // Matmul dimensions: [m, k] x [k, n]
  mlir::Value m = rewriter.create<mlir::memref::DimOp>(loc, lhs, 0);
  mlir::Value k = rewriter.create<mlir::memref::DimOp>(loc, lhs, 1);
  mlir::Value n = rewriter.create<mlir::memref::DimOp>(loc, rhs, 1);

  // Matmul static dimensions if they are known (can be ShapedType::kDynamicSize
  // if not known statically).
  int64_t m_static = lhs.getType().cast<mlir::MemRefType>().getDimSize(0);
  int64_t k_static = lhs.getType().cast<mlir::MemRefType>().getDimSize(1);
  int64_t n_static = rhs.getType().cast<mlir::MemRefType>().getDimSize(1);

  auto one = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexType(),
                                               rewriter.getIndexAttr(1));
  auto m_is_one =
      rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, m, one);
  auto n_is_one =
      rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, n, one);

  auto m_not_one =
      rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, m, one);
  auto n_not_one =
      rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, n, one);

  // linalg.dot: n == 1 && m == 1
  auto is_dot_product = rewriter.create<mlir::AndOp>(loc, m_is_one, n_is_one);
  // linalg.vecmat m == 1 && n != 1
  auto is_vecmat = rewriter.create<mlir::AndOp>(loc, m_is_one, n_not_one);
  // linalg.matvec n == 1 && m != 1
  auto is_matvec = rewriter.create<mlir::AndOp>(loc, n_is_one, m_not_one);

  // Build a linalg.dot operation casting inputs to vectors.
  auto dot = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto lhs_vec = MemrefToVector(builder, nestedLoc, lhs, k, k_static);
    auto rhs_vec = MemrefToVector(builder, nestedLoc, rhs, k, k_static);
    auto out_scalar = MemrefToScalar(builder, nestedLoc, out);

    builder.create<mlir::linalg::DotOp>(nestedLoc,
                                        mlir::ValueRange({lhs_vec, rhs_vec}),
                                        mlir::ValueRange({out_scalar}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a linalg.vecmat operation casting lhs to vector.
  auto vecmat = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto lhs_vec = MemrefToVector(builder, nestedLoc, lhs, k, k_static);
    auto out_vec = MemrefToVector(builder, nestedLoc, out, n, n_static);

    builder.create<mlir::linalg::VecmatOp>(nestedLoc,
                                           mlir::ValueRange({lhs_vec, rhs}),
                                           mlir::ValueRange({out_vec}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a linalg.matvec operation casting rhs to vector.
  auto matvec = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto rhs_vec = MemrefToVector(builder, nestedLoc, rhs, k, k_static);
    auto out_vec = MemrefToVector(builder, nestedLoc, out, m, m_static);

    builder.create<mlir::linalg::MatvecOp>(nestedLoc,
                                           mlir::ValueRange({lhs, rhs_vec}),
                                           mlir::ValueRange({out_vec}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a generic linalg.matmul operation when it can't be matched to any of
  // the specializations.
  auto generic = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    llvm::SmallVector<mlir::Value> inputs = matmul.getInputOperands();
    llvm::SmallVector<mlir::Value> outputs = matmul.getOutputOperands();
    auto specialized =
        builder.create<mlir::linalg::MatmulOp>(nestedLoc, inputs, outputs);
    specialized->setAttr("__tf_cpurt_specialized", rewriter.getUnitAttr());
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // TODO(ezhulenev): Simplify to scf.switch operation.
  // if (is_dot_product) ===>>> linalg.dot    ------------------------------- //
  auto dispatch = rewriter.create<mlir::scf::IfOp>(
      loc, is_dot_product, dot,
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
        // else if (is_vecmat)  ===>>> linalg.vecmat    --------------------- //
        rewriter.create<mlir::scf::IfOp>(
            nestedLoc, is_vecmat, vecmat,
            [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
              // else if (is_matvec)  ===>>> linalg.matvec    --------------- //
              // else                 ===>>> linalg.matmul    --------------- //
              rewriter.create<mlir::scf::IfOp>(nestedLoc, is_matvec, matvec,
                                               generic);
              builder.create<mlir::scf::YieldOp>(nestedLoc);
            });
        builder.create<mlir::scf::YieldOp>(nestedLoc);
      });

  rewriter.replaceOp(matmul, dispatch.results());
  return mlir::success();
}

void LinalgMatmulSpecializationPass::runOnFunction() {
  mlir::FuncOp function = getFunction();
  mlir::MLIRContext* ctx = function.getContext();

  mlir::RewritePatternSet patterns(ctx);
  patterns.insert<LinalgMatmulSpecializationPattern>(ctx);

  (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
}

std::unique_ptr<mlir::FunctionPass> CreateLinalgMatmulSpecializationPass() {
  return std::make_unique<LinalgMatmulSpecializationPass>();
}

// -------------------------------------------------------------------------- //
// Break Tensorflow _Fused{Op} operations into primitive ones.
// -------------------------------------------------------------------------- //

namespace {
struct FissionPass : public FissionBase<FissionPass> {
  void runOnFunction() override;
};

struct FusedMatMulFission
    : public mlir::OpRewritePattern<mlir::TF::_FusedMatMulOp> {
  using OpRewritePattern<mlir::TF::_FusedMatMulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_FusedMatMulOp op,
      mlir::PatternRewriter& rewriter) const override;
};
}  // namespace

mlir::LogicalResult FusedMatMulFission::matchAndRewrite(
    mlir::TF::_FusedMatMulOp op, mlir::PatternRewriter& rewriter) const {
  auto loc = op.getLoc();
  auto type = op.getResult().getType();

  size_t n = op.fused_ops().size();

  // Extract fused operations from the operation attributes.
  mlir::StringAttr fusion0 =
      n > 0 ? op.fused_ops()[0].dyn_cast<mlir::StringAttr>() : nullptr;
  mlir::StringAttr fusion1 =
      n > 1 ? op.fused_ops()[1].dyn_cast<mlir::StringAttr>() : nullptr;

  // Match to supported operations
  bool is_bias_add = fusion0 && fusion0.getValue() == "BiasAdd";
  bool is_relu_activation = fusion1 && fusion1.getValue() == "Relu";

  // Create a simple MatMul operation from the fused one.
  auto matmul = [&]() -> mlir::TF::MatMulOp {
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
    return rewriter.create<mlir::TF::MatMulOp>(
        loc, type, lhs, rhs, op.transpose_a(), op.transpose_b());
  };

  // FusedMatMul[BiasAdd].
  if (n == 1 && is_bias_add) {
    rewriter.replaceOpWithNewOp<mlir::TF::BiasAddOp>(op, type, matmul(),
                                                     op.getOperand(2));
    return mlir::success();
  }

  // FusedMatMul[BiasAdd, Relu].
  if (n == 2 && is_bias_add && is_relu_activation) {
    auto biased = rewriter.create<mlir::TF::BiasAddOp>(loc, type, matmul(),
                                                       op.getOperand(2));
    rewriter.replaceOpWithNewOp<mlir::TF::ReluOp>(op, type, biased);
    return mlir::success();
  }

  return mlir::failure();
}

void FissionPass::runOnFunction() {
  mlir::FuncOp function = getFunction();
  mlir::MLIRContext* ctx = function.getContext();

  mlir::RewritePatternSet patterns(ctx);
  patterns.insert<FusedMatMulFission>(ctx);

  (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
}

std::unique_ptr<mlir::FunctionPass> CreateFissionPass() {
  return std::make_unique<FissionPass>();
}

// -------------------------------------------------------------------------- //
// Custom passes that are missing upstream.
// -------------------------------------------------------------------------- //

namespace {
// TODO(herhut): Remove this once leftover tensor_to_memref are handled in core.
struct RemoveUnusedBufferCastOperations
    : public mlir::PassWrapper<RemoveUnusedBufferCastOperations,
                               mlir::FunctionPass> {
  void runOnFunction() override;
};

// Adds a Tensorflow producer version to the module to enable shape inference.
struct AddTensorflowProducerVersion
    : public mlir::PassWrapper<AddTensorflowProducerVersion,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
};

// Use Linalg CodegenStrategy to tile linalg.matmul, linalg.matvec and
// linalg.vecmat operations.
struct CodegenStrategyForMatMulPass
    : public mlir::PassWrapper<CodegenStrategyForMatMulPass,
                               mlir::FunctionPass> {
  void runOnFunction() override;
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }
};
}  // namespace

void RemoveUnusedBufferCastOperations::runOnFunction() {
  getFunction().walk([](mlir::memref::BufferCastOp op) {
    // Drop all buffer_cast that have no more users. Currently this will
    // not happen, as tensor_to_memref has a side-effect. See
    // https://reviews.llvm.org/D91967 for a discussion.
    if (op.memref().getUsers().empty()) {
      op.erase();
    }
  });
}

void CodegenStrategyForMatMulPass::runOnFunction() {
  // Promote tiles to full buffers allocated on the stack.
  mlir::linalg::LinalgPromotionOptions full_alloca_promotion;
  full_alloca_promotion.setUseFullTileBuffersByDefault(true).setUseAlloca(true);

  // Vector contraction options.
  mlir::vector::VectorTransformsOptions vector_transforms_ops;
  vector_transforms_ops.setVectorTransformsOptions(
      mlir::vector::VectorContractLowering::OuterProduct);

  // Vector transfer options.
  mlir::VectorTransferToSCFOptions vector_transfer_opts;
  vector_transfer_opts.setUnroll(true);

  // TODO(ezhulenev): Set up tiling options depending on the target machine.

  // Tile and vectorize linalg.matmul operations.
  mlir::linalg::LinalgTilingOptions matmul_tiling;
  matmul_tiling.setTileSizes({12, 32, 16});

  mlir::linalg::CodegenStrategy matmul_strategy;
  matmul_strategy.tile<mlir::linalg::MatmulOp>(matmul_tiling)
      .promote<mlir::linalg::MatmulOp>(full_alloca_promotion)
      .vectorize<mlir::linalg::MatmulOp>()
      .setVectorTransformsOptions(vector_transforms_ops)
      .setVectorTransferToSCFOptions(vector_transfer_opts);
  matmul_strategy.transform(getFunction());

  // Tile and vectorize linalg.vecmat operations. Interchange loop order to
  // linearly read from the matrix memref.
  mlir::linalg::LinalgTilingOptions vecmat_tiling;
  vecmat_tiling.setTileSizes({16, 8}).setInterchange({1, 0});

  mlir::linalg::CodegenStrategy vecmat_strategy;
  vecmat_strategy.tile<mlir::linalg::VecmatOp>(vecmat_tiling)
      .promote<mlir::linalg::VecmatOp>(full_alloca_promotion)
      .vectorize<mlir::linalg::VecmatOp>()
      .setVectorTransformsOptions(vector_transforms_ops)
      .setVectorTransferToSCFOptions(vector_transfer_opts);
  vecmat_strategy.transform(getFunction());
}

static std::unique_ptr<mlir::FunctionPass>
CreateCodegenStrategyForMatMulPass() {
  return std::make_unique<CodegenStrategyForMatMulPass>();
}

void AddTensorflowProducerVersion::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // Tensorflow producer version does not really impact anything during the
  // shape inference. Set it to `0` (any random number will do the work) to
  // bypass attribute checks.
  mlir::Builder builder(module);
  auto version = builder.getNamedAttr("producer", builder.getI32IntegerAttr(0));
  module->setAttr("tf.versions", builder.getDictionaryAttr({version}));
}

// -------------------------------------------------------------------------- //
// Cluster operations based on the TF CPURT clustering policy.
// -------------------------------------------------------------------------- //

namespace {
using llvm::ArrayRef;

using mlir::TFDevice::Cluster;
using mlir::TFDevice::ClusteringPolicySet;
using mlir::TFDevice::CreateClusterOp;
using mlir::TFDevice::FindClustersInTheBlock;

struct ClusteringPass : public ClusteringBase<ClusteringPass> {
  ClusteringPass() = default;
  ClusteringPass(ArrayRef<std::string> cluster_oplist, int cluster_min_size) {
    oplist = cluster_oplist;
    min_cluster_size = cluster_min_size;
  }

  void runOnFunction() override;
};
}  // anonymous namespace

void ClusteringPass::runOnFunction() {
  ClusteringPolicySet policies;

  // Parse clustering tier and operations filter from the oplist.
  llvm::DenseSet<llvm::StringRef> opset;
  llvm::Optional<CpurtClusteringTier> tier;

  for (const auto& op : oplist) {
    if (op == "tier1") {
      tier = CpurtClusteringTier::kTier1;
    } else if (op == "all") {
      tier = CpurtClusteringTier::kAll;
    } else {
      opset.insert(op);
    }
  }

  // Run clustering only if the clustering tier or supported operations are
  // explicitly defined by the oplist.
  if (!tier.hasValue() && opset.empty()) return;

  // If the clustering tier is not defined, it means that the opset will later
  // filter supported operations, so it's ok to use `all` tier.
  populateTfCpurtClusteringPolicies(policies,
                                    tier.getValueOr(CpurtClusteringTier::kAll));

  // If opset is not empty restrict operations that are enabled for clustering.
  auto filter = [&](mlir::Operation* op) -> bool {
    return opset.empty() || opset.contains(op->getName().getStringRef());
  };

  // Annotate all formed clusters with an attribute.
  auto policy = mlir::StringAttr::get(&getContext(), "tfrt.auto-fusion");

  getFunction().walk([&](mlir::Block* block) {
    for (Cluster& cluster : FindClustersInTheBlock(block, policies, filter)) {
      // Do not create too small clusters.
      if (cluster.operations.size() < min_cluster_size) continue;
      // Verify that JIT runtime can compile the cluster.
      if (failed(VerifyCluster(cluster))) continue;

      CreateClusterOp(cluster, policy);
    }
  });
}

std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass() {
  return std::make_unique<ClusteringPass>();
}

std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass(
    llvm::ArrayRef<std::string> oplist, int min_cluster_size) {
  return std::make_unique<ClusteringPass>(oplist, min_cluster_size);
}

// -------------------------------------------------------------------------- //
// Assemble a TF-CPURT pipeline to lower from Tensorflow dialects to Linalg on
// buffers via progressive lowering to MHLO and Linalg.
// -------------------------------------------------------------------------- //

void CreateTfCpuRtPipeline(mlir::OpPassManager& pm) {
  // Break Tensorflow fused operations into primitive operations before
  // lowering to HLO.
  pm.addNestedPass<mlir::FuncOp>(CreateFissionPass());

  // Run shape inference to propagate potentially specialized input shapes.
  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Transform TF operation to HLO.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeTFPass());

  // Move up broadcasting operations to allow for more fusion opportunities.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform HLO operations to LinAlg and fuse them.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());

  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateLowerIndexCastPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateShapeSimplification());
  pm.addNestedPass<mlir::FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());

  // Bufferize Linalg on tensors program.
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(
      mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  // Turn tensor constants into global memrefs.
  // TODO(kramerb): Expose the patterns and add them to the bufferize passes.
  pm.addPass(mlir::createTensorConstantBufferizePass());
  // Run canonicalizer for dead code removal.
  pm.addPass(mlir::createCanonicalizerPass());
  // tensor_to_memref is not considered dead currently, fix that directly.
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<RemoveUnusedBufferCastOperations>());
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::kernel_gen::transforms::CreateFinalBufferizePass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgTrivialCopyRemovalPass());

  // Specilize linalg.matmul to linalg.dot, linalg.matvec or linalg.vecmat, and
  // immediately canonicalize to clean up not taken branches.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgMatmulSpecializationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Tile and vectorize linalg operation using Linalg Codegen Strategy.
  pm.addNestedPass<mlir::FuncOp>(CreateCodegenStrategyForMatMulPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static mlir::PassPipelineRegistration<> tf_cpurt_pipeline(
    "tf-cpurt-pipeline",
    "Convert Tensorflow dialect to TFRT's CPURT compatible dialects",
    CreateTfCpuRtPipeline);

}  // namespace tensorflow
