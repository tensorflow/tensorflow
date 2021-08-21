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

#include <sys/types.h>

#include <string>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

using llvm::ArrayRef;
using llvm::DenseMap;
using llvm::SmallVector;

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::ConstantIndexOp;
using mlir::ConstantOp;
using mlir::DenseIntElementsAttr;
using mlir::dyn_cast;
using mlir::dyn_cast_or_null;
using mlir::failure;
using mlir::FuncOp;
using mlir::FunctionPass;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::success;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;

namespace linalg = mlir::linalg;
namespace mhlo = mlir::mhlo;
namespace shape = mlir::shape;
namespace tensor = mlir::tensor;

// -------------------------------------------------------------------------- //
// Collection of utility functions to work with symbolic shapes.
// -------------------------------------------------------------------------- //

using SymbolicShape = SmallVector<int64_t>;
using SymbolicShapes = DenseMap<Value, SymbolicShape>;

// Returns the symbolic shape of the value (fully known static shape is a valid
// symbolic shape without any symbolic dimensions).
llvm::Optional<SymbolicShape> GetSymbolicShape(
    Value value, const SymbolicShapes& symbolic_shapes) {
  // Check if the value type has static shape.
  auto shaped = value.getType().dyn_cast<ShapedType>();
  if (shaped && shaped.hasStaticShape()) {
    auto shape = shaped.getShape();
    return SymbolicShape(shape.begin(), shape.end());
  }

  // Check if we know the symbolic shape from the symbolic shapes map.
  auto it = symbolic_shapes.find(value);
  if (it != symbolic_shapes.end()) {
    return it->getSecond();
  }

  return llvm::None;
}

// Collect symbolic shapes from the shape values.
LogicalResult GetSymbolicShapes(ValueRange shapes,
                                const SymbolicShapes& symbolic_shapes,
                                SmallVector<SymbolicShape>& bcasted_shapes) {
  for (Value operand : shapes) {
    Operation* defined_by_op = operand.getDefiningOp();
    if (!defined_by_op) return failure();

    // Check if the shape is a constant.
    if (auto const_shape = dyn_cast<shape::ConstShapeOp>(defined_by_op)) {
      bcasted_shapes.emplace_back(const_shape.shape().getValues<int64_t>());
      continue;
    }

    // Check if the shape is a result of shape.shape_of operation.
    if (auto shape_of = dyn_cast<shape::ShapeOfOp>(defined_by_op)) {
      if (auto shape = GetSymbolicShape(shape_of.arg(), symbolic_shapes)) {
        bcasted_shapes.emplace_back(std::move(*shape));
        continue;
      }
    }

    // We couldn't find the symbolic shape of the operand.
    return failure();
  }

  return success();
}

// Collect symbolic shapes of the `shape::CstrBroadcastableOp` operands.
LogicalResult GetSymbolicShapes(shape::CstrBroadcastableOp op,
                                const SymbolicShapes& symbolic_shapes,
                                SmallVector<SymbolicShape>& bcasted_shapes) {
  return GetSymbolicShapes(op.shapes(), symbolic_shapes, bcasted_shapes);
}

// Collect symbolic shapes of the `shape::BroadcastOp` operands and by following
// chains of broadcast operations (broadcast the result of the broadcast).
LogicalResult GetSymbolicShapes(shape::BroadcastOp op,
                                const SymbolicShapes& symbolic_shapes,
                                SmallVector<Value>& bcasted_values,
                                SmallVector<SymbolicShape>& bcasted_shapes) {
  SmallVector<shape::BroadcastOp> worklist = {op};
  bcasted_values.reserve(op.getNumOperands());
  bcasted_shapes.reserve(op.getNumOperands());

  while (!worklist.empty()) {
    shape::BroadcastOp bcast = worklist.pop_back_val();

    for (Value shape : bcast.getOperands()) {
      if (auto bcast_arg = shape.getDefiningOp<shape::BroadcastOp>()) {
        worklist.push_back(bcast_arg);
        continue;
      }
      bcasted_values.push_back(shape);
    }
  }

  return GetSymbolicShapes(bcasted_values, symbolic_shapes, bcasted_shapes);
}

// Joins broadcasted symbolic shapes with the `shape` to get the output shape
// after broadcasting. Returns error if can't prove statically that the
// broadcast will be successful.
//
// TODO(ezhulenev): What to do with dimensions statically known to be zero?
// Numpy can only broadcast [0] with [1], however Tensorflow can broadcast [0]
// with any dimension size, and produces dimension of size [0]. Currently we'll
// conservatively return failure and will not proceed with a rewrite.
LogicalResult JoinSymbolicShapes(ArrayRef<SymbolicShape> bcasted_shapes,
                                 SymbolicShape& shape) {
  for (unsigned d = 0; d < shape.size(); ++d) {
    for (const SymbolicShape& bcasted_shape : bcasted_shapes) {
      // Conservatively avoid dealing with empty tensors.
      if (shape[d] == 0 || bcasted_shape[d] == 0) return failure();

      // dim x 1 -> dim
      if (bcasted_shape[d] == 1) continue;

      // 1 x dim -> dim
      if (shape[d] == 1) {
        shape[d] = bcasted_shape[d];
        continue;
      }

      // Can't statically prove that broadcast will be successful.
      if (shape[d] != bcasted_shape[d]) return failure();
    }
  }
  return success();
}

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

// -------------------------------------------------------------------------- //

// Rewrite shape.cstr_broadcastable with constant witness if can prove that
// shapes are broadcastable from the symbolic shapes.

class CstrBroadcastableOpLowering
    : public mlir::OpRewritePattern<shape::CstrBroadcastableOp> {
 public:
  using Base = OpRewritePattern<shape::CstrBroadcastableOp>;

  CstrBroadcastableOpLowering(MLIRContext* ctx,
                              SymbolicShapes& symbolic_shapes);

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                mlir::PatternRewriter& rewriter) const override;

 private:
  SymbolicShapes& symbolic_shapes_;
};

CstrBroadcastableOpLowering::CstrBroadcastableOpLowering(
    MLIRContext* ctx, SymbolicShapes& symbolic_shapes)
    : Base(ctx), symbolic_shapes_(symbolic_shapes) {}

LogicalResult CstrBroadcastableOpLowering::matchAndRewrite(
    shape::CstrBroadcastableOp op, mlir::PatternRewriter& rewriter) const {
  // Collect symbolic shapes of the operands.
  SmallVector<SymbolicShape> bcasted_shapes;
  if (failed(GetSymbolicShapes(op, symbolic_shapes_, bcasted_shapes)))
    return failure();

  // Find the maximum rank of the operands.
  size_t rank = 0;
  for (const SymbolicShape& bcasted_shape : bcasted_shapes)
    rank = std::max(rank, bcasted_shape.size());

  // Prepend `1` to all shapes to match the maximum rank.
  for (size_t i = 0; i < bcasted_shapes.size(); ++i) {
    bcasted_shapes[i].insert(bcasted_shapes[i].begin(),
                             rank - bcasted_shapes[i].size(), 1);
  }

  // Pick the first shape as the initialization value for the output shape, and
  // check if the broadcast can be statically proven to be successful.
  SymbolicShape output_shape = bcasted_shapes[0];
  if (failed(JoinSymbolicShapes(bcasted_shapes, output_shape)))
    return failure();

  // Replace constraint with a true witness.
  rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);

  return success();
}

// -------------------------------------------------------------------------- //

// Rewrite mhlo.dynamic_broadcast_in_dim operation into linalg.generic operation
// if can infer the indexing maps for the operand from the symbolic shapes.
class DynamicBroadcastInDimOpLowering
    : public mlir::OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using Base = OpRewritePattern<mhlo::DynamicBroadcastInDimOp>;

  DynamicBroadcastInDimOpLowering(MLIRContext* ctx,
                                  SymbolicShapes& symbolic_shapes);

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                mlir::PatternRewriter& rewriter) const override;

 private:
  SymbolicShapes& symbolic_shapes_;
};

DynamicBroadcastInDimOpLowering::DynamicBroadcastInDimOpLowering(
    MLIRContext* ctx, SymbolicShapes& symbolic_shapes)
    : Base(ctx), symbolic_shapes_(symbolic_shapes) {}

LogicalResult DynamicBroadcastInDimOpLowering::matchAndRewrite(
    mhlo::DynamicBroadcastInDimOp op, mlir::PatternRewriter& rewriter) const {
  MLIRContext* ctx = getContext();

  auto in_type = op.operand().getType().dyn_cast<RankedTensorType>();
  auto out_type = op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!in_type || !out_type) return failure();

  // Check that broadcast is right-aligned (numpy style), so that operand
  // dimensions broadcasted to match inner-most dimensions of the output.
  auto bcast_dims = op.broadcast_dimensions().getValues<int64_t>();
  auto expected_bcast_dims = llvm::seq<int64_t>(
      out_type.getRank() - in_type.getRank(), out_type.getRank());
  if (!llvm::equal(bcast_dims, expected_bcast_dims)) return failure();

  // Check if the output shape is defined by the broadcast operation.
  Operation* output_dimensions_op = op.output_dimensions().getDefiningOp();
  auto bcast = dyn_cast_or_null<shape::BroadcastOp>(output_dimensions_op);
  if (!bcast) return failure();

  // Collect symbolic shapes (and the values that define these shapes) from the
  // broadcast operation operands.
  SmallVector<Value> bcasted_values;
  SmallVector<SymbolicShape> bcasted_shapes;
  if (failed(GetSymbolicShapes(bcast, symbolic_shapes_, bcasted_values,
                               bcasted_shapes)))
    return failure();

  // Get the symbolic shape of the broadcasted operand.
  auto operand_shape = GetSymbolicShape(op.operand(), symbolic_shapes_);
  if (!operand_shape.hasValue()) return failure();

  SymbolicShape input_shape = std::move(*operand_shape);

  // Find the rank of the broadcast result (maximum rank of the operands).
  size_t rank = input_shape.size();
  for (const SymbolicShape& bcasted_shape : bcasted_shapes)
    rank = std::max(rank, bcasted_shape.size());

  // Prepend `1` to all shapes to match the maximum rank. Keep track of the
  // number of `1` prepended to each shape, we'll need it later to create
  // correct `tensor.dim` operations.
  size_t input_shape_ext = rank - input_shape.size();
  input_shape.insert(input_shape.begin(), input_shape_ext, 1);

  SmallVector<size_t> bcasted_shapes_ext(bcasted_shapes.size());
  for (size_t i = 0; i < bcasted_shapes.size(); ++i) {
    SymbolicShape& bcasted_shape = bcasted_shapes[i];
    bcasted_shapes_ext[i] = rank - bcasted_shape.size();
    bcasted_shape.insert(bcasted_shape.begin(), bcasted_shapes_ext[i], 1);
  }

  // Compute the output symbolic shape.
  SymbolicShape output_shape = input_shape;
  if (failed(JoinSymbolicShapes(bcasted_shapes, output_shape)))
    return failure();

  Location loc = op.getLoc();

  // Resolve symbolic shape dimension to an MLIR Value.
  auto resolve_dimension = [&](int64_t dim) -> Value {
    // Create constant operation for known constant dimensions.
    if (dim >= 0) return rewriter.create<ConstantIndexOp>(loc, dim);

    // Try to find the symbolic dimension in the input operand shape.
    for (unsigned d = 0; d < rank; ++d) {
      if (input_shape[d] == dim) {
        return rewriter.create<tensor::DimOp>(loc, op.operand(),
                                              d - input_shape_ext);
      }
    }

    // Otherwise try to find the symbolic shape in the broadcasted shapes.
    for (unsigned i = 0; i < bcasted_shapes.size(); ++i) {
      for (unsigned d = 0; d < rank; ++d) {
        if (bcasted_shapes[i][d] == dim) {
          Operation* operand_src = bcasted_values[i].getDefiningOp();

          // Shape defined by the shape.const_shape operation.
          if (auto shape = dyn_cast_or_null<shape::ConstShapeOp>(operand_src)) {
            return rewriter.create<ConstantOp>(
                loc, shape.shape().getValue({static_cast<unsigned>(dim)}));
          }

          // Shape defined by the shape.shape_of operation.
          if (auto shape_of = dyn_cast_or_null<shape::ShapeOfOp>(operand_src)) {
            return rewriter.create<tensor::DimOp>(loc, shape_of.arg(),
                                                  d - bcasted_shapes_ext[i]);
          }
        }
      }
    }

    assert(false && "couldn't resolve symbolic shape to a value");
    return Value();
  };

  // Resolve dynamic output dimensions for the `linalg.init_tensor` operation.
  SmallVector<Value> output_dyn_dimensions;
  for (size_t d = 0; d < rank; ++d) {
    int64_t output_dim = output_shape[d];

    // Skip static output dimensions, they will be resolved from the shape.
    if (output_dim >= 0) continue;

    // Resolve the dynamic size of the output dimension.
    Value output_dyn_dim = resolve_dimension(output_shape[d]);
    if (!output_dyn_dim) return failure();

    output_dyn_dimensions.push_back(output_dyn_dim);
  }

  // Create a linalg.tensor_init operation to initialize output.
  Value init = rewriter.create<linalg::InitTensorOp>(loc, output_dyn_dimensions,
                                                     out_type.getShape(),
                                                     out_type.getElementType());

  // Output indexing map is an identity with `rank` number of loops.
  AffineMap output_map = AffineMap::getMultiDimIdentityMap(rank, ctx);

  // For input indexing map replace all broadcasted dimensions with constant `0`
  // affine expression, and all non-broadcasted dimensions with identity.
  SmallVector<AffineExpr> input_map_exprs;
  for (size_t d = input_shape_ext; d < input_shape.size(); ++d) {
    bool extend = input_shape[d] == 1 && output_shape[d] != 1;
    input_map_exprs.push_back(extend ? rewriter.getAffineConstantExpr(0)
                                     : rewriter.getAffineDimExpr(d));
  }

  AffineMap input_map = AffineMap::get(/*dimCount=*/rank,
                                       /*symbolCount=*/0, input_map_exprs, ctx);

  // All iterators are parallel.
  SmallVector<llvm::StringRef> iterator_types(rank, "parallel");

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      op, /*resultTensorTypes=*/TypeRange{init.getType()},
      /*inputs=*/ValueRange{op.operand()},
      /*outputs=*/ValueRange{init},
      /*indexingMaps=*/llvm::makeArrayRef({input_map, output_map}),
      /*iteratorTypes=*/iterator_types,
      [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
        nested_builder.create<linalg::YieldOp>(nested_loc, args[0]);
      });

  return success();
}

// -------------------------------------------------------------------------- //
// Optimize function based on the symbolic shape attributes.
// -------------------------------------------------------------------------- //

// Returns symbolic shapes from the function argument attributes.
//
// Example:
//   func @compute(
//     %arg0: tensor<?xf32> {cpurt.symbolic_shape = dense<-2> : tensor<1xi64>},
//     %arg1: tensor<?xf32> {cpurt.symbolic_shape = dense<-2> : tensor<1xi64>})
//   } { ... }
//
// Symbolic shape is a negative value smaller than `-1`. The concrete value
// is not known at compile time, and in this particular example it is only known
// that both arguments have the same shape.
//
// TODO(ezhulenev): Add symbolic shape attribute verifier to the cpurt dialect.
SymbolicShapes GetOperandsSymbolicShapes(FuncOp func) {
  SymbolicShapes symbolic_shapes;

  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    Value arg = func.getArgument(i);

    // Check if the argument has a symbolic shape attribute.
    auto shape =
        func.getArgAttrOfType<DenseIntElementsAttr>(i, "cpurt.symbolic_shape");
    if (!shape) continue;

    // Check that argument type matches the symbolic shape attribute.
    auto arg_type = arg.getType().dyn_cast<RankedTensorType>();
    assert(arg_type && "argument must be a ranked tensor");
    assert(arg_type.getRank() == shape.getNumElements());
    (void)arg_type;

    auto dims = shape.getValues<ssize_t>();
    symbolic_shapes.try_emplace(arg, SymbolicShape(dims.begin(), dims.end()));
  }

  return symbolic_shapes;
}

struct SymbolicShapeOptimizationPass
    : public SymbolicShapeOptimizationBase<SymbolicShapeOptimizationPass> {
  SymbolicShapeOptimizationPass() = default;

  explicit SymbolicShapeOptimizationPass(bool constraints_only) {
    this->optimize_only_constraints = constraints_only;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();

    // Check if we have any symbolic shape information.
    SymbolicShapes symbolic_shapes = GetOperandsSymbolicShapes(func);
    if (symbolic_shapes.empty()) return;

    MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Rewrite constraints based on the symbolic shapes.
    patterns.insert<CstrBroadcastableOpLowering>(ctx, symbolic_shapes);

    // Move broadcasts up across mhlo operations to enable more opportunities
    // for constraints and broadcasts optimizations. These patterns are only
    // applicable if we do not lower mhlo broadcasts to linalg.generic.
    if (optimize_only_constraints)
      mlir::mhlo::PopulateBroadcastsPropagationPatterns(ctx, &patterns);

    // Rewrite broadcasts based on the symbolic shapes if enabled.
    if (!optimize_only_constraints)
      patterns.insert<DynamicBroadcastInDimOpLowering>(ctx, symbolic_shapes);

    // Add shape dialect canonicalization patterns to fold shape operations
    // after constraints are replaced with constant witness.
    mlir::Dialect* shape_dialect = ctx->getLoadedDialect<shape::ShapeDialect>();
    for (auto* op : ctx->getRegisteredOperations()) {
      if (op->dialect.getTypeID() == shape_dialect->getTypeID())
        op->getCanonicalizationPatterns(patterns, ctx);
    }

    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<FunctionPass> CreateSymbolicShapeOptimizationPass(
    bool constraints_only) {
  return std::make_unique<SymbolicShapeOptimizationPass>(constraints_only);
}

}  // namespace tensorflow
