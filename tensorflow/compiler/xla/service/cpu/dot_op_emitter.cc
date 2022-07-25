/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/mlir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/cpu/tiled_dot_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using llvm_ir::SetToFirstInsertPoint;

namespace cpu {
namespace {
// Returns true if we should call into multi-threaded Eigen routines.
bool ShouldUseMultiThreadedEigen(const HloModuleConfig& config) {
  return config.debug_options().xla_cpu_multi_thread_eigen();
}

// Represents a dot operation.  We use this in lieu of an `HloInstruction`
// because we want to be able to create this for the "inner" dot operation in a
// batch dot, for which there is no separate HLO instruction.
struct DotInfo {
  Shape lhs_shape;
  Shape rhs_shape;
  Shape result_shape;
  DotDimensionNumbers dim_nums;

  DotInfo() = default;

  explicit DotInfo(const HloInstruction& instr) {
    CHECK_EQ(instr.opcode(), HloOpcode::kDot);
    lhs_shape = instr.operand(0)->shape();
    rhs_shape = instr.operand(1)->shape();
    result_shape = instr.shape();
    dim_nums = instr.dot_dimension_numbers();
  }
};

// Dictates how a dot operation is implemented.
enum class DotImplementationStrategy {
  // The dot operation is lowered into LLVM IR that implements a naive nested
  // loop that computes the result one element at a time.  This is our
  // "fallback"; we don't really want this to kick in for any non-trival dot
  // operation.
  kNaiveLlvmIr,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Vector operation.  This strategy also allows fusing in a bias add
  // into the dot.  The matrix can be row major or column major, both are
  // supported.
  kTiledLlvmIrGemv,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Matrix operation.  No fusions are supported.  The two inputs
  // and the output have to be row major.
  kTiledLlvmIrGemm,

  // The dot operation is lowered into linalg.matmul op and lowered to LLVM IR.
  kLinalgMatmul,

  // The dot operation is lowered into a call into an Eigen routine.  No fusions
  // are supported today.  The two inputs and the output have to be row major.
  // However, we do allow transposing either the LHS or the RHS as part of the
  // GEMM -- we expose this flexibility as flexibility in the contraction
  // dimensions, but we can also see this as flexibility in the input layouts.
  kEigen,
};

// Returns the implementation strategy for a dot with the configuration
// `dot_info`.
DotImplementationStrategy GetDotImplementationStrategy(
    const HloModuleConfig& config, const DotInfo& dot_info,
    const TargetMachineFeatures& target_machine_features);

// Helper class for emitting LLVM IR to perform the dot operation.
class DotOpEmitter {
 public:
  explicit DotOpEmitter(DotInfo dot_info, std::string dot_hlo_name,
                        const llvm_ir::IrArray& target_array,
                        const llvm_ir::IrArray& lhs_array,
                        const llvm_ir::IrArray& rhs_array,
                        const llvm_ir::IrArray* addend_array,
                        llvm::Value* executable_run_options_value,
                        llvm::IRBuilder<>* b, mlir::MLIRContext* mlir_context,
                        const HloModuleConfig& hlo_module_config,
                        const TargetMachineFeatures& target_machine_features);

  // Emits the IR to perform the dot operation.
  Status Emit();

  // Emits the IR to perform the batch dot operation.
  Status EmitBatch();

 private:
  // Emits instructions to perform a scalar dot product (a multiply of the
  // LHS and RHS) and store the results in the target.
  Status EmitScalarDot();

  // Emits a call to the CPU runtime to perform the matrix multiply.
  Status EmitCallToRuntime();

  // Emits a call to the CPU runtime to perform the batch matrix multiply.
  Status EmitCallToBatchRuntime();

  // Represents the dimensions of a matrix-matrix multiply operation.
  struct MatMultDims {
    // The number of rows in the LHS.
    int64_t m;

    // The number of columns in the LHS, which is also must be equal to the
    // number of rows in the RHS.
    int64_t k;

    // The number of columns on the RHS.
    int64_t n;

    // True if the LHS matrix is column major.
    bool lhs_column_major;

    // True if the LHS contraction dimension is 1.
    bool lhs_canonical;

    // True if the RHS matrix is column major.
    bool rhs_column_major;

    // True if the RHS contraction dimension is 0.
    bool rhs_canonical;
  };

  // Get the MatMultDims instance for the dot product this DotOpEmitter
  // represents.  Precondition: the dot is of rank 2 (and thus its operands are
  // of rank 2 as well).
  MatMultDims GetMatMultDims() const;

  // Get the MatMultDims instance for the dot product this DotOpEmitter
  // represents.  Precondition: the dot is of rank 3 (and thus its operands are
  // of rank 3 as well).
  MatMultDims GetBatchMatMultDims() const;

  // Lowers the dot operation as a tiled Matrix*Vector loop.
  void EmitTiledLlvmIrGemv();

  // Lowers the dot operation as a tiled Matrix*Matrix loop.
  void EmitTiledLlvmIrGemm();

  // Lowers the dot operation through MLIR's linalg.matmul.
  Status EmitLinalgMatmul();

  // Lowers the dot operation as a naive nested loop that computes the result
  // one element at a time.
  void EmitNaiveLlvmIrGemm();

  // When doing a tiled GEMV in LLVM IR, a "tile" consists of this many vector
  // registers.
  int64_t GetGemvTilingFactor() const {
    const int64_t kDefaultTilingFactor = 8;
    return options::LlvmIrGemvTilingFactor(hlo_module_config_)
        .value_or(kDefaultTilingFactor);
  }

  std::tuple<int64_t, int64_t, int64_t> GetGemmTileSize() const {
    // Tuned for broadwell - Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
    //
    // TODO(b/80093688): Tune for other architectures and centralize this
    // information in one place.
    const std::tuple<int64_t, int64_t, int64_t> kDefaultTileSize =
        std::tuple<int64_t, int64_t, int64_t>(11, 9, 1);
    return options::LlvmIrGemmTileSize(hlo_module_config_)
        .value_or(kDefaultTileSize);
  }

  std::array<int64_t, 3> GetMlirGemmTileSize() const {
    // Tile by 4 x registers x register size. This was picked by running
    // small matmuls on Haswell and Skylake. There's a lot of room for
    // improvement here.
    constexpr int64_t kDefaultTileSizeForM = 4;
    int64_t elements_per_register =
        target_machine_features_.vector_register_num_elements(
            *b_->GetInsertBlock()->getParent(),
            dot_info_.result_shape.element_type());
    int64_t num_registers = target_machine_features_.vector_register_count(
        *b_->GetInsertBlock()->getParent());
    return {{kDefaultTileSizeForM, num_registers, elements_per_register}};
  }

  DotInfo dot_info_;
  std::string dot_hlo_name_;
  const llvm_ir::IrArray& target_array_;
  const llvm_ir::IrArray& lhs_array_;
  const llvm_ir::IrArray& rhs_array_;
  const llvm_ir::IrArray* addend_array_;
  llvm::Value* executable_run_options_value_;
  llvm::IRBuilder<>* b_;
  mlir::MLIRContext* mlir_context_;
  const HloModuleConfig& hlo_module_config_;
  const TargetMachineFeatures& target_machine_features_;
};
}  // namespace

DotOpEmitter::DotOpEmitter(
    DotInfo dot_info, std::string dot_hlo_name,
    const llvm_ir::IrArray& target_array, const llvm_ir::IrArray& lhs_array,
    const llvm_ir::IrArray& rhs_array, const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
    mlir::MLIRContext* mlir_context, const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features)
    : dot_info_(std::move(dot_info)),
      dot_hlo_name_(std::move(dot_hlo_name)),
      target_array_(target_array),
      lhs_array_(lhs_array),
      rhs_array_(rhs_array),
      addend_array_(addend_array),
      executable_run_options_value_(executable_run_options_value),
      b_(b),
      mlir_context_(mlir_context),
      hlo_module_config_(hlo_module_config),
      target_machine_features_(target_machine_features) {}

Status DotOpEmitter::EmitLinalgMatmul() {
  Shape operand_shapes[] = {dot_info_.lhs_shape, dot_info_.rhs_shape};
  llvm::Value* operand_ptrs[] = {lhs_array_.GetBasePointer(),
                                 rhs_array_.GetBasePointer()};
  llvm::Value* target_ptr = target_array_.GetBasePointer();

  // Zero out the output buffer.
  int64_t size_bytes = ShapeUtil::ByteSizeOf(dot_info_.result_shape);
  b_->CreateMemSet(target_ptr, b_->getInt8(0), /*Size=*/size_bytes,
                   /*Align=*/llvm::MaybeAlign(1));

  std::string name =
      absl::StrCat("linalgMatMul_", dot_info_.result_shape.ToString(true), "_",
                   dot_info_.lhs_shape.ToString(true), "_",
                   dot_info_.rhs_shape.ToString(true));

  return EmitMlirFuncAndCall(
      mlir_context_, b_, dot_info_.result_shape, operand_shapes, target_ptr,
      operand_ptrs, name,
      [&](mlir::OpBuilder* builder, mlir::func::FuncOp function) {
        CHECK_EQ(dot_info_.dim_nums.lhs_contracting_dimensions_size(), 1);
        CHECK_EQ(dot_info_.dim_nums.rhs_contracting_dimensions_size(), 1);
        mlir::MLIRContext* context = builder->getContext();
        mlir::Value a = function.getArgument(0), b = function.getArgument(1),
                    c = function.getArgument(2);

        llvm::SmallVector<mlir::AffineExpr, 2> b_exprs(
            dot_info_.lhs_shape.rank());
        llvm::SmallVector<mlir::AffineExpr, 2> c_exprs(
            dot_info_.rhs_shape.rank());

        llvm::SmallVector<mlir::AffineExpr, 2> parallel_exprs;
        mlir::AffineExpr reduce_expr;
        for (int i = 0; i != dot_info_.result_shape.rank(); ++i) {
          parallel_exprs.push_back(mlir::getAffineDimExpr(i, context));
        }
        reduce_expr =
            mlir::getAffineDimExpr(dot_info_.result_shape.rank(), context);

        // The reduction expr is shared for both inputs.
        b_exprs[dot_info_.dim_nums.lhs_contracting_dimensions(0)] = reduce_expr;
        c_exprs[dot_info_.dim_nums.rhs_contracting_dimensions(0)] = reduce_expr;

        // Fill in the remaining parallel exprs.
        int par_expr_num = 0;
        for (auto* v : {&b_exprs, &c_exprs}) {
          for (auto& e : *v) {
            if (!e) {
              e = parallel_exprs[par_expr_num++];
            }
          }
        }

        llvm::SmallVector<llvm::StringRef, 4> iteratorTypes(
            parallel_exprs.size(), toString(mlir::IteratorType::Parallel));
        iteratorTypes.push_back(toString(mlir::IteratorType::Reduction));
        builder->create<mlir::linalg::GenericOp>(
            function.getLoc(),
            /*inputs=*/mlir::ValueRange{b, c},
            /*outputs=*/mlir::ValueRange{a},
            /*indexingMaps=*/
            mlir::AffineMap::inferFromExprList(
                {b_exprs, c_exprs, parallel_exprs}),
            /*iteratorTypes=*/iteratorTypes,
            [](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
              mlir::ArithBuilder ab(b, loc);
              mlir::Value mul = ab.mul(args[0], args[1]);
              mlir::Value add = ab.add(mul, args[2]);
              b.create<mlir::linalg::YieldOp>(loc, add);
            });
        builder->create<mlir::func::ReturnOp>(function.getLoc());

        mlir::linalg::LinalgTilingOptions tilingOptions;
        tilingOptions = tilingOptions.setTileSizes(GetMlirGemmTileSize());
        // TODO: this has been retired upstream, reevaluate whether this 
        // path really needs it or if it is even relevant anymore.
        // int64_t alignment =
        //     target_machine_features_.minimum_alignment_for_allocation(
        //         ShapeUtil::ByteSizeOf(dot_info_.result_shape));
        mlir::linalg::CodegenStrategy strategy;
        strategy
            .tile(mlir::linalg::GenericOp::getOperationName(), tilingOptions)
            // TODO: this has been retired upstream, reevaluate whether this 
            // path really needs it or if it is even relevant anymore.
            // .promote(mlir::linalg::GenericOp::getOperationName(),
            //          mlir::linalg::LinalgPromotionOptions()
            //              .setAlignment(alignment)
            //              .setUseFullTileBuffersByDefault(true)
            //              .setUseAlloca(true))
            .vectorize(mlir::linalg::GenericOp::getOperationName())
            .vectorLowering(
                mlir::linalg::LinalgVectorLoweringOptions()
                    .setVectorTransformsOptions(
                        mlir::vector::VectorTransformsOptions()
                            .setVectorTransformsOptions(
                                mlir::vector::VectorContractLowering::
                                    OuterProduct))
                    .setVectorTransferToSCFOptions(
                        mlir::VectorTransferToSCFOptions().enableFullUnroll()));
        // TODO: this should be within a pass and we should be able to create a
        // nested OpPassManager.
        // Created a nested OpPassManager, populate the strategy and run.
        // mlir::OpPassManager dynamicPM("func.func");
        // strategy.configurePassPipeline(dynamicPM, function.getContext());
        // Propagate pass failure?
        // (void)mlir::runPipeline(dynamicPM, function);
        mlir::PassManager pm(function.getContext(),
                             function.getOperationName());
        strategy.configurePassPipeline(pm, function.getContext());
        // Propagate pass failure?
        (void)pm.run(function);
      });
}

void DotOpEmitter::EmitTiledLlvmIrGemm() {
  PrimitiveType primitive_type = dot_info_.result_shape.element_type();
  MatMultDims mat_mult_dims = GetMatMultDims();

  llvm::Value* lhs = lhs_array_.GetBasePointer();
  llvm::Value* rhs = rhs_array_.GetBasePointer();
  llvm::Value* target = target_array_.GetBasePointer();
  int64_t m = mat_mult_dims.m;
  int64_t k = mat_mult_dims.k;
  int64_t n = mat_mult_dims.n;

  if (mat_mult_dims.lhs_column_major) {
    std::swap(lhs, rhs);
    std::swap(m, n);
  }

  int64_t size_bytes =
      m * n * ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  b_->CreateMemSet(target, b_->getInt8(0), /*Size=*/size_bytes,
                   /*Align=*/llvm::MaybeAlign(1));

  int64_t max_target_vector_width =
      target_machine_features_.vector_register_num_elements(
          *b_->GetInsertBlock()->getParent(), primitive_type);

  int64_t tile_size_m, tile_size_k, tile_size_n_in_vector_width;
  std::tie(tile_size_m, tile_size_k, tile_size_n_in_vector_width) =
      GetGemmTileSize();

  EmitSmallGemm(
      /*scalar_type=*/primitive_type,
      /*m=*/m, /*k=*/k, /*n=*/n,
      /*max_vectorization_width=*/max_target_vector_width,
      /*max_vector_count=*/tile_size_n_in_vector_width,
      /*min_vectorization_width=*/std::min<int64_t>(4, max_target_vector_width),
      /*tile_size_m=*/tile_size_m, /*tile_size_k=*/tile_size_k, /*lhs=*/lhs,
      /*rhs=*/rhs, /*result=*/target, b_, hlo_module_config_);
}

void DotOpEmitter::EmitTiledLlvmIrGemv() {
  PrimitiveType primitive_type = dot_info_.result_shape.element_type();

  CHECK(primitive_util::IsFloatingPointType(primitive_type) ||
        primitive_util::IsIntegralType(primitive_type));

  MatMultDims mat_mult_dims = GetMatMultDims();
  bool is_column_major_matrix_vector_gemv = false;
  bool is_row_major_matrix_vector_gemv = false;

  int64_t m, k;
  bool swap_operands;

  if (mat_mult_dims.m == 1) {
    // Our emitters can only do Matrix*Vector (abbreviated as M*V) but when M=1
    // we actually want V*M.  We implement V*M as follows (Tr(X) = Transpose of
    // X):
    //
    //   V*M = Tr(Tr(V*M))  // Tr(Tr(X)) == X
    //       = Tr(Tr(M) * Tr(V))  // Tr(A * B) == Tr(B) * Tr(A)
    //
    // Since transposing a vector is physically a no-op, this is really
    // equivalent to `Tr(M) * V`.  We further implement Tr(M) by pretending that
    // M is row major if it is actually column major and vice-versa.

    bool rhs_effectively_column_major = mat_mult_dims.rhs_canonical
                                            ? mat_mult_dims.rhs_column_major
                                            : !mat_mult_dims.rhs_column_major;

    if (rhs_effectively_column_major) {
      k = mat_mult_dims.k;
      m = mat_mult_dims.n;

      // We set is_row_major_matrix_vector_gemv and not
      // is_column_major_matrix_vector_gemv to implement the Transpose trick
      // mentioned above.
      is_row_major_matrix_vector_gemv = true;
      swap_operands = true;
    } else {
      k = mat_mult_dims.k;
      m = mat_mult_dims.n;

      // We set is_column_major_matrix_vector_gemv and not
      // is_row_major_matrix_vector_gemv to implement the Transpose trick
      // mentioned above.
      is_column_major_matrix_vector_gemv = true;
      swap_operands = true;
    }
  }

  if (mat_mult_dims.n == 1) {
    bool lhs_effectively_column_major = mat_mult_dims.lhs_canonical
                                            ? mat_mult_dims.lhs_column_major
                                            : !mat_mult_dims.lhs_column_major;

    if (lhs_effectively_column_major) {
      m = mat_mult_dims.m;
      k = mat_mult_dims.k;
      is_column_major_matrix_vector_gemv = true;
      swap_operands = false;
    } else {
      m = mat_mult_dims.m;
      k = mat_mult_dims.k;
      is_row_major_matrix_vector_gemv = true;
      swap_operands = false;
    }
  }

  CHECK(is_column_major_matrix_vector_gemv || is_row_major_matrix_vector_gemv);

  int64_t tiling_factor = GetGemvTilingFactor();
  CHECK_GT(tiling_factor, 0);

  llvm::Value* result_op = target_array_.GetBasePointer();
  llvm::Value* lhs_op =
      swap_operands ? rhs_array_.GetBasePointer() : lhs_array_.GetBasePointer();
  llvm::Value* rhs_op =
      swap_operands ? lhs_array_.GetBasePointer() : rhs_array_.GetBasePointer();

  const int target_vector_register_element_size =
      target_machine_features_.vector_register_num_elements(
          *b_->GetInsertBlock()->getParent(), primitive_type);

  // We may not always know the vector register size for the target we're
  // compiling against, in which case target_vector_register_element_size is 0.
  // In these cases we choose a default LLVM IR register size.
  const int kUnknownTargetVectorRegisterSize = 4;
  const int vector_register_element_size =
      target_vector_register_element_size == 0
          ? kUnknownTargetVectorRegisterSize
          : target_vector_register_element_size;

  if (is_column_major_matrix_vector_gemv) {
    VLOG(2) << "Emitting column major matrix-vector multiply with m = " << m
            << " and k = " << k;
    EmitColumnMajorGemv(
        /*scalar_type=*/primitive_type,
        /*tile_rows=*/vector_register_element_size, /*tile_cols=*/tiling_factor,
        /*m=*/m, /*k=*/k, /*lhs=*/lhs_op, /*rhs=*/rhs_op,
        /*addend=*/addend_array_ ? addend_array_->GetBasePointer() : nullptr,
        /*result=*/result_op, b_, hlo_module_config_);
  } else {
    VLOG(2) << "Emitting row major matrix-vector multiply with m = " << m
            << " and k = " << k;
    EmitRowMajorGemv(
        /*scalar_type=*/primitive_type,
        /*tile_rows=*/tiling_factor,
        /*tile_cols=*/vector_register_element_size,
        /*m=*/m, /*k=*/k, /*lhs=*/lhs_op, /*rhs=*/rhs_op,
        /*addend=*/addend_array_ ? addend_array_->GetBasePointer() : nullptr,
        /*result=*/result_op, b_, hlo_module_config_);
  }
}

Status DotOpEmitter::Emit() {
  // The dot operation performs a sum of products over dimension 0 of the left
  // hand side operand and dimension 1 of the right hand side operand.
  //
  // Let the shapes of lhs and rhs be defined as below:
  //
  //   lhs = [L{n-1} x L{n-2} x ... L{0}]
  //   rhs = [R{m-1} x R{m-2} x ... R{0}]
  //
  // The sum-of-products dimension in the lhs has size L{0} and the dimension in
  // the rhs has size R{1}. Necessarily, then:
  //
  //   L{0} == R{1}
  //
  // The output of the operation has the following shape:
  //
  //   output = [L{n-1} x L{n-2} x ... L{1} x R{m-1} x R{m-2} x ... R{2} x R{0}]
  //
  // To perform the operation we construct a loop nest with one for-loop for
  // each dimension of the output. Inside this loop nest is another for-loop
  // which performs the sum-of-products (the reduction loop) before storing
  // the result in the output buffer.

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();

  if (ShapeUtil::IsScalar(lhs_shape) || ShapeUtil::IsScalar(rhs_shape)) {
    // If the operands are scalar, don't emit any loops.
    TF_RET_CHECK(ShapeUtil::IsScalar(lhs_shape) &&
                 ShapeUtil::IsScalar(rhs_shape));
    return EmitScalarDot();
  }

  switch (GetDotImplementationStrategy(hlo_module_config_, dot_info_,
                                       target_machine_features_)) {
    case DotImplementationStrategy::kNaiveLlvmIr:
      EmitNaiveLlvmIrGemm();
      return OkStatus();

    case DotImplementationStrategy::kTiledLlvmIrGemv:
      EmitTiledLlvmIrGemv();
      return OkStatus();

    case DotImplementationStrategy::kTiledLlvmIrGemm:
      EmitTiledLlvmIrGemm();
      return OkStatus();

    case DotImplementationStrategy::kLinalgMatmul:
      return EmitLinalgMatmul();

    case DotImplementationStrategy::kEigen:
      return EmitCallToRuntime();
  }
}

Status DotOpEmitter::EmitBatch() {
  // The dot operation performs a sum of products over dimension 0 of the left
  // hand side operand and dimension 1 of the right hand side operand.
  //
  // Let the shapes of lhs and rhs be defined as below:
  //
  //   lhs = [L{n-1} x L{n-2} x ... L{0}]
  //   rhs = [R{m-1} x R{m-2} x ... R{0}]
  //
  // The sum-of-products dimension in the lhs has size L{0} and the dimension in
  // the rhs has size R{1}. Necessarily, then:
  //
  //   L{0} == R{1}
  //
  // The output of the operation has the following shape:
  //
  //   output = [L{n-1} x L{n-2} x ... L{1} x R{m-1} x R{m-2} x ... R{2} x R{0}]
  //
  // To perform the operation we construct a loop nest with one for-loop for
  // each dimension of the output. Inside this loop nest is another for-loop
  // which performs the sum-of-products (the reduction loop) before storing
  // the result in the output buffer.

  return EmitCallToBatchRuntime();
}

void DotOpEmitter::EmitNaiveLlvmIrGemm() {
  CHECK_EQ(addend_array_, nullptr);

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  const DotDimensionNumbers& dim_nums = dot_info_.dim_nums;

  // Reduce along dimension 0 of the LHS and 1 of the RHS. Vectors are a special
  // case where the reduction dimension is 0 for both LHS and RHS. This results
  // in a vector dot product producing a scalar.
  int64_t lhs_reduction_dimension = dim_nums.lhs_contracting_dimensions(0);
  int64_t rhs_reduction_dimension = dim_nums.rhs_contracting_dimensions(0);

  // Verify the reduction dimension in the two operands are the same size.
  CHECK_EQ(lhs_shape.dimensions(lhs_reduction_dimension),
           rhs_shape.dimensions(rhs_reduction_dimension));

  bool lhs_reduction_along_minor_dimension =
      lhs_reduction_dimension == LayoutUtil::Minor(lhs_shape.layout(), 0);
  bool rhs_reduction_along_minor_dimension =
      rhs_reduction_dimension == LayoutUtil::Minor(rhs_shape.layout(), 0);

  // Create loop nests which loop through the LHS operand dimensions and the RHS
  // operand dimensions. The reduction dimension of the LHS and RHS are handled
  // in a separate innermost loop which performs the sum of products.
  llvm_ir::ForLoopNest loop_nest(llvm_ir::IrName(dot_hlo_name_), b_);
  std::vector<llvm::Value*> lhs_multi_index =
      loop_nest.EmitOperandArrayLoopNest(
          lhs_array_, /*dimension_to_skip=*/lhs_reduction_dimension, "lhs");
  std::vector<llvm::Value*> rhs_multi_index =
      loop_nest.EmitOperandArrayLoopNest(
          rhs_array_, /*dimension_to_skip=*/rhs_reduction_dimension, "rhs");

  // Create the loop which does the sum of products reduction.
  //
  // The prevent_unrolling bit is working around a deficiency in LLVM's loop
  // vectorization pipeline, wherein in some cases unrolling a loop can prevent
  // effective vectorization.  Since we know that the IR we generate when
  // reducing across the minor dimension in both LHS and RHS is vectorized well
  // by the loop vectorizer, we block unrolling in that case to stop loop unroll
  // from messing up the vectorization.
  std::unique_ptr<llvm_ir::ForLoop> reduction_loop = loop_nest.AddLoop(
      0, lhs_shape.dimensions(lhs_reduction_dimension), "reduction",
      /*unroll_mode=*/
      (lhs_reduction_along_minor_dimension &&
       rhs_reduction_along_minor_dimension)
          ? xla::llvm_ir::UnrollMode::kNoUnroll
          : xla::llvm_ir::UnrollMode::kDefaultUnroll);

  // The final entry in the rhs and lhs indexes is the indvar of the
  // reduction loop.
  lhs_multi_index[lhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  llvm_ir::IrArray::Index lhs_index(lhs_multi_index, lhs_shape,
                                    b_->getInt64Ty());
  rhs_multi_index[rhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  llvm_ir::IrArray::Index rhs_index(rhs_multi_index, rhs_shape,
                                    b_->getInt64Ty());

  // For computing the sum of products we alloca a single location to store the
  // dot product result as we accumulate it within the reduction loop. After the
  // reduction loop we load the result and store into the output array.

  // Function entry basic block.
  // - Emit alloca for accumulator
  llvm::Function* func = reduction_loop->GetPreheaderBasicBlock()->getParent();
  SetToFirstInsertPoint(&func->getEntryBlock(), b_);
  llvm::Type* accum_type = target_array_.GetElementLlvmType();
  llvm::Value* accum_address =
      b_->CreateAlloca(accum_type, /*ArraySize=*/nullptr, "accum_address");

  // Preheader basic block of reduction loop:
  // - Initialize accumulator to zero.
  llvm::BasicBlock* preheader_bb = reduction_loop->GetPreheaderBasicBlock();
  b_->SetInsertPoint(preheader_bb->getTerminator());

  b_->CreateStore(llvm::Constant::getNullValue(accum_type), accum_address);

  // Body basic block of reduction loop:
  // - Load elements from lhs and rhs array.
  // - Multiply lhs-element and rhs-element.
  // - Load accumulator and add to product.
  // - Store sum back into accumulator.
  SetToFirstInsertPoint(reduction_loop->GetBodyBasicBlock(), b_);

  llvm::Value* lhs_element = lhs_array_.EmitReadArrayElement(lhs_index, b_);
  llvm::Value* rhs_element = rhs_array_.EmitReadArrayElement(rhs_index, b_);

  llvm::Value* accum = b_->CreateLoad(accum_type, accum_address);
  llvm::Value* updated_accum;
  if (ShapeUtil::ElementIsComplex(lhs_shape)) {
    auto real = [&](llvm::Value* x) { return b_->CreateExtractValue(x, {0}); };
    auto imag = [&](llvm::Value* x) { return b_->CreateExtractValue(x, {1}); };
    llvm::Value* product_real =
        b_->CreateFSub(b_->CreateFMul(real(lhs_element), real(rhs_element)),
                       b_->CreateFMul(imag(lhs_element), imag(rhs_element)));
    llvm::Value* product_imag =
        b_->CreateFAdd(b_->CreateFMul(real(lhs_element), imag(rhs_element)),
                       b_->CreateFMul(imag(lhs_element), real(rhs_element)));
    updated_accum = b_->CreateInsertValue(
        accum, b_->CreateFAdd(real(accum), product_real), {0});
    updated_accum = b_->CreateInsertValue(
        updated_accum, b_->CreateFAdd(imag(accum), product_imag), {1});
  } else if (ShapeUtil::ElementIsIntegral(lhs_shape)) {
    llvm::Value* product = b_->CreateMul(lhs_element, rhs_element);
    updated_accum = b_->CreateAdd(accum, product);
  } else if (lhs_shape.element_type() == PRED) {
    llvm::Value* product = b_->CreateAnd(lhs_element, rhs_element);
    updated_accum = b_->CreateOr(accum, product);
  } else {
    llvm::Value* product = b_->CreateFMul(lhs_element, rhs_element);
    updated_accum = b_->CreateFAdd(accum, product);
  }
  b_->CreateStore(updated_accum, accum_address);

  // Exit basic block of reduction loop.
  // - Load accumulator value (the result).
  // - Store into output array.
  SetToFirstInsertPoint(reduction_loop->GetExitBasicBlock(), b_);

  llvm::Value* result = b_->CreateLoad(accum_type, accum_address);

  // Create index into target address. The target index is the concatenation of
  // the rhs and lhs indexes with the reduction dimensions removed. The terms
  // from the rhs index are the lower dimensions in the index so we add them
  // first.
  std::vector<llvm::Value*> target_multi_index;
  for (int dimension = 0; dimension < lhs_index.size(); ++dimension) {
    if (dimension != lhs_reduction_dimension) {
      target_multi_index.push_back(lhs_index[dimension]);
    }
  }
  for (int dimension = 0; dimension < rhs_index.size(); ++dimension) {
    if (dimension != rhs_reduction_dimension) {
      target_multi_index.push_back(rhs_index[dimension]);
    }
  }

  llvm_ir::IrArray::Index target_index(
      target_multi_index, target_array_.GetShape(), lhs_index.GetType());
  target_array_.EmitWriteArrayElement(target_index, result, b_);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop.
  b_->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());
}

Status DotOpEmitter::EmitScalarDot() {
  // A scalar dot is just a scalar multiply.
  llvm::Value* result;
  // Use the same index_type for all tensor accesses in the same kernel.
  llvm::Type* index_type = b_->getInt64Ty();
  llvm_ir::IrArray::Index element_index(index_type);
  llvm::Value* lhs_value =
      lhs_array_.EmitReadArrayElement(/*index=*/element_index, b_);
  llvm::Value* rhs_value =
      rhs_array_.EmitReadArrayElement(/*index=*/element_index, b_);
  if (ShapeUtil::ElementIsComplex(lhs_array_.GetShape())) {
    auto get_real = [&](llvm::Value* x) {
      return b_->CreateExtractValue(x, {0});
    };

    auto get_imag = [&](llvm::Value* x) {
      return b_->CreateExtractValue(x, {1});
    };

    llvm::Value* real = b_->CreateFSub(
        b_->CreateFMul(get_real(lhs_value), get_real(rhs_value)),
        b_->CreateFMul(get_imag(lhs_value), get_imag(rhs_value)));
    llvm::Value* imag = b_->CreateFAdd(
        b_->CreateFMul(get_real(lhs_value), get_imag(rhs_value)),
        b_->CreateFMul(get_imag(lhs_value), get_real(rhs_value)));
    result = llvm::ConstantAggregateZero::get(lhs_array_.GetElementLlvmType());
    result = b_->CreateInsertValue(result, real, {0});
    result = b_->CreateInsertValue(result, imag, {1});
  } else {
    result = b_->CreateFMul(lhs_value, rhs_value);
  }
  target_array_.EmitWriteArrayElement(/*index=*/element_index, result, b_);
  return OkStatus();
}

Status DotOpEmitter::EmitCallToRuntime() {
  // The signature of the Eigen runtime matmul function is:
  //
  //   (void)(void* run_options, float* out, float* lhs, float* rhs,
  //          int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
  //          int32_t transpose_rhs);
  // The two transpose_... parameters are actually booleans, but we use int32_t
  // to avoid target-dependent calling convention details.

  bool multi_threaded = ShouldUseMultiThreadedEigen(hlo_module_config_);
  bool use_mkl_dnn = hlo_module_config_.debug_options().xla_cpu_use_mkl_dnn();
  bool use_acl = hlo_module_config_.debug_options().xla_cpu_use_acl();
  PrimitiveType type = target_array_.GetShape().element_type();
  llvm::Function* function = b_->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();
  llvm::Type* float_type;
  const char* fn_name;
  switch (type) {
    case F16:
      fn_name = multi_threaded
                    ? runtime::kEigenMatMulF16SymbolName
                    : runtime::kEigenSingleThreadedMatMulF16SymbolName;
      float_type = b_->getHalfTy();
      break;
    case F32:
      fn_name =
          multi_threaded
              ? (use_mkl_dnn ? runtime::kMKLMatMulF32SymbolName
                             : (use_acl ? runtime::kACLMatMulF32SymbolName
                                        : runtime::kEigenMatMulF32SymbolName))
              : (use_mkl_dnn
                     ? runtime::kMKLSingleThreadedMatMulF32SymbolName
                     : runtime::kEigenSingleThreadedMatMulF32SymbolName);
      float_type = b_->getFloatTy();
      break;
    case F64:
      fn_name = multi_threaded
                    ? (use_mkl_dnn ? runtime::kMKLMatMulF64SymbolName
                                   : runtime::kEigenMatMulF64SymbolName)
                    : (use_mkl_dnn
                           ? runtime::kMKLSingleThreadedMatMulF64SymbolName
                           : runtime::kEigenSingleThreadedMatMulF64SymbolName);
      float_type = b_->getDoubleTy();
      break;
    case C64:
      fn_name = multi_threaded
                    ? runtime::kEigenMatMulC64SymbolName
                    : runtime::kEigenSingleThreadedMatMulC64SymbolName;
      float_type = llvm_ir::PrimitiveTypeToIrType(C64, module);
      break;
    case C128:
      fn_name = multi_threaded
                    ? runtime::kEigenMatMulC128SymbolName
                    : runtime::kEigenSingleThreadedMatMulC128SymbolName;
      float_type = llvm_ir::PrimitiveTypeToIrType(C128, module);
      break;
    case S32:
      fn_name = multi_threaded
                    ? runtime::kEigenMatMulS32SymbolName
                    : runtime::kEigenSingleThreadedMatMulS32SymbolName;
      float_type = b_->getInt32Ty();
      break;
    default:
      return Unimplemented("Invalid type %s for dot operation",
                           PrimitiveType_Name(type));
  }

  llvm::Type* float_ptr_type = float_type->getPointerTo();
  llvm::Type* int64_type = b_->getInt64Ty();
  llvm::Type* int32_type = b_->getInt32Ty();
  llvm::Type* int8_ptr_type = b_->getInt8Ty()->getPointerTo();
  llvm::FunctionType* matmul_type = llvm::FunctionType::get(
      b_->getVoidTy(),
      {int8_ptr_type, float_ptr_type, float_ptr_type, float_ptr_type,
       int64_type, int64_type, int64_type, int32_type, int32_type},
      /*isVarArg=*/false);

  llvm::FunctionCallee matmul_func =
      module->getOrInsertFunction(fn_name, matmul_type);
  if (auto* fn = llvm::dyn_cast<llvm::Function>(matmul_func.getCallee())) {
    fn->setCallingConv(llvm::CallingConv::C);
    fn->setDoesNotThrow();
    fn->setOnlyAccessesArgMemory();
  }

  // The Eigen runtime function expects column-major layout. If the matrices are
  // row major, then use the following identity to compute the product:
  //
  //   (A x B)^T = B^T x A^T
  //
  // The connection between this identity and memory layout is that the
  // transpose operation can also be considered as an operation that changes the
  // memory layout of a matrix from row-major to column-major or vice versa.
  //
  // Effectively this involves swapping the 'lhs' with 'rhs' and 'm' with 'n'.

  MatMultDims mat_mult_dims = GetMatMultDims();

  CHECK_EQ(mat_mult_dims.lhs_column_major, mat_mult_dims.rhs_column_major);

  const llvm_ir::IrArray* lhs = &lhs_array_;
  const llvm_ir::IrArray* rhs = &rhs_array_;
  bool transpose_lhs = !mat_mult_dims.lhs_canonical;
  bool transpose_rhs = !mat_mult_dims.rhs_canonical;

  if (!mat_mult_dims.lhs_column_major) {
    std::swap(mat_mult_dims.m, mat_mult_dims.n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
  }

  b_->CreateCall(
      matmul_func,
      {b_->CreateBitCast(executable_run_options_value_, int8_ptr_type),
       b_->CreateBitCast(target_array_.GetBasePointer(), float_ptr_type),
       b_->CreateBitCast(lhs->GetBasePointer(), float_ptr_type),
       b_->CreateBitCast(rhs->GetBasePointer(), float_ptr_type),
       b_->getInt64(mat_mult_dims.m), b_->getInt64(mat_mult_dims.n),
       b_->getInt64(mat_mult_dims.k), b_->getInt32(transpose_lhs),
       b_->getInt32(transpose_rhs)});
  return OkStatus();
}

Status DotOpEmitter::EmitCallToBatchRuntime() {
  // The signature of the runtime batch matmul function is:
  //
  //   (void)(void* run_options, float* out, float* lhs, float* rhs,
  //          int64_t m, int64_t n, int64_t k, int64_t batch_size, int32_t
  //          transpose_lhs, int32_t transpose_rhs);
  // The two transpose_... parameters are actually booleans, but we use int32_t
  // to avoid target-dependent calling convention details.

  PrimitiveType type = target_array_.GetShape().element_type();
  bool use_acl = hlo_module_config_.debug_options().xla_cpu_use_acl();
  llvm::Function* function = b_->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();
  llvm::Type* float_type;
  const char* fn_name;
  switch (type) {
    case F32:
      fn_name = use_acl ? runtime::kACLBatchMatMulF32SymbolName
                        : runtime::kEigenBatchMatMulF32SymbolName;

      float_type = b_->getFloatTy();
      break;
    default:
      return Unimplemented("Invalid type %s for dot operation",
                           PrimitiveType_Name(type));
  }

  llvm::Type* float_ptr_type = float_type->getPointerTo();
  llvm::Type* int64_type = b_->getInt64Ty();
  llvm::Type* int32_type = b_->getInt32Ty();
  llvm::Type* int8_ptr_type = b_->getInt8Ty()->getPointerTo();
  llvm::FunctionType* matmul_type = llvm::FunctionType::get(
      b_->getVoidTy(),
      {int8_ptr_type, float_ptr_type, float_ptr_type, float_ptr_type,
       int64_type, int64_type, int64_type, int64_type, int32_type, int32_type},
      /*isVarArg=*/false);

  llvm::FunctionCallee matmul_func =
      module->getOrInsertFunction(fn_name, matmul_type);
  if (auto* fn = llvm::dyn_cast<llvm::Function>(matmul_func.getCallee())) {
    fn->setCallingConv(llvm::CallingConv::C);
    fn->setDoesNotThrow();
    fn->setOnlyAccessesArgMemory();
  }

  // The ACL runtime function expects column-major layout. If the matrices are
  // row major, then use the following identity to compute the product:
  //
  //   (A x B)^T = B^T x A^T
  //
  // The connection between this identity and memory layout is that the
  // transpose operation can also be considered as an operation that changes the
  // memory layout of a matrix from row-major to column-major or vice versa.
  //
  // Effectively this involves swapping the 'lhs' with 'rhs' and 'm' with 'n'.

  MatMultDims mat_mult_dims = GetBatchMatMultDims();
  CHECK_EQ(mat_mult_dims.lhs_column_major, mat_mult_dims.rhs_column_major);

  const llvm_ir::IrArray* lhs = &lhs_array_;
  const llvm_ir::IrArray* rhs = &rhs_array_;
  bool transpose_lhs = !mat_mult_dims.lhs_canonical;
  bool transpose_rhs = !mat_mult_dims.rhs_canonical;
  const Shape& lhs_shape = lhs_array_.GetShape();

  if (!mat_mult_dims.lhs_column_major) {
    std::swap(mat_mult_dims.m, mat_mult_dims.n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
  }

  VLOG(1) << "Batch dot emitted with runtime:" << fn_name;

  b_->CreateCall(
      matmul_func,
      {b_->CreateBitCast(executable_run_options_value_, int8_ptr_type),
       b_->CreateBitCast(target_array_.GetBasePointer(), float_ptr_type),
       b_->CreateBitCast(lhs->GetBasePointer(), float_ptr_type),
       b_->CreateBitCast(rhs->GetBasePointer(), float_ptr_type),
       b_->getInt64(mat_mult_dims.m), b_->getInt64(mat_mult_dims.n),
       b_->getInt64(mat_mult_dims.k), b_->getInt64(lhs_shape.dimensions(0)),
       b_->getInt32(static_cast<uint32_t>(transpose_lhs)),
       b_->getInt32(static_cast<uint32_t>(transpose_rhs))});
  return Status::OK();
}

DotOpEmitter::MatMultDims DotOpEmitter::GetMatMultDims() const {
  CHECK_LE(dot_info_.result_shape.dimensions_size(), 2);

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  const DotDimensionNumbers& dim_nums = dot_info_.dim_nums;

  auto is_column_major = [](const Shape& shape) {
    return shape.rank() > 1 && LayoutUtil::Minor(shape.layout(), 0) == 0;
  };

  // Non-contracting dots should never make it here.
  CHECK_GE(dim_nums.lhs_contracting_dimensions_size(), 0);
  CHECK_GE(dim_nums.rhs_contracting_dimensions_size(), 0);

  return {
      /*m=*/lhs_shape.rank() <= 1
          ? 1LL
          : lhs_shape.dimensions(1LL - dim_nums.lhs_contracting_dimensions(0)),
      /*k=*/lhs_shape.dimensions(dim_nums.lhs_contracting_dimensions(0)),
      /*n=*/rhs_shape.rank() <= 1
          ? 1LL
          : rhs_shape.dimensions(1LL - dim_nums.rhs_contracting_dimensions(0)),
      /*lhs_column_major=*/is_column_major(lhs_shape),
      /*lhs_canonical=*/lhs_shape.rank() <= 1 ||
          dim_nums.lhs_contracting_dimensions(0) == 1,
      /*rhs_column_major=*/is_column_major(rhs_shape),
      /*rhs_canonical=*/dim_nums.rhs_contracting_dimensions(0) == 0};
}

DotOpEmitter::MatMultDims DotOpEmitter::GetBatchMatMultDims() const {
  CHECK_LE(dot_info_.result_shape.dimensions_size(), 2);

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  const DotDimensionNumbers& dim_nums = dot_info_.dim_nums;

  auto is_column_major = [](const Shape& shape) {
    return shape.rank() > 1 && LayoutUtil::Minor(shape.layout(), 0) == 0;
  };

  // Non-contracting dots should never make it here.
  CHECK_GE(dim_nums.lhs_contracting_dimensions_size(), 0);
  CHECK_GE(dim_nums.rhs_contracting_dimensions_size(), 0);

  return {
      /*m=*/lhs_shape.rank() <= 1
          ? 1LL
          : lhs_shape.dimensions(2LL - dim_nums.lhs_contracting_dimensions(0)),
      /*k=*/lhs_shape.dimensions(1LL + dim_nums.lhs_contracting_dimensions(0)),
      /*n=*/rhs_shape.rank() <= 1
          ? 1LL
          : rhs_shape.dimensions(2LL - dim_nums.rhs_contracting_dimensions(0)),
      /*lhs_column_major=*/is_column_major(lhs_shape),
      /*lhs_canonical=*/lhs_shape.rank() <= 1 ||
          dim_nums.lhs_contracting_dimensions(0) == 1,
      /*rhs_column_major=*/is_column_major(rhs_shape),
      /*rhs_canonical=*/dim_nums.rhs_contracting_dimensions(0) == 0};
}

// For vector-matrix dot products, it is always profitable to make the Rhs
// column major.
std::optional<int64_t> ProfitableToMakeDotOperandColumnMajor(
    const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kDot && hlo.shape().dimensions_size() <= 1) {
    if (hlo.operand(0)->shape().rank() != 1 ||
        hlo.dot_dimension_numbers().rhs_contracting_dimensions(0) != 0) {
      return {};
    }

    // Don't bother if the other operand is tiny, switching to column major
    // wouldn't use tiling.
    constexpr int kColumnMajorThresholdInBytes = 32;
    int64_t lhs_size =
        ShapeUtil::ByteSizeOfPrimitiveType(hlo.shape().element_type()) *
        ShapeUtil::ElementsIn(hlo.operand(0)->shape());
    if (lhs_size < kColumnMajorThresholdInBytes) {
      return {};
    }

    return 1;
  }

  if (hlo.IsOutputFusion()) {
    auto* fusion_root =
        hlo.fused_instructions_computation()->root_instruction();
    if (fusion_root->opcode() != HloOpcode::kAdd) {
      return {};
    }

    for (auto* fusion_root_op : fusion_root->operands()) {
      if (fusion_root_op->opcode() != HloOpcode::kDot) {
        continue;
      }
      if (auto operand_num =
              ProfitableToMakeDotOperandColumnMajor(*fusion_root_op)) {
        auto* operand = fusion_root_op->operand(*operand_num);
        if (operand->opcode() == HloOpcode::kParameter &&
            operand->user_count() == 1) {
          return operand->parameter_number();
        }
      }
    }
  }

  return {};
}

namespace {
// Return whether the given shape is rank 2.
bool IsRank2(const Shape& shape) { return shape.rank() == 2; }

bool IsSimpleLayout(const Layout& layout) {
  return layout.tiles().empty() && layout.format() == DENSE;
}

// In a gemm operation where output = lhs * rhs, check whether the given shapes
// are valid for the operation.
bool AreGemmShapes(const Shape& lhs_shape, const Shape& rhs_shape,
                   const Shape& output_shape,
                   const TargetMachineFeatures& target_machine_features) {
  CHECK(!lhs_shape.has_layout() || IsSimpleLayout(lhs_shape.layout()))
      << lhs_shape.DebugString();
  CHECK(!rhs_shape.has_layout() || IsSimpleLayout(rhs_shape.layout()))
      << rhs_shape.DebugString();
  CHECK(!output_shape.has_layout() || IsSimpleLayout(output_shape.layout()))
      << output_shape.DebugString();

  switch (output_shape.element_type()) {
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
    case S32:
      return IsRank2(lhs_shape) && IsRank2(rhs_shape) && IsRank2(output_shape);
    default:
      return false;
  }
}

bool IsAlignedGemm(const DotInfo& dot_info,
                   const TargetMachineFeatures& target_machine_features) {
  if (ShapeUtil::IsZeroElementArray(dot_info.lhs_shape) ||
      ShapeUtil::IsZeroElementArray(dot_info.rhs_shape)) {
    return false;
  }

  return AreGemmShapes(dot_info.lhs_shape, dot_info.rhs_shape,
                       dot_info.result_shape, target_machine_features);
}

bool CanEmitTiledLlvmIrGemm(
    const HloModuleConfig& config, const DotInfo& dot_info,
    const TargetMachineFeatures& target_machine_features) {
  CHECK(IsAlignedGemm(dot_info, target_machine_features));

  if (ShouldUseMultiThreadedEigen(config)) {
    return false;
  }

  int m = dot_info.result_shape.dimensions(0);
  int k = dot_info.lhs_shape.dimensions(
      dot_info.dim_nums.lhs_contracting_dimensions(0));
  int n = dot_info.result_shape.dimensions(1);

  if (!options::ForceEnableExperimentalLlvmIrGemm(config)) {
    // TODO(sanjoy):  We should make these numbers micro-arch specific.
    bool small_gemm =
        k <= 128 && ((m <= 32 && n <= 128) || (m <= 128 && n <= 32));
    if (!small_gemm) {
      return false;
    }
  }

  bool lhs_canonical = dot_info.dim_nums.lhs_contracting_dimensions(0) == 1;
  bool rhs_canonical = dot_info.dim_nums.rhs_contracting_dimensions(0) == 0;

  if (!(lhs_canonical && rhs_canonical)) {
    return false;
  }

  if (dot_info.result_shape.element_type() == F16 ||
      dot_info.result_shape.element_type() == C64 ||
      dot_info.result_shape.element_type() == C128) {
    // TODO(sanjoy): This is probably easy to fix, but I want to keep the CL
    // adding this comment NFC.
    return false;
  }

  return true;
}

DotImplementationStrategy GetDotImplementationStrategy(
    const HloModuleConfig& config, const DotInfo& dot_info,
    const TargetMachineFeatures& target_machine_features) {
  PrimitiveType element_type = dot_info.result_shape.element_type();
  // Any Matrix-Vector product of floating point or integral type, or
  // a transpose-dot fusion of the same can be lowered to a tiled LLVM
  // IR implementation.
  if ((dot_info.result_shape.dimensions_size() <= 1 ||
       (dot_info.result_shape.dimensions_size() == 2 &&
        (dot_info.result_shape.dimensions(0) == 1 ||
         dot_info.result_shape.dimensions(1) == 1))) &&
      (primitive_util::IsFloatingPointType(element_type) ||
       primitive_util::IsIntegralType(element_type))) {
    return DotImplementationStrategy::kTiledLlvmIrGemv;
  }

  // MatMul smaller than 3x3 should use naive nested loop.
  if ((dot_info.lhs_shape.dimensions_size() <= 1 ||
       (dot_info.lhs_shape.dimensions_size() == 2 &&
        (dot_info.lhs_shape.dimensions(0) <= 3 ||
         dot_info.lhs_shape.dimensions(1) <= 3))) &&
      (dot_info.rhs_shape.dimensions_size() <= 1 ||
       (dot_info.rhs_shape.dimensions_size() == 2 &&
        (dot_info.rhs_shape.dimensions(0) <= 3 ||
         dot_info.rhs_shape.dimensions(1) <= 3))) &&
      (primitive_util::IsFloatingPointType(element_type) ||
       primitive_util::IsIntegralType(element_type))) {
    return DotImplementationStrategy::kNaiveLlvmIr;
  }

  if (IsAlignedGemm(dot_info, target_machine_features)) {
    if (CanEmitTiledLlvmIrGemm(config, dot_info, target_machine_features)) {
      return DotImplementationStrategy::kTiledLlvmIrGemm;
    }
    return DotImplementationStrategy::kEigen;
  }

  return DotImplementationStrategy::kNaiveLlvmIr;
}

Status EmitNonBatchDotOperation(
    DotInfo dot_info, std::string hlo_name,
    const llvm_ir::IrArray& target_array, const llvm_ir::IrArray& lhs_array,
    const llvm_ir::IrArray& rhs_array, const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
    mlir::MLIRContext* mlir_context, const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features) {
  PrimitiveType type = target_array.GetShape().element_type();
  TF_RET_CHECK(PRED == type || S8 == type || U8 == type || S16 == type ||
               U16 == type || S32 == type || U32 == type || S64 == type ||
               U64 == type || F16 == type || F32 == type || F64 == type ||
               C64 == type || C128 == type);
  DotOpEmitter dot_emitter(std::move(dot_info), std::move(hlo_name),
                           target_array, lhs_array, rhs_array, addend_array,
                           executable_run_options_value, b, mlir_context,
                           hlo_module_config, target_machine_features);
  return dot_emitter.Emit();
}

Shape DropFirstDim(const Shape& shape) {
  absl::Span<int64_t const> array_shape_dims(shape.dimensions());
  array_shape_dims.remove_prefix(1);
  return ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                  array_shape_dims);
}

Shape CollapseFirstNDims(const Shape& shape, int64_t n) {
  absl::Span<int64_t const> input_shape_dims(shape.dimensions());
  int64_t prefix_dim =
      std::accumulate(input_shape_dims.begin(), input_shape_dims.begin() + n,
                      1ll, std::multiplies<int64_t>());
  DimensionVector result_dims;
  result_dims.push_back(prefix_dim);
  std::copy(input_shape_dims.begin() + n, input_shape_dims.end(),
            std::back_inserter(result_dims));
  return ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                  result_dims);
}

llvm_ir::IrArray CollapseFirstNDims(llvm::IRBuilder<>* b,
                                    const llvm_ir::IrArray& array, int64_t n) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  const Shape& shape = array.GetShape();
  CHECK(shape.has_layout() &&
        LayoutUtil::IsMonotonicWithDim0Major(shape.layout()));
  CHECK_GE(shape.dimensions_size(), n);
  Shape new_shape = CollapseFirstNDims(shape, n);
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(new_shape, module);
  llvm::Value* new_value =
      b->CreateBitCast(array.GetBasePointer(), new_ir_type->getPointerTo());
  return llvm_ir::IrArray(new_value, new_ir_type, std::move(new_shape));
}

Status ValidateDotDimensionNumbers(const DotDimensionNumbers& dim_numbers) {
  // Checks some invariants that do not hold in general, but DotDecomposer
  // should have established for us.  This is just a debugging aid.
  TF_RET_CHECK(dim_numbers.lhs_contracting_dimensions_size() == 1);
  std::vector<int64_t> batch_dim_numbers(
      dim_numbers.lhs_batch_dimensions_size());
  absl::c_iota(batch_dim_numbers, 0);
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.lhs_batch_dimensions()));
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.rhs_batch_dimensions()));
  return OkStatus();
}

// Slice out the inner array at batch index `batch_index` from `outer_array`.
llvm_ir::IrArray SliceOutInnerArray(llvm_ir::IrArray outer_array,
                                    llvm::Value* batch_index,
                                    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();

  Shape inner_shape = DropFirstDim(outer_array.GetShape());
  std::vector<llvm::Value*> multidim_index(inner_shape.rank() + 1,
                                           b->getInt64(0));
  multidim_index[0] = batch_index;
  llvm_ir::IrArray::Index slice_index(multidim_index, outer_array.GetShape(),
                                      batch_index->getType());
  llvm::Value* slice_ptr = outer_array.EmitArrayElementAddress(slice_index, b);
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(inner_shape, module);
  llvm::Type* slice_ptr_type = new_ir_type->getPointerTo();
  return llvm_ir::IrArray(b->CreateBitCast(slice_ptr, slice_ptr_type),
                          new_ir_type, std::move(inner_shape));
}

bool PotentiallyImplementedAsEigenMatmul(
    const HloInstruction& dot, const llvm_ir::IrArray& target_array,
    const llvm_ir::IrArray& lhs_array, const llvm_ir::IrArray& rhs_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
    mlir::MLIRContext* mlir_context, const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features, DotInfo& dot_info) {
  int64_t num_batch_dims =
      dot.dot_dimension_numbers().lhs_batch_dimensions_size();

  // TODO(kramerb): Remove this limitation.
  if (num_batch_dims > 1) return false;

  // First reshape the inputs to make sure we only have one batch dimension.
  // This is a no-op bitcast because the operands have to be in row-major layout
  // (enforced in CpuLayoutAssignment), and the batch dimensions are the leading
  // dimensions (established by DotDecomposer and checked by
  // ValidateDotDimensionNumbers above).
  llvm_ir::IrArray lhs_array_reshaped =
      CollapseFirstNDims(b, lhs_array, num_batch_dims);
  llvm_ir::IrArray rhs_array_reshaped =
      CollapseFirstNDims(b, rhs_array, num_batch_dims);
  llvm_ir::IrArray target_array_reshaped =
      CollapseFirstNDims(b, target_array, num_batch_dims);

  DotDimensionNumbers adjusted_dim_numbers = dot.dot_dimension_numbers();
  adjusted_dim_numbers.clear_lhs_batch_dimensions();
  adjusted_dim_numbers.clear_rhs_batch_dimensions();

  // Create a DotInfo representing the batch of "inner" dot operations.
  dot_info.lhs_shape = DropFirstDim(lhs_array_reshaped.GetShape());
  dot_info.rhs_shape = DropFirstDim(rhs_array_reshaped.GetShape());
  dot_info.result_shape = DropFirstDim(target_array_reshaped.GetShape());
  dot_info.dim_nums = dot.dot_dimension_numbers();
  dot_info.dim_nums.clear_lhs_batch_dimensions();
  dot_info.dim_nums.clear_rhs_batch_dimensions();

  dot_info.dim_nums.set_lhs_contracting_dimensions(
      0, dot_info.dim_nums.lhs_contracting_dimensions(0) - num_batch_dims);
  dot_info.dim_nums.set_rhs_contracting_dimensions(
      0, dot_info.dim_nums.rhs_contracting_dimensions(0) - num_batch_dims);

  PrimitiveType type = target_array.GetShape().element_type();
  if (F32 != type) return false;

  if (ShapeUtil::IsScalar(dot_info.lhs_shape) ||
      ShapeUtil::IsScalar(dot_info.rhs_shape)) {
    // If the operands are scalar, don't emit any loops.
    return false;
  }

  DotImplementationStrategy impl_strategy = GetDotImplementationStrategy(
      dot.parent()->parent()->config(), dot_info, target_machine_features);

  return impl_strategy == DotImplementationStrategy::kEigen;
}

Status EmitBatchDotOperation(
    const HloInstruction& dot, const llvm_ir::IrArray& target_array,
    const llvm_ir::IrArray& lhs_array, const llvm_ir::IrArray& rhs_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
    mlir::MLIRContext* mlir_context, const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features) {
  TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(dot.dot_dimension_numbers()));

  // first check if the batch can be rendered directly by the runtime
  // otherwise lower it to a sequence of non-batch dot operations
  DotInfo dot_info;
  if (ShouldUseMultiThreadedEigen(hlo_module_config) &&
      PotentiallyImplementedAsEigenMatmul(
          dot, target_array, lhs_array, rhs_array, executable_run_options_value,
          b, mlir_context, hlo_module_config, target_machine_features,
          dot_info)) {
    DotOpEmitter dot_emitter(dot_info, dot.name(), target_array, lhs_array,
                             rhs_array, nullptr /*addend_array*/,
                             executable_run_options_value, b, mlir_context,
                             hlo_module_config, target_machine_features);

    return dot_emitter.EmitBatch();
  } else {
    // Lower a batch dot into a sequence of non-batch dot operations.

    int64_t num_batch_dims =
        dot.dot_dimension_numbers().lhs_batch_dimensions_size();

    // First reshape the inputs to make sure we only have one batch dimension.
    // This is a no-op bitcast because the operands have to be in row-major
    // layout (enforced in CpuLayoutAssignment), and the batch dimensions are
    // the leading dimensions (established by DotDecomposer and checked by
    // ValidateDotDimensionNumbers above).
    llvm_ir::IrArray lhs_array_reshaped =
        CollapseFirstNDims(b, lhs_array, num_batch_dims);
    llvm_ir::IrArray rhs_array_reshaped =
        CollapseFirstNDims(b, rhs_array, num_batch_dims);
    llvm_ir::IrArray target_array_reshaped =
        CollapseFirstNDims(b, target_array, num_batch_dims);

    int64_t batch_count = lhs_array_reshaped.GetShape().dimensions(0);

    KernelSupportLibrary ksl(b);

    return ksl.ForWithStatus(
        llvm_ir::IrName(&dot, "bdot"), /*start=*/0, /*end=*/batch_count,
        /*step=*/1, [&](llvm::Value* indvar) {
          DotDimensionNumbers adjusted_dim_numbers =
              dot.dot_dimension_numbers();
          adjusted_dim_numbers.clear_lhs_batch_dimensions();
          adjusted_dim_numbers.clear_rhs_batch_dimensions();

          // Create a DotInfo representing the "inner" non-batch dot operation.
          DotInfo dot_info;
          dot_info.lhs_shape = DropFirstDim(lhs_array_reshaped.GetShape());
          dot_info.rhs_shape = DropFirstDim(rhs_array_reshaped.GetShape());
          dot_info.result_shape =
              DropFirstDim(target_array_reshaped.GetShape());
          dot_info.dim_nums = dot.dot_dimension_numbers();
          dot_info.dim_nums.clear_lhs_batch_dimensions();
          dot_info.dim_nums.clear_rhs_batch_dimensions();

          dot_info.dim_nums.set_lhs_contracting_dimensions(
              0,
              dot_info.dim_nums.lhs_contracting_dimensions(0) - num_batch_dims);
          dot_info.dim_nums.set_rhs_contracting_dimensions(
              0,
              dot_info.dim_nums.rhs_contracting_dimensions(0) - num_batch_dims);

          llvm_ir::IrArray lhs_slice =
              SliceOutInnerArray(lhs_array_reshaped, /*batch_index=*/indvar, b);
          llvm_ir::IrArray rhs_slice =
              SliceOutInnerArray(rhs_array_reshaped, /*batch_index=*/indvar, b);
          llvm_ir::IrArray target_slice = SliceOutInnerArray(
              target_array_reshaped, /*batch_index=*/indvar, b);

          // Emit the inner non-batch dot operation.
          return EmitNonBatchDotOperation(
              dot_info, dot.name(), target_slice, lhs_slice, rhs_slice, nullptr,
              executable_run_options_value, b, mlir_context, hlo_module_config,
              target_machine_features);
        });
  }
}

bool IsBatchDot(const HloInstruction& instr) {
  if (auto* dot_instr = DynCast<HloDotInstruction>(&instr)) {
    return dot_instr->dot_dimension_numbers().lhs_batch_dimensions_size() > 0;
  }

  return false;
}
}  // namespace

bool DotImplementationCanHandleTranspose(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features) {
  DotImplementationStrategy impl_strategy =
      GetDotImplementationStrategy(dot_instr.parent()->parent()->config(),
                                   DotInfo(dot_instr), target_machine_features);

  return impl_strategy == DotImplementationStrategy::kNaiveLlvmIr ||
         impl_strategy == DotImplementationStrategy::kTiledLlvmIrGemv ||
         impl_strategy == DotImplementationStrategy::kEigen;
}

bool DotOperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& dot_instr,
    const TargetMachineFeatures& target_machine_features) {
  // Batched dots require the batch dimensions to be major. DotDecomposer always
  // moves batch dimensions to the front of the shape, so force a row-major
  // layout.
  if (IsBatchDot(dot_instr)) {
    return true;
  }

  DotImplementationStrategy impl_strategy =
      GetDotImplementationStrategy(dot_instr.parent()->parent()->config(),
                                   DotInfo(dot_instr), target_machine_features);

  return impl_strategy == DotImplementationStrategy::kTiledLlvmIrGemm ||
         impl_strategy == DotImplementationStrategy::kEigen;
}

Status EmitDotOperation(const HloInstruction& dot,
                        const llvm_ir::IrArray& target_array,
                        const llvm_ir::IrArray& lhs_array,
                        const llvm_ir::IrArray& rhs_array,
                        const llvm_ir::IrArray* addend_array,
                        llvm::Value* executable_run_options_value,
                        llvm::IRBuilder<>* b, mlir::MLIRContext* mlir_context,
                        const HloModuleConfig& hlo_module_config,
                        const TargetMachineFeatures& target_machine_features) {
  // This routine assumes that the dot operation is not in a parallelized
  // enclosing computation.
  CHECK(dot.parent()->root_instruction()->outer_dimension_partitions().empty());

  if (IsBatchDot(dot)) {
    TF_RET_CHECK(addend_array == nullptr);
    return EmitBatchDotOperation(dot, target_array, lhs_array, rhs_array,
                                 executable_run_options_value, b, mlir_context,
                                 hlo_module_config, target_machine_features);
  }

  return EmitNonBatchDotOperation(DotInfo(dot), dot.name(), target_array,
                                  lhs_array, rhs_array, addend_array,
                                  executable_run_options_value, b, mlir_context,
                                  hlo_module_config, target_machine_features);
}
}  // namespace cpu
}  // namespace xla
