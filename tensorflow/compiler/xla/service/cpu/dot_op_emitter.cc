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

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"
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
// Loads a tile of values from a 2D tensor.
class TileLoader {
 public:
  // Constructs a TileLoader that will load a tile consisting of
  // `tile_size_along_major_dim` vectors from the matrix `matrix`, starting at
  // `major_dim_offset` in the major dimension.  The tile size along the minor
  // dimension is the vector size, and that is implicitly determined by `vsl`.
  TileLoader(VectorSupportLibrary* vsl, llvm::IRBuilder<>* ir_builder,
             llvm::Value* matrix, int64 matrix_size_along_minor_dim,
             llvm::Value* major_dim_offset, int64 tile_size_along_major_dim)
      : vsl_(vsl) {
    pointers_.reserve(tile_size_along_major_dim);
    for (int64 i = 0; i < tile_size_along_major_dim; i++) {
      llvm::Value* total_offset = ir_builder->CreateMul(
          ir_builder->getInt64(matrix_size_along_minor_dim),
          ir_builder->CreateAdd(ir_builder->getInt64(i), major_dim_offset));
      pointers_.push_back(vsl_->ComputeOffsetPointer(matrix, total_offset));
    }
  }

  // Load a tile consisting of `tile_size_along_major_dim_` vectors starting at
  // `major_dim_offset_` in the major dimension and `minor_dim_offset` in the
  // minor dimension.
  std::vector<llvm::Value*> LoadTile(llvm::Value* minor_dim_offset) const {
    std::vector<llvm::Value*> result;
    result.reserve(pointers_.size());
    for (const auto& pointer : pointers_) {
      result.push_back(vsl_->LoadVector(pointer, minor_dim_offset));
    }
    return result;
  }

 private:
  VectorSupportLibrary* vsl_;
  std::vector<llvm::Value*> pointers_;
};

// Computes a dot product between "[M,K]{0,1} lhs" with a [K,1] vector (the
// layout of the vector does not matter).  This implementation uses a tiling
// scheme to improve performance.
//
// We logically separate the LHS matrix into four segments:
//
//   +----------------------+---+
//   |                      |   |
//   |                      |   |
//   |         A            | B |
//   |                      |   |
//   |                      |   |
//   |                      |   |
//   +----------------------+---+
//   |         C            | D |
//   +----------------------+---+
//
// where A is the largest submatrix of the LHS that can be evenly dividied into
// tiles.  For each tile in A, assuming tile_rows_ == tile_cols_ == 4, we have:
//
//   +---+---+---+---+       +--+--+--+--+
//   |M00|M10|M20|M30|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M01|M11|M21|M31| and   |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M02|M12|M22|M32|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M03|M13|M23|M33|       |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//
// (Legend: rows are horizontal and columns are vertical; and each column is one
// llvm::Value of a vector type)
//
// where:
//
//   a. The left tile is from the column major left matrix.
//   b. The right tile is an elementwise broadcast of a [V0, V1, V2, V3]
//      vector loaded from the RHS vector.
//
// As we iterate through the column dimension, we compute the change to the
// result vector by an elementwise multiplication between the two tiles above
// followed by a reduction along the major dimension:
//
//                     +-----------------------------------+
//                     | M00*V0 + M10*V1 + M20*V2 + M30*V3 |
//                     +-----------------------------------+
//                     | M01*V0 + M11*V1 + M21*V2 + M31*V3 |
// Result[R:R+4] +=    +-----------------------------------+
//                     | M02*V0 + M12*V1 + M22*V2 + M32*V3 |
//                     +-----------------------------------+
//                     | M03*V0 + M13*V1 + M23*V2 + M33*V3 |
//                     +-----------------------------------+
//
// Where R is the starting row for the tile.
//
// We have an inner epilogue loop to deal with the "C" submatrix and an outer
// epilogue loop to deal with the B,D submarix.
//
// TODO(sanjoy): We should investigate if using gather loads and scatter stores
// can be used here have the same inner loop for both column-major and row-major
// matrix-vector products.
class ColumnMajorMatrixVectorProductEmitter {
 public:
  ColumnMajorMatrixVectorProductEmitter(PrimitiveType scalar_type,
                                        int64 tile_rows, int64 tile_cols,
                                        int64 m, int64 k, llvm::Value* lhs,
                                        llvm::Value* rhs, llvm::Value* addend,
                                        llvm::Value* result,
                                        llvm::IRBuilder<>* ir_builder)
      : scalar_type_(scalar_type),
        tile_rows_(tile_rows),
        tile_cols_(tile_cols),
        m_(m),
        k_(k),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        ir_builder_(ir_builder),
        ksl_(ir_builder_),
        vsl_(scalar_type_, /*vector_size=*/tile_rows_, ir_builder_, "") {
    CHECK(tile_rows_ > 0 && IsPowerOfTwo(static_cast<uint64>(tile_rows_)));
  }

  void Emit();

 private:
  void EmitOuterLoopBody(llvm::Value* column, int64 column_count,
                         bool is_first_column);

  TileLoader GetLhsTileLoader(llvm::Value* column_start, int64 column_count) {
    return TileLoader(&vsl_, ir_builder_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/m_,
                      /*major_dim_offset=*/column_start,
                      /*tile_size_along_major_dim=*/column_count);
  }

  // Load a tile of values from the RHS.  For the RHS a "tile" is a contiguous
  // sequence of `count` values, each one broadcasted to the vector width.
  std::vector<llvm::Value*> LoadRhsTile(llvm::Value* offset, int64 count) {
    llvm::Value* base_pointer = vsl_.ComputeOffsetPointer(rhs_, offset);
    std::vector<llvm::Value*> result;
    result.reserve(count);
    for (int64 i = 0; i < count; i++) {
      result.push_back(vsl_.LoadBroadcast(base_pointer, i));
    }
    return result;
  }

  void EmitInnerLoopTiled(TileLoader* lhs_tile_loader,
                          const std::vector<llvm::Value*>& rhs_tile,
                          int64 columns, bool is_first_column);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_col, int64 columns,
                             bool is_first_tiled_column);

  PrimitiveType scalar_type_;
  int64 tile_rows_;
  int64 tile_cols_;
  int64 m_;
  int64 k_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* ir_builder_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void ColumnMajorMatrixVectorProductEmitter::EmitOuterLoopBody(
    llvm::Value* column, int64 column_count, bool is_first_column) {
  TileLoader lhs_tile_loader = GetLhsTileLoader(/*column_start=*/column,
                                                /*column_count=*/column_count);

  std::vector<llvm::Value*> rhs_tile =
      LoadRhsTile(column, /*count=*/column_count);
  EmitInnerLoopTiled(&lhs_tile_loader, rhs_tile,
                     /*columns=*/column_count, is_first_column);
  EmitInnerLoopEpilogue(column, /*columns=*/column_count, is_first_column);
}

void ColumnMajorMatrixVectorProductEmitter::Emit() {
  // See the comment on the class declaration for the algorithm used here.
  int64 column_remainder = k_ % tile_cols_;
  int64 column_limit = k_ - column_remainder;

  ksl_.For("dot.outer.tiled",
           /*start=*/0, /*end=*/column_limit, /*step=*/tile_cols_,
           [&](llvm::Value* column, bool is_first_column) {
             EmitOuterLoopBody(column, tile_cols_, is_first_column);
           });

  if (column_remainder != 0) {
    EmitOuterLoopBody(ir_builder_->getInt64(column_limit), column_remainder,
                      column_limit == 0);
  }
}

void ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    TileLoader* lhs_tile_loader, const std::vector<llvm::Value*>& rhs_tile,
    int64 columns, bool is_first_column) {
  int64 row_limit = m_ - (m_ % tile_rows_);

  ksl_.For("dot.inner.tiled", /*start=*/0, /*end=*/row_limit,
           /*step=*/tile_rows_, [&](llvm::Value* row) {
             std::vector<llvm::Value*> lhs_tile =
                 lhs_tile_loader->LoadTile(/*minor_dim_offset=*/row);
             llvm::Value* accumulator =
                 is_first_column ? (addend_ ? vsl_.LoadVector(addend_, row)
                                            : vsl_.GetZeroVector())
                                 : vsl_.LoadVector(result_, row);
             for (int i = 0; i < columns; i++) {
               accumulator = vsl_.MulAdd(lhs_tile[i], rhs_tile[i], accumulator);
             }
             vsl_.StoreVector(accumulator, result_, row);
           });
}

void ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue(
    llvm::Value* current_tile_col, int64 columns, bool is_first_tiled_column) {
  int64 row_start = m_ - (m_ % tile_rows_);
  if (row_start == m_) {
    return;
  }

  llvm::Value* columns_llvm = ir_builder_->getInt64(columns);

  // for (col = current_tile_col; col < (columns + current_tile_col); col++)
  //   for (row = row_start, row < m_; row++) {
  //     result[row] += lhs[row, col] * rhs[col]
  //     // Also take into account that if col is 0 then result[row] is not
  //     // initialized.
  //   }

  ksl_.For(
      "dot.inner.epilg.outer", /*start=*/current_tile_col,
      /*end=*/ir_builder_->CreateAdd(columns_llvm, current_tile_col),
      /*step=*/1, /*peel_first_iteration=*/false,
      [&](llvm::Value* col, llvm::Value* is_first_scalar_col) {
        llvm::Value* rhs_element = vsl_.LoadScalar(rhs_, col);
        llvm::Value* total_offset =
            ir_builder_->CreateMul(col, ir_builder_->getInt64(m_));
        llvm::Value* lhs_base_pointer =
            vsl_.ComputeOffsetPointer(lhs_, total_offset);
        ksl_.For(
            "dot.inner.epilg.inner", /*start=*/row_start, /*end=*/m_,
            /*step=*/1, [&](llvm::Value* scalar_row) {
              llvm::Value* product = vsl_.Mul(
                  vsl_.LoadScalar(lhs_base_pointer, scalar_row), rhs_element);
              llvm::Value* setting_result_first_time = ir_builder_->CreateAnd(
                  is_first_scalar_col,
                  ir_builder_->getInt1(is_first_tiled_column));
              ksl_.If(
                  setting_result_first_time,
                  /*true_block_generator=*/
                  [&]() {
                    if (addend_) {
                      vsl_.StoreScalar(
                          vsl_.Add(vsl_.LoadScalar(addend_, scalar_row),
                                   product),
                          result_, scalar_row);
                    } else {
                      vsl_.StoreScalar(product, result_, scalar_row);
                    }
                  },
                  /*false_block_generator=*/
                  [&]() {
                    vsl_.StoreScalar(
                        vsl_.Add(vsl_.LoadScalar(result_, scalar_row), product),
                        result_, scalar_row);
                  });
            });
      });
}

// Computes a dot product between "[M,K]{1,0} lhs" with a [K,1] vector (the
// layout of the vector does not matter).  This implementation uses a tiling
// scheme to improve performance.
//
// We logically separate the LHS matrix into four segments:
//
//   +----------------------+---+
//   |                      |   |
//   |                      |   |
//   |         A            | B |
//   |                      |   |
//   |                      |   |
//   |                      |   |
//   +----------------------+---+
//   |         C            | D |
//   +----------------------+---+
//
// where A is the largest submatrix of the LHS that can be evenly dividied into
// tiles.  For each tile in A, assuming tile_rows_ == tile_cols_ == 4, we have:
//
//   +---+---+---+---+
//   |M00|M10|M20|M30|
//   +---+---+---+---+       +--+--+--+--+
//   |M01|M11|M21|M31| and   |V0|V1|V2|V3|
//   +---+---+---+---+       +--+--+--+--+
//   |M02|M12|M22|M32|
//   +---+---+---+---+
//   |M03|M13|M23|M33|
//   +---+---+---+---+
//
// (Legend: rows are horizontal and columns are vertical; and each row is one
// llvm::Value of a vector type)
//
// where:
//
//   a. The left tile is loaded from the row major left matrix.
//   b. The right vector is loaded from the RHS vector.
//
// We keep 4 vector accumulators accumulating the following four vector
// expressions as we iterate over the row dimension:
//
//   +------+------+------+------+
//   |M0I*V0|M1I*V1|M2I*V2|M3I*V3|  for I in [0,4)
//   +------+------+------+------+
//
// In the end we do a horizontal reduction over these 4 vector accumulators to
// get 4 values in the result vector.
//
// We have an inner epilogue loop to deal with the "B" sub-matrix and an outer
// epilogue loop to deal with the C,D submatrix.
class RowMajorMatrixVectorProductEmitter {
 public:
  RowMajorMatrixVectorProductEmitter(PrimitiveType scalar_type, int64 tile_rows,
                                     int64 tile_cols, int64 m, int64 k,
                                     llvm::Value* lhs, llvm::Value* rhs,
                                     llvm::Value* addend, llvm::Value* result,
                                     llvm::IRBuilder<>* ir_builder)
      : scalar_type_(scalar_type),
        tile_rows_(tile_rows),
        tile_cols_(tile_cols),
        m_(m),
        k_(k),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        ir_builder_(ir_builder),
        ksl_(ir_builder_),
        vsl_(scalar_type_, /*vector_size=*/tile_cols_, ir_builder_, "") {
    CHECK(tile_cols_ > 0 && IsPowerOfTwo(static_cast<uint64>(tile_cols_)));
  }

  void Emit();

 private:
  TileLoader GetLhsTileLoader(llvm::Value* row_start, int64 row_count) {
    return TileLoader(&vsl_, ir_builder_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/k_,
                      /*major_dim_offset=*/row_start,
                      /*tile_size_along_major_dim=*/row_count);
  }

  void EmitOuterLoopBody(llvm::Value* row, int64 row_count);

  void EmitInnerLoopTiled(TileLoader* lhs_tile_loader, int64 rows,
                          std::vector<VectorVariable>* vector_accumulators);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_row, int64 rows,
                             std::vector<ScalarVariable>* scalar_accumulators);

  PrimitiveType scalar_type_;
  int64 tile_rows_;
  int64 tile_cols_;
  int64 m_;
  int64 k_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* ir_builder_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void RowMajorMatrixVectorProductEmitter::EmitOuterLoopBody(llvm::Value* row,
                                                           int64 row_count) {
  TileLoader lhs_tile_loader = GetLhsTileLoader(/*row_start=*/row,
                                                /*row_count=*/row_count);
  std::vector<VectorVariable> vector_accumulators;
  std::vector<ScalarVariable> scalar_accumulators;
  for (int i = 0; i < row_count; i++) {
    vector_accumulators.emplace_back(&vsl_, vsl_.GetZeroVector());
    scalar_accumulators.emplace_back(&vsl_, vsl_.GetZeroScalar());
  }
  EmitInnerLoopTiled(&lhs_tile_loader, /*rows=*/row_count,
                     &vector_accumulators);
  EmitInnerLoopEpilogue(/*current_tile_row=*/row, /*rows=*/row_count,
                        &scalar_accumulators);

  std::vector<llvm::Value*> accumulator_values;
  std::transform(
      vector_accumulators.begin(), vector_accumulators.end(),
      std::back_inserter(accumulator_values),
      [](const VectorVariable& vector_var) { return vector_var.Get(); });

  std::vector<llvm::Value*> horizontal_sums;
  if (row_count == vsl_.vector_size()) {
    if (addend_) {
      horizontal_sums = vsl_.ComputeHorizontalSums(
          std::move(accumulator_values), vsl_.LoadVector(addend_, row));
    } else {
      horizontal_sums =
          vsl_.ComputeHorizontalSums(std::move(accumulator_values));
    }
  } else {
    horizontal_sums = vsl_.ComputeHorizontalSums(std::move(accumulator_values));
  }

  for (int i = 0; i < row_count; i++) {
    llvm::Value* result_value =
        vsl_.Add(horizontal_sums[i], scalar_accumulators[i].Get());
    llvm::Value* offset = ir_builder_->CreateAdd(ir_builder_->getInt64(i), row);
    if (addend_ && row_count != vsl_.vector_size()) {
      result_value = vsl_.Add(vsl_.LoadScalar(addend_, offset), result_value);
    }
    vsl_.StoreScalar(result_value, result_, offset);
  }
}

void RowMajorMatrixVectorProductEmitter::Emit() {
  // See the comment on the class declaration for the algorithm used here.
  int64 row_remainder = m_ % tile_rows_;
  int64 row_limit = m_ - row_remainder;

  ksl_.For("dot.outer.tiled",
           /*start=*/0, /*end=*/row_limit, /*step=*/tile_rows_,
           [&](llvm::Value* row) { EmitOuterLoopBody(row, tile_rows_); });

  if (row_remainder != 0) {
    EmitOuterLoopBody(ir_builder_->getInt64(row_limit), row_remainder);
  }
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    TileLoader* lhs_tile_loader, int64 rows,
    std::vector<VectorVariable>* vector_accumulators) {
  int64 column_limit = k_ - (k_ % tile_cols_);

  ksl_.For("dot.inner.tiled", /*start=*/0, /*end=*/column_limit,
           /*step=*/tile_cols_, [&](llvm::Value* col) {
             std::vector<llvm::Value*> lhs_tile =
                 lhs_tile_loader->LoadTile(/*minor_dim_offset=*/col);
             llvm::Value* rhs_value = vsl_.LoadVector(rhs_, col);
             for (int i = 0; i < rows; i++) {
               llvm::Value* old_sum = (*vector_accumulators)[i].Get();
               (*vector_accumulators)[i].Set(
                   vsl_.Add(old_sum, vsl_.Mul(rhs_value, lhs_tile[i])));
             }
           });
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue(
    llvm::Value* current_tile_row, int64 rows,
    std::vector<ScalarVariable>* scalar_accumulators) {
  int64 column_start = k_ - (k_ % tile_cols_);
  if (column_start == k_) {
    return;
  }

  for (int r = 0; r < rows; r++) {
    llvm::Value* total_offset = ir_builder_->CreateMul(
        ir_builder_->CreateAdd(ir_builder_->getInt64(r), current_tile_row),
        ir_builder_->getInt64(k_));
    llvm::Value* lhs_base_pointer =
        vsl_.ComputeOffsetPointer(lhs_, total_offset);
    ksl_.For("dot.inner.epilg.inner", /*start=*/column_start, /*end=*/k_,
             /*step=*/1, [&](llvm::Value* scalar_col) {
               llvm::Value* product =
                   vsl_.Mul(vsl_.LoadScalar(lhs_base_pointer, scalar_col),
                            vsl_.LoadScalar(rhs_, scalar_col));
               llvm::Value* old_value = (*scalar_accumulators)[r].Get();
               (*scalar_accumulators)[r].Set(vsl_.Add(old_value, product));
             });
  }
}

}  // namespace

DotOpEmitter::DotOpEmitter(
    const HloInstruction& dot, bool transpose_lhs, bool transpose_rhs,
    const llvm_ir::IrArray& target_array, const llvm_ir::IrArray& lhs_array,
    const llvm_ir::IrArray& rhs_array, const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* ir_builder,
    const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features)
    : dot_(dot),
      transpose_lhs_(transpose_lhs),
      transpose_rhs_(transpose_rhs),
      target_array_(target_array),
      lhs_array_(lhs_array),
      rhs_array_(rhs_array),
      addend_array_(addend_array),
      executable_run_options_value_(executable_run_options_value),
      ir_builder_(ir_builder),
      hlo_module_config_(hlo_module_config),
      target_machine_features_(target_machine_features) {}

/* static */ tensorflow::Status DotOpEmitter::EmitDotOperation(
    const HloInstruction& dot, bool transpose_lhs, bool transpose_rhs,
    const llvm_ir::IrArray& target_array, const llvm_ir::IrArray& lhs_array,
    const llvm_ir::IrArray& rhs_array, const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* ir_builder,
    const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features) {
  PrimitiveType type = target_array.GetShape().element_type();
  TF_RET_CHECK(F16 == type || F32 == type || F64 == type || C64 == type);
  DotOpEmitter dot_emitter(dot, transpose_lhs, transpose_rhs, target_array,
                           lhs_array, rhs_array, addend_array,
                           executable_run_options_value, ir_builder,
                           hlo_module_config, target_machine_features);
  return dot_emitter.Emit();
}

bool DotOpEmitter::ShapesAreLegalForRuntimeDot() const { return true; }

bool DotOpEmitter::EmitLlvmIrDotIfProfitable() {
  if (dot_.shape().dimensions_size() != 2) {
    return false;
  }

  PrimitiveType primitive_type = dot_.shape().element_type();

  if (!primitive_util::IsFloatingPointType(primitive_type) &&
      !primitive_util::IsIntegralType(primitive_type)) {
    return false;
  }

  MatMultDims mat_mult_dims = GetMatMultDims();
  bool is_column_major_matrix_vector = false;
  bool is_row_major_matrix_vector = false;

  int64 m, k;
  bool swap_operands;

  if (mat_mult_dims.m == 1) {
    bool rhs_effectively_row_major =
        transpose_rhs_ ^ !mat_mult_dims.rhs_column_major;
    if (rhs_effectively_row_major) {
      k = mat_mult_dims.k;
      m = mat_mult_dims.n;
      is_column_major_matrix_vector = true;
      swap_operands = true;
    } else {
      k = mat_mult_dims.k;
      m = mat_mult_dims.n;
      is_row_major_matrix_vector = true;
      swap_operands = true;
    }
  }

  if (mat_mult_dims.n == 1) {
    bool lhs_effectively_column_major =
        transpose_lhs_ ^ mat_mult_dims.lhs_column_major;
    if (lhs_effectively_column_major) {
      m = mat_mult_dims.m;
      k = mat_mult_dims.k;
      is_column_major_matrix_vector = true;
      swap_operands = false;
    } else {
      m = mat_mult_dims.m;
      k = mat_mult_dims.k;
      is_row_major_matrix_vector = true;
      swap_operands = false;
    }
  }

  if (!is_column_major_matrix_vector && !is_row_major_matrix_vector) {
    return false;
  }

  int64 tiling_factor = GetGemvTilingFactor();
  CHECK_GT(tiling_factor, 0);

  llvm::Value* result_op = target_array_.GetBasePointer();
  llvm::Value* lhs_op =
      swap_operands ? rhs_array_.GetBasePointer() : lhs_array_.GetBasePointer();
  llvm::Value* rhs_op =
      swap_operands ? lhs_array_.GetBasePointer() : rhs_array_.GetBasePointer();

  const bool enable_fast_math =
      hlo_module_config_.debug_options().xla_enable_fast_math();
  const bool optimize_for_size =
      options::OptimizeForSizeRequested(hlo_module_config_);

  const int target_vector_register_element_size =
      target_machine_features_.vector_register_num_elements(
          *ir_builder_->GetInsertBlock()->getParent(), primitive_type);

  // We may not always know the vector register size for the target we're
  // compiling against, in which case target_vector_register_element_size is 0.
  // In these cases we choose a default LLVM IR register size.
  const int kUnknownTargetVectorRegisterSize = 4;
  const int vector_register_element_size =
      target_vector_register_element_size == 0
          ? kUnknownTargetVectorRegisterSize
          : target_vector_register_element_size;

  if (is_column_major_matrix_vector) {
    VLOG(2) << "Emitting column major matrix-vector multiply with m = " << m
            << " and k = " << k;
    int64 tile_rows = vector_register_element_size;
    int64 tile_cols = tiling_factor;

    string kernel_name = tensorflow::strings::StrCat(
        "col_major_gemv_", PrimitiveType_Name(primitive_type), "_", tile_rows,
        "_", tile_cols, "_", m, "_", k, addend_array_ ? "_with_addend" : "");

    KernelSupportLibrary::EmitAndCallOutlinedKernel(
        /*enable_fast_math=*/enable_fast_math,
        /*optimize_for_size=*/optimize_for_size, ir_builder_, kernel_name,
        lhs_op, rhs_op,
        addend_array_ ? addend_array_->GetBasePointer() : nullptr, result_op,
        [this, tile_rows, tile_cols, m, k, primitive_type](
            llvm::Value* lhs_op, llvm::Value* rhs_op, llvm::Value* addend_op,
            llvm::Value* result_op) {
          ColumnMajorMatrixVectorProductEmitter emitter(
              primitive_type, tile_rows, tile_cols, m, k, lhs_op, rhs_op,
              addend_op, result_op, ir_builder_);
          emitter.Emit();
        });
  } else {
    VLOG(2) << "Emitting row major matrix-vector multiply with m = " << m
            << " and k = " << k;
    int64 tile_rows = tiling_factor;
    int64 tile_cols = vector_register_element_size;

    string kernel_name = tensorflow::strings::StrCat(
        "row_major_gemv_", PrimitiveType_Name(primitive_type), "_", tile_rows,
        "_", tile_cols, "_", m, "_", k, addend_array_ ? "_with_addend" : "");

    KernelSupportLibrary::EmitAndCallOutlinedKernel(
        /*enable_fast_math=*/enable_fast_math,
        /*optimize_for_size=*/optimize_for_size, ir_builder_, kernel_name,
        lhs_op, rhs_op,
        addend_array_ ? addend_array_->GetBasePointer() : nullptr, result_op,
        [this, tile_rows, tile_cols, m, k, primitive_type](
            llvm::Value* lhs_op, llvm::Value* rhs_op, llvm::Value* addend_op,
            llvm::Value* result_op) {
          RowMajorMatrixVectorProductEmitter emitter(
              primitive_type, tile_rows, tile_cols, m, k, lhs_op, rhs_op,
              addend_op, result_op, ir_builder_);
          emitter.Emit();
        });
  }

  return true;
}

tensorflow::Status DotOpEmitter::Emit() {
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

  if (EmitLlvmIrDotIfProfitable()) {
    return Status::OK();
  }

  CHECK_EQ(addend_array_, nullptr);

  if (PotentiallyImplementedAsEigenDot(dot_)) {
    return EmitCallToRuntime();
  }

  // Reduce along dimension 0 of the LHS and 1 of the RHS. Vectors are a special
  // case where the reduction dimension is 0 for both LHS and RHS. This results
  // in a vector dot product producing a scalar.
  int64 lhs_reduction_dimension = 0;
  if (ShapeUtil::Rank(lhs_shape) >= 2) {
    lhs_reduction_dimension =
        ShapeUtil::GetDimensionNumber(lhs_shape, transpose_lhs_ ? -2 : -1);
  }
  int64 rhs_reduction_dimension = 0;
  if (ShapeUtil::Rank(rhs_shape) >= 2) {
    rhs_reduction_dimension =
        ShapeUtil::GetDimensionNumber(rhs_shape, transpose_rhs_ ? -1 : -2);
  }

  // Verify the reduction dimension in the two operands are the same size.
  TF_RET_CHECK(lhs_shape.dimensions(lhs_reduction_dimension) ==
               rhs_shape.dimensions(rhs_reduction_dimension));

  bool lhs_reduction_along_minor_dimension =
      lhs_reduction_dimension == LayoutUtil::Minor(lhs_shape.layout(), 0);
  bool rhs_reduction_along_minor_dimension =
      rhs_reduction_dimension == LayoutUtil::Minor(rhs_shape.layout(), 0);

  // Create loop nests which loop through the LHS operand dimensions and the RHS
  // operand dimensions. The reduction dimension of the LHS and RHS are handled
  // in a separate innermost loop which performs the sum of products.
  llvm_ir::ForLoopNest loop_nest(llvm_ir::IrName(&dot_), ir_builder_);
  llvm_ir::IrArray::Index lhs_index = EmitOperandArrayLoopNest(
      &loop_nest, lhs_array_, lhs_reduction_dimension, "lhs");
  llvm_ir::IrArray::Index rhs_index = EmitOperandArrayLoopNest(
      &loop_nest, rhs_array_, rhs_reduction_dimension, "rhs");

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
      /*prevent_unrolling=*/lhs_reduction_along_minor_dimension &&
          rhs_reduction_along_minor_dimension);

  // The final entry in the rhs and lhs indexes is the indvar of the
  // reduction loop.
  lhs_index[lhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  rhs_index[rhs_reduction_dimension] = reduction_loop->GetIndVarValue();

  // For computing the sum of products we alloca a single location to store the
  // dot product result as we accumulate it within the reduction loop. After the
  // reduction loop we load the result and store into the output array.

  // Function entry basic block.
  // - Emit alloca for accumulator
  llvm::Function* func = reduction_loop->GetPreheaderBasicBlock()->getParent();
  SetToFirstInsertPoint(&func->getEntryBlock(), ir_builder_);
  llvm::Type* accum_type = target_array_.GetElementLlvmType();
  llvm::Value* accum_address = ir_builder_->CreateAlloca(
      accum_type, /*ArraySize=*/nullptr, "accum_address");

  // Preheader basic block of reduction loop:
  // - Initialize accumulator to zero.
  llvm::BasicBlock* preheader_bb = reduction_loop->GetPreheaderBasicBlock();
  ir_builder_->SetInsertPoint(preheader_bb->getTerminator());

  ir_builder_->CreateStore(llvm::Constant::getNullValue(accum_type),
                           accum_address);

  // Body basic block of reduction loop:
  // - Load elements from lhs and rhs array.
  // - Multiply lhs-element and rhs-element.
  // - Load accumulator and add to product.
  // - Store sum back into accumulator.
  SetToFirstInsertPoint(reduction_loop->GetBodyBasicBlock(), ir_builder_);

  llvm::Value* lhs_element =
      lhs_array_.EmitReadArrayElement(lhs_index, ir_builder_);
  llvm::Value* rhs_element =
      rhs_array_.EmitReadArrayElement(rhs_index, ir_builder_);

  llvm::Value* accum = ir_builder_->CreateLoad(accum_address);
  llvm::Value* updated_accum;
  if (ShapeUtil::ElementIsComplex(lhs_shape)) {
    auto real = [&](llvm::Value* x) {
      return ir_builder_->CreateExtractValue(x, {0});
    };
    auto imag = [&](llvm::Value* x) {
      return ir_builder_->CreateExtractValue(x, {1});
    };
    llvm::Value* product_real = ir_builder_->CreateFSub(
        ir_builder_->CreateFMul(real(lhs_element), real(rhs_element)),
        ir_builder_->CreateFMul(imag(lhs_element), imag(rhs_element)));
    llvm::Value* product_imag = ir_builder_->CreateFAdd(
        ir_builder_->CreateFMul(real(lhs_element), imag(rhs_element)),
        ir_builder_->CreateFMul(imag(lhs_element), real(rhs_element)));
    updated_accum = ir_builder_->CreateInsertValue(
        accum, ir_builder_->CreateFAdd(real(accum), product_real), {0});
    updated_accum = ir_builder_->CreateInsertValue(
        updated_accum, ir_builder_->CreateFAdd(imag(accum), product_imag), {1});
  } else {
    llvm::Value* product = ir_builder_->CreateFMul(lhs_element, rhs_element);
    updated_accum = ir_builder_->CreateFAdd(accum, product);
  }
  ir_builder_->CreateStore(updated_accum, accum_address);

  // Exit basic block of reduction loop.
  // - Load accumulator value (the result).
  // - Store into output array.
  SetToFirstInsertPoint(reduction_loop->GetExitBasicBlock(), ir_builder_);

  llvm::Value* result = ir_builder_->CreateLoad(accum_address);

  // Create index into target address. The target index is the concatenation of
  // the rhs and lhs indexes with the reduction dimensions removed. The terms
  // from the rhs index are the lower dimensions in the index so we add them
  // first.
  llvm_ir::IrArray::Index target_index;
  for (int dimension = 0; dimension < lhs_index.size(); ++dimension) {
    if (dimension != lhs_reduction_dimension) {
      target_index.push_back(lhs_index[dimension]);
    }
  }
  for (int dimension = 0; dimension < rhs_index.size(); ++dimension) {
    if (dimension != rhs_reduction_dimension) {
      target_index.push_back(rhs_index[dimension]);
    }
  }

  target_array_.EmitWriteArrayElement(target_index, result, ir_builder_);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop.
  ir_builder_->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return tensorflow::Status::OK();
}

tensorflow::Status DotOpEmitter::EmitScalarDot() {
  // A scalar dot is just a scalar multiply.
  llvm::Value* result;
  llvm::Value* lhs_value =
      lhs_array_.EmitReadArrayElement(/*index=*/{}, ir_builder_);
  llvm::Value* rhs_value =
      rhs_array_.EmitReadArrayElement(/*index=*/{}, ir_builder_);
  if (ShapeUtil::ElementIsComplex(lhs_array_.GetShape())) {
#define REAL(x) ir_builder_->CreateExtractValue(x, {0})
#define IMAG(x) ir_builder_->CreateExtractValue(x, {1})
    llvm::Value* real = ir_builder_->CreateFSub(
        ir_builder_->CreateFMul(REAL(lhs_value), REAL(rhs_value)),
        ir_builder_->CreateFMul(IMAG(lhs_value), IMAG(rhs_value)));
    llvm::Value* imag = ir_builder_->CreateFAdd(
        ir_builder_->CreateFMul(REAL(lhs_value), IMAG(rhs_value)),
        ir_builder_->CreateFMul(IMAG(lhs_value), REAL(rhs_value)));
#undef IMAG
#undef REAL
    result = llvm::ConstantAggregateZero::get(lhs_array_.GetElementLlvmType());
    result = ir_builder_->CreateInsertValue(result, real, {0});
    result = ir_builder_->CreateInsertValue(result, imag, {1});
  } else {
    result = ir_builder_->CreateFMul(lhs_value, rhs_value);
  }
  target_array_.EmitWriteArrayElement(/*index=*/{}, result, ir_builder_);
  return tensorflow::Status::OK();
}

tensorflow::Status DotOpEmitter::EmitCallToRuntime() {
  DCHECK(ShapesAreLegalForRuntimeDot());

  // The signature of the Eigen runtime matmul function is:
  //
  //   (void)(void* run_options, float* out, float* lhs, float* rhs,
  //          int64 m, int64 n, int64 k, int32 transpose_lhs,
  //          int32 transpose_rhs);
  // The two transpose_... parameters are actually booleans, but we use int32
  // to avoid target-dependent calling convention details.

  bool multi_threaded_eigen =
      hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
  PrimitiveType type = target_array_.GetShape().element_type();
  llvm::Type* float_type;
  const char* fn_name;
  switch (type) {
    case F32:
      fn_name = multi_threaded_eigen
                    ? runtime::kEigenMatMulF32SymbolName
                    : runtime::kEigenSingleThreadedMatMulF32SymbolName;
      float_type = ir_builder_->getFloatTy();
      break;
    case F64:
      fn_name = multi_threaded_eigen
                    ? runtime::kEigenMatMulF64SymbolName
                    : runtime::kEigenSingleThreadedMatMulF64SymbolName;
      float_type = ir_builder_->getDoubleTy();
      break;
    default:
      return Unimplemented("Invalid type %s for dot operation",
                           PrimitiveType_Name(type).c_str());
  }

  llvm::Type* float_ptr_type = float_type->getPointerTo();
  llvm::Type* int64_type = ir_builder_->getInt64Ty();
  llvm::Type* int32_type = ir_builder_->getInt32Ty();
  llvm::Type* int8_ptr_type = ir_builder_->getInt8Ty()->getPointerTo();
  llvm::FunctionType* matmul_type = llvm::FunctionType::get(
      ir_builder_->getVoidTy(),
      {int8_ptr_type, float_ptr_type, float_ptr_type, float_ptr_type,
       int64_type, int64_type, int64_type, int32_type, int32_type},
      /*isVarArg=*/false);

  llvm::Function* function = ir_builder_->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();

  llvm::Function* matmul_func = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, matmul_type));
  matmul_func->setCallingConv(llvm::CallingConv::C);
  matmul_func->setDoesNotThrow();
  matmul_func->setOnlyAccessesArgMemory();

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
  bool transpose_lhs = transpose_lhs_;
  bool transpose_rhs = transpose_rhs_;

  if (!mat_mult_dims.lhs_column_major) {
    std::swap(mat_mult_dims.m, mat_mult_dims.n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
  }

  ir_builder_->CreateCall(
      matmul_func,
      {ir_builder_->CreateBitCast(executable_run_options_value_, int8_ptr_type),
       ir_builder_->CreateBitCast(target_array_.GetBasePointer(),
                                  float_ptr_type),
       ir_builder_->CreateBitCast(lhs->GetBasePointer(), float_ptr_type),
       ir_builder_->CreateBitCast(rhs->GetBasePointer(), float_ptr_type),
       ir_builder_->getInt64(mat_mult_dims.m),
       ir_builder_->getInt64(mat_mult_dims.n),
       ir_builder_->getInt64(mat_mult_dims.k),
       ir_builder_->getInt32(transpose_lhs),
       ir_builder_->getInt32(transpose_rhs)});
  return tensorflow::Status::OK();
}

DotOpEmitter::MatMultDims DotOpEmitter::GetMatMultDims() const {
  CHECK_EQ(dot_.shape().dimensions_size(), 2);

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();

  return {lhs_shape.dimensions(transpose_lhs_ ? 1 : 0),
          lhs_shape.dimensions(transpose_lhs_ ? 0 : 1),
          rhs_shape.dimensions(transpose_rhs_ ? 0 : 1),
          LayoutUtil::Minor(lhs_shape.layout(), 0) == 0,
          LayoutUtil::Minor(rhs_shape.layout(), 0) == 0};
}

llvm_ir::IrArray::Index DotOpEmitter::EmitOperandArrayLoopNest(
    llvm_ir::ForLoopNest* loop_nest, const llvm_ir::IrArray& operand_array,
    int64 reduction_dimension, tensorflow::StringPiece name_suffix) {
  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // reduction dimension.
  std::vector<int64> dimensions;
  const Shape& shape = operand_array.GetShape();
  for (int i = LayoutUtil::MinorToMajor(shape).size() - 1; i >= 0; --i) {
    int64 dimension = LayoutUtil::Minor(shape.layout(), i);
    if (dimension != reduction_dimension) {
      dimensions.push_back(dimension);
    }
  }

  // Create loop nest with one for-loop for each dimension of the
  // output.
  llvm_ir::IrArray::Index index =
      loop_nest->AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix);
  // Verify every dimension except the reduction dimension was set in the index.
  for (int dimension = 0; dimension < index.size(); ++dimension) {
    if (dimension == reduction_dimension) {
      DCHECK_EQ(nullptr, index[dimension]);
    } else {
      DCHECK_NE(nullptr, index[dimension]);
    }
  }
  return index;
}

// Return whether the given shape is a matrix with no padding.
static bool IsRank2WithNoPadding(const Shape& shape) {
  return ShapeUtil::Rank(shape) == 2 && !LayoutUtil::IsPadded(shape);
}

// In a gemm operation where output = lhs * rhs, check whether the given shapes
// are valid for the operation.
static bool AreValidGemmShapes(const Shape& lhs_shape, const Shape& rhs_shape,
                               const Shape& output_shape) {
  // The inputs and the output must
  // 1) be matrices with no padding, and
  // 2) have an allowed element type.
  return output_shape.element_type() == F32 &&
         IsRank2WithNoPadding(lhs_shape) && IsRank2WithNoPadding(rhs_shape) &&
         IsRank2WithNoPadding(output_shape);
}

bool PotentiallyImplementedAsEigenDot(const HloInstruction& hlo) {
  // For certain types of Dot, we can call Eigen
  if (hlo.opcode() == HloOpcode::kDot) {
    const Shape& lhs_shape = hlo.operand(0)->shape();
    const Shape& rhs_shape = hlo.operand(1)->shape();

    if (ShapeUtil::HasZeroElements(lhs_shape) ||
        ShapeUtil::HasZeroElements(rhs_shape)) {
      return false;
    }

    if (ProfitableToImplementDotInTiledLlvmIr(hlo)) {
      return false;
    }

    // If gemm can accept the operand shapes, use it rather than a custom
    // kernel.
    if (AreValidGemmShapes(lhs_shape, rhs_shape, hlo.shape())) {
      // The size of the reduction dimension should match. The shape inference
      // guarantees this invariant, so the check here is for programming
      // errors.
      CHECK_EQ(lhs_shape.dimensions(1), rhs_shape.dimensions(0));
      return true;
    }
  }

  if (hlo.opcode() == HloOpcode::kFusion &&
      hlo.fusion_kind() == HloInstruction::FusionKind::kTransposeDot &&
      hlo.fused_expression_root()->opcode() == HloOpcode::kDot) {
    auto* dot = hlo.fused_expression_root();
    const Shape& lhs_shape = dot->operand(0)->shape();
    const Shape& rhs_shape = dot->operand(1)->shape();
    if (ShapeUtil::HasZeroElements(lhs_shape) ||
        ShapeUtil::HasZeroElements(rhs_shape)) {
      return false;
    }
    return true;
  }

  return false;
}

// For vector-matrix dot products, it is always profitable to make the Rhs
// column major.
tensorflow::gtl::optional<int64> ProfitableToMakeDotOperandColumnMajor(
    const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kDot && hlo.shape().dimensions_size() == 2 &&
      hlo.shape().dimensions(0) == 1) {
    if (hlo.dot_dimension_numbers().rhs_contracting_dimensions(0) == 0) {
      return 1;
    }
    return {};
  }

  if (hlo.opcode() == HloOpcode::kFusion &&
      hlo.fusion_kind() == HloInstruction::FusionKind::kOutput) {
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

bool ProfitableToImplementDotInTiledLlvmIr(const HloInstruction& dot) {
  // Any Matrix-Vector product of floating point or integral type, or
  // a transpose-dot fusion of the same can be lowered to a tiled LLVM
  // IR implementation.
  const Shape& shape = dot.shape();
  return shape.dimensions_size() == 2 &&
         (shape.dimensions(0) == 1 || shape.dimensions(1) == 1) &&
         (primitive_util::IsFloatingPointType(shape.element_type()) ||
          primitive_util::IsIntegralType(shape.element_type()));
}

}  // namespace cpu
}  // namespace xla
