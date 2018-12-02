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
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
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
// Provides tiled access to an in-memory rank 2 array.
class MemoryTile {
 public:
  // Constructs a MemoryTile that can operate on tiles consisting of
  // `tile_size_along_major_dim` vectors from the matrix `matrix`, starting at
  // `major_dim_offset` in the major dimension.  The tile size along the minor
  // dimension is the vector size, and that is implicitly determined by `vsl`.
  MemoryTile(VectorSupportLibrary* vsl, llvm::IRBuilder<>* b,
             llvm::Value* matrix, int64 matrix_size_along_minor_dim,
             llvm::Value* major_dim_offset, int64 tile_size_along_major_dim)
      : vsl_(vsl), b_(b) {
    pointers_.reserve(tile_size_along_major_dim);
    for (int64 i = 0; i < tile_size_along_major_dim; i++) {
      llvm::Value* total_offset =
          b->CreateMul(b->getInt64(matrix_size_along_minor_dim),
                       b->CreateAdd(b->getInt64(i), major_dim_offset));
      pointers_.push_back(vsl_->ComputeOffsetPointer(matrix, total_offset));
    }
  }

  // Load a tile consisting of `tile_size_along_major_dim` vectors from position
  // {major: `major_dim_offset`, minor: `minor_dim_offset`}.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  std::vector<llvm::Value*> LoadTile(llvm::Value* minor_dim_offset) const {
    std::vector<llvm::Value*> result;
    result.reserve(pointers_.size());
    for (const auto& pointer : pointers_) {
      result.push_back(vsl_->LoadVector(pointer, minor_dim_offset));
    }
    return result;
  }

  // Stores `tile` to position {major: `major_dim_offset`, minor:
  // `minor_dim_offset`}.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  void StoreTile(absl::Span<llvm::Value* const> tile,
                 llvm::Value* minor_dim_offset) const {
    CHECK_EQ(tile.size(), pointers_.size());
    for (int64 i = 0; i < pointers_.size(); i++) {
      vsl_->StoreVector(tile[i], pointers_[i], minor_dim_offset);
    }
  }

  // Loads a tile of size [`tile_size_along_major_dim`,
  // `tile_size_along_middle_dim`] from position {major: `major_dim_offset`,
  // minor: `minor_dim_offset`} and then broadcasts each element into a vector
  // of size vsl_.vector_size().  The (i,j)'th element of the return value is
  // the (i,j)'th element in the tile broadcasted into an LLVM vector.
  //
  // Note: `major_dim_offset` is a parameter to the constructor.
  std::vector<std::vector<llvm::Value*>> LoadBroadcastTile(
      llvm::Value* minor_dim_offset, int64 tile_size_along_middle_dim) const {
    std::vector<std::vector<llvm::Value*>> result;
    result.resize(pointers_.size());
    for (int64 i = 0; i < pointers_.size(); i++) {
      for (int64 j = 0; j < tile_size_along_middle_dim; j++) {
        result[i].push_back(vsl_->LoadBroadcast(
            pointers_[i], b_->CreateAdd(minor_dim_offset, b_->getInt64(j))));
      }
    }
    return result;
  }

 private:
  VectorSupportLibrary* vsl_;
  llvm::IRBuilder<>* b_;
  std::vector<llvm::Value*> pointers_;
};

// The base class for the classes representing the GEMV emitter configurations.
//
// The IR emitted (modulo the LLVM values representing the input and output
// buffers) by the row major and column major GEMV emitters should be a function
// of their configuration.  This is important because their configuration is
// used as a key to cache the generated IR.
class GemvConfig {
 public:
  // Mixin for convenience.
  template <typename T>
  struct User {
   public:
    PrimitiveType scalar_type() const {
      return derived().config().scalar_type();
    }
    int64 tile_rows() const { return derived().config().tile_rows(); }
    int64 tile_cols() const { return derived().config().tile_cols(); }
    int64 m() const { return derived().config().m(); }
    int64 k() const { return derived().config().k(); }
    int64 has_addend() const { return derived().config().has_addend(); }

   private:
    const T& derived() const { return *static_cast<const T*>(this); }
  };

  PrimitiveType scalar_type() const { return scalar_type_; }
  int64 tile_rows() const { return tile_rows_; }
  int64 tile_cols() const { return tile_cols_; }
  int64 m() const { return m_; }
  int64 k() const { return k_; }
  bool has_addend() const { return has_addend_; }

  string GetCacheKey() const {
    return absl::StrCat(name_, "_", PrimitiveType_Name(scalar_type()), "_",
                        tile_rows(), "_", tile_cols(), "_", m(), "_", k(),
                        has_addend() ? "_with_addend" : "");
  }

 protected:
  explicit GemvConfig(string name, PrimitiveType scalar_type, int64 tile_rows,
                      int64 tile_cols, int64 m, int64 k, bool has_addend)
      : name_(std::move(name)),
        scalar_type_(scalar_type),
        tile_rows_(tile_rows),
        tile_cols_(tile_cols),
        m_(m),
        k_(k),
        has_addend_(has_addend) {}

 private:
  string name_;
  PrimitiveType scalar_type_;
  int64 tile_rows_;
  int64 tile_cols_;
  int64 m_;
  int64 k_;
  bool has_addend_;
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
class ColumnMajorMatrixVectorProductEmitter
    : public GemvConfig::User<ColumnMajorMatrixVectorProductEmitter> {
 public:
  class Config : public GemvConfig {
   public:
    explicit Config(PrimitiveType scalar_type, int64 tile_rows, int64 tile_cols,
                    int64 m, int64 k, bool has_addend)
        : GemvConfig(/*name=*/"col_major_gemv", scalar_type,
                     /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols, /*m=*/m,
                     /*k=*/k, /*has_addend=*/has_addend) {}
  };

  ColumnMajorMatrixVectorProductEmitter(const Config& config, llvm::Value* lhs,
                                        llvm::Value* rhs, llvm::Value* addend,
                                        llvm::Value* result,
                                        llvm::IRBuilder<>* b)
      : config_(config),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        b_(b),
        ksl_(b_),
        vsl_(config.scalar_type(), /*vector_size=*/config.tile_rows(), b_, "") {
    CHECK(tile_rows() > 0 && IsPowerOfTwo(static_cast<uint64>(tile_rows())));
    CHECK(!has_addend() || addend != nullptr);
  }

  void Emit();

  const Config& config() const { return config_; }

 private:
  void EmitOuterLoopBody(llvm::Value* column, int64 column_count,
                         bool is_first_column);

  MemoryTile GetLhsMemoryTile(llvm::Value* column_start, int64 column_count) {
    return MemoryTile(&vsl_, b_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/m(),
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

  void EmitInnerLoopTiled(MemoryTile* lhs_memory_tile,
                          const std::vector<llvm::Value*>& rhs_tile,
                          int64 columns, bool is_first_column);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_col, int64 columns,
                             bool is_first_tiled_column);

  Config config_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void ColumnMajorMatrixVectorProductEmitter::EmitOuterLoopBody(
    llvm::Value* column, int64 column_count, bool is_first_column) {
  MemoryTile lhs_memory_tile = GetLhsMemoryTile(/*column_start=*/column,
                                                /*column_count=*/column_count);

  std::vector<llvm::Value*> rhs_tile =
      LoadRhsTile(column, /*count=*/column_count);
  EmitInnerLoopTiled(&lhs_memory_tile, rhs_tile,
                     /*columns=*/column_count, is_first_column);
  EmitInnerLoopEpilogue(column, /*columns=*/column_count, is_first_column);
}

void ColumnMajorMatrixVectorProductEmitter::Emit() {
  // See the comment on the class declaration for the algorithm used here.
  int64 column_remainder = k() % tile_cols();
  int64 column_limit = k() - column_remainder;

  ksl_.ForReturnVoid("dot.outer.tiled",
                     /*start=*/0, /*end=*/column_limit, /*step=*/tile_cols(),
                     [&](llvm::Value* column, bool is_first_column) {
                       EmitOuterLoopBody(column, tile_cols(), is_first_column);
                     });

  if (column_remainder != 0) {
    EmitOuterLoopBody(b_->getInt64(column_limit), column_remainder,
                      column_limit == 0);
  }
}

void ColumnMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    MemoryTile* lhs_memory_tile, const std::vector<llvm::Value*>& rhs_tile,
    int64 columns, bool is_first_column) {
  int64 row_limit = m() - (m() % tile_rows());

  ksl_.ForReturnVoid(
      "dot.inner.tiled", /*start=*/0, /*end=*/row_limit,
      /*step=*/tile_rows(), [&](llvm::Value* row) {
        std::vector<llvm::Value*> lhs_tile =
            lhs_memory_tile->LoadTile(/*minor_dim_offset=*/row);
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
  int64 row_start = m() - (m() % tile_rows());
  if (row_start == m()) {
    return;
  }

  llvm::Value* columns_llvm = b_->getInt64(columns);

  // for (col = current_tile_col; col < (columns + current_tile_col); col++)
  //   for (row = row_start, row < m_; row++) {
  //     result[row] += lhs[row, col] * rhs[col]
  //     // Also take into account that if col is 0 then result[row] is not
  //     // initialized.
  //   }

  ksl_.ForReturnVoid(
      "dot.inner.epilg.outer", /*start=*/current_tile_col,
      /*end=*/b_->CreateAdd(columns_llvm, current_tile_col),
      /*step=*/1, /*peel_first_iteration=*/false,
      [&](llvm::Value* col, llvm::Value* is_first_scalar_col) {
        llvm::Value* rhs_element = vsl_.LoadScalar(rhs_, col);
        llvm::Value* total_offset = b_->CreateMul(col, b_->getInt64(m()));
        llvm::Value* lhs_base_pointer =
            vsl_.ComputeOffsetPointer(lhs_, total_offset);
        ksl_.ForReturnVoid(
            "dot.inner.epilg.inner", /*start=*/row_start, /*end=*/m(),
            /*step=*/1, [&](llvm::Value* scalar_row) {
              llvm::Value* product = vsl_.Mul(
                  vsl_.LoadScalar(lhs_base_pointer, scalar_row), rhs_element);
              llvm::Value* setting_result_first_time = b_->CreateAnd(
                  is_first_scalar_col, b_->getInt1(is_first_tiled_column));
              ksl_.IfReturnVoid(
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
class RowMajorMatrixVectorProductEmitter
    : public GemvConfig::User<RowMajorMatrixVectorProductEmitter> {
 public:
  class Config : public GemvConfig {
   public:
    explicit Config(PrimitiveType scalar_type, int64 tile_rows, int64 tile_cols,
                    int64 m, int64 k, bool has_addend)
        : GemvConfig(/*name=*/"row_major_gemv", scalar_type,
                     /*tile_rows=*/tile_rows, /*tile_cols=*/tile_cols, /*m=*/m,
                     /*k=*/k, /*has_addend=*/has_addend) {}
  };

  RowMajorMatrixVectorProductEmitter(const Config& config, llvm::Value* lhs,
                                     llvm::Value* rhs, llvm::Value* addend,
                                     llvm::Value* result, llvm::IRBuilder<>* b)
      : config_(config),
        lhs_(lhs),
        rhs_(rhs),
        addend_(addend),
        result_(result),
        b_(b),
        ksl_(b_),
        vsl_(scalar_type(), /*vector_size=*/tile_cols(), b_, "") {
    CHECK(tile_cols() > 0 && IsPowerOfTwo(static_cast<uint64>(tile_cols())));
    CHECK(!has_addend() || addend != nullptr);
  }

  void Emit();

  const Config& config() const { return config_; }

 private:
  MemoryTile GetLhsMemoryTile(llvm::Value* row_start, int64 row_count) {
    return MemoryTile(&vsl_, b_, /*matrix=*/lhs_,
                      /*matrix_size_along_minor_dim=*/k(),
                      /*major_dim_offset=*/row_start,
                      /*tile_size_along_major_dim=*/row_count);
  }

  void EmitOuterLoopBody(llvm::Value* row, int64 row_count);

  void EmitInnerLoopTiled(MemoryTile* lhs_memory_tile, int64 rows,
                          std::vector<VectorVariable>* vector_accumulators);

  void EmitInnerLoopEpilogue(llvm::Value* current_tile_row, int64 rows,
                             std::vector<ScalarVariable>* scalar_accumulators);

  Config config_;
  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* addend_;
  llvm::Value* result_;
  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
  VectorSupportLibrary vsl_;
};

void RowMajorMatrixVectorProductEmitter::EmitOuterLoopBody(llvm::Value* row,
                                                           int64 row_count) {
  MemoryTile lhs_memory_tile = GetLhsMemoryTile(/*row_start=*/row,
                                                /*row_count=*/row_count);
  std::vector<VectorVariable> vector_accumulators;
  std::vector<ScalarVariable> scalar_accumulators;
  for (int i = 0; i < row_count; i++) {
    vector_accumulators.emplace_back(&vsl_, vsl_.GetZeroVector());
    scalar_accumulators.emplace_back(&vsl_, vsl_.GetZeroScalar());
  }
  EmitInnerLoopTiled(&lhs_memory_tile, /*rows=*/row_count,
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
    llvm::Value* offset = b_->CreateAdd(b_->getInt64(i), row);
    if (addend_ && row_count != vsl_.vector_size()) {
      result_value = vsl_.Add(vsl_.LoadScalar(addend_, offset), result_value);
    }
    vsl_.StoreScalar(result_value, result_, offset);
  }
}

void RowMajorMatrixVectorProductEmitter::Emit() {
  // See the comment on the class declaration for the algorithm used here.
  int64 row_remainder = m() % tile_rows();
  int64 row_limit = m() - row_remainder;

  ksl_.ForReturnVoid(
      "dot.outer.tiled",
      /*start=*/0, /*end=*/row_limit, /*step=*/tile_rows(),
      [&](llvm::Value* row) { EmitOuterLoopBody(row, tile_rows()); });

  if (row_remainder != 0) {
    EmitOuterLoopBody(b_->getInt64(row_limit), row_remainder);
  }
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopTiled(
    MemoryTile* lhs_memory_tile, int64 rows,
    std::vector<VectorVariable>* vector_accumulators) {
  int64 column_limit = k() - (k() % tile_cols());

  ksl_.ForReturnVoid("dot.inner.tiled", /*start=*/0, /*end=*/column_limit,
                     /*step=*/tile_cols(), [&](llvm::Value* col) {
                       std::vector<llvm::Value*> lhs_tile =
                           lhs_memory_tile->LoadTile(/*minor_dim_offset=*/col);
                       llvm::Value* rhs_value = vsl_.LoadVector(rhs_, col);
                       for (int i = 0; i < rows; i++) {
                         llvm::Value* old_sum = (*vector_accumulators)[i].Get();
                         (*vector_accumulators)[i].Set(vsl_.Add(
                             old_sum, vsl_.Mul(rhs_value, lhs_tile[i])));
                       }
                     });
}

void RowMajorMatrixVectorProductEmitter::EmitInnerLoopEpilogue(
    llvm::Value* current_tile_row, int64 rows,
    std::vector<ScalarVariable>* scalar_accumulators) {
  int64 column_start = k() - (k() % tile_cols());
  if (column_start == k()) {
    return;
  }

  for (int r = 0; r < rows; r++) {
    llvm::Value* total_offset = b_->CreateMul(
        b_->CreateAdd(b_->getInt64(r), current_tile_row), b_->getInt64(k()));
    llvm::Value* lhs_base_pointer =
        vsl_.ComputeOffsetPointer(lhs_, total_offset);
    ksl_.ForReturnVoid(
        "dot.inner.epilg.inner", /*start=*/column_start, /*end=*/k(),
        /*step=*/1, [&](llvm::Value* scalar_col) {
          llvm::Value* product =
              vsl_.Mul(vsl_.LoadScalar(lhs_base_pointer, scalar_col),
                       vsl_.LoadScalar(rhs_, scalar_col));
          llvm::Value* old_value = (*scalar_accumulators)[r].Get();
          (*scalar_accumulators)[r].Set(vsl_.Add(old_value, product));
        });
  }
}

// This class implements a tiled matrix multiplication algorithm, intended for
// multiplying small matrices that don't need cache tiling.
//
// In the future this can be used as the innermost GEBP loop in a GEMM kernel as
// described in "Goto, Kazushige, and Robert A. Geijn. "Anatomy of
// high-performance matrix multiplication." ACM Transactions on Mathematical
// Software (TOMS) 34.3 (2008): 12.".
//
// This only supports canonical dot operations (i.e. where the lhs contraction
// dimension is 1 and the rhs contraction dimension is 0) over row major
// matrices.
class TiledSmallGemmEmitter {
 public:
  // Describe the dimensions of the kernel.
  class Dimensions {
   public:
    explicit Dimensions(int64 m, int64 k, int64 n) : m_(m), k_(k), n_(n) {}

    int64 m() const { return m_; }
    int64 k() const { return k_; }
    int64 n() const { return n_; }

    string ToString() const { return absl::StrCat(m(), "x", k(), "x", n()); }

   private:
    const int64 m_;
    const int64 k_;
    const int64 n_;
  };

  // Represents the configuration of the emitter.  The LLVM IR emitted by the
  // emitter, modulo the LLVM values holding the input and output buffers, must
  // be a function of the instance of `Config` passed to it.
  //
  // `dims` holds the matrix multiplication dimensions.
  //
  // `max_vectorization_width` is the maximum vector width (i.e. the width of
  // the largest vector register we will use).  This can be larger than the
  // largest vector register supported by the machine -- LLVM will legalize
  // these large vector widths into legally sized vectors.
  //
  // `max_vector_count` is the maximum number of vectors of size
  // `max_vectorization_width` that we will attempt to process at once.
  //
  // `min_vectorization_width` is the smallest vector width the emitter will use
  // -- below that it will devolve to using a scalar loop.
  //
  // The innermost reduction loop executes the matrix multiply in tiles of size
  // [`tile_size_m`, `tile_size_k`] from the LHS and [`tile_size_k`,
  // <vectorization width>] in the RHS.
  class Config {
   public:
    explicit Config(PrimitiveType scalar_type, Dimensions dims,
                    int64 max_vectorization_width, int64 max_vector_count,
                    int64 min_vectorization_width, int64 tile_size_m,
                    int64 tile_size_k)
        : scalar_type_(scalar_type),
          dims_(dims),
          max_vectorization_width_(max_vectorization_width),
          max_vector_count_(max_vector_count),
          min_vectorization_width_(min_vectorization_width),
          tile_size_m_(tile_size_m),
          tile_size_k_(tile_size_k) {}

    string GetCacheKey() const {
      return absl::StrCat("gemm_", PrimitiveType_Name(scalar_type()), "_",
                          dims().ToString(), "_", max_vectorization_width(),
                          "_", min_vectorization_width(), "_", tile_size_m(),
                          "_", tile_size_k());
    }

    PrimitiveType scalar_type() const { return scalar_type_; }
    Dimensions dims() const { return dims_; }
    int64 max_vectorization_width() const { return max_vectorization_width_; }
    int64 max_vector_count() const { return max_vector_count_; }
    int64 min_vectorization_width() const { return min_vectorization_width_; }

    int64 tile_size_m() const { return tile_size_m_; }
    int64 tile_size_k() const { return tile_size_k_; }

   private:
    PrimitiveType scalar_type_;
    Dimensions dims_;
    int64 max_vectorization_width_;
    int64 max_vector_count_;
    int64 min_vectorization_width_;
    int64 tile_size_m_;
    int64 tile_size_k_;
  };

  // Creates an instance of TiledSmallGemmEmitter that matrix-multiplies
  // `lhs` with `rhs` and stores the result in `result`.
  explicit TiledSmallGemmEmitter(Config config, llvm::Value* lhs,
                                 llvm::Value* rhs, llvm::Value* result,
                                 llvm::IRBuilder<>* b)
      : lhs_(lhs),
        rhs_(rhs),
        result_(result),
        config_(config),
        b_(b),
        ksl_(b_) {
    CHECK(max_vectorization_width() > 0 &&
          IsPowerOfTwo(static_cast<uint64>(max_vectorization_width())));
    CHECK_GT(max_vector_count(), 0);
    CHECK(min_vectorization_width() > 0 &&
          IsPowerOfTwo(static_cast<uint64>(min_vectorization_width())));
    CHECK_GE(max_vectorization_width(), min_vectorization_width());
    CHECK_GT(tile_size_k(), 0);
  }

  void Emit();

 private:
  // The HandleResiduesOnX helpers split the iteration space for dimension X
  // into a multiple of the tile size on dimension X and an epilogue.  These
  // helpers ultimately call into `EmitTiledGemm` for emitting the
  // tiled GEMM kernel.

  void HandleResiduesOnN();
  void HandleResiduesOnK(VectorSupportLibrary* vsl, llvm::Value* n_start,
                         llvm::Value* n_end);
  void HandleResiduesOnM(VectorSupportLibrary* vsl, int64 tile_size_k,
                         llvm::Value* k_start, llvm::Value* k_end,
                         llvm::Value* n_start, llvm::Value* n_end);

  // This emits a tiled GEMM kernel.  For a detailed description see the comment
  // on the implementation.
  void EmitTiledGemm(VectorSupportLibrary* vsl, int64 tile_size_k,
                     llvm::Value* k_start, llvm::Value* k_end,
                     llvm::Value* n_start, llvm::Value* n_end,
                     int64 tile_size_m, llvm::Value* m_start,
                     llvm::Value* m_end);

  llvm::Value* GetInt64(int64 value) { return b_->getInt64(value); }

  Config config() const { return config_; }
  Dimensions dims() const { return config().dims(); }

  int64 max_vectorization_width() const {
    return config().max_vectorization_width();
  }
  int64 max_vector_count() const { return config().max_vector_count(); }
  int64 min_vectorization_width() const {
    return config().min_vectorization_width();
  }
  int64 tile_size_m() const { return config().tile_size_m(); }
  int64 tile_size_k() const { return config().tile_size_k(); }
  PrimitiveType scalar_type() const { return config().scalar_type(); }

  llvm::Value* lhs_;
  llvm::Value* rhs_;
  llvm::Value* result_;
  Config config_;

  llvm::IRBuilder<>* b_;
  KernelSupportLibrary ksl_;
};

void TiledSmallGemmEmitter::Emit() { HandleResiduesOnN(); }

void TiledSmallGemmEmitter::HandleResiduesOnN() {
  // We can only iterate the `n` dimension for an extent that is divisible by
  // the vectorization width.  So we emit an outer loop that first processes the
  // largest extent in `n` that is divisible by max_vectorization_width, then
  // the largest remaining extent that is divisible by max_vectorization_width /
  // 2 etc.

  int64 current_vectorization_width =
      max_vector_count() * max_vectorization_width();
  int64 current_vector_count = max_vector_count();

  int64 n_start = 0;
  while (n_start != dims().n() &&
         current_vectorization_width >= min_vectorization_width()) {
    int64 n_end = dims().n() - (dims().n() % current_vectorization_width);
    if (n_start != n_end) {
      VectorSupportLibrary vsl(scalar_type(), current_vectorization_width, b_,
                               "gemm");
      HandleResiduesOnK(&vsl, GetInt64(n_start), GetInt64(n_end));
      n_start = n_end;
    }
    if (current_vector_count == 1) {
      current_vectorization_width /= 2;
    } else {
      current_vector_count--;
      current_vectorization_width =
          current_vector_count * max_vectorization_width();
    }
  }

  if (n_start != dims().n()) {
    VectorSupportLibrary vsl(scalar_type(), 1, b_, "gemm");
    ksl_.ForReturnVoid("epi.n", n_start, dims().n(), 1, [&](llvm::Value* n_i) {
      llvm::Value* n_i_next = b_->CreateAdd(n_i, b_->getInt64(1));
      HandleResiduesOnK(&vsl, n_i, n_i_next);
    });
  }
}

void TiledSmallGemmEmitter::HandleResiduesOnK(VectorSupportLibrary* vsl,
                                              llvm::Value* n_start,
                                              llvm::Value* n_end) {
  int64 k_start = 0;
  int64 k_end = dims().k() - (dims().k() % tile_size_k());
  if (k_end != k_start) {
    HandleResiduesOnM(vsl, tile_size_k(), GetInt64(k_start), GetInt64(k_end),
                      n_start, n_end);
    k_start = k_end;
  }

  if (k_start != dims().k()) {
    HandleResiduesOnM(vsl, dims().k() - k_start, GetInt64(k_start),
                      GetInt64(dims().k()), n_start, n_end);
  }
}

void TiledSmallGemmEmitter::HandleResiduesOnM(
    VectorSupportLibrary* vsl, int64 tile_size_k, llvm::Value* k_start,
    llvm::Value* k_end, llvm::Value* n_start, llvm::Value* n_end) {
  const int64 m_end = dims().m() - dims().m() % tile_size_m();
  EmitTiledGemm(vsl, tile_size_k, k_start, k_end, n_start, n_end, tile_size_m(),
                GetInt64(0), GetInt64(m_end));

  if (m_end != dims().m()) {
    EmitTiledGemm(vsl, tile_size_k, k_start, k_end, n_start, n_end,
                  dims().m() - m_end, GetInt64(m_end), GetInt64(dims().m()));
  }
}

// The loop structure is:
//
// Iterate over dimension M as m:
//   Iterate over dimension N as n:
//     Iterate over dimension K as k:
//       OutputTile[m,n] += Dot(LhsTile[m,k], RhsTile[k,n])
//
// I.e. a just a tiled version of a "naive" GEMM.
//
// The tiling scheme is as follows:
//
// Let the LHS be:
//
//   +----+----+----+
//   | a0 | b0 | c0 | .
//   +----+----+----+ .
//   | a1 | b1 | c1 | .
//   +----+----+----+
//     ..     ..
//
// and the RHS be:
//
//   +----+----+----+----+
//   | p0 | p1 | p2 | p3 | .
//   +----+----+----+----+ .
//   | q0 | q1 | q2 | q3 | .
//   +----+----+----+----+
//   | r0 | r1 | r2 | r3 | .
//   +----+----+----+----+ .
//     ......    ......
//
// and let tile_size_m=2, tile_size_k=3 and the vector width (implicitly denoted
// by `vsl`) be 4.  Then we want to matrix multiply this tile to get a [2,4]
// matrix that we can increment the result matrix by.
//
// First broadcast the rows row in LHS to 3 vectors of width 4, giving us a rank
// 3 array, L, of dimension [2,3,4]:
//
//       L[0,_,_]           *      L[1,_,_]
//                          *
//   +----+----+----+----+  *  +----+----+----+----+
//   | a0 | a0 | a0 | a0 |  *  | a1 | a1 | a1 | a1 |
//   +----+----+----+----+  *  +----+----+----+----+
//   | b0 | b0 | b0 | b0 |  *  | b1 | b1 | b1 | b1 |
//   +----+----+----+----+  *  +----+----+----+----+
//   | c0 | c0 | c0 | c0 |  *  | c1 | c1 | c1 | c1 |
//   +----+----+----+----+  *  +----+----+----+----+
//
//
// Then we FMA L[0,_,_] with the RHS to get the first row of the result and
// L[1,_,_] with the RHS to get the second row of the result.  For example,
// L[0,_,_] is computed as:
//
//   +----+----+----+----+   +----+----+----+----+
//   | a0 | a0 | a0 | a0 | * | p0 | p1 | p2 | p3 |   +
//   +----+----+----+----+   +----+----+----+----+
//
//   +----+----+----+----+   +----+----+----+----+
//   | b0 | b0 | b0 | b0 | * | q0 | q1 | q2 | q3 |   +
//   +----+----+----+----+   +----+----+----+----+
//
//   +----+----+----+----+   +----+----+----+----+
//   | c0 | c0 | c0 | c0 | * | r0 | r1 | r2 | r3 |
//   +----+----+----+----+   +----+----+----+----+
//
// to get:
//
//   +-------------------+-------------------+-------------------+---------
//   | a0*p0+b0*q0+c0*r0 | a0*p1+b0*q1+c0*r1 | a0*p2+b0*q2+c0*r2 |  ...
//   +-------------------+-------------------+-------------------+---------
void TiledSmallGemmEmitter::EmitTiledGemm(
    VectorSupportLibrary* vsl, int64 tile_size_k, llvm::Value* k_start,
    llvm::Value* k_end, llvm::Value* n_start, llvm::Value* n_end,
    int64 tile_size_m, llvm::Value* m_start, llvm::Value* m_end) {
  ksl_.ForReturnVoid(
      "dot.m", m_start, m_end, tile_size_m, [&](llvm::Value* m_i) {
        MemoryTile result_memory_tile(
            vsl, b_, /*matrix=*/result_,
            /*matrix_size_along_minor_dim=*/dims().n(),
            /*major_dim_offset=*/m_i,
            /*tile_size_along_major_dim=*/tile_size_m);
        MemoryTile lhs_memory_tile(vsl, b_, /*matrix=*/lhs_,
                                   /*matrix_size_along_minor_dim=*/dims().k(),
                                   /*major_dim_offset=*/m_i,
                                   /*tile_size_along_major_dim=*/tile_size_m);
        ksl_.ForReturnVoid(
            "dot.n", n_start, n_end, vsl->vector_size(), [&](llvm::Value* n_i) {
              TileVariable result_tile_var(vsl,
                                           result_memory_tile.LoadTile(n_i));
              ksl_.ForReturnVoid(
                  "dot.k", k_start, k_end, tile_size_k, [&](llvm::Value* k_i) {
                    MemoryTile rhs_memory_tile(vsl, b_, rhs_, dims().n(), k_i,
                                               tile_size_k);
                    std::vector<std::vector<llvm::Value*>> lhs_tile =
                        lhs_memory_tile.LoadBroadcastTile(k_i, tile_size_k);
                    std::vector<llvm::Value*> rhs_tile =
                        rhs_memory_tile.LoadTile(n_i);
                    std::vector<llvm::Value*> result_tile =
                        result_tile_var.Get();
                    for (int64 r_m_i = 0; r_m_i < tile_size_m; r_m_i++) {
                      for (int64 r_k_i = 0; r_k_i < tile_size_k; r_k_i++) {
                        result_tile[r_m_i] =
                            vsl->MulAdd(lhs_tile[r_m_i][r_k_i], rhs_tile[r_k_i],
                                        result_tile[r_m_i]);
                      }
                    }
                    result_tile_var.Set(result_tile);
                  });

              result_memory_tile.StoreTile(result_tile_var.Get(), n_i);
            });
      });
}

}  // namespace

DotOpEmitter::DotOpEmitter(const HloInstruction& dot,
                           const llvm_ir::IrArray& target_array,
                           const llvm_ir::IrArray& lhs_array,
                           const llvm_ir::IrArray& rhs_array,
                           const llvm_ir::IrArray* addend_array,
                           llvm::Value* executable_run_options_value,
                           llvm::IRBuilder<>* b,
                           const HloModuleConfig& hlo_module_config,
                           const TargetMachineFeatures& target_machine_features)
    : dot_(dot),
      target_array_(target_array),
      lhs_array_(lhs_array),
      rhs_array_(rhs_array),
      addend_array_(addend_array),
      executable_run_options_value_(executable_run_options_value),
      b_(b),
      hlo_module_config_(hlo_module_config),
      target_machine_features_(target_machine_features) {}

/* static */ Status DotOpEmitter::EmitDotOperation(
    const HloInstruction& dot, const llvm_ir::IrArray& target_array,
    const llvm_ir::IrArray& lhs_array, const llvm_ir::IrArray& rhs_array,
    const llvm_ir::IrArray* addend_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* b,
    const HloModuleConfig& hlo_module_config,
    const TargetMachineFeatures& target_machine_features) {
  PrimitiveType type = target_array.GetShape().element_type();
  TF_RET_CHECK(F16 == type || F32 == type || F64 == type || C64 == type);
  DotOpEmitter dot_emitter(dot, target_array, lhs_array, rhs_array,
                           addend_array, executable_run_options_value, b,
                           hlo_module_config, target_machine_features);
  return dot_emitter.Emit();
}

bool DotOpEmitter::EmitSmallGemmIfProfitable(
    const DotOpEmitter::MatMultDims& mat_mult_dims) {
  if (ShouldUseMultiThreadedEigen()) {
    return false;
  }

  if (!EnableExperimentalLlvmIrGemm()) {
    // TODO(sanjoy):  We should make these numbers micro-arch specific.
    bool small_gemm = mat_mult_dims.k <= 128 &&
                      ((mat_mult_dims.m <= 32 && mat_mult_dims.n <= 128) ||
                       (mat_mult_dims.m <= 128 && mat_mult_dims.n <= 32));
    if (!small_gemm) {
      return false;
    }
  }

  if (mat_mult_dims.lhs_non_canonical || mat_mult_dims.rhs_non_canonical) {
    return false;
  }

  PrimitiveType primitive_type = dot_.shape().element_type();

  switch (primitive_type) {
    default:
      return false;

    case F32:
    case F64:
    case S32:
    case S64:
      break;
  }

  if (!(mat_mult_dims.lhs_column_major == mat_mult_dims.rhs_column_major &&
        mat_mult_dims.rhs_column_major == mat_mult_dims.target_column_major)) {
    return false;
  }

  llvm::Value* lhs = lhs_array_.GetBasePointer();
  llvm::Value* rhs = rhs_array_.GetBasePointer();
  llvm::Value* target = target_array_.GetBasePointer();
  int64 m = mat_mult_dims.m;
  int64 k = mat_mult_dims.k;
  int64 n = mat_mult_dims.n;

  if (mat_mult_dims.lhs_column_major) {
    std::swap(lhs, rhs);
    std::swap(m, n);
  }

  int64 size_bytes = m * n * ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  b_->CreateMemSet(
      target, b_->getInt8(0), size_bytes,
      target_machine_features_.minimum_alignment_for_allocation(size_bytes));

  int64 max_target_vector_width =
      target_machine_features_.vector_register_num_elements(
          *b_->GetInsertBlock()->getParent(), primitive_type);

  int64 tile_size_m, tile_size_k, tile_size_n_in_vector_width;
  std::tie(tile_size_m, tile_size_k, tile_size_n_in_vector_width) =
      GetGemmTileSize();

  TiledSmallGemmEmitter::Config config(
      /*scalar_type=*/primitive_type,
      TiledSmallGemmEmitter::Dimensions{/*m=*/m, /*k=*/k, /*n=*/n},
      /*max_vectorization_width=*/max_target_vector_width,
      /*max_vector_count=*/tile_size_n_in_vector_width,
      /*min_vectorization_width=*/std::min<int64>(4, max_target_vector_width),
      /*tile_size_m=*/tile_size_m, /*tile_size_k=*/tile_size_k);

  VLOG(2) << "Emitting GEMM kernel in LLVM IR with config "
          << config.GetCacheKey();

  const bool enable_fast_math =
      hlo_module_config_.debug_options().xla_cpu_enable_fast_math();
  const bool optimize_for_size =
      options::OptimizeForSizeRequested(hlo_module_config_);

  KernelSupportLibrary::EmitAndCallOutlinedKernel(
      /*enable_fast_math=*/enable_fast_math,
      /*optimize_for_size=*/optimize_for_size, b_, config.GetCacheKey(), lhs,
      rhs, target,
      [this, config](llvm::Value* lhs, llvm::Value* rhs, llvm::Value* target) {
        TiledSmallGemmEmitter small_gemm_emitter(config, /*lhs=*/lhs,
                                                 /*rhs=*/rhs,
                                                 /*result=*/target, b_);
        small_gemm_emitter.Emit();
      });

  return true;
}

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
        mat_mult_dims.rhs_non_canonical ^ !mat_mult_dims.rhs_column_major;
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
        mat_mult_dims.lhs_non_canonical ^ mat_mult_dims.lhs_column_major;
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
    return EmitSmallGemmIfProfitable(mat_mult_dims);
  }

  int64 tiling_factor = GetGemvTilingFactor();
  CHECK_GT(tiling_factor, 0);

  llvm::Value* result_op = target_array_.GetBasePointer();
  llvm::Value* lhs_op =
      swap_operands ? rhs_array_.GetBasePointer() : lhs_array_.GetBasePointer();
  llvm::Value* rhs_op =
      swap_operands ? lhs_array_.GetBasePointer() : rhs_array_.GetBasePointer();

  const bool enable_fast_math =
      hlo_module_config_.debug_options().xla_cpu_enable_fast_math();
  const bool optimize_for_size =
      options::OptimizeForSizeRequested(hlo_module_config_);

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

  if (is_column_major_matrix_vector) {
    VLOG(2) << "Emitting column major matrix-vector multiply with m = " << m
            << " and k = " << k;
    ColumnMajorMatrixVectorProductEmitter::Config config(
        /*scalar_type=*/primitive_type,
        /*tile_rows=*/vector_register_element_size, /*tile_cols=*/tiling_factor,
        /*m=*/m, /*k=*/k, /*has_addend=*/addend_array_ != nullptr);

    KernelSupportLibrary::EmitAndCallOutlinedKernel(
        /*enable_fast_math=*/enable_fast_math,
        /*optimize_for_size=*/optimize_for_size, b_, config.GetCacheKey(),
        lhs_op, rhs_op,
        addend_array_ ? addend_array_->GetBasePointer() : nullptr, result_op,
        [this, config](llvm::Value* lhs_op, llvm::Value* rhs_op,
                       llvm::Value* addend_op, llvm::Value* result_op) {
          ColumnMajorMatrixVectorProductEmitter emitter(
              config, lhs_op, rhs_op, addend_op, result_op, b_);
          emitter.Emit();
        });
  } else {
    VLOG(2) << "Emitting row major matrix-vector multiply with m = " << m
            << " and k = " << k;
    RowMajorMatrixVectorProductEmitter::Config config(
        /*scalar_type=*/primitive_type,
        /*tile_rows=*/tiling_factor, /*tile_cols=*/vector_register_element_size,
        /*m=*/m, /*k=*/k, /*has_addend=*/addend_array_ != nullptr);

    KernelSupportLibrary::EmitAndCallOutlinedKernel(
        /*enable_fast_math=*/enable_fast_math,
        /*optimize_for_size=*/optimize_for_size, b_, config.GetCacheKey(),
        lhs_op, rhs_op,
        addend_array_ ? addend_array_->GetBasePointer() : nullptr, result_op,
        [this, config](llvm::Value* lhs_op, llvm::Value* rhs_op,
                       llvm::Value* addend_op, llvm::Value* result_op) {
          RowMajorMatrixVectorProductEmitter emitter(config, lhs_op, rhs_op,
                                                     addend_op, result_op, b_);
          emitter.Emit();
        });
  }

  return true;
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

  // This routine assumes that the dot operation is not in a parallelized
  // enclosing computation.
  CHECK(
      dot_.parent()->root_instruction()->outer_dimension_partitions().empty());

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

  if (PotentiallyImplementedAsEigenDot(dot_, target_machine_features_)) {
    return EmitCallToRuntime();
  }

  // Reduce along dimension 0 of the LHS and 1 of the RHS. Vectors are a special
  // case where the reduction dimension is 0 for both LHS and RHS. This results
  // in a vector dot product producing a scalar.
  int64 lhs_reduction_dimension =
      dot_.dot_dimension_numbers().lhs_contracting_dimensions(0);
  int64 rhs_reduction_dimension =
      dot_.dot_dimension_numbers().rhs_contracting_dimensions(0);

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
  llvm_ir::ForLoopNest loop_nest(llvm_ir::IrName(&dot_), b_);
  llvm_ir::IrArray::Index lhs_index = loop_nest.EmitOperandArrayLoopNest(
      lhs_array_, /*dimension_to_skip=*/lhs_reduction_dimension, "lhs");
  llvm_ir::IrArray::Index rhs_index = loop_nest.EmitOperandArrayLoopNest(
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
  lhs_index[lhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  rhs_index[rhs_reduction_dimension] = reduction_loop->GetIndVarValue();

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

  llvm::Value* accum = b_->CreateLoad(accum_address);
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
  } else {
    llvm::Value* product = b_->CreateFMul(lhs_element, rhs_element);
    updated_accum = b_->CreateFAdd(accum, product);
  }
  b_->CreateStore(updated_accum, accum_address);

  // Exit basic block of reduction loop.
  // - Load accumulator value (the result).
  // - Store into output array.
  SetToFirstInsertPoint(reduction_loop->GetExitBasicBlock(), b_);

  llvm::Value* result = b_->CreateLoad(accum_address);

  // Create index into target address. The target index is the concatenation of
  // the rhs and lhs indexes with the reduction dimensions removed. The terms
  // from the rhs index are the lower dimensions in the index so we add them
  // first.
  llvm_ir::IrArray::Index target_index(lhs_index.GetType());
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

  target_array_.EmitWriteArrayElement(target_index, result, b_);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop.
  b_->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return Status::OK();
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
#define REAL(x) b_->CreateExtractValue(x, {0})
#define IMAG(x) b_->CreateExtractValue(x, {1})
    llvm::Value* real =
        b_->CreateFSub(b_->CreateFMul(REAL(lhs_value), REAL(rhs_value)),
                       b_->CreateFMul(IMAG(lhs_value), IMAG(rhs_value)));
    llvm::Value* imag =
        b_->CreateFAdd(b_->CreateFMul(REAL(lhs_value), IMAG(rhs_value)),
                       b_->CreateFMul(IMAG(lhs_value), REAL(rhs_value)));
#undef IMAG
#undef REAL
    result = llvm::ConstantAggregateZero::get(lhs_array_.GetElementLlvmType());
    result = b_->CreateInsertValue(result, real, {0});
    result = b_->CreateInsertValue(result, imag, {1});
  } else {
    result = b_->CreateFMul(lhs_value, rhs_value);
  }
  target_array_.EmitWriteArrayElement(/*index=*/element_index, result, b_);
  return Status::OK();
}

Status DotOpEmitter::EmitCallToRuntime() {
  // The signature of the Eigen runtime matmul function is:
  //
  //   (void)(void* run_options, float* out, float* lhs, float* rhs,
  //          int64 m, int64 n, int64 k, int32 transpose_lhs,
  //          int32 transpose_rhs);
  // The two transpose_... parameters are actually booleans, but we use int32
  // to avoid target-dependent calling convention details.

  bool multi_threaded = ShouldUseMultiThreadedEigen();
  bool use_mkl_dnn = hlo_module_config_.debug_options().xla_cpu_use_mkl_dnn();
  PrimitiveType type = target_array_.GetShape().element_type();
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
      fn_name = multi_threaded
                    ? (use_mkl_dnn ? runtime::kMKLMatMulF32SymbolName
                                   : runtime::kEigenMatMulF32SymbolName)
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

  llvm::Function* function = b_->GetInsertBlock()->getParent();
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
  bool transpose_lhs = mat_mult_dims.lhs_non_canonical;
  bool transpose_rhs = mat_mult_dims.rhs_non_canonical;

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
  return Status::OK();
}

DotOpEmitter::MatMultDims DotOpEmitter::GetMatMultDims() const {
  CHECK_EQ(dot_.shape().dimensions_size(), 2);

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  const DotDimensionNumbers& dim_nums = dot_.dot_dimension_numbers();

  return {
      /*m=*/lhs_shape.dimensions(1 - dim_nums.lhs_contracting_dimensions(0)),
      /*k=*/lhs_shape.dimensions(dim_nums.lhs_contracting_dimensions(0)),
      /*n=*/rhs_shape.dimensions(1 - dim_nums.rhs_contracting_dimensions(0)),
      /*lhs_column_major=*/LayoutUtil::Minor(lhs_shape.layout(), 0) == 0,
      /*lhs_non_canonical=*/dim_nums.lhs_contracting_dimensions(0) == 0,
      /*rhs_column_major=*/LayoutUtil::Minor(rhs_shape.layout(), 0) == 0,
      /*rhs_non_canonical=*/dim_nums.rhs_contracting_dimensions(0) == 1,
      /*target_column_major=*/
      LayoutUtil::Minor(target_array_.GetShape().layout(), 0) == 0};
}

// Return whether the given shape is rank 2.
static bool IsRank2(const Shape& shape) { return ShapeUtil::Rank(shape) == 2; }

// In a gemm operation where output = lhs * rhs, check whether the given shapes
// are valid for the operation.
static bool AreValidGemmShapes(
    const Shape& lhs_shape, const Shape& rhs_shape, const Shape& output_shape,
    const TargetMachineFeatures& target_machine_features) {
  // The inputs and the output must
  // 1) be matrices with no padding, and
  // 2) have an allowed element type.
  PrimitiveType output_primitive_type = output_shape.element_type();
  if (!(output_primitive_type == F64 || output_primitive_type == F32 ||
        output_primitive_type == F16)) {
    return false;
  }

  if (!(IsRank2(lhs_shape) && IsRank2(rhs_shape) && IsRank2(output_shape))) {
    return false;
  }

  auto is_aligned = [&](const Shape& shape) {
    return GetMinimumAlignmentForArray(shape, target_machine_features) >=
           TargetMachineFeatures::kEigenExpectedTensorAlignment;
  };

  if (!is_aligned(lhs_shape) || !is_aligned(rhs_shape) ||
      !is_aligned(output_shape)) {
    return false;
  }

  return true;
}

bool PotentiallyImplementedAsEigenDot(
    const HloInstruction& hlo,
    const TargetMachineFeatures& target_machine_features) {
  // For certain types of Dot, we can call Eigen
  if (hlo.opcode() == HloOpcode::kDot) {
    const Shape& lhs_shape = hlo.operand(0)->shape();
    const Shape& rhs_shape = hlo.operand(1)->shape();

    if (ShapeUtil::IsZeroElementArray(lhs_shape) ||
        ShapeUtil::IsZeroElementArray(rhs_shape)) {
      return false;
    }

    if (ProfitableToImplementDotInTiledLlvmIr(hlo)) {
      return false;
    }

    // If gemm can accept the operand shapes, use it rather than a custom
    // kernel.
    if (AreValidGemmShapes(lhs_shape, rhs_shape, hlo.shape(),
                           target_machine_features)) {
      const DotDimensionNumbers& dim_numbers = hlo.dot_dimension_numbers();
      // The size of the reduction dimension should match. The shape inference
      // guarantees this invariant, so the check here is for programming
      // errors.
      CHECK_EQ(lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions(0)),
               rhs_shape.dimensions(dim_numbers.rhs_contracting_dimensions(0)));
      return true;
    }
  }

  return false;
}

// For vector-matrix dot products, it is always profitable to make the Rhs
// column major.
absl::optional<int64> ProfitableToMakeDotOperandColumnMajor(
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
