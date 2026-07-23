/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/large_constant_fold_pass.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {

template <typename Functor>
LogicalResult DispatchElementType(Type elem_type, Functor&& func) {
  if (elem_type.isF32()) {
    func.template operator()<float>();
    return success();
  } else if (elem_type.isBF16()) {
    func.template operator()<Eigen::bfloat16>();
    return success();
  } else if (elem_type.isF16()) {
    func.template operator()<Eigen::half>();
    return success();
  } else if (elem_type.isInteger(32)) {
    func.template operator()<int32_t>();
    return success();
  } else if (elem_type.isInteger(64)) {
    func.template operator()<int64_t>();
    return success();
  } else if (elem_type.isInteger(8)) {
    func.template operator()<int8_t>();
    return success();
  } else if (elem_type.isInteger(16)) {
    func.template operator()<int16_t>();
    return success();
  }
  return failure();
}

// Retrieves the underlying AsmResourceBlob from a resource attribute via direct
// handle lookup or resolved resource fallback.
static AsmResourceBlob* GetBlob(DenseResourceElementsAttr attr) {
  if (AsmResourceBlob* blob = attr.getRawHandle().getBlob()) return blob;
  if (auto* resource = attr.getRawHandle().getResource())
    return resource->getBlob();
  return nullptr;
}

static std::string GetResourceUniqueKey(DenseResourceElementsAttr attr) {
  std::string key = attr.getRawHandle().getKey().str();
  constexpr absl::string_view kPrefix = "dense_resource_off_";
  if (absl::StartsWith(key, kPrefix)) {
    absl::string_view suffix = absl::string_view(key).substr(kPrefix.size());
    size_t end_pos = suffix.find_first_not_of("0123456789");
    if (end_pos != 0) {
      return std::string(suffix.substr(0, end_pos));
    }
  }
  AsmResourceBlob* blob = GetBlob(attr);
  if (blob && !blob->getData().empty()) {
    return absl::StrCat("ptr_",
                        reinterpret_cast<uintptr_t>(blob->getData().data()));
  }
  return key;
}

struct FoldResourceCast : public OpRewritePattern<CastOp> {
  explicit FoldResourceCast(MLIRContext* ctx, bool fold_fp16_resource_casts)
      : OpRewritePattern<CastOp>(ctx),
        fold_fp16_resource_casts(fold_fp16_resource_casts) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter& rewriter) const override {
    ElementsAttr input_attr;
    if (!matchPattern(op.getInput(), m_Constant(&input_attr))) {
      return failure();
    }

    if (auto dense_attr =
            mlir::dyn_cast_or_null<DenseElementsAttr>(input_attr)) {
      auto in_type = dense_attr.getElementType();
      auto out_type = op.getType().getElementType();
      auto result_type = mlir::cast<ShapedType>(op.getType());
      if (in_type == out_type) {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, dense_attr);
        return success();
      }
      size_t num_elements = result_type.getNumElements();
      if (in_type.isBF16() && out_type.isF32()) {
        auto src_data = mlir::TFL::GetValues<Eigen::bfloat16>(dense_attr);
        if (src_data.empty()) return failure();
        bool is_splat = dense_attr.isSplat() || (src_data.size() == 1);
        if (is_splat) {
          float val = static_cast<float>(src_data[0]);
          auto new_attr =
              DenseElementsAttr::get(result_type, llvm::ArrayRef<float>(val));
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
          return success();
        }
        SmallVector<float, 8> dst_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
          dst_data[i] = static_cast<float>(src_data[i]);
        auto new_attr =
            DenseElementsAttr::get(result_type, llvm::ArrayRef(dst_data));
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
        return success();
      } else if (in_type.isF16() && out_type.isF32()) {
        auto src_data = mlir::TFL::GetValues<Eigen::half>(dense_attr);
        if (src_data.empty()) return failure();
        bool is_splat = dense_attr.isSplat() || (src_data.size() == 1);
        if (is_splat) {
          float val = static_cast<float>(src_data[0]);
          auto new_attr =
              DenseElementsAttr::get(result_type, llvm::ArrayRef<float>(val));
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
          return success();
        }
        SmallVector<float, 8> dst_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
          dst_data[i] = static_cast<float>(src_data[i]);
        auto new_attr =
            DenseElementsAttr::get(result_type, llvm::ArrayRef(dst_data));
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
        return success();
      } else if (in_type.isF32() && out_type.isBF16()) {
        auto src_data = mlir::TFL::GetValues<float>(dense_attr);
        if (src_data.empty()) return failure();
        bool is_splat = dense_attr.isSplat() || (src_data.size() == 1);
        if (is_splat) {
          Eigen::bfloat16 val = static_cast<Eigen::bfloat16>(src_data[0]);
          auto new_attr = DenseElementsAttr::get(
              result_type, llvm::ArrayRef<Eigen::bfloat16>(val));
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
          return success();
        }
        SmallVector<Eigen::bfloat16, 8> dst_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
          dst_data[i] = static_cast<Eigen::bfloat16>(src_data[i]);
        auto new_attr =
            DenseElementsAttr::get(result_type, llvm::ArrayRef(dst_data));
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
        return success();
      }
      return failure();
    }

    auto resource_attr =
        mlir::dyn_cast_or_null<DenseResourceElementsAttr>(input_attr);
    if (!resource_attr) return failure();

    auto in_type = resource_attr.getElementType();
    auto out_type = op.getType().getElementType();
    auto result_type = mlir::cast<ShapedType>(op.getType());

    if (!fold_fp16_resource_casts && (in_type.isBF16() || in_type.isF16()) &&
        out_type.isF32()) {
      return failure();
    }

    if (in_type == out_type) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resource_attr);
      return success();
    }

    AsmResourceBlob* blob = GetBlob(resource_attr);
    if (!blob) return failure();

    size_t out_elem_size = (out_type.getIntOrFloatBitWidth() + 7) / 8;

    bool is_splat = resource_attr.isSplat();
    if (!is_splat) {
      if (in_type.isBF16() && blob->getDataAs<Eigen::bfloat16>().size() == 1)
        is_splat = true;
      else if (in_type.isF16() && blob->getDataAs<Eigen::half>().size() == 1)
        is_splat = true;
      else if (in_type.isF32() && blob->getDataAs<float>().size() == 1)
        is_splat = true;
    }

    if (!is_splat) {
      auto new_attr = DenseResourceElementsAttr::get(
          result_type, resource_attr.getRawHandle());
      Operation* orig_op = op.getInput().getDefiningOp();
      auto new_const_op =
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
      new_const_op->setAttr("litert.target_element_type",
                            TypeAttr::get(out_type));
      if (orig_op) {
        if (auto perm_attr = orig_op->getAttrOfType<ArrayAttr>(
                "litert.layout_permutation")) {
          new_const_op->setAttr("litert.layout_permutation", perm_attr);
        }
      }
      return success();
    }

    auto new_blob = mlir::HeapAsmResourceBlob::allocate(
        out_elem_size, /*align=*/64, /*dataIsMutable=*/true);

    if (in_type.isBF16() && out_type.isF32()) {
      auto src_data = blob->getDataAs<Eigen::bfloat16>();
      if (src_data.empty()) return failure();
      const_cast<float*>(new_blob.getDataAs<float>().data())[0] =
          static_cast<float>(src_data[0]);
    } else if (in_type.isF32() && out_type.isBF16()) {
      auto src_data = blob->getDataAs<float>();
      if (src_data.empty()) return failure();
      const_cast<Eigen::bfloat16*>(
          new_blob.getDataAs<Eigen::bfloat16>().data())[0] =
          static_cast<Eigen::bfloat16>(src_data[0]);
    } else if (in_type.isF16() && out_type.isF32()) {
      auto src_data = blob->getDataAs<Eigen::half>();
      if (src_data.empty()) return failure();
      const_cast<float*>(new_blob.getDataAs<float>().data())[0] =
          static_cast<float>(src_data[0]);
    } else if (in_type.isF32() && out_type.isF16()) {
      auto src_data = blob->getDataAs<float>();
      if (src_data.empty()) return failure();
      const_cast<Eigen::half*>(new_blob.getDataAs<Eigen::half>().data())[0] =
          static_cast<Eigen::half>(src_data[0]);
    } else if (in_type.isBF16() && out_type.isF16()) {
      auto src_data = blob->getDataAs<Eigen::bfloat16>();
      if (src_data.empty()) return failure();
      const_cast<Eigen::half*>(new_blob.getDataAs<Eigen::half>().data())[0] =
          static_cast<Eigen::half>(static_cast<float>(src_data[0]));
    } else if (in_type.isF16() && out_type.isBF16()) {
      auto src_data = blob->getDataAs<Eigen::half>();
      if (src_data.empty()) return failure();
      const_cast<Eigen::bfloat16*>(
          new_blob.getDataAs<Eigen::bfloat16>().data())[0] =
          static_cast<Eigen::bfloat16>(static_cast<float>(src_data[0]));
    } else {
      return failure();
    }

    std::string res_key = GetResourceUniqueKey(resource_attr);
    absl::string_view out_type_str = out_type.isF32()    ? "f32"
                                     : out_type.isBF16() ? "bf16"
                                     : out_type.isF16()  ? "f16"
                                                         : "int";
    std::string new_key = absl::StrCat("cast_", res_key, "_", out_type_str);
    auto new_attr = DenseResourceElementsAttr::get(result_type, new_key,
                                                   std::move(new_blob));
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
    return success();
  }

  bool fold_fp16_resource_casts = true;
};

template <typename T, typename OpFn>
void RunBroadcastBinary(ArrayRef<T> lhs, ArrayRef<T> rhs,
                        MutableArrayRef<T> dst, ArrayRef<int64_t> lhs_shape,
                        ArrayRef<int64_t> rhs_shape,
                        ArrayRef<int64_t> dst_shape, bool lhs_splat,
                        bool rhs_splat, OpFn&& op_fn) {
  size_t num_elements = dst.size();
  if (lhs_splat && rhs_splat) {
    T val = op_fn(lhs[0], rhs[0]);
    for (size_t i = 0; i < num_elements; ++i) dst[i] = val;
    return;
  }
  if (lhs_splat) {
    T val = lhs[0];
    for (size_t i = 0; i < num_elements; ++i) dst[i] = op_fn(val, rhs[i]);
    return;
  }
  if (rhs_splat) {
    T val = rhs[0];
    for (size_t i = 0; i < num_elements; ++i) dst[i] = op_fn(lhs[i], val);
    return;
  }
  if (lhs_shape == rhs_shape) {
    for (size_t i = 0; i < num_elements; ++i) dst[i] = op_fn(lhs[i], rhs[i]);
    return;
  }
  int rank = dst_shape.size();
  SmallVector<int64_t, 4> padded_lhs(rank, 1), padded_rhs(rank, 1);
  for (int i = 0; i < lhs_shape.size(); ++i)
    padded_lhs[rank - lhs_shape.size() + i] = lhs_shape[i];
  for (int i = 0; i < rhs_shape.size(); ++i)
    padded_rhs[rank - rhs_shape.size() + i] = rhs_shape[i];

  SmallVector<int64_t, 4> coord(rank, 0);
  for (size_t i = 0; i < num_elements; ++i) {
    size_t lhs_idx = 0, rhs_idx = 0;
    size_t lhs_stride = 1, rhs_stride = 1;
    for (int r = rank - 1; r >= 0; --r) {
      lhs_idx += (coord[r] % padded_lhs[r]) * lhs_stride;
      rhs_idx += (coord[r] % padded_rhs[r]) * rhs_stride;
      lhs_stride *= padded_lhs[r];
      rhs_stride *= padded_rhs[r];
    }
    dst[i] = op_fn(lhs[lhs_idx], rhs[rhs_idx]);
    for (int r = rank - 1; r >= 0; --r) {
      if (++coord[r] < dst_shape[r]) break;
      coord[r] = 0;
    }
  }
}

template <typename OpType, typename BinaryFunctor>
struct FoldResourceBinaryOp : public OpRewritePattern<OpType> {
  explicit FoldResourceBinaryOp(
      MLIRContext* ctx,
      absl::flat_hash_map<std::string, DenseResourceElementsAttr>& cache)
      : OpRewritePattern<OpType>(ctx), cache(cache) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    if (op.getFusedActivationFunction() != "NONE") return failure();

    if ((op.getLhs().getDefiningOp() &&
         op.getLhs().getDefiningOp()->hasAttr("litert.layout_permutation")) ||
        (op.getRhs().getDefiningOp() &&
         op.getRhs().getDefiningOp()->hasAttr("litert.layout_permutation"))) {
      return failure();
    }

    ElementsAttr lhs_attr, rhs_attr;
    if (!matchPattern(op.getLhs(), m_Constant(&lhs_attr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhs_attr))) {
      return failure();
    }
    bool lhs_is_res = mlir::isa<DenseResourceElementsAttr>(lhs_attr);
    bool rhs_is_res = mlir::isa<DenseResourceElementsAttr>(rhs_attr);
    if (!lhs_is_res && !rhs_is_res) return failure();

    auto result_type = mlir::cast<ShapedType>(op.getType());
    auto elem_type = result_type.getElementType();

    if (std::is_same_v<OpType, DivOp> && mlir::isa<IntegerType>(elem_type)) {
      bool has_zero = false;
      (void)DispatchElementType(elem_type, [&]<typename T>() {
        ArrayRef<T> rhs_data = mlir::TFL::GetValues<T>(rhs_attr);
        for (T val : rhs_data) {
          if (val == 0) {
            has_zero = true;
            break;
          }
        }
      });
      if (has_zero) return failure();
    }

    llvm::StringRef op_name = op.getOperationName();
    op_name.consume_front("tfl.");
    absl::string_view op_name_sv(op_name.data(), op_name.size());

    // Hash raw data for non-resource attributes to prevent cache collisions
    // across different dense attributes of the same size.
    auto get_operand_key = [](ElementsAttr attr) -> std::string {
      if (auto res_attr = mlir::dyn_cast<DenseResourceElementsAttr>(attr)) {
        return res_attr.getRawHandle().getKey().str();
      }
      if (auto dense_attr = mlir::dyn_cast<DenseElementsAttr>(attr)) {
        llvm::StringRef raw_data(dense_attr.getRawData().data(),
                                 dense_attr.getRawData().size());
        return absl::StrCat("dense_", dense_attr.getNumElements(), "_",
                            static_cast<uint64_t>(llvm::hash_value(raw_data)));
      }
      return absl::StrCat("attr_", attr.getNumElements());
    };

    std::string lhs_key = get_operand_key(lhs_attr);
    std::string rhs_key = get_operand_key(rhs_attr);
    std::string cache_key =
        absl::StrCat(op_name_sv, ":", lhs_key, ":", rhs_key);

    auto it = cache.find(cache_key);
    if (it != cache.end()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, it->second);
      return success();
    }

    size_t num_elements = result_type.getNumElements();
    size_t elem_size = (elem_type.getIntOrFloatBitWidth() + 7) / 8;

    auto new_blob = mlir::HeapAsmResourceBlob::allocate(
        num_elements * elem_size, /*align=*/64, /*dataIsMutable=*/true);

    auto lhs_type = mlir::cast<ShapedType>(lhs_attr.getType());
    auto rhs_type = mlir::cast<ShapedType>(rhs_attr.getType());

    LogicalResult status = DispatchElementType(elem_type, [&]<typename T>() {
      ArrayRef<T> lhs_data = mlir::TFL::GetValues<T>(lhs_attr);
      ArrayRef<T> rhs_data = mlir::TFL::GetValues<T>(rhs_attr);
      if (lhs_data.empty() || rhs_data.empty()) return;
      bool lhs_splat = lhs_attr.isSplat() || (lhs_data.size() == 1);
      bool rhs_splat = rhs_attr.isSplat() || (rhs_data.size() == 1);
      MutableArrayRef<T> dst_data(
          const_cast<T*>(new_blob.getDataAs<T>().data()), num_elements);
      RunBroadcastBinary<T>(lhs_data, rhs_data, dst_data, lhs_type.getShape(),
                            rhs_type.getShape(), result_type.getShape(),
                            lhs_splat, rhs_splat, BinaryFunctor{});
    });
    if (failed(status)) return failure();

    std::string new_key = absl::StrCat(op_name_sv, "_", lhs_key, "_", rhs_key);
    auto new_attr = DenseResourceElementsAttr::get(result_type, new_key,
                                                   std::move(new_blob));
    cache[cache_key] = new_attr;
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
    return success();
  }

  absl::flat_hash_map<std::string, DenseResourceElementsAttr>& cache;
};

struct AddFunctor {
  template <typename T>
  T operator()(T a, T b) const {
    return a + b;
  }
};
struct SubFunctor {
  template <typename T>
  T operator()(T a, T b) const {
    return a - b;
  }
};
struct MulFunctor {
  template <typename T>
  T operator()(T a, T b) const {
    return a * b;
  }
};
struct DivFunctor {
  template <typename T>
  T operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      if (b == 0) return T(0);
    }
    return a / b;
  }
};

using FoldResourceAdd = FoldResourceBinaryOp<AddOp, AddFunctor>;
using FoldResourceSub = FoldResourceBinaryOp<SubOp, SubFunctor>;
using FoldResourceMul = FoldResourceBinaryOp<MulOp, MulFunctor>;
using FoldResourceDiv = FoldResourceBinaryOp<DivOp, DivFunctor>;

struct FoldResourceTranspose : public OpRewritePattern<TransposeOp> {
  explicit FoldResourceTranspose(MLIRContext* ctx)
      : OpRewritePattern<TransposeOp>(ctx) {}

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter& rewriter) const override {
    ElementsAttr input_attr, perm_attr;
    if (!matchPattern(op.getOperand(0), m_Constant(&input_attr)) ||
        !matchPattern(op.getOperand(1), m_Constant(&perm_attr))) {
      return failure();
    }
    auto resource_attr =
        mlir::dyn_cast_or_null<DenseResourceElementsAttr>(input_attr);
    if (!resource_attr) return failure();

    auto input_type = mlir::cast<ShapedType>(input_attr.getType());
    auto result_type = mlir::cast<ShapedType>(op.getType());
    int rank = input_type.getRank();

    if (rank <= 1 || resource_attr.isSplat()) {
      auto new_attr = DenseResourceElementsAttr::get(
          result_type, resource_attr.getRawHandle());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
      return success();
    }

    auto int_perm_attr = mlir::dyn_cast<DenseIntElementsAttr>(perm_attr);
    if (!int_perm_attr) return failure();

    auto perms = llvm::to_vector<4>(
        llvm::map_range(int_perm_attr.getValues<APInt>(),
                        [](const APInt& val) { return val.getSExtValue(); }));
    if (perms.size() != rank) return failure();

    Operation* const_op = op.getOperand(0).getDefiningOp();
    if (!const_op) return failure();

    SmallVector<int64_t, 4> final_perms = perms;
    if (auto existing_attr =
            const_op->getAttrOfType<ArrayAttr>("litert.layout_permutation")) {
      SmallVector<int64_t, 4> old_perms;
      for (auto attr : existing_attr.getValue()) {
        old_perms.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
      if (old_perms.size() == perms.size()) {
        for (size_t i = 0; i < perms.size(); ++i) {
          final_perms[i] = old_perms[perms[i]];
        }
      }
    }
    const_op->setAttr("litert.layout_permutation",
                      rewriter.getI64ArrayAttr(final_perms));
    if (auto res_attr =
            const_op->getAttrOfType<DenseResourceElementsAttr>("value")) {
      const_op->setAttr("value", DenseResourceElementsAttr::get(
                                     result_type, res_attr.getRawHandle()));
    }
    const_op->getResult(0).setType(result_type);
    rewriter.replaceOp(op, const_op->getResult(0));
    return success();
  }
};

struct FoldResourceReshape : public OpRewritePattern<ReshapeOp> {
  explicit FoldResourceReshape(MLIRContext* ctx)
      : OpRewritePattern<ReshapeOp>(ctx) {}

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getInput().getDefiningOp() &&
        op.getInput().getDefiningOp()->hasAttr("litert.layout_permutation")) {
      return failure();
    }

    ElementsAttr input_attr;
    if (!matchPattern(op.getInput(), m_Constant(&input_attr))) {
      return failure();
    }
    auto resource_attr =
        mlir::dyn_cast_or_null<DenseResourceElementsAttr>(input_attr);
    if (!resource_attr) return failure();

    auto result_type = mlir::cast<ShapedType>(op.getType());
    auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
    if (result_type.getNumElements() != input_type.getNumElements()) {
      return failure();
    }

    auto new_attr = DenseResourceElementsAttr::get(
        result_type, resource_attr.getRawHandle());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
    return success();
  }
};

}  // namespace

void LargeConstantFoldPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();
  absl::flat_hash_map<std::string, DenseResourceElementsAttr> cache;

  patterns.add<FoldResourceCast>(ctx, GetOptions().fold_fp16_resource_casts);
  if (GetOptions().fold_elementwise_ops) {
    patterns.add<FoldResourceAdd>(ctx, cache);
    patterns.add<FoldResourceSub>(ctx, cache);
    patterns.add<FoldResourceMul>(ctx, cache);
    patterns.add<FoldResourceDiv>(ctx, cache);
  }
  patterns.add<FoldResourceTranspose>(ctx);
  patterns.add<FoldResourceReshape>(ctx);

  GreedyRewriteConfig config;
  config.enableFolding(false);

  if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
    module.emitError() << "large-constant-fold failed.";
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateLargeConstantFoldPass(
    bool fold_fp16_resource_casts, bool fold_elementwise_ops) {
  return std::make_unique<LargeConstantFoldPass>(fold_fp16_resource_casts,
                                                 fold_elementwise_ops);
}

}  // namespace TFL
}  // namespace mlir
