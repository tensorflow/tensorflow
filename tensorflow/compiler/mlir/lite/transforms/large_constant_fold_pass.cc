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
#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_in_place_transpose.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir::TFL {
namespace {

// Retrieves the underlying AsmResourceBlob from a resource attribute.
static AsmResourceBlob* GetBlob(DenseResourceElementsAttr attr) {
  if (AsmResourceBlob* blob = attr.getRawHandle().getBlob()) return blob;
  if (auto* resource = attr.getRawHandle().getResource())
    return resource->getBlob();
  return nullptr;
}

static std::string GetResourceKey(DenseResourceElementsAttr attr) {
  AsmResourceBlob* blob = GetBlob(attr);
  if (blob && !blob->getData().empty()) {
    return absl::StrCat("ptr_",
                        reinterpret_cast<uintptr_t>(blob->getData().data()));
  }
  return attr.getRawHandle().getKey().str();
}

template <typename OpT>
struct ResourceOpFolder;

template <>
struct ResourceOpFolder<TFL::TransposeOp> {
  using OpType = TFL::TransposeOp;

  static bool Match(OpType op, Operation*& const_op,
                    DenseResourceElementsAttr& resource_attr) {
    Value input = op.getInput();
    ElementsAttr input_attr;
    if (!matchPattern(input, m_Constant(&input_attr))) return false;
    resource_attr =
        mlir::dyn_cast_or_null<DenseResourceElementsAttr>(input_attr);
    if (!resource_attr) return false;

    ElementsAttr perm_attr;
    if (!matchPattern(op.getPerm(), m_Constant(&perm_attr))) return false;
    Type perm_elem = perm_attr.getElementType();
    if (!perm_elem.isInteger(32) && !perm_elem.isInteger(64)) return false;

    const_op = input.getDefiningOp();
    if (!const_op) return false;

    AsmResourceBlob* blob = GetBlob(resource_attr);
    if (!blob || !blob->isMutable()) return false;

    auto input_type = mlir::cast<ShapedType>(resource_attr.getType());
    if (!input_type.hasStaticShape()) return false;
    if (!input_type.getElementType().isIntOrIndexOrFloat()) return false;

    return true;
  }

  static std::string GetSignature(OpType op) {
    ElementsAttr perm_attr;
    if (!matchPattern(op.getPerm(), m_Constant(&perm_attr))) return "";
    std::string sig = "transpose";
    if (perm_attr.getElementType().isInteger(32)) {
      for (int32_t val : GetValues<int32_t>(perm_attr)) {
        absl::StrAppend(&sig, "_", val);
      }
    } else if (perm_attr.getElementType().isInteger(64)) {
      for (int64_t val : GetValues<int64_t>(perm_attr)) {
        absl::StrAppend(&sig, "_", val);
      }
    }
    return sig;
  }

  static LogicalResult Fold(OpType op, DenseResourceElementsAttr resource_attr,
                            AsmResourceBlob* blob,
                            DenseResourceElementsAttr& new_attr,
                            bool in_place) {
    auto input_type = mlir::cast<ShapedType>(resource_attr.getType());
    ElementsAttr perm_attr;
    if (!matchPattern(op.getPerm(), m_Constant(&perm_attr))) return failure();

    SmallVector<int64_t> perms;
    if (perm_attr.getElementType().isInteger(32)) {
      auto perm_values = GetValues<int32_t>(perm_attr);
      perms.assign(perm_values.begin(), perm_values.end());
    } else if (perm_attr.getElementType().isInteger(64)) {
      auto perm_values = GetValues<int64_t>(perm_attr);
      perms.assign(perm_values.begin(), perm_values.end());
    } else {
      return failure();
    }

    int bit_width = input_type.getElementType().getIntOrFloatBitWidth();
    ArrayRef<int64_t> input_shape = input_type.getShape();
    ShapedType new_type = GetResultType(op);

    if (in_place) {
      void* raw_data = blob->getMutableData().data();
      if (!raw_data) return failure();

      if (!InPlaceTranspose(raw_data, input_shape, perms, bit_width)) {
        return failure();
      }
      new_attr = DenseResourceElementsAttr::get(new_type,
                                                resource_attr.getRawHandle());
      return success();
    }

    ArrayRef<char> src_bytes = blob->getData();
    auto new_blob = mlir::HeapAsmResourceBlob::allocate(
        src_bytes.size(), /*align=*/64, /*dataIsMutable=*/true);
    void* dst_data = new_blob.getMutableData().data();
    if (!dst_data) return failure();

    if (!TransposeBuffer(src_bytes.data(), dst_data, input_shape, perms,
                         bit_width)) {
      return failure();
    }
    std::string new_key = absl::StrCat(
        resource_attr.getRawHandle().getKey().str(), "_", GetSignature(op));
    new_attr =
        DenseResourceElementsAttr::get(new_type, new_key, std::move(new_blob));
    return success();
  }

  static ShapedType GetResultType(OpType op) {
    return mlir::cast<ShapedType>(op.getType());
  }
};

template <>
struct ResourceOpFolder<TFL::CastOp> {
  using OpType = TFL::CastOp;

  static bool Match(OpType op, Operation*& const_op,
                    DenseResourceElementsAttr& resource_attr) {
    Value input = op.getInput();
    ElementsAttr input_attr;
    if (!matchPattern(input, m_Constant(&input_attr))) return false;
    resource_attr =
        mlir::dyn_cast_or_null<DenseResourceElementsAttr>(input_attr);
    if (!resource_attr) return false;

    const_op = input.getDefiningOp();
    if (!const_op) return false;

    AsmResourceBlob* blob = GetBlob(resource_attr);
    if (!blob) return false;

    auto input_type = mlir::dyn_cast<ShapedType>(resource_attr.getType());
    auto output_type = mlir::dyn_cast<ShapedType>(op.getType());
    if (!input_type || !output_type || !input_type.hasStaticShape() ||
        !output_type.hasStaticShape()) {
      return false;
    }

    Type in_elem = input_type.getElementType();
    Type out_elem = output_type.getElementType();

    if ((in_elem.isBF16() || in_elem.isF16() || in_elem.isF32()) &&
        (out_elem.isBF16() || out_elem.isF16() || out_elem.isF32())) {
      return true;
    }

    return false;
  }

  static std::string GetSignature(OpType op) {
    auto output_type = mlir::dyn_cast<ShapedType>(op.getType());
    if (!output_type) return "";
    Type out_elem = output_type.getElementType();
    if (out_elem.isF32()) return "cast_f32";
    if (out_elem.isF16()) return "cast_f16";
    if (out_elem.isBF16()) return "cast_bf16";
    return "cast_unknown";
  }

  static LogicalResult Fold(OpType op, DenseResourceElementsAttr resource_attr,
                            AsmResourceBlob* blob,
                            DenseResourceElementsAttr& new_attr,
                            bool /*in_place*/) {
    auto input_type = mlir::cast<ShapedType>(resource_attr.getType());
    auto output_type = mlir::cast<ShapedType>(op.getType());
    Type in_elem = input_type.getElementType();
    Type out_elem = output_type.getElementType();

    size_t num_elements = input_type.getNumElements();
    if (num_elements == 0) return success();

    const void* src_data = blob->getData().data();
    if (!src_data) return failure();

    size_t out_elem_size = (out_elem.getIntOrFloatBitWidth() + 7) / 8;
    size_t new_size_bytes = num_elements * out_elem_size;

    auto new_blob = mlir::HeapAsmResourceBlob::allocate(
        new_size_bytes, /*align=*/64, /*dataIsMutable=*/true);
    void* dst_data = new_blob.getMutableData().data();
    if (!dst_data) return failure();

    if (in_elem.isBF16() && out_elem.isF32()) {
      const uint16_t* in = reinterpret_cast<const uint16_t*>(src_data);
      uint32_t* out = reinterpret_cast<uint32_t*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        out[i] = static_cast<uint32_t>(in[i]) << 16;
      }
    } else if (in_elem.isF16() && out_elem.isF32()) {
      const uint16_t* in = reinterpret_cast<const uint16_t*>(src_data);
      float* out = reinterpret_cast<float*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        Eigen::half h = Eigen::numext::bit_cast<Eigen::half>(in[i]);
        out[i] = static_cast<float>(h);
      }
    } else if (in_elem.isF32() && out_elem.isBF16()) {
      const uint32_t* in = reinterpret_cast<const uint32_t*>(src_data);
      uint16_t* out = reinterpret_cast<uint16_t*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        uint32_t val = in[i];
        uint32_t lsb = (val >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        out[i] = static_cast<uint16_t>((val + rounding_bias) >> 16);
      }
    } else if (in_elem.isF32() && out_elem.isF16()) {
      const float* in = reinterpret_cast<const float*>(src_data);
      uint16_t* out = reinterpret_cast<uint16_t*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        Eigen::half h(in[i]);
        out[i] = Eigen::numext::bit_cast<uint16_t>(h);
      }
    } else if (in_elem.isBF16() && out_elem.isF16()) {
      const uint16_t* in = reinterpret_cast<const uint16_t*>(src_data);
      uint16_t* out = reinterpret_cast<uint16_t*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        uint32_t val = static_cast<uint32_t>(in[i]) << 16;
        float f = *reinterpret_cast<float*>(&val);
        Eigen::half h(f);
        out[i] = Eigen::numext::bit_cast<uint16_t>(h);
      }
    } else if (in_elem.isF16() && out_elem.isBF16()) {
      const uint16_t* in = reinterpret_cast<const uint16_t*>(src_data);
      uint16_t* out = reinterpret_cast<uint16_t*>(dst_data);
      for (size_t i = 0; i < num_elements; ++i) {
        Eigen::half h = Eigen::numext::bit_cast<Eigen::half>(in[i]);
        float f = static_cast<float>(h);
        uint32_t val = *reinterpret_cast<uint32_t*>(&f);
        uint32_t lsb = (val >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        out[i] = static_cast<uint16_t>((val + rounding_bias) >> 16);
      }
    } else {
      return failure();
    }

    std::string new_key = absl::StrCat(
        resource_attr.getRawHandle().getKey().str(), "_", GetSignature(op));
    new_attr = DenseResourceElementsAttr::get(output_type, new_key,
                                              std::move(new_blob));
    return success();
  }

  static ShapedType GetResultType(OpType op) {
    return mlir::cast<ShapedType>(op.getType());
  }
};

template <typename OpT, typename Folder = ResourceOpFolder<OpT>>
LogicalResult FoldResourceOpPattern(ModuleOp module) {
  struct Candidate {
    OpT op;
    Operation* const_op;
    DenseResourceElementsAttr resource_attr;
    std::string resource_key;
    std::string signature;
  };

  llvm::SmallVector<Candidate> candidates;
  absl::flat_hash_map<std::string, llvm::SmallVector<size_t>>
      resource_to_candidate_indices;

  module.walk([&](OpT op) {
    Operation* const_op = nullptr;
    DenseResourceElementsAttr resource_attr;
    if (Folder::Match(op, const_op, resource_attr)) {
      std::string key = GetResourceKey(resource_attr);
      std::string sig = Folder::GetSignature(op);
      size_t idx = candidates.size();
      candidates.push_back({op, const_op, resource_attr, key, sig});
      resource_to_candidate_indices[key].push_back(idx);
    }
  });

  if (candidates.empty()) return success();

  absl::flat_hash_map<std::string, llvm::SmallPtrSet<Operation*, 4>>
      resource_to_all_const_ops;
  module.walk([&](Operation* op) {
    ElementsAttr attr;
    if (matchPattern(op, m_Constant(&attr))) {
      if (auto res_attr = mlir::dyn_cast<DenseResourceElementsAttr>(attr)) {
        resource_to_all_const_ops[GetResourceKey(res_attr)].insert(op);
      }
    }
  });

  llvm::SmallVector<Operation*> ops_to_erase;
  llvm::SmallVector<Operation*> const_ops_to_erase;

  for (auto& [key, indices] : resource_to_candidate_indices) {
    if (indices.empty()) continue;

    AsmResourceBlob* blob = GetBlob(candidates[indices[0]].resource_attr);
    if (!blob) continue;

    llvm::DenseSet<Operation*> candidate_ops;
    llvm::DenseSet<Operation*> unique_candidate_const_ops;
    for (size_t idx : indices) {
      candidate_ops.insert(candidates[idx].op.getOperation());
      unique_candidate_const_ops.insert(candidates[idx].const_op);
    }

    absl::flat_hash_map<std::string, llvm::SmallVector<size_t>>
        sig_to_candidate_indices;
    for (size_t idx : indices) {
      sig_to_candidate_indices[candidates[idx].signature].push_back(idx);
    }

    const auto& all_module_const_ops = resource_to_all_const_ops[key];
    bool all_users_are_candidates =
        (unique_candidate_const_ops.size() == all_module_const_ops.size());
    if (all_users_are_candidates) {
      for (Operation* const_op : unique_candidate_const_ops) {
        for (Operation* user : const_op->getUsers()) {
          if (!candidate_ops.contains(user)) {
            all_users_are_candidates = false;
            break;
          }
        }
        if (!all_users_are_candidates) break;
      }
    }

    bool is_single_signature = (sig_to_candidate_indices.size() == 1);
    bool can_fold_in_place = all_users_are_candidates && is_single_signature;

    if (can_fold_in_place) {
      const auto& first_cand = candidates[indices[0]];
      DenseResourceElementsAttr new_attr;
      if (failed(Folder::Fold(first_cand.op, first_cand.resource_attr, blob,
                              new_attr, /*in_place=*/true))) {
        continue;
      }

      ShapedType new_type = Folder::GetResultType(first_cand.op);
      for (Operation* const_op : unique_candidate_const_ops) {
        const_op->setAttr("value", new_attr);
        const_op->getResult(0).setType(new_type);
      }

      for (size_t idx : indices) {
        candidates[idx].op->replaceAllUsesWith(candidates[idx].const_op);
        ops_to_erase.push_back(candidates[idx].op.getOperation());
      }

      if (!(new_attr.getRawHandle() ==
            first_cand.resource_attr.getRawHandle())) {
        *blob = mlir::AsmResourceBlob();
      }
    } else {
      bool all_signatures_folded = true;
      for (auto& [sig, sig_indices] : sig_to_candidate_indices) {
        if (sig_indices.empty()) continue;
        const auto& first_cand = candidates[sig_indices[0]];

        DenseResourceElementsAttr new_attr;
        if (failed(Folder::Fold(first_cand.op, first_cand.resource_attr, blob,
                                new_attr, /*in_place=*/false))) {
          all_signatures_folded = false;
          continue;
        }

        ShapedType new_type = Folder::GetResultType(first_cand.op);
        absl::flat_hash_map<Operation*, Operation*> orig_const_to_new_const;

        for (size_t idx : sig_indices) {
          auto& cand = candidates[idx];
          Operation*& new_cst_op = orig_const_to_new_const[cand.const_op];
          if (!new_cst_op) {
            OpBuilder builder(cand.const_op);
            new_cst_op = builder.create<arith::ConstantOp>(
                cand.const_op->getLoc(), new_type, new_attr);
          }
          cand.op->replaceAllUsesWith(new_cst_op);
          ops_to_erase.push_back(cand.op.getOperation());
        }
      }

      if (all_users_are_candidates && all_signatures_folded) {
        for (Operation* const_op : unique_candidate_const_ops) {
          const_ops_to_erase.push_back(const_op);
        }
        *blob = mlir::AsmResourceBlob();
      }
    }
  }

  for (Operation* op : ops_to_erase) {
    op->erase();
  }
  for (Operation* op : const_ops_to_erase) {
    if (op->use_empty()) {
      op->erase();
    }
  }

  return success();
}

}  // namespace

void LargeConstantFoldPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(FoldResourceOpPattern<TFL::TransposeOp>(module))) {
    signalPassFailure();
  }
  if (GetOptions().fold_fp16_resource_casts) {
    if (failed(FoldResourceOpPattern<TFL::CastOp>(module))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateLargeConstantFoldPass(
    bool fold_fp16_resource_casts, bool fold_elementwise_ops) {
  return std::make_unique<LargeConstantFoldPass>(fold_fp16_resource_casts,
                                                 fold_elementwise_ops);
}

}  // namespace mlir::TFL
