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

// Folding of unary ops on `DenseResourceElementsAttr` constants that accounts
// for blobs shared by several constants.
//
// Large weights are stored in resource blobs that can be referenced by more
// than one constant op, possibly in different functions. Folding an op on such
// a constant has to leave the other users intact, while keeping as few copies
// of the weight alive as possible.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_CONSTANT_FOLD_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_CONSTANT_FOLD_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::TFL {

// Returns the blob backing `attr`, or null if it has none.
inline AsmResourceBlob* GetResourceBlob(DenseResourceElementsAttr attr) {
  if (AsmResourceBlob* blob = attr.getRawHandle().getBlob()) return blob;
  if (auto* resource = attr.getRawHandle().getResource()) {
    return resource->getBlob();
  }
  return nullptr;
}

// Returns a key that identifies the data backing `attr`. Attributes that share
// a blob, even under different resource keys, get the same key.
inline std::string GetResourceKey(DenseResourceElementsAttr attr) {
  AsmResourceBlob* blob = GetResourceBlob(attr);
  if (blob && !blob->getData().empty()) {
    return absl::StrCat("ptr_",
                        reinterpret_cast<uintptr_t>(blob->getData().data()));
  }
  return attr.getRawHandle().getKey().str();
}

// Folds every op of type `OpT` whose operand is a resource constant, as
// described by `Folder`:
//
//   // Returns true if `op` can be folded, and sets the constant op and
//   // resource attribute it reads.
//   static bool Match(OpT op, Operation*& const_op,
//                     DenseResourceElementsAttr& resource_attr);
//   // Returns a string that is equal for ops that compute the same result
//   // from the same input.
//   static std::string GetSignature(OpT op);
//   // Computes the folded value. With `in_place`, `blob` may be overwritten
//   // and the result may reuse its handle.
//   static LogicalResult Fold(OpT op, DenseResourceElementsAttr resource_attr,
//                             AsmResourceBlob* blob,
//                             DenseResourceElementsAttr& new_attr,
//                             bool in_place);
//   static ShapedType GetResultType(OpT op);
//   // Creates a constant op holding `value`.
//   static Operation* CreateConstant(OpBuilder& builder, Location loc,
//                                    ShapedType type, ElementsAttr value);
//
// Ops are grouped by the blob they read. If every user of every constant that
// references a blob is a candidate computing the same result, the blob is
// folded once, in place when it is mutable, and the constants are updated.
// Otherwise one folded copy is created per distinct result and the source blob
// is released once nothing references it anymore.
//
// Sets `*changed` to true if any op was folded.
template <typename OpT, typename Folder>
LogicalResult FoldResourceOpPattern(ModuleOp module, bool* changed = nullptr) {
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
      // Constants sharing a blob can have different types, e.g. after a
      // reshape was folded by reusing the blob. The same op then computes a
      // different result, so the input type is part of the signature.
      std::string sig;
      llvm::raw_string_ostream os(sig);
      os << Folder::GetSignature(op) << "|" << resource_attr.getType();
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

    AsmResourceBlob* blob =
        GetResourceBlob(candidates[indices[0]].resource_attr);
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
    bool can_fold_in_place =
        all_users_are_candidates && is_single_signature && blob->isMutable();

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
      if (changed) *changed = true;

      if (!(new_attr.getRawHandle() ==
            first_cand.resource_attr.getRawHandle())) {
        *blob = mlir::AsmResourceBlob();
      }
    } else {
      bool all_signatures_folded = true;
      bool blob_still_referenced = false;
      for (auto& [sig, sig_indices] : sig_to_candidate_indices) {
        if (sig_indices.empty()) continue;
        const auto& first_cand = candidates[sig_indices[0]];

        DenseResourceElementsAttr new_attr;
        if (failed(Folder::Fold(first_cand.op, first_cand.resource_attr, blob,
                                new_attr, /*in_place=*/false))) {
          all_signatures_folded = false;
          continue;
        }
        if (GetResourceBlob(new_attr) == blob) blob_still_referenced = true;

        ShapedType new_type = Folder::GetResultType(first_cand.op);
        absl::flat_hash_map<Operation*, Operation*> orig_const_to_new_const;

        for (size_t idx : sig_indices) {
          auto& cand = candidates[idx];
          Operation*& new_cst_op = orig_const_to_new_const[cand.const_op];
          if (!new_cst_op) {
            OpBuilder builder(cand.const_op);
            new_cst_op = Folder::CreateConstant(
                builder, cand.const_op->getLoc(), new_type, new_attr);
          }
          cand.op->replaceAllUsesWith(new_cst_op);
          ops_to_erase.push_back(cand.op.getOperation());
        }
        if (changed) *changed = true;
      }

      if (all_users_are_candidates && all_signatures_folded) {
        for (Operation* const_op : unique_candidate_const_ops) {
          const_ops_to_erase.push_back(const_op);
        }
        if (!blob_still_referenced) *blob = mlir::AsmResourceBlob();
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

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_CONSTANT_FOLD_UTILS_H_
