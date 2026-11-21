/* Copyright 2024 The OpenXLA Authors.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep
#include "xla/python/ifrt/memory.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTVERIFYDONATIONPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// Returns the effective memory-kind name of `type`.
llvm::StringRef GetMemoryKind(IfrtArrayType type) {
  const mlir::StringAttr attr = type.getMemoryKindAttr();
  return attr != nullptr ? attr.getValue()
                         : llvm::StringRef(MemoryKind().value());
}

// Returns true if `lhs` and `rhs` have the same effective `MemoryKind`.
bool HasSameMemoryKind(IfrtArrayType lhs, IfrtArrayType rhs) {
  return lhs.getMemoryKindAttr() == rhs.getMemoryKindAttr() ||
         GetMemoryKind(lhs) == GetMemoryKind(rhs);
}

// Sentinel `ArrayStateRef::num_shards` indicating that `ArrayStateRef::index`
// stores a single `ArrayState` ID inline rather than a `shard_arena_` offset.
constexpr int32_t kInlineArray = -1;

// Sentinel index indicating an absent `ArrayState`.
constexpr int32_t kNoArray = -1;

// Tracks provenance and donation state across array values. Each array starts
// as a single `ArrayState` and lazily splits into contiguous shard
// `ArrayState`s only when an operation aliases individual shards.
class DonationTracker {
 public:
  // Index-based per-shard view of a materialized array. Remains valid across
  // `arrays_` and `shard_arena_` reallocations.
  struct ArrayView {
    int32_t base;
    bool is_contiguous;
  };

  explicit DonationTracker(mlir::func::FuncOp func_op) {
    size_t num_arrays = func_op.getNumArguments();
    for (mlir::Operation& op : func_op.getOps()) {
      const unsigned num_results = op.getNumResults();
      if (num_results > 1) {
        num_arrays += num_results - 1;
      }
    }
    arrays_.reserve(num_arrays);
    array_to_ref_.reserve(num_arrays);

    const mlir::StringAttr donated_attr =
        mlir::StringAttr::get(func_op.getContext(), kIfrtDonatedArgAttrName);
    for (const mlir::BlockArgument arg : func_op.getArguments()) {
      if (!arg.use_empty() && mlir::isa<IfrtArrayType>(arg.getType())) {
        const bool is_donated =
            func_op.getArgAttr(arg.getArgNumber(), donated_attr) != nullptr;
        RegisterNewArray(arg, /*is_donatable=*/is_donated);
      }
    }
  }

  // Registers each used array in `arrays` as an array allocation.
  void RegisterArrays(mlir::ValueRange arrays) {
    for (const mlir::Value array : arrays) {
      RegisterNewArray(array, /*is_donatable=*/true);
    }
  }

  // Registers `out_array` (if used) as an alias of `in_array`.
  void RegisterArrayAlias(mlir::Value in_array, mlir::Value out_array) {
    if (!out_array.use_empty()) {
      RegisterArray(out_array, Lookup(in_array));
    }
  }

  // Registers a single-device `out_array` (if used) as an alias of
  // `shard_array_id`.
  void RegisterShardAlias(mlir::Value out_array, int32_t shard_array_id) {
    RegisterArray(out_array, ArrayStateRef{shard_array_id, kInlineArray});
  }

  // Allocates a standalone shard `ArrayState`.
  int32_t AllocateShard() { return AllocateArray(/*is_donatable=*/true); }

  // Materializes per-device shard `ArrayState`s for `array` if needed and
  // returns an index-based `ArrayView`. Never modifies `shard_arena_`.
  ArrayView MaterializeShards(mlir::Value array) {
    return MaterializeShards(array, Lookup(array));
  }

  // Returns the `ArrayState` ID for `shard_index` within `view`.
  int32_t GetArrayIdForShard(ArrayView view, size_t shard_index) const {
    CHECK_GE(view.base, 0);
    const size_t offset = static_cast<size_t>(view.base) + shard_index;
    if (view.is_contiguous) {
      CHECK_LT(offset, arrays_.size());
      return static_cast<int32_t>(offset);
    }
    CHECK_LT(offset, shard_arena_.size());
    return shard_arena_[offset];
  }

  // Allocates and registers contiguous output shard slices in `shard_arena_`
  // for all arrays in `outputs`.
  llvm::SmallVector<llvm::MutableArrayRef<int32_t>, 4> AllocateOutputShards(
      mlir::ValueRange outputs) {
    size_t num_shards = 0;
    for (const mlir::Value output : outputs) {
      num_shards += GetArrayType(output).getDevices().size();
    }

    llvm::SmallVector<llvm::MutableArrayRef<int32_t>, 4> slices;
    slices.reserve(outputs.size());
    size_t offset = shard_arena_.size();
    CHECK_LE(num_shards,
             static_cast<size_t>(std::numeric_limits<int32_t>::max()) - offset);
    shard_arena_.resize(offset + num_shards);
    for (const mlir::Value output : outputs) {
      const size_t num_shards = GetArrayType(output).getDevices().size();
      RegisterArray(output, ArrayStateRef{static_cast<int32_t>(offset),
                                          static_cast<int32_t>(num_shards)});
      slices.push_back(llvm::MutableArrayRef<int32_t>(
          shard_arena_.data() + offset, num_shards));
      offset += num_shards;
    }
    return slices;
  }

  // Returns an `Operation*` that already donated any shard of `array`, or
  // `nullptr` if none have been donated yet.
  mlir::Operation* FindDonatedBy(mlir::Value array) const {
    return FindDonatedBy(Lookup(array));
  }

  // Verifies that `array` has not already been donated and is not an undonated
  // program input, then marks `array` donated by `op`. Sets `donated_by` to the
  // prior donating op if already donated, or `nullptr` otherwise.
  mlir::LogicalResult DonateInput(mlir::Operation* op, size_t input_index,
                                  mlir::Value array,
                                  mlir::Operation*& donated_by) {
    const ArrayStateRef ref = Lookup(array);
    if (donated_by = FindDonatedBy(ref); donated_by != nullptr) {
      return mlir::failure();
    }
    return MarkDonated(op, input_index, ref);
  }

  // Verifies that no shard of any input in `inputs` has already been donated,
  // and marks all inputs as donated by `op` when `donated` is true. When
  // `allow_duplicates` is false, overlapping inputs within `inputs` are
  // rejected.
  mlir::LogicalResult DonateInputs(mlir::Operation* op, mlir::ValueRange inputs,
                                   bool donated, bool allow_duplicates) {
    llvm::SmallVector<ArrayStateRef, 8> refs;
    return ResolveAndDonateInputs(op, inputs, donated, allow_duplicates, refs);
  }

  // Verifies (and when `donated` is true, donates) single-device `inputs` and
  // registers `output` in a single lookup per input.
  mlir::LogicalResult AssembleInputs(mlir::Operation* op,
                                     mlir::ValueRange inputs,
                                     mlir::Value output, bool donated) {
    llvm::SmallVector<ArrayStateRef, 8> refs;
    if (mlir::failed(ResolveAndDonateInputs(
            op, inputs, donated, /*allow_duplicates=*/false, refs))) {
      return mlir::failure();
    }
    if (donated) {
      RegisterNewArray(output, /*is_donatable=*/true);
    } else if (!output.use_empty()) {
      llvm::MutableArrayRef<int32_t> out_shards =
          AllocateOutputShards(output).front();
      for (size_t input_index = 0; input_index < refs.size(); ++input_index) {
        const ArrayStateRef ref = refs[input_index];
        if (ref.num_shards == kInlineArray) {
          out_shards[input_index] = ref.index;
        } else {
          const llvm::ArrayRef<int32_t> shard_ids = GetShardArenaIds(ref);
          CHECK_EQ(shard_ids.size(), 1);
          out_shards[input_index] = shard_ids.front();
        }
      }
    }
    return mlir::success();
  }

  // Verifies and donates `inputs` (when `donated` is true) in a single lookup
  // per input. Populates `in_views` with per-input `ArrayView`s unless
  // `donated` is true and all inputs are disjoint unaliased arrays.
  mlir::LogicalResult VerifyAndMaterializeInputs(
      mlir::Operation* op, mlir::ValueRange inputs, bool donated,
      llvm::SmallVectorImpl<ArrayView>& in_views) {
    in_views.clear();
    llvm::SmallVector<ArrayStateRef, 8> refs;
    if (mlir::failed(ResolveAndDonateInputs(op, inputs, donated,
                                            /*allow_duplicates=*/true, refs))) {
      return mlir::failure();
    }
    if (donated && AreDisjointUnaliasedArrays(refs)) {
      return mlir::success();
    }
    in_views.reserve(inputs.size());
    for (const auto [input_index, input] : llvm::enumerate(inputs)) {
      in_views.push_back(MaterializeShards(input, refs[input_index]));
    }
    return mlir::success();
  }

 private:
  // Handle to an array's backing `ArrayState`: either one inline `ArrayState`
  // (`num_shards == kInlineArray`) or a slice `[index, index + num_shards)` in
  // `shard_arena_`.
  struct ArrayStateRef {
    int32_t index;
    int32_t num_shards;
  };

  // Unified provenance and donation state for either a whole array or a shard
  // (`parent_array_id != kNoArray`).
  struct ArrayState {
    mlir::Operation* donated_by;
    int32_t parent_array_id;
    int32_t first_shard;
    int32_t num_shards;
    bool is_donatable;
  };

  // Registers `array` (if used) as backed by `ref`.
  void RegisterArray(mlir::Value array, ArrayStateRef ref) {
    if (!array.use_empty()) {
      CHECK(array_to_ref_.insert({array, ref}).second);
    }
  }

  // Allocates a whole-array `ArrayState` and registers `array` (if used).
  void RegisterNewArray(mlir::Value array, bool is_donatable) {
    if (!array.use_empty()) {
      const int32_t array_id = AllocateArray(is_donatable);
      CHECK(array_to_ref_.insert({array, ArrayStateRef{array_id, kInlineArray}})
                .second);
    }
  }

  // Looks up the `ArrayStateRef` registered for `array`.
  ArrayStateRef Lookup(mlir::Value array) const {
    auto it = array_to_ref_.find(array);
    CHECK(it != array_to_ref_.end());
    return it->second;
  }

  // Materializes per-device shard `ArrayState`s for `array` (backed by `ref`)
  // if needed and returns an index-based `ArrayView`. Never modifies
  // `shard_arena_`.
  ArrayView MaterializeShards(mlir::Value array, ArrayStateRef ref) {
    if (ref.num_shards != kInlineArray) {
      return ArrayView{ref.index, /*is_contiguous=*/false};
    }
    const size_t num_shards = GetArrayType(array).getDevices().size();
    if (num_shards <= 1) {
      return ArrayView{ref.index, /*is_contiguous=*/true};
    }
    CHECK_GE(ref.index, 0);
    const size_t parent_index = static_cast<size_t>(ref.index);
    CHECK_EQ(arrays_[parent_index].parent_array_id, kNoArray);
    if (arrays_[parent_index].first_shard == kNoArray) {
      const size_t first_shard = arrays_.size();
      CHECK_LE(num_shards,
               static_cast<size_t>(std::numeric_limits<int32_t>::max()) -
                   first_shard);
      mlir::Operation* const donated_by = arrays_[parent_index].donated_by;
      const bool is_donatable = arrays_[parent_index].is_donatable;
      arrays_[parent_index].first_shard = static_cast<int32_t>(first_shard);
      arrays_[parent_index].num_shards = static_cast<int32_t>(num_shards);
      for (size_t i = 0; i < num_shards; ++i) {
        arrays_.push_back(ArrayState{donated_by, ref.index,
                                     /*first_shard=*/kNoArray,
                                     /*num_shards=*/0, is_donatable});
      }
    }
    return ArrayView{arrays_[parent_index].first_shard, /*is_contiguous=*/true};
  }

  // Returns true if every entry in `refs` is backed by a distinct whole
  // `ArrayState` with no shard aliases (so no two inputs share any shards).
  bool AreDisjointUnaliasedArrays(llvm::ArrayRef<ArrayStateRef> refs) const {
    auto is_unaliased_array = [&](ArrayStateRef ref) -> bool {
      if (ref.num_shards != kInlineArray) {
        return false;
      }
      CHECK_GE(ref.index, 0);
      const ArrayState& array = arrays_[static_cast<size_t>(ref.index)];
      return array.parent_array_id == kNoArray && array.first_shard == kNoArray;
    };
    if (refs.size() == 1) {
      return is_unaliased_array(refs.front());
    }
    llvm::SmallDenseSet<int32_t, 8> seen_array_ids;
    for (const ArrayStateRef ref : refs) {
      if (!is_unaliased_array(ref) ||
          !seen_array_ids.insert(ref.index).second) {
        return false;
      }
    }
    return true;
  }

  // Resolves each input in `inputs` once into `refs`, verifies that no shard
  // has already been donated, and marks all inputs as donated by `op` when
  // `donated` is true (rejecting overlapping inputs within `inputs` when
  // `allow_duplicates` is false).
  mlir::LogicalResult ResolveAndDonateInputs(
      mlir::Operation* op, mlir::ValueRange inputs, bool donated,
      bool allow_duplicates, llvm::SmallVectorImpl<ArrayStateRef>& refs) {
    refs.clear();
    refs.reserve(inputs.size());
    for (const auto [input_index, input] : llvm::enumerate(inputs)) {
      const ArrayStateRef ref = Lookup(input);
      if (mlir::Operation* donated_by = FindDonatedBy(ref);
          donated_by != nullptr) {
        return op->emitOpError()
               << "input #" << input_index << " of op at " << op->getLoc()
               << " was already donated to the op at " << donated_by->getLoc();
      }
      if (donated && !allow_duplicates &&
          mlir::failed(MarkDonated(op, input_index, ref))) {
        return mlir::failure();
      }
      refs.push_back(ref);
    }
    if (donated && allow_duplicates) {
      for (const auto [input_index, ref] : llvm::enumerate(refs)) {
        if (mlir::failed(MarkDonated(op, input_index, ref))) {
          return mlir::failure();
        }
      }
    }
    return mlir::success();
  }

  // Returns the donating `Operation*` for `array_id`, or `nullptr` if not yet
  // donated.
  mlir::Operation* GetDonatedBy(int32_t array_id) const {
    CHECK_GE(array_id, 0);
    return arrays_[static_cast<size_t>(array_id)].donated_by;
  }

  // Returns an `Operation*` that already donated any shard of `ref`, or
  // `nullptr` if none have been donated yet.
  mlir::Operation* FindDonatedBy(ArrayStateRef ref) const {
    if (ref.num_shards == kInlineArray) {
      return GetDonatedBy(ref.index);
    }
    for (const int32_t array_id : GetShardArenaIds(ref)) {
      if (mlir::Operation* donated_by = GetDonatedBy(array_id);
          donated_by != nullptr) {
        return donated_by;
      }
    }
    return nullptr;
  }

  // Verifies that `ref` does not originate from an undonated program input and
  // marks all backing `ArrayState`s as donated by `op`.
  mlir::LogicalResult MarkDonated(mlir::Operation* op, size_t input_index,
                                  ArrayStateRef ref) {
    if (ref.num_shards == kInlineArray) {
      return MarkArrayDonated(op, input_index, ref.index);
    }
    for (const int32_t array_id : GetShardArenaIds(ref)) {
      if (mlir::failed(MarkArrayDonated(op, input_index, array_id))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  // Appends a new `ArrayState` to `arrays_` and returns its ID.
  int32_t AllocateArray(bool is_donatable) {
    const size_t array_id = arrays_.size();
    CHECK_LT(array_id,
             static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    arrays_.push_back(ArrayState{/*donated_by=*/nullptr,
                                 /*parent_array_id=*/kNoArray,
                                 /*first_shard=*/kNoArray,
                                 /*num_shards=*/0, is_donatable});
    return static_cast<int32_t>(array_id);
  }

  // Returns the `shard_arena_` slice backing a non-inline `ref`.
  llvm::ArrayRef<int32_t> GetShardArenaIds(ArrayStateRef ref) const {
    CHECK_GE(ref.index, 0);
    CHECK_GE(ref.num_shards, 0);
    return llvm::ArrayRef<int32_t>(
        shard_arena_.data() + static_cast<size_t>(ref.index),
        static_cast<size_t>(ref.num_shards));
  }

  // Verifies that `array_id` does not originate from an undonated program input
  // and marks `array_id` as donated by `op`, propagating to any split shards
  // or parent summary state.
  mlir::LogicalResult MarkArrayDonated(mlir::Operation* op, size_t input_index,
                                       int32_t array_id) {
    CHECK_GE(array_id, 0);
    ArrayState& array = arrays_[static_cast<size_t>(array_id)];
    if (!array.is_donatable) {
      return op->emitOpError() << "input #" << input_index
                               << " has not been donated to the program.";
    }
    array.donated_by = op;
    if (array.first_shard != kNoArray) {
      CHECK_GE(array.first_shard, 0);
      CHECK_GE(array.num_shards, 0);
      const size_t first_shard = static_cast<size_t>(array.first_shard);
      const size_t num_shards = static_cast<size_t>(array.num_shards);
      for (size_t i = 0; i < num_shards; ++i) {
        arrays_[first_shard + i].donated_by = op;
      }
    }
    if (array.parent_array_id != kNoArray) {
      CHECK_GE(array.parent_array_id, 0);
      ArrayState& parent = arrays_[static_cast<size_t>(array.parent_array_id)];
      if (parent.donated_by == nullptr) {
        parent.donated_by = op;
      }
    }
    return mlir::success();
  }

  std::vector<ArrayState> arrays_;
  std::vector<int32_t> shard_arena_;
  llvm::DenseMap<mlir::Value, ArrayStateRef> array_to_ref_;
};

// Verifies input donations and `io_aliases` for `IfrtCallOpInterface` ops. For
// these ops, `donated_input_indices` and `io_aliases` have identical
// input-consumption semantics: a donated or aliased input array cannot be used
// after the call, and overlapping shards (including the same SSA input array)
// cannot be donated or aliased more than once or also passed as a non-donated
// input to the same call.
mlir::LogicalResult VerifyCallOpAliasesAndDonations(IfrtCallOpInterface op,
                                                    DonationTracker& tracker) {
  auto emit_already_donated =
      [&](size_t input_index,
          mlir::Operation* donated_by) -> mlir::LogicalResult {
    return op.emitOpError()
           << "input #" << input_index << " of " << op.getCalleeAttr()
           << " was already donated or aliased to the op at "
           << donated_by->getLoc();
  };

  const size_t num_inputs_size = op.getInputs().size();
  CHECK_LE(num_inputs_size,
           static_cast<size_t>(std::numeric_limits<unsigned>::max()));
  const unsigned num_inputs = static_cast<unsigned>(num_inputs_size);
  llvm::SmallBitVector donated_or_aliased_inputs(num_inputs);
  for (const int32_t input_index : op.getDonatedInputIndices()) {
    CHECK_GE(input_index, 0);
    CHECK_LT(static_cast<unsigned>(input_index), num_inputs);
    donated_or_aliased_inputs.set(static_cast<unsigned>(input_index));
  }
  for (const IoAlias& io_alias : op.getIoAliasPairs()) {
    CHECK_GE(io_alias.input_index, 0);
    CHECK_LT(static_cast<unsigned>(io_alias.input_index), num_inputs);
    donated_or_aliased_inputs.set(static_cast<unsigned>(io_alias.input_index));
  }

  for (const unsigned input_index : donated_or_aliased_inputs.set_bits()) {
    mlir::Operation* donated_by = nullptr;
    if (mlir::failed(tracker.DonateInput(op.getOperation(), input_index,
                                         op.getInputs()[input_index],
                                         donated_by))) {
      return donated_by != nullptr
                 ? emit_already_donated(input_index, donated_by)
                 : mlir::failure();
    }
  }

  // Verify non-donated, non-aliased inputs after donated and aliased inputs
  // have been marked so that intra-call conflicts (e.g. `ifrt.Call(%0, %0)`
  // with one input donated or aliased) are also caught.
  for (unsigned input_index = 0; input_index < num_inputs; ++input_index) {
    if (!donated_or_aliased_inputs.test(input_index)) {
      if (mlir::Operation* donated_by =
              tracker.FindDonatedBy(op.getInputs()[input_index]);
          donated_by != nullptr) {
        return emit_already_donated(input_index, donated_by);
      }
    }
  }

  tracker.RegisterArrays(op.getOutputs());
  return mlir::success();
}

// Verifies donations for `BitcastArraysOp`. `BitcastArraysOp` destructively
// consumes each input buffer and therefore disallows donating overlapping
// shards more than once within the same op.
mlir::LogicalResult VerifyBitcastArraysOp(BitcastArraysOp op,
                                          DonationTracker& tracker) {
  CHECK(op.getDonated());
  if (mlir::failed(tracker.DonateInputs(op, op.getInputs(), /*donated=*/true,
                                        /*allow_duplicates=*/false))) {
    return mlir::failure();
  }
  tracker.RegisterArrays(op.getOutputs());
  return mlir::success();
}

// Verifies donations for `ReshardOp`. Duplicate inputs within the same
// `ReshardOp` are permitted when `donated=true`; outputs always own shards.
mlir::LogicalResult VerifyReshardOp(ReshardOp op, DonationTracker& tracker) {
  if (mlir::failed(tracker.DonateInputs(op, op.getInputs(), op.getDonated(),
                                        /*allow_duplicates=*/true))) {
    return mlir::failure();
  }
  tracker.RegisterArrays(op.getOutputs());
  return mlir::success();
}

// Verifies donations and per-shard buffer reuse aliasing for `CopyArraysOp`.
// In-place buffer reuse occurs on each shard where input and output devices
// and effective memory kind match, for both `reuse=true` (output shard aliases
// input shard) and `donated=true` (overlapping donated inputs share the reused
// output shard).
mlir::LogicalResult VerifyCopyArraysOp(CopyArraysOp op,
                                       DonationTracker& tracker) {
  const bool donated = op.getDonated();
  const bool reuse = op.getReuse();
  // `CopyArraysOp::verify()` guarantees all inputs share the same devices and
  // memory kind, and all outputs share the same devices and memory kind.
  const IfrtArrayType first_in_array = GetArrayType(op.getInputs().front());
  const IfrtArrayType first_out_array = GetArrayType(op.getOutputs().front());
  const llvm::ArrayRef<int> in_devices = first_in_array.getDevices();
  const llvm::ArrayRef<int> out_devices = first_out_array.getDevices();
  const bool same_memory_kind =
      (donated || reuse) && HasSameMemoryKind(first_in_array, first_out_array);
  const bool all_devices_equal =
      same_memory_kind &&
      first_in_array.getDevicesAttr() == first_out_array.getDevicesAttr();

  if (!donated && all_devices_equal) {
    if (mlir::failed(tracker.DonateInputs(op, op.getInputs(), /*donated=*/false,
                                          /*allow_duplicates=*/true))) {
      return mlir::failure();
    }
    for (const auto [in_array, out_array] :
         llvm::zip_equal(op.getInputs(), op.getOutputs())) {
      tracker.RegisterArrayAlias(in_array, out_array);
    }
    return mlir::success();
  }

  bool any_device_equal = all_devices_equal;
  if (same_memory_kind && !any_device_equal) {
    const size_t min_shards = std::min(in_devices.size(), out_devices.size());
    for (size_t shard_index = 0; shard_index < min_shards; ++shard_index) {
      if (in_devices[shard_index] == out_devices[shard_index]) {
        any_device_equal = true;
        break;
      }
    }
  }

  if (!any_device_equal) {
    if (mlir::failed(tracker.DonateInputs(op, op.getInputs(), donated,
                                          /*allow_duplicates=*/true))) {
      return mlir::failure();
    }
    tracker.RegisterArrays(op.getOutputs());
    return mlir::success();
  }

  llvm::SmallVector<DonationTracker::ArrayView, 4> in_views;
  if (mlir::failed(tracker.VerifyAndMaterializeInputs(op, op.getInputs(),
                                                      donated, in_views))) {
    return mlir::failure();
  }
  if (in_views.empty()) {
    tracker.RegisterArrays(op.getOutputs());
    return mlir::success();
  }

  llvm::SmallVector<llvm::MutableArrayRef<int32_t>, 4> out_shards_list =
      tracker.AllocateOutputShards(op.getOutputs());
  llvm::DenseMap<int32_t, int32_t> copied_donated_shard_id;
  for (const auto [in_view, out_shards] :
       llvm::zip_equal(in_views, out_shards_list)) {
    for (size_t shard_index = 0; shard_index < out_shards.size();
         ++shard_index) {
      if (shard_index < in_devices.size() &&
          in_devices[shard_index] == out_devices[shard_index]) {
        int32_t dst_array_id = tracker.GetArrayIdForShard(in_view, shard_index);
        if (donated) {
          auto [it, inserted] =
              copied_donated_shard_id.insert({dst_array_id, kNoArray});
          if (inserted) {
            it->second = tracker.AllocateShard();
          }
          dst_array_id = it->second;
        }
        out_shards[shard_index] = dst_array_id;
      } else {
        out_shards[shard_index] = tracker.AllocateShard();
      }
    }
  }
  return mlir::success();
}

// Verifies donations and shard-level aliasing for `RemapArraysOp`.
mlir::LogicalResult VerifyRemapArraysOp(RemapArraysOp op,
                                        DonationTracker& tracker) {
  const bool donated = op.getDonated();
  llvm::SmallVector<DonationTracker::ArrayView, 4> in_views;
  if (mlir::failed(tracker.VerifyAndMaterializeInputs(op, op.getInputs(),
                                                      donated, in_views))) {
    return mlir::failure();
  }
  if (in_views.empty()) {
    // `RemapArraysOp::verify()` guarantees each shard within an input array is
    // mapped at most once. When all inputs are disjoint unaliased arrays,
    // every output shard maps from a unique physical input shard, so outputs
    // own pairwise-disjoint buffers without walking `mappings`.
    tracker.RegisterArrays(op.getOutputs());
    return mlir::success();
  }

  llvm::SmallVector<llvm::MutableArrayRef<int32_t>, 4> out_shards_list =
      tracker.AllocateOutputShards(op.getOutputs());

  llvm::DenseMap<int32_t, int32_t> remapped_donated_shard_id;
  for (const IfrtArrayMappingAttr array_mapping :
       op.getMappings().getAsRange<IfrtArrayMappingAttr>()) {
    CHECK_GE(array_mapping.getInArrayIndex(), 0);
    CHECK_GE(array_mapping.getOutArrayIndex(), 0);
    const DonationTracker::ArrayView in_view =
        in_views[static_cast<size_t>(array_mapping.getInArrayIndex())];
    llvm::MutableArrayRef<int32_t> out_shards =
        out_shards_list[static_cast<size_t>(array_mapping.getOutArrayIndex())];
    for (const IfrtMappingAttr mapping :
         array_mapping.getMappings().getAsRange<IfrtMappingAttr>()) {
      const IfrtIntervalAttr in_interval = mapping.getFromShards();
      const IfrtIntervalAttr out_interval = mapping.getToShards();
      int64_t in_shard = in_interval.getStart();
      int64_t out_shard = out_interval.getStart();
      while (in_shard < in_interval.getEnd()) {
        CHECK_GE(in_shard, 0);
        CHECK_GE(out_shard, 0);
        CHECK_LE(static_cast<uint64_t>(in_shard),
                 static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
        CHECK_LE(static_cast<uint64_t>(out_shard),
                 static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
        int32_t dst_array_id =
            tracker.GetArrayIdForShard(in_view, static_cast<size_t>(in_shard));
        if (donated) {
          auto [it, inserted] =
              remapped_donated_shard_id.insert({dst_array_id, kNoArray});
          if (inserted) {
            it->second = tracker.AllocateShard();
          }
          dst_array_id = it->second;
        }
        out_shards[static_cast<size_t>(out_shard)] = dst_array_id;
        in_shard += in_interval.getStep();
        out_shard += out_interval.getStep();
      }
    }
  }
  return mlir::success();
}

// Verifies donations and per-shard aliasing for `AssembleOp`.
mlir::LogicalResult VerifyAssembleOp(AssembleOp op, DonationTracker& tracker) {
  return tracker.AssembleInputs(op, op.getInputs(), op.getOutput(),
                                op.getDonated());
}

// Verifies donations and per-shard aliasing for `DisassembleOp`.
mlir::LogicalResult VerifyDisassembleOp(DisassembleOp op,
                                        DonationTracker& tracker) {
  if (mlir::failed(tracker.DonateInputs(op, op.getInput(), op.getDonated(),
                                        /*allow_duplicates=*/false))) {
    return mlir::failure();
  }
  if (op.getDonated()) {
    tracker.RegisterArrays(op.getOutputs());
    return mlir::success();
  }
  const DonationTracker::ArrayView in_view =
      tracker.MaterializeShards(op.getInput());
  for (const auto [output_index, output] : llvm::enumerate(op.getOutputs())) {
    tracker.RegisterShardAlias(
        output, tracker.GetArrayIdForShard(in_view, output_index));
  }
  return mlir::success();
}

// Verifies that no returned array operand has already been donated.
mlir::LogicalResult VerifyReturnOp(mlir::func::ReturnOp return_op,
                                   const DonationTracker& tracker) {
  for (const auto [result_index, result] :
       llvm::enumerate(return_op.getOperands())) {
    if (mlir::isa<IfrtArrayType>(result.getType())) {
      if (mlir::Operation* donated_by = tracker.FindDonatedBy(result);
          donated_by != nullptr) {
        return return_op.emitOpError()
               << "result #" << result_index << " of op at "
               << return_op.getLoc() << " was already donated to the op at "
               << donated_by->getLoc();
      }
    }
  }
  return mlir::success();
}

// Verifies that no array shard is donated or used after being donated, and that
// all donated shards originating from inputs to the IFRT `FuncOp` have the
// `ifrt.donated` attribute set.
class IfrtVerifyDonationPass
    : public impl::IfrtVerifyDonationPassBase<IfrtVerifyDonationPass> {
 public:
  void runOnOperation() override;
};

void IfrtVerifyDonationPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!IsIfrtFunction(func_op)) {
    return;
  }

  DonationTracker tracker(func_op);
  for (mlir::Operation& op : func_op.getOps()) {
    const mlir::LogicalResult op_result =
        llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(&op)
            .Case<IfrtCallOpInterface>([&](IfrtCallOpInterface call_op) {
              return VerifyCallOpAliasesAndDonations(call_op, tracker);
            })
            .Case<BitcastArraysOp>([&](BitcastArraysOp bitcast_op) {
              return VerifyBitcastArraysOp(bitcast_op, tracker);
            })
            .Case<ReshardOp>([&](ReshardOp reshard_op) {
              return VerifyReshardOp(reshard_op, tracker);
            })
            .Case<CopyArraysOp>([&](CopyArraysOp copy_op) {
              return VerifyCopyArraysOp(copy_op, tracker);
            })
            .Case<RemapArraysOp>([&](RemapArraysOp remap_op) {
              return VerifyRemapArraysOp(remap_op, tracker);
            })
            .Case<AssembleOp>([&](AssembleOp assemble_op) {
              return VerifyAssembleOp(assemble_op, tracker);
            })
            .Case<DisassembleOp>([&](DisassembleOp disassemble_op) {
              return VerifyDisassembleOp(disassemble_op, tracker);
            })
            .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp return_op) {
              return VerifyReturnOp(return_op, tracker);
            })
            .Default(mlir::success());

    if (mlir::failed(op_result)) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
