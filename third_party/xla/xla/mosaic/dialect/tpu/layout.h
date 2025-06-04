/* Copyright 2023 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>

#include "absl/log/check.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tpu {

// TODO(apaszke): Optimize this to encode the optional in the value itself
// and use a narrower type.
// An offset is nullopt when the value is replicated along sublanes or lanes.
using LayoutOffset = std::optional<int64_t>;
using LayoutOffsets = std::array<LayoutOffset, 2>;

enum class Direction { kSublanes, kLanes, kSubelements };

struct VRegDataBounds {
  // TODO(tlongeri): Should get{Vector, Sublane}Mask take a Location?
  virtual ~VRegDataBounds() = default;
  // Determines whether all indices along a direction contain useful data.
  virtual bool maskVariesAlong(Direction direction,
                               std::array<int64_t, 2> target_shape) const = 0;

  bool isComplete(const std::array<int64_t, 2> target_shape) const {
    return !maskVariesAlong(Direction::kSublanes, target_shape) &&
           !maskVariesAlong(Direction::kLanes, target_shape) &&
           !maskVariesAlong(Direction::kSubelements, target_shape);
  }

  // Constructs a vector mask value that is true iff the entry contains useful
  // data.
  //
  // The returned value can be an int32 bitmask too, when the target does not
  // have sufficiently expressive vector masks.
  //
  // Args:
  //   generation: The target TPU generation.
  virtual FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder &builder, Location loc, int generation,
      std::array<int64_t, 2> target_shape) const = 0;

  // Constructs a DenseBoolArrayAttr containing a sublane mask for the vreg.
  //
  // The sublane mask should never have true for sublanes that do not contain
  // useful data, but having an unmasked sublane doesn't imply that all bits
  // in that sublane are used to represent data (relevant for packed layouts).
  virtual DenseBoolArrayAttr getSublaneMask(
      MLIRContext *ctxt, std::array<int64_t, 2> target_shape) const = 0;
};

// Represents a rectangular region of data within a vector register.
//
// This class is very limited in its power and should only be used for 32-bit
// values with native tiling.
//
// Attributes:
//   bounds: A TargetTuple of slices encoding the bounds of the rectangular
//     data region.
// TODO(tlongeri): Can this be removed in favor of the more general
//  TiledRectangularVregBounds?
class RectangularVregBounds : public VRegDataBounds {
 public:
  RectangularVregBounds(const std::array<int64_t, 2> starts,
                        const std::array<int64_t, 2> ends)
      : starts_(starts), ends_(ends) {}

  // See base class.
  bool maskVariesAlong(Direction direction,
                       std::array<int64_t, 2> target_shape) const override;

  // See base class.
  FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder &builder, Location loc, int generation,
      std::array<int64_t, 2> target_shape) const override;

  // See base class.
  DenseBoolArrayAttr getSublaneMask(
      MLIRContext *mlir_ctxt,
      std::array<int64_t, 2> target_shape) const override;

 private:
  std::array<int64_t, 2> starts_;
  std::array<int64_t, 2> ends_;
};

// VectorLayout describes a mapping of an arbitrarily sized values into vregs.
//
// First, let us consider the simplest case, when implicit_dim is None, bitwidth
// is 32, and tiling matches the vreg shape. Then, the two last dimensions of a
// vector are tiled over sublanes and lanes respectively. If a value is too
// large to fit within a single vreg, then it continues in another vector
// register. For example purposes, we assume that vregs have 4 sublanes and 5
// lanes from now on. A matrix with elements:
//
//   a b c d e
//   f g h i j
//   k l m n o
//   p q r s t
//
// laid out with offsets (1, 2) will use four vregs as follows:
//
//   vreg 1      vreg 2
// . . . . .    . . . . .
// . . a b c    d e . . .
// . . f g h    i j . . .
// . . k l m    n o . . .
//
//   vreg 3      vreg 4
// . . p q r    s t . . .
// . . . . .    . . . . .
// . . . . .    . . . . .
// . . . . .    . . . . .
//
// The dot character indicates padding. Nothing should be assumed about the
// value of those entries.
//
// If a value with this layout has rank >2, the leading dimensions will be
// unrolled over vregs. That is, the total number of vregs used to represent
// a value is equal to the product of all leading dimension sizes, and the
// number of vregs necessary to lay out the last two dimensions (as in the
// example).
//
// ---
//
// The implicit_dim attribute makes it possible to tile only the last dimension
// of a value, by implicitly inserting a singleton dimension that is tiled over
// sublanes (when implicit_dim is kMinor) or lanes (when implicit_dim is
// kSecondMinor).
//
// When the value has only one dimension, implicit_dim must be specified.
//
// ---
//
// The tiling attribute makes it possible to subdivide a single vector register
// into multiple sub-tiles that traverse the last dimension of a value. For
// example, consider vregs of shape (4, 5) on (2, 10) array:
//
//   a b c d e f g h i j
//   k l m n o p q r s t
//
// If we used a tiling of (4, 5), we would need two vregs to store this value,
// with the lower half of every register containing padding. But, if we use a
// tiling of (2, 5), both tiles fit into a single vreg:
//
//   vreg 0
// a b c d e | tile 0
// k l m n o |
// f g h i j    | tile 1
// p q r s t    |
//
// Tiling is especially useful for compact storage of 1D values. Without it,
// we could use at most one sublane of every vector register. But, with a tiling
// of (1, 128) and implicit_dim being kSecondMinor, we can use all entries in a
// register to store long vectors.
//
// ---
//
// Finally, when the element bitwidth becomes smaller than 32, we use a two
// level tiling scheme, where elements of consecutive rows are packed into
// subelements. In TPU documentation this is often called a compressed layout.
// Note that this puts restrictions on the tile sizes, as they cannot have fewer
// rows than the packing factor (32 / bitwidth).
//
// Attributes:
//   bitwidth: The bitwidth of the stored values.
//   offsets: The coordinates of the first valid element. If an offset is
//     replicated (nullopt), then any offset is valid as the value does not vary
//     across sublanes or lanes respectively.
//   tiling: The tiling used to lay out values (see the XLA docs). For values of
//     bitwidth < 32, an implicit (32 / bitwidth, 1) tiling is appended to the
//     one specified as an attribute.
//   implicit_dim: If specified, the value has an implicit dim inserted in
//     either minormost or second minormost position.
//
// Note: There is a special case when VectorLayout is used for an mlir::Value
// of i1 type. In this case, we use it to represent a vmask, which has a smaller
// bitwidth than a vreg. For these types, the packing() is accurate but the
// bitwidth() is a lie, and the i1 value is replicated for every bit.
// For example, if the vmask is 8 x 128 x 4 bits and packing() == 2, each 4-bit
// register contains two logical bool values which are represented as either b11
// or b00. Its usage is currently limited to MLIR arith.cmp and arith.select ops
// but we might want to split out a separate class if it gets used more widely.
class VectorLayout {
 public:
  enum class ImplicitDim {
    kNone = 0,  // To make if (implicit_dim) work.
    // Also want to do dims[dims.size() - xla::to_underlying(implicit_dim)]
    kMinor = 1,
    kSecondMinor = 2,
  };
  VectorLayout(const int8_t bitwidth, const LayoutOffsets offsets,
               const std::array<int64_t, 2> tiling,
               const ImplicitDim implicit_dim = ImplicitDim::kNone)
      : offsets_(offsets),
        tiling_(tiling),
        bitwidth_(bitwidth),
        implicit_dim_(implicit_dim) {
    // TODO(b/275751535): Allow more bitwidths.
    CHECK(llvm::has_single_bit<unsigned>(bitwidth_) && bitwidth_ <= 32);
    CHECK_GT(tiling_[0], 0);
    CHECK_GT(tiling_[1], 0);
    CHECK_GE(offsets_[0].value_or(0), 0);
    CHECK_GE(offsets_[1].value_or(0), 0);
    CHECK_LT(offsets_[0].value_or(0), tiling_[0]);
  }

  static int num_implicit_dims(const ImplicitDim implicit_dim) {
    switch (implicit_dim) {
      case ImplicitDim::kNone:
        return 0;
      case ImplicitDim::kMinor:
      case ImplicitDim::kSecondMinor:
        return 1;
    }
  }

  // The number of non-implicit dimensions that are tiled.
  static int layout_rank(const ImplicitDim implicit_dim) {
    return 2 - num_implicit_dims(implicit_dim);
  }

  int8_t bitwidth() const { return bitwidth_; }
  const LayoutOffsets &offsets() const { return offsets_; }
  const LayoutOffsets getCanonicalOffsets(
      const ArrayRef<int64_t> shape,
      const std::array<int64_t, 2> target_shape) const {
    // For (1, n) tiling with a single row, 2nd minor replication does not
    // change anything about the layout - it is equivalent to an offset of 0.
    // We choose a replicated offset as "canonical".
    const std::array<int64_t, 2> tiled_ishape = getImplicitTiledDims(shape, 1);
    return {
        (tiling_[0] == 1 && tiled_ishape[0] == 1) ? std::nullopt : offsets_[0],
        offsets_[1]};
  }
  const std::array<int64_t, 2> &tiling() const { return tiling_; }
  ImplicitDim implicit_dim() const { return implicit_dim_; }
  int packing() const { return 32 / bitwidth_; }
  int num_implicit_dims() const { return num_implicit_dims(implicit_dim_); }
  int layout_rank() const { return layout_rank(implicit_dim_); }

  bool operator==(const VectorLayout &other) const;
  bool operator!=(const VectorLayout &other) const { return !(*this == other); }

  static int64_t tilesPerVreg(const std::array<int64_t, 2> target_shape,
                              const int8_t bitwidth,
                              const std::array<int64_t, 2> tiling) {
    CHECK_NE(0, bitwidth) << "bitwidth cannot be 0";
    const int64_t tile_elems = tiling[0] * tiling[1];
    const int64_t vreg_capacity =
        (32 / bitwidth) * target_shape[0] * target_shape[1];
    const auto [tiles_per_vreg, rem] = std::div(vreg_capacity, tile_elems);
    CHECK_EQ(rem, 0);
    return tiles_per_vreg;
  }
  // How many tiles fit in each vector register.
  int64_t tilesPerVreg(const std::array<int64_t, 2> target_shape) const {
    return VectorLayout::tilesPerVreg(target_shape, bitwidth_, tiling_);
  }

  int64_t sublanesPerTile(const std::array<int64_t, 2> target_shape) const {
    auto [sublanes_per_tile, rem] =
        std::div(target_shape[0], tilesPerVreg(target_shape));
    CHECK_EQ(rem, 0);
    return sublanes_per_tile;
  }

  // Returns the size of a window contained in a single vreg.
  //
  // We never reuse the same vector register to store data of multiple rows,
  // so only the minormost dimension can increase.
  static std::array<int64_t, 2> vregSlice(std::array<int64_t, 2> target_shape,
                                          const int8_t bitwidth,
                                          const std::array<int64_t, 2> tiling) {
    return {
        tiling[0],
        VectorLayout::tilesPerVreg(target_shape, bitwidth, tiling) * tiling[1]};
  }

  std::array<int64_t, 2> vregSlice(std::array<int64_t, 2> target_shape) const {
    return VectorLayout::vregSlice(target_shape, bitwidth_, tiling_);
  }

  template <typename T>
  void insertImplicit(SmallVectorImpl<T> &vec, T value) const {
    CHECK_GE(vec.size(), layout_rank());
    switch (implicit_dim_) {
      case ImplicitDim::kNone:
        break;
      case ImplicitDim::kMinor:
      case ImplicitDim::kSecondMinor:
        vec.insert(vec.end() - (static_cast<int64_t>(implicit_dim_) - 1),
                   value);
        break;
    }
  }

  template <typename T>
  void eraseImplicit(SmallVectorImpl<T> &vec) const {
    CHECK_GE(vec.size(), 2);
    switch (implicit_dim_) {
      case ImplicitDim::kNone:
        break;
      case ImplicitDim::kMinor:
      case ImplicitDim::kSecondMinor:
        vec.erase(vec.end() - static_cast<int64_t>(implicit_dim_));
        break;
    }
  }

  static std::array<int64_t, 2> getImplicitTiledDims(
      const ImplicitDim implicit_dim, const ArrayRef<int64_t> arr,
      const int64_t implicit_value) {
    CHECK_GE(arr.size(), layout_rank(implicit_dim));
    switch (implicit_dim) {
      case ImplicitDim::kNone:
        return {*(arr.end() - 2), *(arr.end() - 1)};
      case ImplicitDim::kMinor:
        return {*(arr.end() - 1), implicit_value};
      case ImplicitDim::kSecondMinor:
        return {implicit_value, *(arr.end() - 1)};
    }
  }

  // Returns the value of the tiled (2 minormost) dimensions of the given array
  // with implicit dims inserted.
  //
  // Roughly equivalent to the following (but avoids vector allocation):
  //
  //   SmallVector<int64_t> vec = arr;
  //   insertImplicit(arr, implicit_value);
  //   return {*(vec.end() - 2), *(vec.end() - 1)};
  std::array<int64_t, 2> getImplicitTiledDims(
      const ArrayRef<int64_t> arr, const int64_t implicit_value) const {
    return getImplicitTiledDims(implicit_dim_, arr, implicit_value);
  }

  SmallVector<int64_t> implicitShape(ArrayRef<int64_t> shape) const;

  // Returns the shape of ndarray of vregs needed to represent a value.
  //
  // All but the last two dimensions are unrolled over vregs. In the last two
  // dims we need as many vregs as indicated by dividing the point at which
  // the value ends (given by the start offset plus the dim size) divided by
  // the respective vreg capacity in that dim (and a ceiling if non-integral).
  // If a value is replicated, then any offset is valid and we pick 0 to
  // minimize the number of vregs.
  //
  // Args:
  //   src_is_implicit: If true, the input shape already has implicit dimensions
  //     inserted.
  //   res_is_implicit: If true, the output shape will have implicit dimensions
  //     inserted.
  //   shape: The shape of the full vector this layout applies to, possibly
  //     with implicit dimensions inserted.
  SmallVector<int64_t> tileArrayShape(
      bool src_is_implicit, bool res_is_implicit, ArrayRef<int64_t> shape,
      std::array<int64_t, 2> target_shape) const {
    return tileArrayShape(src_is_implicit, res_is_implicit,
                          SmallVector<int64_t>(shape), target_shape);
  }
  SmallVector<int64_t> tileArrayShape(
      bool src_is_implicit, bool res_is_implicit, SmallVector<int64_t> &&shape,
      std::array<int64_t, 2> target_shape) const;

  SmallVector<int64_t> tileArrayImplicitShape(
      ArrayRef<int64_t> shape, std::array<int64_t, 2> target_shape) const {
    return tileArrayShape(false, true, shape, target_shape);
  }

  SmallVector<int64_t> tileArrayShape(
      ArrayRef<int64_t> shape, std::array<int64_t, 2> target_shape) const {
    return tileArrayShape(false, false, shape, target_shape);
  }

  // Returns the bounds of the given tile that hold useful data.
  //
  // Arguments:
  //   full_shape: The shape of the full vector this layout applies to.
  //   ixs: The indices into an array of tiles representing the full vector
  //     (see tile_array_shape for bounds) selecting the tile for which the
  //     bounds are queried.
  //   allow_replicated: If False, no offset is allowed to be replicated. If
  //     True, offsets are allowed to be replicated, but the bounds will span
  //     the full dimension of the tile (i.e. potentially multiple repeats of
  //     the actual data).
  //
  // Returns:
  //   A TargetTuple of slices, indicating the span of useful data within the
  //   tile selected by idx.
  std::unique_ptr<VRegDataBounds> tileDataBounds(
      MLIRContext *mlir_ctxt, ArrayRef<int64_t> full_shape,
      ArrayRef<int64_t> idxs, std::array<int64_t, 2> target_shape,
      std::array<bool, 2> allow_replicated) const;
  std::unique_ptr<VRegDataBounds> tileDataBounds(
      MLIRContext *mlir_ctxt, ArrayRef<int64_t> full_shape,
      ArrayRef<int64_t> idxs, std::array<int64_t, 2> target_shape,
      bool allow_replicated = false) const {
    return tileDataBounds(mlir_ctxt, full_shape, idxs, target_shape,
                          {allow_replicated, allow_replicated});
  }

  // True if every vector register has a layout without jumps.
  //
  // By without jumps we mean that traversing vregs over (sub)lanes always leads
  // to a contiguous traversal of the (second) minormost dimension of data. This
  // is only true for 32-bit types, since narrower types use two level tiling.
  bool hasNaturalTopology(const std::array<int64_t, 2> target_shape) const {
    return bitwidth_ == 32 && llvm::equal(tiling_, target_shape) &&
           implicit_dim_ == ImplicitDim::kNone;
  }
  // True if every vector register has a natural "packed" topology.
  //
  // This is equivalent to has_natural_topology for 32-bit types, but
  // generalizes it to narrower values with packed layouts too.
  bool hasNativeTiling(std::array<int64_t, 2> target_shape) const;

  // Returns true if the other layout is a special case of this one.
  //
  // In here, other is considered "a special case" when the set of vector
  // register entries that represent a value in that layout is also the set of
  // entries in which this stores the value. This is of course true for layouts
  // that are equivalent, but it does not need to hold both ways. For example,
  // a layout that implies the value does not change along an axis of the vector
  // register is more general than the layout that picks a fixed starting point
  // for the value and does not encode that assumption.
  //
  // The generalization relation is a non-strict partial order. You can think of
  // it as a partial <= on vector layouts, but we don't overload operators since
  // there's no clear way to decide where the bottom and top should be.
  //
  // Args:
  //   other: The layout compared against this.
  //   shape: A optional shape of the vector to which both layouts apply.
  //     If shape.data() == nullptr, then return whether it generalizes across
  //     all shapes.
  //     The generalization relation is larger than usual for some shapes. That
  //     is, if generalizes(other) then also generalizes(other, shape) for any
  //     shape, but that implication does not hold the other way around for some
  //     shapes.
  bool generalizes(const VectorLayout &other, ArrayRef<int64_t> shape,
                   std::array<int64_t, 2> target_shape) const;

  // Returns True if the two layouts are equivalent.
  //
  // That is, when all potential vector entries where the value can be stored
  // (there might be multiple choices for some layouts!) are equal in both
  // self and other.
  //
  // Args:
  //   other: The layout compared against self.
  //   shape: An optional shape of the vector to which both layouts apply. More
  //     layouts are considered equivalent when the shape is specified. Also see
  //     the docstring of the generalizes method.
  bool equivalentTo(const VectorLayout &other, const ArrayRef<int64_t> shape,
                    const std::array<int64_t, 2> target_shape) const {
    return generalizes(other, shape, target_shape) &&
           other.generalizes(*this, shape, target_shape);
  }

  template <typename Stream>
  void print(Stream &os) const;

  static std::optional<VectorLayout> join(const VectorLayout &l,
                                          const VectorLayout &r,
                                          ArrayRef<int64_t> shape);

  static std::optional<VectorLayout> parse(StringRef *data);

  // Check conditions that depend on the target shape. Invariants that are
  // independent of it are checked in the constructor.
  bool isValid(const std::array<int64_t, 2> target_shape) const {
    // Offsets should fall within the vreg slice, or else we have vregs that
    // only contain padding.
    for (auto [o, vs] : llvm::zip(offsets_, vregSlice(target_shape))) {
      if (o.has_value() && (*o < 0 || vs <= *o)) {
        return false;
      }
    }

    // Tiling should neatly divide the target shape, so that every vector
    // register ends up having the same structure.
    // Also, every tile should occupy a fixed number of sublanes.
    auto [num_sublanes, rem] =
        std::div(tiling_[0] * tiling_[1], packing() * target_shape[1]);
    return rem == 0 && target_shape[0] % num_sublanes == 0;
  }

 private:
  std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
             int8_t, ImplicitDim>
  as_tuple() const;

  friend llvm::hash_code hash_value(const VectorLayout &layout);

  LayoutOffsets offsets_;
  std::array<int64_t, 2> tiling_;
  int8_t bitwidth_;
  ImplicitDim implicit_dim_;
};

using Layout = std::optional<VectorLayout>;
extern const Layout kNoLayout;

std::ostream &operator<<(std::ostream &os, const Layout &v);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Layout &v);
llvm::hash_code hash_value(const VectorLayout &layout);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const Layout &v);
std::ostream &operator<<(std::ostream &os, VectorLayout::ImplicitDim dim);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag,
                             VectorLayout::ImplicitDim dim);

std::optional<Layout> parseLayout(mlir::AsmParser &parser);

}  // namespace mlir::tpu

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
