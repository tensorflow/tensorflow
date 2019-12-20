//===- Region.h - MLIR Region Class -----------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the Region class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGION_H
#define MLIR_IR_REGION_H

#include "mlir/IR/Block.h"

namespace mlir {
class BlockAndValueMapping;

/// This class contains a list of basic blocks and a link to the parent
/// operation it is attached to.
class Region {
public:
  Region() = default;
  explicit Region(Operation *container);
  ~Region();

  /// Return the context this region is inserted in.  The region must have a
  /// valid parent container.
  MLIRContext *getContext();

  /// Return a location for this region. This is the location attached to the
  /// parent container. The region must have a valid parent container.
  Location getLoc();

  using BlockListType = llvm::iplist<Block>;
  BlockListType &getBlocks() { return blocks; }

  // Iteration over the blocks in the region.
  using iterator = BlockListType::iterator;
  using reverse_iterator = BlockListType::reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }

  bool empty() { return blocks.empty(); }
  void push_back(Block *block) { blocks.push_back(block); }
  void push_front(Block *block) { blocks.push_front(block); }

  Block &back() { return blocks.back(); }
  Block &front() { return blocks.front(); }

  /// getSublistAccess() - Returns pointer to member of region.
  static BlockListType Region::*getSublistAccess(Block *) {
    return &Region::blocks;
  }

  /// Return the region containing this region or nullptr if the region is
  /// attached to a top-level operation.
  Region *getParentRegion();

  /// Return the parent operation this region is attached to.
  Operation *getParentOp();

  /// Find the first parent operation of the given type, or nullptr if there is
  /// no ancestor operation.
  template <typename ParentT> ParentT getParentOfType() {
    auto *region = this;
    do {
      if (auto parent = dyn_cast_or_null<ParentT>(region->container))
        return parent;
    } while ((region = region->getParentRegion()));
    return ParentT();
  }

  /// Return the number of this region in the parent operation.
  unsigned getRegionNumber();

  /// Return true if this region is a proper ancestor of the `other` region.
  bool isProperAncestor(Region *other);

  /// Return true if this region is ancestor of the `other` region.  A region
  /// is considered as its own ancestor, use `isProperAncestor` to avoid this.
  bool isAncestor(Region *other) {
    return this == other || isProperAncestor(other);
  }

  /// Clone the internal blocks from this region into dest. Any
  /// cloned blocks are appended to the back of dest. If the mapper
  /// contains entries for block arguments, these arguments are not included
  /// in the respective cloned block.
  void cloneInto(Region *dest, BlockAndValueMapping &mapper);
  /// Clone this region into 'dest' before the given position in 'dest'.
  void cloneInto(Region *dest, Region::iterator destPos,
                 BlockAndValueMapping &mapper);

  /// Takes body of another region (that region will have no body after this
  /// operation completes).  The current body of this region is cleared.
  void takeBody(Region &other) {
    blocks.clear();
    blocks.splice(blocks.end(), other.getBlocks());
  }

  /// Check that this does not use any value defined outside it.
  /// Emit errors if `noteLoc` is provided; this location is used to point
  /// to the operation containing the region, the actual error is reported at
  /// the operation with an offending use.
  bool isIsolatedFromAbove(Optional<Location> noteLoc = llvm::None);

  /// Drop all operand uses from operations within this region, which is
  /// an essential step in breaking cyclic dependences between references when
  /// they are to be deleted.
  void dropAllReferences();

  /// Walk the operations in this region in postorder, calling the callback for
  /// each operation. This method is invoked for void-returning callbacks.
  /// See Operation::walk for more details.
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<std::is_same<RetT, void>::value, RetT>::type
  walk(FnT &&callback) {
    for (auto &block : *this)
      block.walk(callback);
  }

  /// Walk the operations in this region in postorder, calling the callback for
  /// each operation. This method is invoked for interruptible callbacks.
  /// See Operation::walk for more details.
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<std::is_same<RetT, WalkResult>::value, RetT>::type
  walk(FnT &&callback) {
    for (auto &block : *this)
      if (block.walk(callback).wasInterrupted())
        return WalkResult::interrupt();
    return WalkResult::advance();
  }

  /// Displays the CFG in a window. This is for use from the debugger and
  /// depends on Graphviz to generate the graph.
  /// This function is defined in ViewRegionGraph and only works with that
  /// target linked.
  void viewGraph(const Twine &regionName);
  void viewGraph();

private:
  BlockListType blocks;

  /// This is the object we are part of.
  Operation *container;
};

/// This class provides an abstraction over the different types of ranges over
/// Regions. In many cases, this prevents the need to explicitly materialize a
/// SmallVector/std::vector. This class should be used in places that are not
/// suitable for a more derived type (e.g. ArrayRef) or a template range
/// parameter.
class RegionRange
    : public detail::indexed_accessor_range_base<
          RegionRange, PointerUnion<Region *, const std::unique_ptr<Region> *>,
          Region *, Region *, Region *> {
  /// The type representing the owner of this range. This is either a list of
  /// values, operands, or results.
  using OwnerT = PointerUnion<Region *, const std::unique_ptr<Region> *>;

public:
  using RangeBaseT::RangeBaseT;

  RegionRange(MutableArrayRef<Region> regions = llvm::None);

  template <typename Arg,
            typename = typename std::enable_if_t<std::is_constructible<
                ArrayRef<std::unique_ptr<Region>>, Arg>::value>>
  RegionRange(Arg &&arg)
      : RegionRange(ArrayRef<std::unique_ptr<Region>>(std::forward<Arg>(arg))) {
  }
  RegionRange(ArrayRef<std::unique_ptr<Region>> regions);

private:
  /// See `detail::indexed_accessor_range_base` for details.
  static OwnerT offset_base(const OwnerT &owner, ptrdiff_t index);
  /// See `detail::indexed_accessor_range_base` for details.
  static Region *dereference_iterator(const OwnerT &owner, ptrdiff_t index);

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend RangeBaseT;
};

} // end namespace mlir

#endif // MLIR_IR_REGION_H
