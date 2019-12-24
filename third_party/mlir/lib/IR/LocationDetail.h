//===- LocationDetail.h - MLIR Location storage details ---------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of the location attributes.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_IR_LOCATIONDETAIL_H_
#define MLIR_IR_LOCATIONDETAIL_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

namespace detail {

struct CallSiteLocationStorage : public AttributeStorage {
  CallSiteLocationStorage(Location callee, Location caller)
      : callee(callee), caller(caller) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<Location, Location>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(callee, caller);
  }

  /// Construct a new storage instance.
  static CallSiteLocationStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<CallSiteLocationStorage>())
        CallSiteLocationStorage(key.first, key.second);
  }

  Location callee, caller;
};

struct FileLineColLocationStorage : public AttributeStorage {
  FileLineColLocationStorage(Identifier filename, unsigned line,
                             unsigned column)
      : filename(filename), line(line), column(column) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<Identifier, unsigned, unsigned>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(filename, line, column);
  }

  /// Construct a new storage instance.
  static FileLineColLocationStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<FileLineColLocationStorage>())
        FileLineColLocationStorage(std::get<0>(key), std::get<1>(key),
                                   std::get<2>(key));
  }

  Identifier filename;
  unsigned line, column;
};

struct FusedLocationStorage final
    : public AttributeStorage,
      public llvm::TrailingObjects<FusedLocationStorage, Location> {
  FusedLocationStorage(unsigned numLocs, Attribute metadata)
      : numLocs(numLocs), metadata(metadata) {}

  ArrayRef<Location> getLocations() const {
    return ArrayRef<Location>(getTrailingObjects<Location>(), numLocs);
  }

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<Location>, Attribute>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getLocations(), metadata);
  }

  /// Construct a new storage instance.
  static FusedLocationStorage *construct(AttributeStorageAllocator &allocator,
                                         const KeyTy &key) {
    ArrayRef<Location> locs = key.first;

    auto byteSize = totalSizeToAlloc<Location>(locs.size());
    auto rawMem = allocator.allocate(byteSize, alignof(FusedLocationStorage));
    auto result = new (rawMem) FusedLocationStorage(locs.size(), key.second);

    std::uninitialized_copy(locs.begin(), locs.end(),
                            result->getTrailingObjects<Location>());
    return result;
  }

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<FusedLocationStorage, Location>;
  size_t numTrailingObjects(OverloadToken<Location>) const { return numLocs; }

  /// Number of trailing location objects.
  unsigned numLocs;

  /// Metadata used to reason about the generation of this fused location.
  Attribute metadata;
};

struct NameLocationStorage : public AttributeStorage {
  NameLocationStorage(Identifier name, Location child)
      : name(name), child(child) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<Identifier, Location>;
  bool operator==(const KeyTy &key) const { return key == KeyTy(name, child); }

  /// Construct a new storage instance.
  static NameLocationStorage *construct(AttributeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<NameLocationStorage>())
        NameLocationStorage(key.first, key.second);
  }

  Identifier name;
  Location child;
};

struct OpaqueLocationStorage : public AttributeStorage {
  OpaqueLocationStorage(uintptr_t underlyingLocation, ClassID *classId,
                        Location fallbackLocation)
      : underlyingLocation(underlyingLocation), classId(classId),
        fallbackLocation(fallbackLocation) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<uintptr_t, ClassID *, Location>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(underlyingLocation, classId, fallbackLocation);
  }

  /// Construct a new storage instance.
  static OpaqueLocationStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<OpaqueLocationStorage>())
        OpaqueLocationStorage(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  /// Pointer to the corresponding object.
  uintptr_t underlyingLocation;

  /// A unique pointer for each type of underlyingLocation.
  ClassID *classId;

  /// An additional location that can be used if the external one is not
  /// suitable.
  Location fallbackLocation;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_IR_LOCATIONDETAIL_H_
