//===- TypeDetail.h - QuantOps Type detail ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TYPE_DETAIL_H_
#define TYPE_DETAIL_H_

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/bit.h"

namespace mlir {
namespace quant {
namespace detail {

struct QuantizedTypeStorage : public mlir::TypeStorage {
  QuantizedTypeStorage(unsigned flags, Type storageType, Type expressedType,
                       int64_t storageTypeMin, int64_t storageTypeMax)
      : flags(flags), storageType(storageType), expressedType(expressedType),
        storageTypeMin(storageTypeMin), storageTypeMax(storageTypeMax) {}

  /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
  unsigned flags;

  // Integral type for the storage point representation.
  Type storageType;

  // Floating point type that the quantized type approximates.
  Type expressedType;

  // The minimum value storageType can take.
  int64_t storageTypeMin;

  // The maximum value storageType can take.
  int64_t storageTypeMax;
};

struct AnyQuantizedTypeStorage : public QuantizedTypeStorage {
  struct KeyTy {
    KeyTy(unsigned flags, Type storageType, Type expressedType,
          int64_t storageTypeMin, int64_t storageTypeMax)
        : flags(flags), storageType(storageType), expressedType(expressedType),
          storageTypeMin(storageTypeMin), storageTypeMax(storageTypeMax) {}
    unsigned flags;
    Type storageType;
    Type expressedType;
    int64_t storageTypeMin;
    int64_t storageTypeMax;

    // Check for equality of two structures that share KeyTy data members
    // (by name).
    template <typename T, typename U>
    static bool genericIsEqual(const T &lhs, const U &rhs) {
      return lhs.flags == rhs.flags && lhs.storageType == rhs.storageType &&
             lhs.expressedType == rhs.expressedType &&
             lhs.storageTypeMin == rhs.storageTypeMin &&
             lhs.storageTypeMax == rhs.storageTypeMax;
    }

    bool operator==(const KeyTy &other) const {
      return genericIsEqual(*this, other);
    }

    unsigned getHashValue() const {
      return llvm::hash_combine(flags, storageType, expressedType,
                                storageTypeMin, storageTypeMax);
    }
  };

  AnyQuantizedTypeStorage(const KeyTy &key)
      : QuantizedTypeStorage(key.flags, key.storageType, key.expressedType,
                             key.storageTypeMin, key.storageTypeMax) {}

  bool operator==(const KeyTy &key) const {
    return KeyTy::genericIsEqual(*this, key);
  }

  /// Construction.
  static AnyQuantizedTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    return new (allocator.allocate<AnyQuantizedTypeStorage>())
        AnyQuantizedTypeStorage(key);
  }

  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }
};

struct UniformQuantizedTypeStorage : public QuantizedTypeStorage {
  struct KeyTy {
    KeyTy(unsigned flags, Type storageType, Type expressedType, double scale,
          int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax)
        : flags(flags), storageType(storageType), expressedType(expressedType),
          scale(scale), zeroPoint(zeroPoint), storageTypeMin(storageTypeMin),
          storageTypeMax(storageTypeMax) {}
    /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
    unsigned flags;

    // Integral type for the storage point representation.
    Type storageType;

    // Floating point type that the quantized type approximates.
    Type expressedType;

    double scale;
    int64_t zeroPoint;
    int64_t storageTypeMin;
    int64_t storageTypeMax;

    // Check for equality of two structures that share KeyTy data members
    // (by name).
    template <typename T, typename U>
    static bool genericIsEqual(const T &lhs, const U &rhs) {
      return lhs.flags == rhs.flags && lhs.storageType == rhs.storageType &&
             lhs.expressedType == rhs.expressedType && lhs.scale == rhs.scale &&
             lhs.zeroPoint == rhs.zeroPoint &&
             lhs.storageTypeMin == rhs.storageTypeMin &&
             lhs.storageTypeMax == rhs.storageTypeMax;
    }

    bool operator==(const KeyTy &other) const {
      return genericIsEqual(*this, other);
    }

    unsigned getHashValue() const {
      int64_t scaleBits = llvm::bit_cast<int64_t>(scale);
      return llvm::hash_combine(flags, storageType, expressedType, scaleBits,
                                zeroPoint, storageTypeMin, storageTypeMax);
    }
  };

  UniformQuantizedTypeStorage(const KeyTy &key)
      : QuantizedTypeStorage(key.flags, key.storageType, key.expressedType,
                             key.storageTypeMin, key.storageTypeMax),
        scale(key.scale), zeroPoint(key.zeroPoint) {}

  bool operator==(const KeyTy &key) const {
    return KeyTy::genericIsEqual(*this, key);
  }

  /// Construction.
  static UniformQuantizedTypeStorage *construct(TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
    return new (allocator.allocate<UniformQuantizedTypeStorage>())
        UniformQuantizedTypeStorage(key);
  }

  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }

  double scale;
  int64_t zeroPoint;
};

struct UniformQuantizedPerAxisTypeStorage : public QuantizedTypeStorage {
  struct KeyTy {
    KeyTy(unsigned flags, Type storageType, Type expressedType,
          ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
          int32_t quantizedDimension, int64_t storageTypeMin,
          int64_t storageTypeMax)
        : flags(flags), storageType(storageType), expressedType(expressedType),
          scales(scales), zeroPoints(zeroPoints),
          quantizedDimension(quantizedDimension),
          storageTypeMin(storageTypeMin), storageTypeMax(storageTypeMax) {}
    /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
    unsigned flags;

    // Integral type for the storage point representation.
    Type storageType;

    // Floating point type that the quantized type approximates.
    Type expressedType;

    ArrayRef<double> scales;
    ArrayRef<int64_t> zeroPoints;
    int32_t quantizedDimension;
    int64_t storageTypeMin;
    int64_t storageTypeMax;

    ArrayRef<double> getScales() const { return scales; }

    ArrayRef<int64_t> getZeroPoints() const { return zeroPoints; }

    // Check for equality of two structures that share KeyTy data members
    // (by name).
    template <typename T, typename U>
    static bool genericIsEqual(const T &lhs, const U &rhs) {
      return lhs.flags == rhs.flags && lhs.storageType == rhs.storageType &&
             lhs.expressedType == rhs.expressedType &&
             lhs.getScales() == rhs.getScales() &&
             lhs.getZeroPoints() == rhs.getZeroPoints() &&
             lhs.quantizedDimension == rhs.quantizedDimension &&
             lhs.storageTypeMin == rhs.storageTypeMin &&
             lhs.storageTypeMax == rhs.storageTypeMax;
    }

    bool operator==(const KeyTy &other) const {
      return genericIsEqual(*this, other);
    }

    unsigned getHashValue() const {
      int64_t *scalesCast = llvm::bit_cast<int64_t *>(scales.data());
      ArrayRef<int64_t> scalesBits(scalesCast, scales.size());
      return llvm::hash_combine(
          flags, storageType, expressedType,
          llvm::hash_combine_range(scalesBits.begin(), scalesBits.end()),
          llvm::hash_combine_range(zeroPoints.begin(), zeroPoints.end()),
          storageTypeMin, storageTypeMax);
    }
  };

  // We pass scales and zeroPoints in directly rather than relying on KeyTy
  // because we have to create new reallocated versions in `construct` below.
  UniformQuantizedPerAxisTypeStorage(const KeyTy &key, ArrayRef<double> scales,
                                     ArrayRef<int64_t> zeroPoints)
      : QuantizedTypeStorage(key.flags, key.storageType, key.expressedType,
                             key.storageTypeMin, key.storageTypeMax),
        scaleElements(scales.data()), zeroPointElements(zeroPoints.data()),
        quantParamsSize(scales.size()),
        quantizedDimension(key.quantizedDimension) {}

  bool operator==(const KeyTy &key) const {
    return KeyTy::genericIsEqual(*this, key);
  }

  /// Construction.
  static UniformQuantizedPerAxisTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ArrayRef<double> scales = allocator.copyInto(key.scales);
    ArrayRef<int64_t> zeroPoints = allocator.copyInto(key.zeroPoints);
    return new (allocator.allocate<UniformQuantizedPerAxisTypeStorage>())
        UniformQuantizedPerAxisTypeStorage(key, scales, zeroPoints);
  }

  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }

  ArrayRef<double> getScales() const {
    return ArrayRef<double>(scaleElements, quantParamsSize);
  }

  ArrayRef<int64_t> getZeroPoints() const {
    return ArrayRef<int64_t>(zeroPointElements, quantParamsSize);
  }

  const double *scaleElements;
  const int64_t *zeroPointElements;
  unsigned quantParamsSize;
  int32_t quantizedDimension;
};

} // namespace detail
} // namespace quant
} // namespace mlir

#endif // TYPE_DETAIL_H_
