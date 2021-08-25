/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPES_H_

#include <map>
#include <set>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Variant;

// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
enum MemoryType {
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1,
};

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType {
 public:
  DeviceType(const char* type)  // NOLINT(runtime/explicit)
      : type_(type) {}

  explicit DeviceType(StringPiece type) : type_(type.data(), type.size()) {}

  const char* type() const { return type_.c_str(); }
  const std::string& type_string() const { return type_; }

  bool operator<(const DeviceType& other) const;
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const { return !(*this == other); }

 private:
  std::string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

// Convenient constants that can be passed to a DeviceType constructor
TF_EXPORT extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
TF_EXPORT extern const char* const DEVICE_CPU;         // "CPU"
TF_EXPORT extern const char* const DEVICE_GPU;         // "GPU"
TF_EXPORT extern const char* const DEVICE_TPU;         // "TPU"
TF_EXPORT extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM"

template <typename Device>
struct DeviceName {};

template <>
struct DeviceName<Eigen::ThreadPoolDevice> {
  static const std::string value;
};

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
struct DeviceName<Eigen::GpuDevice> {
  static const std::string value;
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;
typedef gtl::InlinedVector<std::pair<DeviceType, int32>, 4>
    PrioritizedDeviceTypeVector;

// Convert the enums to strings for errors:
std::string DataTypeString(DataType dtype);
std::string DeviceTypeString(const DeviceType& device_type);
std::string DataTypeSliceString(const DataTypeSlice dtypes);
inline std::string DataTypeVectorString(const DataTypeVector& dtypes) {
  return DataTypeSliceString(dtypes);
}

// DataTypeSet represents a set of DataType values as a simple and efficient
// bit mask.  Note that DataTypeSet cannot represent all DataType values; it
// cannot represent any of the DT_*_REF values.
class DataTypeSet {
 private:
  const uint32 mask_;

  static constexpr uint32 kNumBits = 32;

 public:
  constexpr DataTypeSet(const DataTypeSet& other) : mask_(other.mask_) {}
  explicit constexpr DataTypeSet(uint32 mask) : mask_(mask) {}

  constexpr bool Contains(DataType dt) const {
    return (static_cast<uint32>(dt) < kNumBits) &&
           ((mask_ >> static_cast<uint32>(dt)) & 1u) != 0u;
  }

  class Iterator {
    const DataTypeSet& set_;
    uint32 pos_;

   public:
    Iterator(const DataTypeSet& set, uint32 pos) : set_(set), pos_(pos) {
      DCHECK_LE(pos, kNumBits);
    }
    DataType operator*() const { return static_cast<DataType>(pos_); }
    Iterator& operator++() {
      ++pos_;
      DCHECK_LE(pos_, kNumBits);
      if (pos_ < kNumBits) {
        uint32 remaining_mask = set_.mask_ >> pos_;
        if (remaining_mask != 0u) {
          pos_ += ctz_uint32(remaining_mask);
        }
      }
      DCHECK_LE(pos_, kNumBits);
      return *this;
    }
    bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }
    size_t operator-(const Iterator& other) const {
      return this->pos_ - other.pos_;
    }
  };

  static uint32 ctz_uint32(uint32 x) {
    DCHECK_NE(x, 0u);
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    uint32 n = 0u;
    while ((x & 1u) == 0u) {
      x >>= 1;
      ++n;
    }
    return n;
#endif
  }

  static uint32 clz_uint32(uint32 x) {
    DCHECK_NE(x, 0u);
#ifdef __GNUC__
    return __builtin_clz(x);
#else
    uint32 n = 0u;
    while ((x >> (kNumBits - 1u)) == 0u) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }

  Iterator begin() const {
    // The begin position is the index of the first bit set to 1 in the entire
    // bit mask. If there are no bits set to 1, then the index is 0.
    if (mask_ != 0) {
      return Iterator(*this, ctz_uint32(mask_));
    }
    // The set is empty.
    return Iterator(*this, 0);
  }

  Iterator end() const {
    // The end position is the index of the highest bit that is set, plus 1.
    // If there are no bits set to 1, then the index is 0.
    if (mask_ != 0) {
      return Iterator(*this, kNumBits - clz_uint32(mask_));
    }
    // The set is empty.
    return Iterator(*this, 0);
  }

  size_t size() const {
#if defined(__GNUC__)
    return __builtin_popcount(mask_);
#else
    size_t n = 0;
    uint32 x = mask_;
    while (x > 0) {
      n += x & 1u;
      x >>= 1;
    }
    return n;
#endif
  }

  constexpr DataTypeSet operator|(const DataTypeSet& other) const {
    return DataTypeSet(mask_ | other.mask_);
  }
};

// If "sp" names a valid type, store it in "*dt" and return true.  Otherwise,
// return false.
bool DataTypeFromString(StringPiece sp, DataType* dt);

constexpr inline DataTypeSet ToSet(DataType dt) {
  return DataTypeSet(1u << static_cast<uint32>(dt));
}

// DT_FLOAT + kDataTypeRefOffset == DT_FLOAT_REF, etc.
enum { kDataTypeRefOffset = 100 };
inline bool IsRefType(DataType dtype) {
  return dtype > static_cast<DataType>(kDataTypeRefOffset);
}
inline DataType MakeRefType(DataType dtype) {
  DCHECK(!IsRefType(dtype));
  return static_cast<DataType>(dtype + kDataTypeRefOffset);
}
inline DataType RemoveRefType(DataType dtype) {
  DCHECK(IsRefType(dtype));
  return static_cast<DataType>(dtype - kDataTypeRefOffset);
}
inline DataType BaseType(DataType dtype) {
  return IsRefType(dtype) ? RemoveRefType(dtype) : dtype;
}

// Returns true if the actual type is the same as or ref of the expected type.
inline bool TypesCompatible(DataType expected, DataType actual) {
  return expected == actual || expected == BaseType(actual);
}

// Does not include _ref types.
constexpr DataTypeSet kAllTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_UINT8) |
    ToSet(DT_INT16) | ToSet(DT_UINT16) | ToSet(DT_INT8) | ToSet(DT_STRING) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_INT64) |
    ToSet(DT_BOOL) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) | ToSet(DT_QINT16) |
    ToSet(DT_QUINT16) | ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_RESOURCE) |
    ToSet(DT_VARIANT) | ToSet(DT_UINT32) | ToSet(DT_UINT64) |
    ToSet(DT_BFLOAT16);
inline const DataTypeSet& AllTypes() { return kAllTypes; }

#if !defined(IS_MOBILE_PLATFORM) || defined(SUPPORT_SELECTIVE_REGISTRATION)

// Types that support '<' and '>'.
constexpr DataTypeSet kRealNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
    ToSet(DT_UINT8) | ToSet(DT_INT16) | ToSet(DT_INT8) | ToSet(DT_UINT16) |
    ToSet(DT_HALF) | ToSet(DT_UINT32) | ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
inline const DataTypeSet RealNumberTypes() { return kRealNumberTypes; }

// Return the list of all numeric types.
// Includes complex and quantized types.
// NOTE: On Android, we only include the float and int32 types for now.
const DataTypeSet kNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT64) | ToSet(DT_INT32) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_UINT32) |
    ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
inline const DataTypeSet& NumberTypes() { return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline const DataTypeSet& QuantizedTypes() { return kQuantizedTypes; }

// Types that support '<' and '>', including quantized types.
const DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_QINT8) | ToSet(DT_QUINT8) | ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
    ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_BFLOAT16);
inline const DataTypeSet& RealAndQuantizedTypes() {
  return kRealAndQuantizedTypes;
}

#elif defined(__ANDROID_TYPES_FULL__)

constexpr DataTypeSet kRealNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_HALF);
inline DataTypeSet RealNumberTypes() { return kRealNumberTypes; }

constexpr DataTypeSet kNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT32) | ToSet(DT_HALF);
inline DataTypeSet NumberTypes() { return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline DataTypeSet QuantizedTypes() { return kQuantizedTypes; }

constexpr DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32) |
    ToSet(DT_HALF);
inline DataTypeSet RealAndQuantizedTypes() { return kRealAndQuantizedTypes; }

#else  // defined(IS_MOBILE_PLATFORM) && !defined(__ANDROID_TYPES_FULL__)

constexpr DataTypeSet kRealNumberTypes = ToSet(DT_FLOAT) | ToSet(DT_INT32);
inline DataTypeSet RealNumberTypes() { return kRealNumberTypes; }

constexpr DataTypeSet kNumberTypes = ToSet(DT_FLOAT) | ToSet(DT_INT32) |
                                     ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                     ToSet(DT_QINT32);
inline DataTypeSet NumberTypes() { return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline DataTypeSet QuantizedTypes() { return kQuantizedTypes; }

constexpr DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
    ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32);
inline DataTypeSet RealAndQuantizedTypes() { return kRealAndQuantizedTypes; }

#endif  // defined(IS_MOBILE_PLATFORM)

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidDataType;

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                 \
  template <>                                           \
  struct DataTypeToEnum<TYPE> {                         \
    static DataType v() { return ENUM; }                \
    static DataType ref() { return MakeRefType(ENUM); } \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template <>                                           \
  struct IsValidDataType<TYPE> {                        \
    static constexpr bool value = true;                 \
  };                                                    \
  template <>                                           \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
MATCH_TYPE_AND_ENUM(tstring, DT_STRING);
MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
MATCH_TYPE_AND_ENUM(complex128, DT_COMPLEX128);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
MATCH_TYPE_AND_ENUM(qint16, DT_QINT16);
MATCH_TYPE_AND_ENUM(quint16, DT_QUINT16);
MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);
MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
MATCH_TYPE_AND_ENUM(ResourceHandle, DT_RESOURCE);
MATCH_TYPE_AND_ENUM(Variant, DT_VARIANT);

template <>
struct DataTypeToEnum<long> {
  static DataType v() { return value; }
  static DataType ref() { return MakeRefType(value); }
  static constexpr DataType value = sizeof(long) == 4 ? DT_INT32 : DT_INT64;
};
template <>
struct IsValidDataType<long> {
  static constexpr bool value = true;
};
template <>
struct EnumToDataType<DT_INT64> {
  typedef int64_t Type;
};

template <>
struct DataTypeToEnum<unsigned long> {
  static DataType v() { return value; }
  static DataType ref() { return MakeRefType(value); }
  static constexpr DataType value =
      sizeof(unsigned long) == 4 ? DT_UINT32 : DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long> {
  static constexpr bool value = true;
};
template <>
struct EnumToDataType<DT_UINT64> {
  typedef tensorflow::uint64 Type;
};

template <>
struct DataTypeToEnum<long long> {
  static DataType v() { return DT_INT64; }
  static DataType ref() { return MakeRefType(DT_INT64); }
  static constexpr DataType value = DT_INT64;
};
template <>
struct IsValidDataType<long long> {
  static constexpr bool value = true;
};

template <>
struct DataTypeToEnum<unsigned long long> {
  static DataType v() { return DT_UINT64; }
  static DataType ref() { return MakeRefType(DT_UINT64); }
  static constexpr DataType value = DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long long> {
  static constexpr bool value = true;
};

#undef MATCH_TYPE_AND_ENUM

// All types not specialized are marked invalid.
template <class T>
struct IsValidDataType {
  static constexpr bool value = false;
};

// Extra validity checking; not part of public API.
static_assert(IsValidDataType<int64_t>::value, "Incorrect impl for int64");
static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");

// TODO(jeff): Maybe unify this with Tensor::CanUseDMA, or the underlying
// is_simple<T> in tensor.cc (and possible choose a more general name?)
constexpr DataTypeSet kDataTypesCanUseMemcpy =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_UINT32) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_INT64) |
    ToSet(DT_UINT64) | ToSet(DT_BOOL) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
    ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32) |
    ToSet(DT_BFLOAT16) | ToSet(DT_HALF);
inline bool DataTypeCanUseMemcpy(DataType dt) {
  return kDataTypesCanUseMemcpy.Contains(dt);
}

// Returns true iff 'dt' is a real, non-quantized floating point type.
constexpr DataTypeSet kDataTypeIsFloating =
    ToSet(DT_HALF) | ToSet(DT_BFLOAT16) | ToSet(DT_FLOAT) | ToSet(DT_DOUBLE);
inline bool DataTypeIsFloating(DataType dt) {
  return kDataTypeIsFloating.Contains(dt);
}

// Returns true iff 'dt' is a complex type.
constexpr DataTypeSet kDataTypeIsComplex =
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128);
inline bool DataTypeIsComplex(DataType dt) {
  return kDataTypeIsComplex.Contains(dt);
}

inline bool DataTypeIsQuantized(DataType dt) {
  return kQuantizedTypes.Contains(dt);
}

// Is the dtype nonquantized integral?
constexpr DataTypeSet kDataTypeIsInteger =
    ToSet(DT_INT8) | ToSet(DT_UINT8) | ToSet(DT_INT16) | ToSet(DT_UINT16) |
    ToSet(DT_INT32) | ToSet(DT_UINT32) | ToSet(DT_INT64) | ToSet(DT_UINT64);
inline bool DataTypeIsInteger(DataType dt) {
  return kDataTypeIsInteger.Contains(dt);
}

// Is the dtype a signed integral type?
constexpr DataTypeSet kDataTypeIsSigned =
    ToSet(DT_INT8) | ToSet(DT_INT16) | ToSet(DT_INT32) | ToSet(DT_INT64);
inline bool DataTypeIsSigned(DataType dt) {
  return kDataTypeIsSigned.Contains(dt);
}

// Is the dtype an unsigned integral type?
constexpr DataTypeSet kDataTypeIsUnsigned =
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_UINT32) | ToSet(DT_UINT64);
inline bool DataTypeIsUnsigned(DataType dt) {
  return kDataTypeIsUnsigned.Contains(dt);
}

// Returns a 0 on failure
int DataTypeSize(DataType dt);

// Returns HOST_MEMORY if `dtype` is always on host or is a DT_INT32,
// DEVICE_MEMORY otherwise.
MemoryType MTypeFromDType(const DataType dtype);

// Returns HOST_MEMORY if `dtype` is always on host, DEVICE_MEMORY otherwise.
// The reason we have MTypeFromDType() and MTypeFromDTypeIntsOnDevice(): for
// GPUs, we would like to keep int operations on host for performance concerns.
// But for TPUs (and other devices), int operations are placed on device.
MemoryType MTypeFromDTypeIntsOnDevice(const DataType dtype);

// Types that always sit on host: DT_STRING, DT_STRING_REF, DT_RESOURCE.
// For DT_RESOURCE, the handle always sits on host (even if the underlying
// object has device-allocated resources).
bool DataTypeAlwaysOnHost(DataType dt);

// FullType implementation.

// Reference container for a type definition. These values are usually interned.
// These containers admit a notion of ordering for efficient access. The
// ordering has no semantic otherwise.
struct TypeRef {
  std::shared_ptr<FullTypeDef> full_type;

  bool operator==(const TypeRef& other) const {
    // TODO(mdan): This should be more efficient.
    return full_type->SerializeAsString() ==
           other.full_type->SerializeAsString();
  }
  bool operator<(const TypeRef& other) const {
    return full_type->SerializeAsString() <
           other.full_type->SerializeAsString();
  }
};

struct TypeHasher {
  std::size_t operator()(const TypeRef& k) const {
    return std::hash<std::string>()(k.full_type->SerializeAsString());
  }
};

// Maps a legacy DType proto enum to an equivalent FullType Tensor.
void map_dtype_to_tensor(const DataType& dtype, FullTypeDef* t);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPES_H_
