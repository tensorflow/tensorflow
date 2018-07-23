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

#ifndef TENSORFLOW_FRAMEWORK_TYPES_H_
#define TENSORFLOW_FRAMEWORK_TYPES_H_

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
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

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
  const string& type_string() const { return type_; }

  bool operator<(const DeviceType& other) const;
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const { return !(*this == other); }

 private:
  string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

// Convenient constants that can be passed to a DeviceType constructor
TF_EXPORT extern const char* const DEVICE_CPU;   // "CPU"
TF_EXPORT extern const char* const DEVICE_GPU;   // "GPU"
TF_EXPORT extern const char* const DEVICE_SYCL;  // "SYCL"

template <typename Device>
struct DeviceName {};

template <>
struct DeviceName<Eigen::ThreadPoolDevice> {
  static const std::string value;
};

#if GOOGLE_CUDA
template <>
struct DeviceName<Eigen::GpuDevice> {
  static const std::string value;
};
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
template <>
struct DeviceName<Eigen::SyclDevice> {
  static const std::string value;
};
#endif  // TENSORFLOW_USE_SYCL

typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;

// Convert the enums to strings for errors:
string DataTypeString(DataType dtype);
string DeviceTypeString(const DeviceType& device_type);
string DataTypeSliceString(const DataTypeSlice dtypes);
inline string DataTypeVectorString(const DataTypeVector& dtypes) {
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
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_UINT16) | ToSet(DT_INT8) |
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
template <class T, typename T2 = void>
struct IsValidDataType {
  static constexpr bool value = false;
};

template <>
struct IsValidDataType<string> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<complex64> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<complex128> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<bool> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<Eigen::half> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<bfloat16> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<ResourceHandle> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<Variant> {
  static constexpr bool value = true;
};

template <class T>
struct IsValidDataType<
    T, typename std::enable_if<std::is_floating_point<T>::value &&
                               (sizeof(T) == 4)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<
    T, typename std::enable_if<std::is_floating_point<T>::value &&
                               (sizeof(T) == 8)>::type> {
  static constexpr bool value = true;
};

template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  std::is_signed<T>::value &&
                                                  (sizeof(T) == 1)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  std::is_signed<T>::value &&
                                                  (sizeof(T) == 2)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  std::is_signed<T>::value &&
                                                  (sizeof(T) == 4)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  std::is_signed<T>::value &&
                                                  (sizeof(T) == 8)>::type> {
  static constexpr bool value = true;
};

template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  !std::is_signed<T>::value &&
                                                  (sizeof(T) == 1)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  !std::is_signed<T>::value &&
                                                  (sizeof(T) == 2)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  !std::is_signed<T>::value &&
                                                  (sizeof(T) == 4)>::type> {
  static constexpr bool value = true;
};
template <class T>
struct IsValidDataType<T, typename std::enable_if<std::is_integral<T>::value &&
                                                  !std::is_signed<T>::value &&
                                                  (sizeof(T) == 8)>::type> {
  static constexpr bool value = true;
};

template <>
struct IsValidDataType<qint8> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<qint16> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<qint32> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<quint8> {
  static constexpr bool value = true;
};
template <>
struct IsValidDataType<quint16> {
  static constexpr bool value = true;
};

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T, typename T2 = void>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

template <>
struct DataTypeToEnum<string> {
  static DataType v() { return DT_STRING; }
  static DataType ref() { return MakeRefType(DT_STRING); }
  static constexpr DataType value = DT_STRING;
};
template <>
struct DataTypeToEnum<complex64> {
  static DataType v() { return DT_COMPLEX64; }
  static DataType ref() { return MakeRefType(DT_COMPLEX64); }
  static constexpr DataType value = DT_COMPLEX64;
};
template <>
struct DataTypeToEnum<complex128> {
  static DataType v() { return DT_COMPLEX128; }
  static DataType ref() { return MakeRefType(DT_COMPLEX128); }
  static constexpr DataType value = DT_COMPLEX128;
};
template <>
struct DataTypeToEnum<bool> {
  static DataType v() { return DT_BOOL; }
  static DataType ref() { return MakeRefType(DT_BOOL); }
  static constexpr DataType value = DT_BOOL;
};
template <>
struct DataTypeToEnum<Eigen::half> {
  static DataType v() { return DT_HALF; }
  static DataType ref() { return MakeRefType(DT_HALF); }
  static constexpr DataType value = DT_HALF;
};
template <>
struct DataTypeToEnum<bfloat16> {
  static DataType v() { return DT_BFLOAT16; }
  static DataType ref() { return MakeRefType(DT_BFLOAT16); }
  static constexpr DataType value = DT_BFLOAT16;
};
template <>
struct DataTypeToEnum<ResourceHandle> {
  static DataType v() { return DT_RESOURCE; }
  static DataType ref() { return MakeRefType(DT_RESOURCE); }
  static constexpr DataType value = DT_RESOURCE;
};
template <>
struct DataTypeToEnum<Variant> {
  static DataType v() { return DT_VARIANT; }
  static DataType ref() { return MakeRefType(DT_VARIANT); }
  static constexpr DataType value = DT_VARIANT;
};

template <class T>
struct DataTypeToEnum<
    T, typename std::enable_if<std::is_floating_point<T>::value &&
                               (sizeof(T) == 4)>::type> {
  static DataType v() { return DT_FLOAT; }
  static DataType ref() { return MakeRefType(DT_FLOAT); }
  static constexpr DataType value = DT_FLOAT;
};
template <class T>
struct DataTypeToEnum<
    T, typename std::enable_if<std::is_floating_point<T>::value &&
                               (sizeof(T) == 8)>::type> {
  static DataType v() { return DT_DOUBLE; }
  static DataType ref() { return MakeRefType(DT_DOUBLE); }
  static constexpr DataType value = DT_DOUBLE;
};

template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 std::is_signed<T>::value &&
                                                 (sizeof(T) == 1)>::type> {
  static DataType v() { return DT_INT8; }
  static DataType ref() { return MakeRefType(DT_INT8); }
  static constexpr DataType value = DT_INT8;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 std::is_signed<T>::value &&
                                                 (sizeof(T) == 2)>::type> {
  static DataType v() { return DT_INT16; }
  static DataType ref() { return MakeRefType(DT_INT16); }
  static constexpr DataType value = DT_INT16;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 std::is_signed<T>::value &&
                                                 (sizeof(T) == 4)>::type> {
  static DataType v() { return DT_INT32; }
  static DataType ref() { return MakeRefType(DT_INT32); }
  static constexpr DataType value = DT_INT32;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 std::is_signed<T>::value &&
                                                 (sizeof(T) == 8)>::type> {
  static DataType v() { return DT_INT64; }
  static DataType ref() { return MakeRefType(DT_INT64); }
  static constexpr DataType value = DT_INT64;
};

template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 !std::is_signed<T>::value &&
                                                 (sizeof(T) == 1)>::type> {
  static DataType v() { return DT_UINT8; }
  static DataType ref() { return MakeRefType(DT_UINT8); }
  static constexpr DataType value = DT_UINT8;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 !std::is_signed<T>::value &&
                                                 (sizeof(T) == 2)>::type> {
  static DataType v() { return DT_UINT16; }
  static DataType ref() { return MakeRefType(DT_UINT16); }
  static constexpr DataType value = DT_UINT16;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 !std::is_signed<T>::value &&
                                                 (sizeof(T) == 4)>::type> {
  static DataType v() { return DT_UINT32; }
  static DataType ref() { return MakeRefType(DT_UINT32); }
  static constexpr DataType value = DT_UINT32;
};
template <class T>
struct DataTypeToEnum<T, typename std::enable_if<std::is_integral<T>::value &&
                                                 !std::is_signed<T>::value &&
                                                 (sizeof(T) == 8)>::type> {
  static DataType v() { return DT_UINT64; }
  static DataType ref() { return MakeRefType(DT_UINT64); }
  static constexpr DataType value = DT_UINT64;
};

template <>
struct DataTypeToEnum<qint8> {
  static DataType v() { return DT_QINT8; }
  static DataType ref() { return MakeRefType(DT_QINT8); }
  static constexpr DataType value = DT_QINT8;
};
template <>
struct DataTypeToEnum<qint16> {
  static DataType v() { return DT_QINT16; }
  static DataType ref() { return MakeRefType(DT_QINT16); }
  static constexpr DataType value = DT_QINT16;
};
template <>
struct DataTypeToEnum<qint32> {
  static DataType v() { return DT_QINT32; }
  static DataType ref() { return MakeRefType(DT_QINT32); }
  static constexpr DataType value = DT_QINT32;
};
template <>
struct DataTypeToEnum<quint8> {
  static DataType v() { return DT_QUINT8; }
  static DataType ref() { return MakeRefType(DT_QUINT8); }
  static constexpr DataType value = DT_QUINT8;
};
template <>
struct DataTypeToEnum<quint16> {
  static DataType v() { return DT_QUINT16; }
  static DataType ref() { return MakeRefType(DT_QUINT16); }
  static constexpr DataType value = DT_QUINT16;
};

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType;  // Specializations below

template <>
struct EnumToDataType<DT_STRING> {
  typedef string Type;
};
template <>
struct EnumToDataType<DT_COMPLEX64> {
  typedef complex64 Type;
};
template <>
struct EnumToDataType<DT_COMPLEX128> {
  typedef complex128 Type;
};
template <>
struct EnumToDataType<DT_BOOL> {
  typedef bool Type;
};
template <>
struct EnumToDataType<DT_HALF> {
  typedef Eigen::half Type;
};
template <>
struct EnumToDataType<DT_BFLOAT16> {
  typedef bfloat16 Type;
};
template <>
struct EnumToDataType<DT_RESOURCE> {
  typedef ResourceHandle Type;
};
template <>
struct EnumToDataType<DT_VARIANT> {
  typedef Variant Type;
};

template <>
struct EnumToDataType<DT_FLOAT> {
  typedef float Type;
};
template <>
struct EnumToDataType<DT_DOUBLE> {
  typedef double Type;
};

template <>
struct EnumToDataType<DT_INT8> {
  typedef int8 Type;
};
template <>
struct EnumToDataType<DT_INT16> {
  typedef int16 Type;
};
template <>
struct EnumToDataType<DT_INT32> {
  typedef int32 Type;
};
template <>
struct EnumToDataType<DT_INT64> {
  typedef int64 Type;
};
template <>
struct EnumToDataType<DT_UINT8> {
  typedef uint8 Type;
};
template <>
struct EnumToDataType<DT_UINT16> {
  typedef uint16 Type;
};
template <>
struct EnumToDataType<DT_UINT32> {
  typedef uint32 Type;
};
template <>
struct EnumToDataType<DT_UINT64> {
  typedef uint64 Type;
};

template <>
struct EnumToDataType<DT_QINT8> {
  typedef qint8 Type;
};
template <>
struct EnumToDataType<DT_QINT16> {
  typedef qint16 Type;
};
template <>
struct EnumToDataType<DT_QINT32> {
  typedef qint32 Type;
};
template <>
struct EnumToDataType<DT_QUINT8> {
  typedef quint8 Type;
};
template <>
struct EnumToDataType<DT_QUINT16> {
  typedef quint16 Type;
};

// Extra validity checking; not part of public API.
static_assert(IsValidDataType<int64>::value, "Incorrect impl for int64");
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

// Types that always sit on host: DT_STRING, DT_STRING_REF, DT_RESOURCE.
// For DT_RESOURCE, the handle always sits on host (even if the underlying
// object has device-allocated resources).
bool DataTypeAlwaysOnHost(DataType dt);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TYPES_H_
