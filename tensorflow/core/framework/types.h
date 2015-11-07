#ifndef TENSORFLOW_FRAMEWORK_TYPES_H_
#define TENSORFLOW_FRAMEWORK_TYPES_H_

#include <map>
#include <set>
#include <string>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"

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

  bool operator<(const DeviceType& other) const;
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const { return !(*this == other); }

 private:
  string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

// Convenient constants that can be passed to a DeviceType constructor
extern const char* const DEVICE_CPU;  // "CPU"
extern const char* const DEVICE_GPU;  // "GPU"

typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;

// Convert the enums to strings for errors:
string DataTypeString(DataType dtype);
string DeviceTypeString(DeviceType device_type);
string DataTypeSliceString(const DataTypeSlice dtypes);
inline string DataTypeVectorString(const DataTypeVector& dtypes) {
  return DataTypeSliceString(dtypes);
}

// If "sp" names a valid type, store it in "*dt" and return true.  Otherwise,
// return false.
bool DataTypeFromString(StringPiece sp, DataType* dt);

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
DataTypeVector AllTypes();

// Return the list of all numeric types.
// NOTE: On Android, we only include the float and int32 types for now.
DataTypeVector RealNumberTypes();  // Types that support '<' and '>'.
DataTypeVector NumberTypes();      // Includes complex and quantized types.

DataTypeVector QuantizedTypes();
DataTypeVector RealAndQuantizedTypes();  // Types that support '<' and
                                         // '>', including quantized
                                         // types

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

// We use Eigen's QInt implementations for our quantized int types.
typedef Eigen::QInt8 qint8;
typedef Eigen::QUInt8 quint8;
typedef Eigen::QInt32 qint32;

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
MATCH_TYPE_AND_ENUM(string, DT_STRING);
MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
MATCH_TYPE_AND_ENUM(int64, DT_INT64);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);

#undef MATCH_TYPE_AND_ENUM

bool DataTypeCanUseMemcpy(DataType dt);

bool DataTypeIsQuantized(DataType dt);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TYPES_H_
