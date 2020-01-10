#ifndef TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_
#define TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_

#include <limits>
#include <utility>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Functions to define quantization attribute of types.
struct true_type {
  static const bool value = true;
};
struct false_type {
  static const bool value = false;
};

// Default is_quantized is false.
template <typename T>
struct is_quantized : false_type {};

// Specialize the quantized types.
template <>
struct is_quantized<qint8> : true_type {};
template <>
struct is_quantized<quint8> : true_type {};
template <>
struct is_quantized<qint32> : true_type {};

// All types not specialized are marked invalid.
template <class T>
struct IsValidDataType {
  static constexpr bool value = false;
};

// Extra validity checking; not part of public API.
struct TestIsValidDataType {
  static_assert(IsValidDataType<int64>::value, "Incorrect impl for int64");
  static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");
};

}  // namespace tensorflow

// Define numeric limits for our quantized as subclasses of the
// standard types.
namespace std {
template <>
class numeric_limits<tensorflow::qint8>
    : public numeric_limits<tensorflow::int8> {};
template <>
class numeric_limits<tensorflow::quint8>
    : public numeric_limits<tensorflow::uint8> {};
template <>
class numeric_limits<tensorflow::qint32>
    : public numeric_limits<tensorflow::int32> {};

// Specialize is_signed for quantized types.
template <>
struct is_signed<tensorflow::qint8> : public is_signed<tensorflow::int8> {};
template <>
struct is_signed<tensorflow::quint8> : public is_signed<tensorflow::uint8> {};
template <>
struct is_signed<tensorflow::qint32> : public is_signed<tensorflow::int32> {};

}  // namespace std

#endif  // TENSORFLOW_FRAMEWORK_TYPE_TRAITS_H_
