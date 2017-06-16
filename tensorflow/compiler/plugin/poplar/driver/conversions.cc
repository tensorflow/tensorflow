#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"

namespace xla {
namespace poplarplugin {

std::vector<char> ConvertInt64ToInt32(const void* src, int64 size) {
  int64 count = size / sizeof(int64);
  std::vector<char> result(count * sizeof(int32));
  const int64* src64 = reinterpret_cast<const int64*>(src);
  int32* dst32 = reinterpret_cast<int32*>(result.data());
  for (int64 i=0; i<count; i++) {
     *dst32++ = *src64++;
  }
  return result;
}


std::vector<char> ConvertInt32ToInt64(const void* src, int64 size) {
  int64 count = size / sizeof(int32);
  std::vector<char> result(count * sizeof(int64));
  const int32* src32 = reinterpret_cast<const int32*>(src);
  int64* dst64 = reinterpret_cast<int64*>(result.data());
  for (int64 i=0; i<count; i++) {
    *dst64++ = *src32++;
  }
  return result;
}

sep::ConversionFn GetInputConversionFunction(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case S64:
    case U64:
      return ConvertInt64ToInt32;
    default:
      return nullptr;
  }
}

sep::ConversionFn GetOutputConversionFunction(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case S64:
    case U64:
      return ConvertInt32ToInt64;
    default:
      return nullptr;
  }
}

}
}
