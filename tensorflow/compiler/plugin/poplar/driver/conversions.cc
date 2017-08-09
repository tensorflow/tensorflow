#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"

namespace perftools {
namespace gputools {
namespace poplarplugin {

std::vector<char> ConvInt64ToInt32(const void *src, int64 ssize, int64 dsize) {
  int64 count = ssize / sizeof(int64);
  if (count == 0) {
    count = dsize / sizeof(int32);
  }
  std::vector<char> result(count * sizeof(int32));
  const int64 *src64 = reinterpret_cast<const int64 *>(src);
  int32 *dst32 = reinterpret_cast<int32 *>(result.data());
  for (int64 i = 0; i < count; i++) {
    *dst32++ = *src64++;
  }
  return result;
}


std::vector<char> ConvInt32ToInt64(const void *src, int64 ssize, int64 dsize) {
  int64 count = ssize / sizeof(int32);
  if (count == 0) {
    count = dsize / sizeof(int64);
  }
  std::vector<char> result(count * sizeof(int64));
  const int32 *src32 = reinterpret_cast<const int32 *>(src);
  int64 *dst64 = reinterpret_cast<int64 *>(result.data());
  for (int64 i = 0; i < count; i++) {
    *dst64++ = *src32++;
  }
  return result;
}

sep::ConversionFn GetInputConversionFunction(const xla::Shape &shape) {
  switch (shape.element_type()) {
    case xla::S64:
    case xla::U64:
      return ConvInt64ToInt32;
    default:
      return nullptr;
  }
}

sep::ConversionFn GetOutputConversionFunction(const xla::Shape &shape) {
  switch (shape.element_type()) {
    case xla::S64:
    case xla::U64:
      return ConvInt32ToInt64;
    default:
      return nullptr;
  }
}

}
}
}
