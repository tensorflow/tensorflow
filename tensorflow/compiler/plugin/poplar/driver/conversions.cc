#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Engine.hpp>
#include <popstd/ActivationMapping.hpp>

namespace xla {
namespace poplarplugin {

std::vector<char> ConvertInt64ToInt32(void* src, int64 size) {
  int64 count = size / sizeof(int64);
  std::vector<char> result(count * sizeof(int32));
  int64* src64 = reinterpret_cast<int64*>(src);
  int32* dst32 = reinterpret_cast<int32*>(result.data());
  for (int64 i=0; i<count; i++) {
     *dst32++ = *src64++;
  }
  return result;
}


std::vector<char> ConvertInt32ToInt64(void* src, int64 size) {
  int64 count = size / sizeof(int32);
  std::vector<char> result(count * sizeof(int64));
  int32* src32 = reinterpret_cast<int32*>(src);
  int64* dst64 = reinterpret_cast<int64*>(result.data());
  for (int64 i=0; i<count; i++) {
    *dst64++ = *src32++;
  }
  return result;
}

}
}
