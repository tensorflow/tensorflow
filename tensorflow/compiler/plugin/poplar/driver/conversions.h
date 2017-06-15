#ifndef IPU_CONVERSIONS_H_
#define IPU_CONVERSIONS_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/stream_executor/executor.h"

namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

std::vector<char> ConvertInt64ToInt32(void* src, int64 size);
std::vector<char> ConvertInt32ToInt64(void* src, int64 size);

sep::ConversionFn GetInputConversionFunction(const xla::Shape&);
sep::ConversionFn GetOutputConversionFunction(const xla::Shape&);


}
}

#endif
