#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONVERSIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONVERSIONS_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"

namespace sep = ::perftools::gputools::poplarplugin;

namespace perftools {
namespace gputools {
namespace poplarplugin {

// NOTE: There is an assumption in executor.cc that the device representation
//       of the data is smaller than the host side.  During the copy from
//       device to host, the device format data is first copied into the host
//       buffer, then converted through an intermediate buffer.

// NOTE: for these convertors, either the source size or the dest size will be
//       non-zero.  The convertor needs to work out the number of items to
//       transfer by considering both.
std::vector<char> ConvInt64ToInt32(const void *src, int64 ssize, int64 dsize);
std::vector<char> ConvInt32ToInt64(const void *src, int64 ssize, int64 dsize);

sep::ConversionFn GetInputConversionFunction(const xla::Shape &);
sep::ConversionFn GetOutputConversionFunction(const xla::Shape &);


}
}
}

#endif
