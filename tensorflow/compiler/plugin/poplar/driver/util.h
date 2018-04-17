#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/types.h"


namespace port = ::perftools::gputools::port;

namespace xla {

class Shape;
class Literal;

namespace poplarplugin {

template<typename To, typename From>
To convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
  }
  return out;
};

int64
CountShapes(const Shape& shape);

std::vector<xla::Shape>
FlattenedXlaShape(const xla::Shape& shape);

StatusOr<std::vector<int64>>
LiteralVectorToInt64Vector(const xla::Literal& lit);

}
}

#endif
