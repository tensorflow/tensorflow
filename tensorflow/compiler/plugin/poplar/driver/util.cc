#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace poplarplugin {

int64 CountShapes(const Shape& shape) {
  int64 n = 0;
  if (ShapeUtil::IsTuple(shape)) {
    for (int64 i=0; i<ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
};

}
}
