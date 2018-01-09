#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
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

std::vector<xla::Shape>
FlattenedXlaShape(const xla::Shape& shape) {
  std::vector<xla::Shape> out;
  if (ShapeUtil::IsTuple(shape)) {
    for (int i=0; i<ShapeUtil::TupleElementCount(shape); i++) {
      std::vector<xla::Shape> shapes =
        FlattenedXlaShape(ShapeUtil::GetTupleElementShape(shape, i));
      out.insert(out.end(), shapes.begin(), shapes.end());
    }
  } else {
    out.push_back(shape);
  }

  return out;
}


port::StatusOr<std::vector<int64>>
LiteralVectorToInt64Vector(const xla::Literal& lit) {
  if (lit.shape().dimensions_size() != 1) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Literal rank != 1");
  }

  std::unique_ptr<Literal> s64_lit;
  TF_ASSIGN_OR_RETURN(s64_lit, lit.Convert(S64));

  const int64* start = static_cast<const int64*>(s64_lit->untyped_data());
  return std::vector<int64>(start, start + s64_lit->shape().dimensions(0));
}

}
}
