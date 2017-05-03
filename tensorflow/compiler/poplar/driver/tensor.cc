#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Engine.hpp>
#include <popstd/ActivationMapping.hpp>

namespace xla {
namespace poplarplugin {

port::StatusOr<std::string>
PoplarDataType(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case PRED:
      return std::string("bool");
    case S8:
      return std::string("char");
    case S16:
      return std::string("short");
    case S32:
      return std::string("int");
    case U8:
      return std::string("unsigned char");
    case U16:
      return std::string("unsigned short");
    case U32:
      return std::string("unsigned int");
    //case F16:
    //  return std::string("half");
    case F32:
      return std::string("float");
    default:
      return port::Status(port::error::FAILED_PRECONDITION,
                          port::StrCat("unsupported primitive type in poplar ",
                                       shape.element_type()));
  }
}

std::vector <size_t>
PoplarShapeFromXlaShape(const xla::Shape &xla_shape) {
  std::vector <size_t> shape;
  for (auto d : xla_shape.dimensions()) {
    shape.push_back(d);
  }
  return shape;
}

port::StatusOr<poplar::Tensor>
AddTensor(poplar::Graph& graph,
          const std::string& name,
          const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  std::string poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addTensor(poplar_type, dim, name);
  popstd::mapTensor(graph, out);
  return out;
}

template<typename TYPE>
static void
AddConstantTensorTyped(poplar::Graph&graph,
                       const xla::Literal &literal,
                       const xla::Shape &shape,
                       const std::string &type,
                       poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const TYPE* data(static_cast<const TYPE*>(LiteralUtil::InternalData(literal)));

  if (num_elements == 0) {
    tensor = graph.addConstantTensor(type, {0}, 0);
  } else if (num_elements == 1) {
    tensor = graph.addConstantTensor(type, dim, data[0]);
  } else {
    tensor = graph.addConstantTensor(type, dim, data);
  }
}

template void
AddConstantTensorTyped<float>(poplar::Graph&,
                              const xla::Literal &,
                              const xla::Shape &,
                              const std::string &,
                              poplar::Tensor& tensor);

template void
AddConstantTensorTyped<int>(poplar::Graph&,
                            const xla::Literal &,
                            const xla::Shape &,
                            const std::string &,
                            poplar::Tensor& tensor);

template void
AddConstantTensorTyped<bool>(poplar::Graph&,
                             const xla::Literal &,
                             const xla::Shape &,
                             const std::string &,
                             poplar::Tensor& tensor);

port::StatusOr<poplar::Tensor>
AddConstantTensor(poplar::Graph& graph,
                  const std::string& name,
                  const xla::Shape& shape,
                  const xla::Literal& literal) {
  poplar::Tensor tensor;

  std::string type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(literal.shape()));

  switch (literal.shape().element_type()) {
    case PRED:
      AddConstantTensorTyped<bool>(graph, literal, shape, type, tensor);
      break;
    case S8:
    case S16:
    case S32:
    case U8:
    case U16:
    case U32:
      AddConstantTensorTyped<int>(graph, literal, shape, type, tensor);
      break;
    case F16:
      // No fp16 support in XLA yet
      break;
    case F32:
      AddConstantTensorTyped<float>(graph, literal, shape, type, tensor);
      break;
    default:
      // The unsupported cases were caught in the call to PoplarDataType above
      break;
  }

  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  return tensor.reshape(dim);
}

template<typename T>
poplar::Tensor
TileTensor(const T &multiples, const poplar::Tensor &in) {
  poplar::Tensor out = in;
  for (unsigned d = 0; d < multiples.size(); d++) {
    int m = multiples[d];
    if (m == 0) {
      out = out.slice(0, 0, d);
    } else if (m > 1) {
      poplar::Tensor n = out;

      out = out.slice(0, 0, d);

      while (m != 0) {
        if (m & 1) {
          out = poplar::concat(out, n, d);
        }
        n = poplar::concat(n, n, d);
        m = m >> 1;
      }
    }
  }
  return out;
}

template poplar::Tensor
TileTensor<tensorflow::BCast::Vec>(const tensorflow::BCast::Vec &,
                                   const poplar::Tensor &);

template poplar::Tensor
TileTensor<std::vector<std::size_t>>(const std::vector<std::size_t> &,
                                     const poplar::Tensor &);

port::StatusOr<poplar::Tensor>
PadTensor(const PaddingConfig& cfg,
          const poplar::Tensor &in,
          const poplar::Tensor& pad) {
  if (pad.numElements() != 1) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "PadTensor: pad tensor is not single valued");
  }

  poplar::Tensor p(pad.reshape(std::vector<std::size_t>(in.rank(), 1)));

  poplar::Tensor out = in;
  for (unsigned d = 0; d < in.shape().size(); d++) {
    std::vector<std::size_t> shape(out.shape());

    if (cfg.dimensions(d).interior_padding() > 0) {
      shape[d] = cfg.dimensions(d).interior_padding();
      poplar::Tensor padded = TileTensor(shape, p);
      poplar::Tensor interleaved = out.slice(0,1,d);
      for (unsigned int slice=1; slice<out.dim(d); slice++) {
        interleaved = poplar::concat(interleaved, padded, d);
        interleaved = poplar::concat(interleaved, out.slice(slice, slice+1, d), d);
      }
      out = interleaved;
    }

    if (cfg.dimensions(d).edge_padding_low() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_low();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(padded, out, d);
    }

    if (cfg.dimensions(d).edge_padding_high() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_high();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(out, padded, d);
    }
  }

  return out;
}

port::StatusOr<poplar::Tensor>
ReverseTensor(const poplar::Tensor &in,
              const std::vector<int64>& dimensions) {
  poplar::Tensor out = in;
  if (in.numElements() > 0) {
    for (int64 d : dimensions) {
      poplar::Tensor slice = out.slice(0, 1, d);
      for (size_t s = 1; s < in.dim(d); s++) {
        slice = poplar::concat(out.slice(s, s+1, d), slice, d);
      }
      out = slice;
    }
  }
  return out;
}

port::StatusOr<poplar::Tensor>
BroadcastTensor(const poplar::Tensor &in,
                const xla::Shape& out,
                const std::vector<int64>& dimensions) {
  tensorflow::BCast::Vec bcast_shape =
          convert_array<tensorflow::BCast::Vec>(out.dimensions());

  tensorflow::BCast::Vec tensor_shape(ShapeUtil::Rank(out), 1);
  if (dimensions.size() > 0) {
    for (size_t d=0; d<dimensions.size(); d++) {
      tensor_shape[dimensions[d]] = in.dim(d);
    }
  } else {
    for (size_t d=0; d<in.rank(); d++) {
      tensor_shape[d] = in.dim(d);
    }
  }

  tensorflow::BCast bcast(tensor_shape, bcast_shape);
  if (!bcast.IsValid()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Incompatible broadcast");
  }

  poplar::Tensor r0 =
          in.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));

  return TileTensor(bcast.x_bcast(), r0);
}

bool
PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                           const xla::Shape& shape) {
  if (tensor.rank() != ShapeUtil::Rank(shape)) return false;
  for (size_t d=0; d<tensor.rank(); d++) {
    if (tensor.dim(d) != shape.dimensions(d)) return false;
  }

  return true;
}


}
}
