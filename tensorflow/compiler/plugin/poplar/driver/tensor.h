#ifndef IPU_TENSOR_H_
#define IPU_TENSOR_H_

#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace poplar {
  class Tensor;
  class Graph;
}

namespace port = ::perftools::gputools::port;

namespace xla {
namespace poplarplugin {

port::StatusOr<std::string>
PoplarDataType(const xla::Shape& shape);

std::vector <size_t>
PoplarShapeFromXlaShape(const xla::Shape &xla_shape);

port::StatusOr<poplar::Tensor>
AddTensor(poplar::Graph& graph,
          const std::string& name,
          const xla::Shape& shape);

port::StatusOr<poplar::Tensor>
AddConstantTensor(poplar::Graph& graph,
                  const std::string& name,
                  const xla::Shape& shape,
                  const xla::Literal& literal);

template <typename T>
poplar::Tensor
TileTensor(const T& multiples,
           const poplar::Tensor& in);

port::StatusOr<poplar::Tensor>
PadTensor(const PaddingConfig& cfg,
          const poplar::Tensor& in,
          const poplar::Tensor& pad);

port::StatusOr<poplar::Tensor>
ReverseTensor(const poplar::Tensor &in,
              const std::vector<int64>& dimensions);

port::StatusOr<poplar::Tensor>
BroadcastTensor(const poplar::Tensor &in,
                const xla::Shape& out,
                const std::vector<int64>& dimensions = {});

bool
PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                           const xla::Shape& shape);

}
}

#endif
