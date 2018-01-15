#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"

#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace poplar {
  class Tensor;
  class Graph;
  class Type;
}

namespace port = ::perftools::gputools::port;

namespace xla {

namespace poplarplugin {

struct CompilerResources;

port::StatusOr<poplar::Type>
PoplarDataType(const xla::Shape& shape);

std::vector<size_t>
PoplarShapeFromXlaShape(const xla::Shape &xla_shape);

xla::Shape
XlaShapeFromPoplarShape(PrimitiveType element_type,
                        const std::vector<size_t> &poplar_shape);

poplar::Tensor
ConvertToDeviceLayout(const Shape& shape, const poplar::Tensor& tensor);

poplar::Tensor
ConvertFromDeviceLayout(const Shape& shape, const poplar::Tensor& tensor);

bool
PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                           const xla::Shape& shape);

port::StatusOr<poplar::Tensor>
AddPlainTensor(poplar::Graph& graph,
               const HloInstruction* inst,
               const xla::Shape& shape);

port::StatusOr<poplar::Tensor>
AddTensor(poplar::Graph& graph,
          const TensorSource& src,
          const xla::Shape& shape,
          CompilerResources& resources);

port::StatusOr<poplar::Tensor>
AddConstantTensor(poplar::Graph& graph,
                  const TensorSource& src,
                  const xla::Shape& shape,
                  const xla::Literal& literal,
                  CompilerResources& resources);

template <typename T>
poplar::Tensor
TileTensor(const T& multiples,
           const poplar::Tensor& in);

port::StatusOr<poplar::Tensor>
PadTensor(const PaddingConfig& cfg,
          const poplar::Tensor& in,
          const poplar::Tensor& pad);

port::StatusOr<poplar::Tensor>
PadWithConstantZero(poplar::Graph& graph,
                    const PaddingConfig& cfg,
                    const poplar::Tensor &in);

port::StatusOr<poplar::Tensor>
ReverseTensor(const poplar::Tensor &in,
              const std::vector<int64>& dimensions);

port::StatusOr<poplar::Tensor>
BroadcastTensor(const poplar::Tensor &in,
                const xla::Shape& out,
                const std::vector<int64>& dimensions = {});

}
}

#endif
