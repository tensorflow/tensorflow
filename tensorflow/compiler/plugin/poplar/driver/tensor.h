#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace poplar {
class Tensor;
class Graph;
class Type;
}  // namespace poplar

namespace xla {
namespace poplarplugin {

struct CompilerResources;

StatusOr<poplar::Type> PoplarDataType(const xla::Shape& shape);

std::vector<size_t> PoplarShapeFromXlaShape(const xla::Shape& xla_shape);

xla::Shape XlaShapeFromPoplarShape(PrimitiveType element_type,
                                   const std::vector<size_t>& poplar_shape);

poplar::Tensor ConvertToDeviceLayout(const Shape& shape,
                                     const poplar::Tensor& tensor);

poplar::Tensor ConvertFromDeviceLayout(const Shape& shape,
                                       const poplar::Tensor& tensor);

bool PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                                const xla::Shape& shape);

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla);

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla,
    poplar::Tensor& physical_layout);

StatusOr<poplar::Tensor> AddPlainTensor(poplar::Graph& graph,
                                        const std::string& debug_name,
                                        const xla::Shape& shape);

StatusOr<poplar::Tensor> AddTensor(poplar::Graph& graph,
                                   const TensorSource& src,
                                   const xla::Shape& shape,
                                   CompilerResources& resources,
                                   const TensorMap& tensor_map);

StatusOr<poplar::Tensor> AddConstantTensor(poplar::Graph& graph,
                                           const TensorSource& src,
                                           const xla::Shape& shape,
                                           const xla::Literal& literal,
                                           CompilerResources& resources,
                                           const TensorMap& tensor_map);

StatusOr<poplar::Tensor> AddIotaTensor(poplar::Graph& graph,
                                       const TensorSource& src,
                                       const xla::Shape& shape,
                                       int64 iota_dimension,
                                       CompilerResources& resources,
                                       const TensorMap& tensor_map);

template <typename T>
poplar::Tensor TileTensor(const T& multiples, const poplar::Tensor& in);

StatusOr<poplar::Tensor> PadTensor(const PaddingConfig& cfg,
                                   const poplar::Tensor& in,
                                   const poplar::Tensor& pad);

StatusOr<poplar::Tensor> ReverseTensor(const poplar::Tensor& in,
                                       const std::vector<int64>& dimensions);

StatusOr<poplar::Tensor> BroadcastTensor(
    const poplar::Tensor& in, const xla::Shape& out,
    const std::vector<int64>& dimensions = {});

}  // namespace poplarplugin
}  // namespace xla

#endif
