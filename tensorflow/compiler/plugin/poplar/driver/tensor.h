#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"

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

StatusOr<poplar::Type> PoplarDataType(const xla::PrimitiveType& element_type);

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

StatusOr<poplar::Tensor> AddNormScaleTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* layout, int64 layout_output_idx,
    const unsigned feature_dimension,
    std::vector<const HloInstruction*> forward_path,
    const TensorMap& tensor_map);

StatusOr<poplar::Tensor> AddNormOffsetTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* layout, int64 layout_output_idx,
    const unsigned feature_dimension,
    std::vector<const HloInstruction*> forward_path,
    const TensorMap& tensor_map);

// Returns true if the given tensor source has a special layout allocation
// target.
bool HasTensorAllocationTarget(const TensorSource& src,
                               const CompilerResources& resources);

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

Status AddOutputTensor(TensorMap& map, const HloInstruction* inst, int64 n,
                       const poplar::Tensor& tensor);

/* Returns a pair of numbers representing the half-open range of indicies
 * which a particular input to a tuple represents in the flattened output.
 *
 * eg.
 *   a = tuple(f32[], tuple(f32[], f32[]), f32)(b c d)
 *
 *   a is a tuple containing a scalar, a tuple of 2 scalars, and another scalar
 *   and flattened it has 4 tensors
 *
 *   FindTupleInputIndices(a, 0) = (0,1)
 *   FindTupleInputIndices(a, 1) = (1,3)
 *   FindTupleInputIndices(a, 2) = (3,4)
 */
std::pair<int64, int64> FindTupleInputIndices(const HloInstruction* tuple,
                                              int64 input);

/* This returns the vector of all poplar tensors which are part of the n'th
 * member of the tuple which is the input to the instruction.
 */
ArgVector FindTupleInInstructionInput(TensorMap& map, CompilerResources& res,
                                      const HloInstruction* inst, int64 input,
                                      int64 n, poplar::program::Sequence& seq,
                                      const bool expand_constants = true);

/* This returns a vector of all poplar tensors which are outputs of the inst
 * operand index `input` in range [range.first, range.second).
 */
ArgVector FindInstructionInputsInRange(TensorMap& map, CompilerResources& res,
                                       const HloInstruction* inst, int64 input,
                                       std::pair<int64, int64> range,
                                       poplar::program::Sequence& seq,
                                       const bool expand_constants = true);

/* This returns the single poplar tensor which is the non-tuple input to the
 * input to the instruction
 */
StatusOr<poplar::Tensor> FindInstructionInput(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, poplar::program::Sequence& seq,
    const bool expand_constants = true);

/* This returns a vector of all poplar tensors which are part of the tuple
 * or non-tuple on the input to the instruction
 */
ArgVector FindInstructionInputs(TensorMap& map, CompilerResources& res,
                                const HloInstruction* inst, int64 input,
                                poplar::program::Sequence& seq,
                                const bool expand_constants = true);

/* Sometimes an inplace op cannot be performed because the input/output tensor
 * is not parallel writable or because further analysis has shown that the op
 * can no longer be in place. If that's the case, this function will add an
 * extra tensor copy and use that tensor as the input/output tensor.
 */
StatusOr<ArgVectors> GetInplaceOutputTensors(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq, const bool expand_constants = true);

/* This returns a vector of poplar tensors which are all of the outputs from
 * the given instruction
 */
OutVector FindInstructionOutputs(const TensorMap& map,
                                 const HloInstruction* inst);

/* This returns a vector of poplar tensors which are all of the outputs from
 * the given instruction - any wide constants are expanded - TODO T5364
 */
OutVector FindExpandedInstructionOutputs(TensorMap& map, CompilerResources& res,
                                         const HloInstruction* inst,
                                         poplar::program::Sequence& seq);

/* Generate a JSON struture describing the tensor mappings
 */
std::string GetTensorMappingJson(const poplar::Graph& graph,
                                 const TensorMaps& tensor_map);

}  // namespace poplarplugin
}  // namespace xla

#endif
