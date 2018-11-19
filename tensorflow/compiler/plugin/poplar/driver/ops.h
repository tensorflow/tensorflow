#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#include <poplar/Program.hpp>
#include <poplin/Convolution.hpp>
#include <popops/Expr.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>
#include "absl/container/inlined_vector.h"

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;
class HloComputation;
class Literal;

namespace poplarplugin {

struct CompilerResources;

using TensorKey = std::pair<std::string, int64>;
using TensorMap = std::map<TensorKey, poplar::Tensor>;
using TensorMaps = std::map<std::string, TensorMap>;

using OutVector = std::vector<poplar::Tensor>;
using ArgVector = std::vector<poplar::Tensor>;
using ArgVectors = std::vector<ArgVector>;

typedef void (*popops_inplace_fn)(poplar::Graph& graph, poplar::Tensor A,
                                  poplar::Tensor B,
                                  poplar::program::Sequence& prog,
                                  const std::string& debugPrefix);

StatusOr<popops::expr::UnaryOpType> LookupUnaryFn(const HloInstruction*);

StatusOr<popops::expr::BinaryOpType> LookupBinaryFn(const HloInstruction*);

Status SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                      const Literal& literal);

poplar::Graph& GetGraph(CompilerResources&, const HloInstruction*);

// Convert a poplar/poplibs exception to a Tensorflow error Status
Status PoplarExceptionToTensorflowStatus(const std::string& prefix,
                                         const std::exception& e);

StatusOr<poplin::ConvParams> GetConvolutionParameters(
    const HloInstruction* operand_op, const HloInstruction* conv_op,
    int64 input_index, int64 kernel_index);

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionInputToPoplar(const HloInstruction* inst,
                                               const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionWeightsToPoplar(const HloInstruction* inst,
                                                 const poplar::Tensor& tensor,
                                                 bool swap_features);

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionOutputToPoplar(const HloInstruction* inst,
                                                const poplar::Tensor& tensor);

poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t,
                                                bool flipped);

poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped);

/* Ops */

StatusOr<poplar::program::Program> CreateUnaryElementwiseOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBinaryElementwiseOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateScaledInplace(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateConvScaledInplace(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateMatMulForDotOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSelectOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCastOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateClampOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreatePoplibsWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBwdMaxPool(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateParallelMap(CompilerResources& res,
                                                     const HloInstruction* inst,
                                                     const xla::Shape& output,
                                                     TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCallOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateFusionOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateConv2D(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBiasAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> TruncatedNormal(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> RandomNormalScale(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> RandomUniformScale(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> RandomNormal(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> RandomUniform(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleSelectAndScatter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSliceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSliceOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateDynamicSliceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateDynamicSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReluOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReluGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSigmoidOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSigmoidGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> Create2DConvWithReverse(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateDepthwiseBackpropFilter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> ConvBiasApply(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateWideConstant(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateIfOp(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output,
                                              TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateZeroPadOp(CompilerResources& res,
                                                   const HloInstruction* inst,
                                                   const xla::Shape& output,
                                                   TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreatePaddingReduceWindow(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSort(CompilerResources& res,
                                              const HloInstruction* inst,
                                              TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBatchNormInf(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBatchNormTraining(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateBatchNormGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map);

/* Optimization tests */

bool IsPoplibsPool(const HloInstruction*, const HloComputation*);

bool IsSimpleSelection(const HloComputation*);

bool IsReducableArtithmetic(const HloComputation*);

StatusOr<bool> IsParallelMap(const HloInstruction*, const HloComputation*);

/* Op Creation Helpers */

StatusOr<poplar::program::Sequence> CreateSort(
    poplar::Graph& graph, poplar::Tensor input, const int64 dimension,
    const std::string& debug_name = "");

StatusOr<poplar::program::Sequence> CreateSort(
    poplar::Graph& graph, poplar::Tensor key, poplar::Tensor value,
    const int64 dimension, const std::string& debug_name = "");

}  // namespace poplarplugin
}  // namespace xla

#endif
