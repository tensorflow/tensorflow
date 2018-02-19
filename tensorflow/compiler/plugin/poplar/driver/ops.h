#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

#include <poplar/Program.hpp>
#include <popconv/Convolution.hpp>

namespace poplar {
  class Graph;
  class Tensor;
}

namespace port = ::perftools::gputools::port;

namespace xla {
class HloInstruction;
class HloComputation;
class Literal;

namespace poplarplugin {

struct CompilerResources;

using TensorMap = std::map<std::pair<std::string, int64>, poplar::Tensor>;

using OutVector = std::vector<poplar::Tensor>;
using ArgVector = std::vector<poplar::Tensor>;
using ArgVectors = std::vector<ArgVector>;

typedef poplar::Tensor (*popops_unary_fn)(poplar::Graph &graph,
                                          poplar::Tensor A,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugPrefix);

typedef poplar::Tensor (*popops_binary_fn)(poplar::Graph &graph,
                                           poplar::Tensor A, poplar::Tensor B,
                                           poplar::program::Sequence &prog,
                                           const std::string &debugPrefix);

typedef void (*popops_inplace_fn)(poplar::Graph &graph,
                                  poplar::Tensor A, poplar::Tensor B,
                                  poplar::program::Sequence &prog,
                                  const std::string &debugPrefix);

port::StatusOr<popops_unary_fn>  LookupUnaryFn(const HloInstruction*);
port::StatusOr<popops_binary_fn> LookupBinaryFn(const HloInstruction*);
port::StatusOr<popops_inplace_fn> LookupBinaryInPlaceFn(const HloInstruction*);

port::Status SetVertexField(poplar::Graph &graph,
                            const poplar::FieldRef &field,
                            const Literal &literal);

port::StatusOr<popconv::ConvParams>
GetConvolutionParameters(const HloInstruction* inst, bool depthwise);

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInputToTensorflow(const HloInstruction* inst,
                                    const poplar::Tensor& tensor);

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeightsToTensorflow(const HloInstruction* inst,
                                      const poplar::Tensor& tensor);

poplar::Tensor RemoveGroupsDimensionFromWeights(const poplar::Tensor& t, bool);

poplar::Tensor AddGroupsDimensionToWeights(const poplar::Tensor& t, bool);

port::Status
AddOutputTensor(TensorMap& map,
                const HloInstruction* inst,
                int64 n,
                const poplar::Tensor& tensor);

/* This returns the vector of all poplar tensors which are part of the n'th
 * member of the tuple which is the input to the instruction.
 */
ArgVector
FindTupleInInstructionInput(const TensorMap& map,
                            const HloInstruction* inst,
                            int64 input,
                            int64 n);

/* This returns the single poplar tensor which is the non-tuple input to the
 * input to the instruction
 */
port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     int64 input);

/* This returns a vector of all poplar tensors which are part of the tuple
 * or non-tuple on the input to the instruction
 */
ArgVector
FindInstructionInputs(const TensorMap& map,
                      const HloInstruction* inst,
                      int64 input);

/* This returns a vector of poplar tensors which are all of the outputs from
 * the given instruction
 */
OutVector
FindInstructionOutputs(const TensorMap& map,
                       const HloInstruction* inst);

/* Ops */

port::StatusOr<poplar::program::Program>
CreateUnaryElementwiseOp(poplar::Graph &graph,
                         CompilerResources& res,
                         const HloInstruction *inst,
                         const xla::Shape& output,
                         TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateBinaryElementwiseOp(poplar::Graph &graph,
                          CompilerResources& res,
                          const HloInstruction *inst,
                          const xla::Shape& output,
                          TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateMatMulOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output,
               TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSelectOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output,
               TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateCastOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output,
             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateClampOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSimpleReduction(poplar::Graph &graph,
                      CompilerResources& res,
                      const HloInstruction *inst,
                      const xla::Shape& output,
                      TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSimpleWindowReduction(poplar::Graph &graph,
                            CompilerResources& res,
                            const HloInstruction *inst,
                            const xla::Shape& output_shape,
                            TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreatePoplibsWindowReduction(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateParallelMap(poplar::Graph &graph,
                  CompilerResources& res,
                  const HloInstruction *inst,
                  const xla::Shape& output,
                  TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateCallOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output,
             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateFusionOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output,
               TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateWhileOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateBiasAddOp(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape& output_shape,
                TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateBiasAddBcastOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape& output_shape,
                     TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
TruncatedNormalScale(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape& output_shape,
                     TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
TruncatedNormal(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape& output_shape,
                TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
RandomNormalScale(poplar::Graph &graph,
                  CompilerResources& res,
                  const HloInstruction *inst,
                  const xla::Shape& output_shape,
                  TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
RandomUniformScale(poplar::Graph &graph,
                   CompilerResources& res,
                   const HloInstruction *inst,
                   const xla::Shape& output_shape,
                   TensorMap& tensor_map) ;

port::StatusOr<poplar::program::Program>
RandomNormal(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
RandomUniform(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSimpleSelectAndScatter(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSliceUpdateOp(poplar::Graph &graph,
                    CompilerResources& res,
                    const HloInstruction *inst,
                    const xla::Shape& output_shape,
                    TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSliceOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateDynamicSliceUpdateOp(poplar::Graph &graph,
                           CompilerResources& res,
                           const HloInstruction *inst,
                           const xla::Shape& output_shape,
                           TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateDynamicSliceOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape& output_shape,
                     TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateReluOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateReluGradOp(poplar::Graph &graph,
                 CompilerResources& res,
                 const HloInstruction *inst,
                 const xla::Shape& output_shape,
                 TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSigmoidOp(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape& output_shape,
                TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateDepthwiseConvolutionOp(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
Create2DConvWithReverse(poplar::Graph &graph,
                        CompilerResources& res,
                        const HloInstruction *inst,
                        const xla::Shape& output_shape,
                        TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
ConvBiasApply(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateWideConstant(poplar::Graph &graph,
                   CompilerResources& res,
                   const HloInstruction *inst,
                   const xla::Shape& output_shape,
                   TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateIfOp(poplar::Graph &graph,
           CompilerResources& res,
           const HloInstruction *inst,
           const xla::Shape& output,
           TensorMap& tensor_map);

/* Optimization tests */

bool
IsPoplibsPool(const HloInstruction*, const HloComputation*);

bool
IsSimpleSelection(const HloComputation*);

bool
IsReducableArtithmetic(const HloComputation*);

port::StatusOr<bool>
IsParallelMap(const HloInstruction*, const HloComputation*);

}
}

#endif
