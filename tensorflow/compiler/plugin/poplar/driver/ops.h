#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_H_

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

namespace poplarplugin {

struct CompilerResources;

using TensorMap = std::map<std::pair<std::string, int64>, poplar::Tensor>;

template<typename To, typename From>
To convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
  }
  return out;
};

port::StatusOr<popconv::ConvParams>
GetConvolutionParameters(const HloInstruction* inst);

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInput(const HloInstruction* inst,
                        const poplar::Tensor& tensor);

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeights(const HloInstruction* inst,
                          const poplar::Tensor& tensor);

port::Status
AddOutputTensor(TensorMap& map,
                const HloInstruction* inst,
                uint64 n,
                const poplar::Tensor& tensor);

port::StatusOr<poplar::Tensor>
FindInstructionInput(const TensorMap& map,
                     const HloInstruction* inst,
                     uint64 input,
                     uint64 n);

port::StatusOr<poplar::Tensor>
FindInstructionOutput(const TensorMap& map,
                      const HloInstruction* inst,
                      uint64 n);

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
CreateRandomOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output,
               TensorMap& tensor_map);

port::StatusOr<poplar::program::Program>
CreateSimpleSelectAndScatter(poplar::Graph &graph,
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
CreateSliceUpdateOp(poplar::Graph &graph,
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

/* Optimization tests */

bool
IsPoplibsPool(const HloInstruction*, const HloComputation*);

bool
IsSimpleSelection(const HloInstruction*, const HloComputation*);

bool
IsReducableArtithmetic(const HloInstruction*, const HloComputation*);

port::StatusOr<bool>
IsParallelMap(const HloInstruction*, const HloComputation*);

}
}

#endif
