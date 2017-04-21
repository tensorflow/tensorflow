#include <algorithm>

#include "tensorflow/compiler/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

static port::StatusOr<poplar::program::Program>
CreateRandomUniformOp(poplar::Graph &graph,
                      CompilerResources& res,
                      const HloInstruction *inst,
                      const xla::Shape& output_shape,
                      TensorMap& tensor_map) {

  // Find the input tensor
  poplar::Tensor l;
  TF_ASSIGN_OR_RETURN(l, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor u;
  TF_ASSIGN_OR_RETURN(u, FindInstructionInput(tensor_map, inst, 1, 0));

  std::string poplar_data_type;
  TF_ASSIGN_OR_RETURN(poplar_data_type, PoplarDataType(inst->shape()));

  std::string vertex_name = templateVertex("RandomUniform", poplar_data_type);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{"lower", l},
                              {"upper", u},
                              {"out", out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

static port::StatusOr<poplar::program::Program>
CreateRandomNormalOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape& output_shape,
                     TensorMap& tensor_map) {

  // Find the input tensor
  poplar::Tensor mean;
  TF_ASSIGN_OR_RETURN(mean, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor sd;
  TF_ASSIGN_OR_RETURN(sd, FindInstructionInput(tensor_map, inst, 1, 0));

  std::string poplar_data_type;
  TF_ASSIGN_OR_RETURN(poplar_data_type, PoplarDataType(inst->shape()));

  std::string vertex_name = templateVertex("RandomNormal", poplar_data_type);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{"mean", mean},
                              {"sd", sd},
                              {"out", out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

static port::StatusOr<poplar::program::Program>
CreateRandomBernoulliOp(poplar::Graph &graph,
                        CompilerResources& res,
                        const HloInstruction *inst,
                        const xla::Shape& output_shape,
                        TensorMap& tensor_map) {

  // Find the input tensor
  poplar::Tensor mean;
  TF_ASSIGN_OR_RETURN(mean, FindInstructionInput(tensor_map, inst, 0, 0));

  std::string poplar_data_type;
  TF_ASSIGN_OR_RETURN(poplar_data_type, PoplarDataType(inst->shape()));

  std::string vertex_name = templateVertex("RandomBernoulli", poplar_data_type);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{"mean", mean},
                              {"out", out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateRandomOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  switch (inst->random_distribution()) {
    case RandomDistribution::RNG_UNIFORM:
      return CreateRandomUniformOp(graph, res, inst, output_shape, tensor_map);
    case RandomDistribution::RNG_NORMAL:
      return CreateRandomNormalOp(graph, res, inst, output_shape, tensor_map);
    case RandomDistribution::RNG_BERNOULLI:
      return CreateRandomBernoulliOp(graph, res, inst, output_shape, tensor_map);
    default:
      return port::Status(port::error::FAILED_PRECONDITION,
                          port::StrCat("Invalid random distribution on ",
                                       inst->name()));
  }
}

}
}

