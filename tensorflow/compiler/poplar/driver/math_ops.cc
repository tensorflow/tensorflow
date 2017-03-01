#include <algorithm>

#include "tensorflow/compiler/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string c_conn("c");
static const std::string out_conn("out");

port::StatusOr<poplar::program::Program>
CreateUnaryElementwiseOp(poplar::Graph &graph,
                         const HloInstruction *inst,
                         const xla::Shape& output_shape,
                         TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));
  in = in.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(in));

  std::string vrtxTemplate;
  TF_ASSIGN_OR_RETURN(vrtxTemplate, LookupPoplarVertexName(inst->opcode()));
  std::string vertex_name = templateVertex(vrtxTemplate, poplar_data_type);

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
                             {{a_conn, in.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateBinaryElementwiseOp(poplar::Graph &graph,
                          const HloInstruction *inst,
                          const xla::Shape& output_shape,
                          TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1, 0));

  const std::string& poplar_data_type(graph.getTensorElementType(in0));

  std::string vrtxTemplate;
  TF_ASSIGN_OR_RETURN(vrtxTemplate, LookupPoplarVertexName(inst->opcode()));
  std::string vertex_name = templateVertex(vrtxTemplate, poplar_data_type);

  if (in0.shape() != in1.shape()) {

    tensorflow::BCast::Vec shape1 =
            convert_array<tensorflow::BCast::Vec>(in0.shape());
    tensorflow::BCast::Vec shape2 =
            convert_array<tensorflow::BCast::Vec>(in1.shape());

    tensorflow::BCast bcast(shape1, shape2);
    if (!bcast.IsValid()) {
      return port::Status(port::error::FAILED_PRECONDITION,
                          port::StrCat("Incompatible broadcast on ",
                                       inst->name()));
    }

    poplar::Tensor r0 =
            in0.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));
    poplar::Tensor r1 =
            in1.reshape(convert_array<std::vector<size_t>>(bcast.y_reshape()));

    in0 = TileTensor(bcast.x_bcast(), r0);
    in1 = TileTensor(bcast.y_bcast(), r1);
  }

  // And now flatten
  in0 = in0.flatten();
  in1 = in1.flatten();

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->name());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, in0.slice(begin, end)},
                              {b_conn, in1.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

// TODO - extend this to the semantics of XLA matmul
// TODO - which is that the last dimension of in0 is done 'dot'
// TODO - with the last but one of in1.
port::StatusOr<poplar::program::Program>
CreateMatMulOp(poplar::Graph &graph,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1, 0));

  const std::string& poplar_data_type(graph.getTensorElementType(in0));

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  uint64 dims = out.rank();
  auto num_rows = out.dim(dims-2);
  auto num_cols = out.dim(dims-1);

  poplar::ComputeSet cs = graph.addComputeSet(inst->name());
  const auto &device_info = graph.getDevice().getDeviceInfo();
  unsigned num_tiles = device_info.getNumTiles();

  poplar::Tensor a = in0;
  poplar::Tensor b = in1.dimShuffle({1,0});

  std::string vertex_name = templateVertex("Dot", poplar_data_type);

  for (unsigned r = 0; r < num_rows; ++r) {
    for (unsigned c = 0; c < num_cols; ++c) {
      auto v = graph.addVertex(cs, vertex_name,
                               {{a_conn, a[r]},
                                {b_conn, b[c]},
                                {out_conn, out[r].slice(c,c+1)}});

      graph.setTileMapping(v, (r + c * num_rows)  % num_tiles);
    }
  }

  return poplar::program::Execute(cs);
}

// TODO - select needs to support scalars
port::StatusOr<poplar::program::Program>
CreateSelectOp(poplar::Graph &graph,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor pred;
  TF_ASSIGN_OR_RETURN(pred, FindInstructionInput(tensor_map, inst, 0, 0));
  pred = pred.flatten();

  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 1, 0));
  in0 = in0.flatten();

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 2, 0));
  in1 = in1.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(in0));

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

  if (pred.dim(0) == 1) {
    std::string vertex_name = templateVertex("ScalarSelect", poplar_data_type);

    for (unsigned i = 0; i < num_workers; ++i) {
      const auto begin = i * N / num_workers;
      const auto end = (i + 1) * N / num_workers;
      auto v = graph.addVertex(cs, vertex_name,
                               {{"pred", pred[0]},
                                {a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)},
                                {out_conn, out.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    }
  } else {
    std::string vertex_name = templateVertex("Select", poplar_data_type);

    for (unsigned i = 0; i < num_workers; ++i) {
      const auto begin = i * N / num_workers;
      const auto end = (i + 1) * N / num_workers;
      auto v = graph.addVertex(cs, vertex_name,
                               {{"pred", pred.slice(begin, end)},
                                {a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)},
                                {out_conn, out.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    }

  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateCastOp(poplar::Graph &graph,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));
  in = in.flatten();

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  const std::string& poplar_in_data_type(graph.getTensorElementType(in));
  const std::string& poplar_out_data_type(graph.getTensorElementType(out));

  std::string vertex_name = templateVertex("Cast",
                                           poplar_in_data_type,
                                           poplar_out_data_type);

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, in.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateClampOp(poplar::Graph &graph,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {

  poplar::Tensor min;
  TF_ASSIGN_OR_RETURN(min, FindInstructionInput(tensor_map, inst, 0, 0));
  if (!PoplarShapeMatchesXLAShape(min, output_shape)) {
    TF_ASSIGN_OR_RETURN(min, BroadcastTensor(min, output_shape));
  }
  min = min.flatten();

  poplar::Tensor arg;
  TF_ASSIGN_OR_RETURN(arg, FindInstructionInput(tensor_map, inst, 1, 0));
  if (!PoplarShapeMatchesXLAShape(arg, output_shape)) {
    TF_ASSIGN_OR_RETURN(arg, BroadcastTensor(arg, output_shape));
  }
  arg = arg.flatten();

  poplar::Tensor max;
  TF_ASSIGN_OR_RETURN(max, FindInstructionInput(tensor_map, inst, 2, 0));
  if (!PoplarShapeMatchesXLAShape(max, output_shape)) {
    TF_ASSIGN_OR_RETURN(max, BroadcastTensor(max, output_shape));
  }
  max = max.flatten();

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(arg));

  std::string vertex_name = templateVertex("Clamp", poplar_data_type);

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, min.slice(begin, end)},
                              {b_conn, arg.slice(begin, end)},
                              {c_conn, max.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

}
}

