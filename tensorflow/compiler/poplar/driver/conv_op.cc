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

#include <popstd/ActivationMapping.hpp>
#include <popconv/Convolution.hpp>
#include <popconv/ConvPlan.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {


port::StatusOr <poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             const HloInstruction *inst,
             const xla::Shape &output_shape,
             TensorMap &tensor_map) {

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));

  // Find the input tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 1, 0));

  const Window& window(inst->window());

  if (in.rank() != 4 || kernel.rank() != 4) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Poplar supports 2D convolution only: ", inst->name()));
  }

  if (window.dimensions().size() != 2) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Invalid window dimension count on ", inst->name()));
  }

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  const std::string& dtype(graph.getTensorElementType(in));

  const std::vector<size_t> &input_dims = in.shape();
  const std::vector<size_t> &kernel_dims = kernel.shape();
  const std::vector<size_t> &output_dims = out.shape();

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.batch_dimension()];
  unsigned int n_i = input_dims[dims.feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];
  unsigned int n_y = input_dims[dims.spatial_dimensions(0)];
  unsigned int n_x = input_dims[dims.spatial_dimensions(1)];
  unsigned int f_y = kernel_dims[dims.kernel_spatial_dimensions(0)];
  unsigned int f_x = kernel_dims[dims.kernel_spatial_dimensions(1)];
  unsigned int o_y = output_dims[dims.spatial_dimensions(0)];
  unsigned int o_x = output_dims[dims.spatial_dimensions(1)];

  // Create a plan
  popconv::Planner planner;
  popconv::Plan plan = planner.createPlan(n_y, n_x, n_i, f_y, f_x,
                                          window.dimensions(0).stride(),
                                          window.dimensions(1).stride(),
                                          window.dimensions(0).padding_low(),
                                          window.dimensions(1).padding_low(),
                                          n_o, n_b, dtype, dtype,
                                          false, graph,
                                          {});

  const unsigned in_chan_groups = n_i / plan.inChansPerGroup;
  const unsigned out_chan_groups = n_o / plan.partialChansPerGroup;

  in = in.dimShuffle({(unsigned int)dims.batch_dimension(),
                      (unsigned int)dims.feature_dimension(),
                      (unsigned int)dims.spatial_dimensions(0),
                      (unsigned int)dims.spatial_dimensions(1)});
  in = in.reshape({n_b,
                   in_chan_groups,
                   plan.inChansPerGroup,
                   n_y,
                   n_x});
  in = in.dimShuffle({0, 1, 3, 4, 2});
  kernel = kernel.dimShuffle({(unsigned int)dims.kernel_spatial_dimensions(0),
                              (unsigned int)dims.kernel_spatial_dimensions(1),
                              (unsigned int)dims.kernel_input_feature_dimension(),
                              (unsigned int)dims.kernel_output_feature_dimension()});
  kernel = kernel.reshape({f_y, f_x,
                           in_chan_groups,
                           plan.inChansPerGroup,
                           out_chan_groups,
                           plan.partialChansPerGroup});
  kernel = kernel.dimShuffle({4, 2, 0, 1, 5, 3});

  out = out.dimShuffle({(unsigned int)dims.batch_dimension(),
                        (unsigned int)dims.feature_dimension(),
                        (unsigned int)dims.spatial_dimensions(0),
                        (unsigned int)dims.spatial_dimensions(1)});
  out = out.reshape({n_b,
                   n_o,
                   1,
                   o_y,
                   o_x});
  out = out.dimShuffle({0, 1, 3, 4, 2});

  // TODO - ideally don't even have a biases tensor
  poplar::Tensor biases = graph.addConstantTensor(dtype, {n_o}, 0);

  popstd::mapActivations(graph, in);
  popconv::mapWeights(kernel, graph, plan, input_dims[0]);
  popconv::mapBiases(biases, graph, out);
  popstd::mapActivations(graph, out);

  // Add the convolution
  poplar::program::Program prog;
  prog = popconv::convolution(graph, plan,
                              window.dimensions(0).stride(), // stride y
                              window.dimensions(1).stride(), // stride x
                              window.dimensions(0).padding_low(), // padding y
                              window.dimensions(1).padding_low(), // padding x
                              in,
                              kernel,
                              biases,
                              out,
                              dtype,
                              false);

  return prog;
}

}
}
