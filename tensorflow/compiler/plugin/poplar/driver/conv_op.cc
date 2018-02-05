#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <popconv/Convolution.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

port::StatusOr<popconv::ConvParams>
GetConvolutionParameters(const HloInstruction* inst) {

  const Shape& input = inst->operand(0)->shape();
  const Shape& kernel = inst->operand(1)->shape();

  const Window& window(inst->window());

  poplar::Type dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];

  std::vector<std::size_t> n_s;
  std::vector<std::size_t> f_s;
  std::vector<unsigned int> w_s;
  std::vector<unsigned int> p_l;
  std::vector<unsigned int> p_u;
  std::vector<unsigned int> t_l;
  std::vector<unsigned int> t_u;
  std::vector<unsigned int> d_i;
  std::vector<unsigned int> d_w;
  std::vector<unsigned int> zeros;
  std::vector<bool> falses;

  for (int64 i=0; i < window.dimensions().size(); i++) {
    n_s.push_back(input_dims[dims.input_spatial_dimensions(i)]);
    f_s.push_back(kernel_dims[dims.kernel_spatial_dimensions(i)]);
    w_s.push_back(window.dimensions(i).stride());
    if (window.dimensions(i).padding_low() < 0) {
      unsigned int p = -window.dimensions(i).padding_low();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d-1) / d;
      unsigned int pad = p % d;
      t_l.push_back(trunc);
      p_l.push_back(pad);
    } else {
      p_l.push_back(window.dimensions(i).padding_low());
      t_l.push_back(0);
    }
    if (window.dimensions(i).padding_high() < 0) {
      unsigned int p = -window.dimensions(i).padding_high();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d-1) / d;
      unsigned int pad = p % d;
      t_u.push_back(trunc);
      p_u.push_back(pad);
    } else {
      p_u.push_back(window.dimensions(i).padding_high());
      t_u.push_back(0);
    }
    d_i.push_back(window.dimensions(i).base_dilation());
    d_w.push_back(window.dimensions(i).window_dilation());
    falses.push_back(false);
    zeros.push_back(0);
  }

  popconv::ConvParams params(dtype, n_b, n_s, f_s, n_i, n_o, 1,
                             t_l, t_u, d_i, p_l, p_u, falses,
                             zeros, zeros, d_w, zeros, zeros, falses,
                             zeros, zeros, w_s, zeros, zeros);

  return params;
}

port::StatusOr<popconv::ConvParams>
GetDepthConvolutionParameters(const HloInstruction* inst) {

  const Shape& input = inst->operand(0)->shape();
  const Shape& kernel = inst->operand(1)->shape();

  const Window& window(inst->window());

  poplar::Type dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];

  std::vector<std::size_t> n_s;
  std::vector<std::size_t> f_s;
  std::vector<unsigned int> w_s;
  std::vector<unsigned int> p_l;
  std::vector<unsigned int> p_u;
  std::vector<unsigned int> t_l;
  std::vector<unsigned int> t_u;
  std::vector<unsigned int> d_i;
  std::vector<unsigned int> d_w;
  std::vector<unsigned int> zeros;
  std::vector<bool> falses;

  for (int64 i=0; i < window.dimensions().size(); i++) {
    n_s.push_back(input_dims[dims.input_spatial_dimensions(i)]);
    f_s.push_back(kernel_dims[dims.kernel_spatial_dimensions(i)]);
    w_s.push_back(window.dimensions(i).stride());
    if (window.dimensions(i).padding_low() < 0) {
      unsigned int p = -window.dimensions(i).padding_low();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d-1) / d;
      unsigned int pad = p % d;
      t_l.push_back(trunc);
      p_l.push_back(pad);
    } else {
      p_l.push_back(window.dimensions(i).padding_low());
      t_l.push_back(0);
    }
    if (window.dimensions(i).padding_high() < 0) {
      unsigned int p = -window.dimensions(i).padding_high();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d-1) / d;
      unsigned int pad = p % d;
      t_u.push_back(trunc);
      p_u.push_back(pad);
    } else {
      p_u.push_back(window.dimensions(i).padding_high());
      t_u.push_back(0);
    }
    d_i.push_back(window.dimensions(i).base_dilation());
    d_w.push_back(window.dimensions(i).window_dilation());
    falses.push_back(false);
    zeros.push_back(0);
  }

  n_o = n_o / n_i;

  popconv::ConvParams params(dtype, n_b, n_s, f_s, 1, n_o, n_i,
                             t_l, t_u, d_i, p_l, p_u, falses,
                             zeros, zeros, d_w, zeros, zeros, falses,
                             zeros, zeros, w_s, zeros, zeros);

  return params;
}

static popconv::Pass GetConvolutionPass(const HloInstruction* inst) {
  if (IsForwardConvolution(inst)) {
    return popconv::Pass::TRAINING_FWD;
  }
  if (IsGradientConvolution(inst)) {
    return popconv::Pass::TRAINING_BWD;
  }
  if (IsWeightUpdateConvolution(inst)) {
    return popconv::Pass::TRAINING_WU;
  }
  return popconv::Pass::NONE;
}

static bool is_identity_shuffle(const std::vector<unsigned int> shuffle) {
  for (unsigned int i=0; i<4; i++) {
    if (shuffle[i] != i) return false;
  }
  return true;
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInputToPoplar(const HloInstruction* inst,
                                const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(2 + d.input_spatial_dimensions_size());
  shuffle[0] = d.input_batch_dimension();
  shuffle[1] = d.input_feature_dimension();
  for (int64 i = 0; i < d.input_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = d.input_spatial_dimensions(i);
  }

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeightsToPoplar(const HloInstruction* inst,
                                  const poplar::Tensor& tensor,
                                  bool swap_features) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(2 + d.kernel_spatial_dimensions_size());
  if (swap_features) {
    shuffle[0] = d.kernel_input_feature_dimension();
    shuffle[1] = d.kernel_output_feature_dimension();
  } else {
    shuffle[0] = d.kernel_output_feature_dimension();
    shuffle[1] = d.kernel_input_feature_dimension();
  }
  for (int64 i = 0; i < d.kernel_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = d.kernel_spatial_dimensions(i);
  }

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInputToTensorflow(const HloInstruction* inst,
                                    const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(2 + d.input_spatial_dimensions_size());
  shuffle[d.input_batch_dimension()] = 0;
  shuffle[d.input_feature_dimension()] = 1;
  for (int64 i = 0; i < d.input_spatial_dimensions_size(); i++) {
    shuffle[d.input_spatial_dimensions(i)] = i + 2;
  }

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeightsToTensorflow(const HloInstruction* inst,
                                      const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(2 + d.kernel_spatial_dimensions_size());
  shuffle[d.kernel_output_feature_dimension()] = 0;
  shuffle[d.kernel_input_feature_dimension()] = 1;
  for (int64 i = 0; i < d.kernel_spatial_dimensions_size(); i++) {
    shuffle[d.kernel_spatial_dimensions(i)] = i + 2;
  }

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionOutputToTensorflow(const HloInstruction* inst,
                                     const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(2 + d.output_spatial_dimensions_size());
  shuffle[d.output_batch_dimension()] = 0;
  shuffle[d.output_feature_dimension()] = 1;
  for (int64 i = 0; i < d.output_spatial_dimensions_size(); i++) {
    shuffle[d.output_spatial_dimensions(i)] = i + 2;
  }

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

poplar::Tensor RemoveGroupsDimensionFromWeights(const poplar::Tensor& t) {
  std::vector<std::size_t> shape;
  for (int64 i = 1; i < t.rank(); i++) {
    shape.push_back(t.dim(i));
  }
  return t.reshape(shape);
}

poplar::Tensor AddGroupsDimensionToWeights(const poplar::Tensor& t) {
  std::vector<std::size_t> shape;
  shape.push_back(1);
  for (int64 i = 0; i < t.rank(); i++) {
    shape.push_back(t.dim(i));
  }
  return t.reshape(shape);
}

port::StatusOr <poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape &output_shape,
             TensorMap &tensor_map) {

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 1));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = GetConvolutionPass(inst);

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst));

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(in, ShuffleConvolutionInputToPoplar(inst, in));

  TF_ASSIGN_OR_RETURN(kernel, ShuffleConvolutionWeightsToPoplar(inst, kernel,
                                                                false));

  kernel = AddGroupsDimensionToWeights(kernel);

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, in, kernel, params,
                                            false, prog, inst->name(), opts);

  TF_ASSIGN_OR_RETURN(out, ShuffleConvolutionOutputToTensorflow(inst, out));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

port::StatusOr <poplar::program::Program>
CreateBiasAddOp(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape &output_shape,
                TensorMap &tensor_map) {
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor bias;
  TF_ASSIGN_OR_RETURN(bias, FindInstructionInput(tensor_map, inst, 1));

  // Should this be taken from the convolution dimension numbers?
  poplar::Tensor shuffled_in = in.dimShuffle({0, 3, 1, 2});

  poplar::program::Sequence prog;
  popconv::addBias(graph, shuffled_in, bias, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

port::StatusOr <poplar::program::Program>
CreateBiasAddBcastOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape &output_shape,
                     TensorMap &tensor_map) {
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  poplar::Tensor bias;
  TF_ASSIGN_OR_RETURN(bias, FindInstructionInput(tensor_map, inst, 0));

  // Should this be taken from the convolution dimension numbers?
  poplar::Tensor shuffled_in = in.dimShuffle({0, 3, 1, 2});

  poplar::program::Sequence prog;
  popconv::addBias(graph, shuffled_in, bias, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

port::StatusOr<poplar::program::Program>
ConvBiasApply(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {

  const HloInstruction* root =
          inst->to_apply()->root_instruction();

  // Find the deltas
  poplar::Tensor deltas;
  TF_ASSIGN_OR_RETURN(deltas, FindInstructionInput(tensor_map, inst, 0));

  // Find the biases
  poplar::Tensor biases;
  TF_ASSIGN_OR_RETURN(biases, FindInstructionInput(tensor_map, inst, 1));

  // Find the learning rate constant
  const auto& literal = root->operand(1)->operand(0)->operand(0)->literal();

  std::unique_ptr<Literal> float_lit;
  TF_ASSIGN_OR_RETURN(float_lit, literal.Convert(F32));

  float learning_rate = float_lit->GetFirstElement<float>();

  poplar::program::Sequence prog;
  popconv::convolutionBiasUpdate(graph, deltas, biases, learning_rate,
                                 poplar::FLOAT, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, biases));

  return prog;
}

port::StatusOr<poplar::program::Program>
CreateDepthwiseConvolutionOp(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {
  const HloInstruction* root =
          inst->to_apply()->root_instruction();

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = GetConvolutionPass(root);

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetDepthConvolutionParameters(root));

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(in, ShuffleConvolutionInputToPoplar(root, in));

  TF_ASSIGN_OR_RETURN(kernel, ShuffleConvolutionWeightsToPoplar(root, kernel,
                                                                false));

  kernel = AddGroupsDimensionToWeights(kernel);

  // Swap IN and GROUPS
  std::vector<unsigned int> shuffle(kernel.rank());
  std::iota(shuffle.begin(), shuffle.end(), 0);
  shuffle[0] = 2;
  shuffle[2] = 0;
  kernel = kernel.dimShuffle(shuffle);
  
  auto out = popconv::convolution(graph, in, kernel, params, false, prog,
                                  inst->name(), opts);

  TF_ASSIGN_OR_RETURN(out, ShuffleConvolutionOutputToTensorflow(root, out));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

port::StatusOr<poplar::program::Program>
Create2DConvWithReverse(poplar::Graph &graph,
                        CompilerResources& res,
                        const HloInstruction *inst,
                        const xla::Shape& output_shape,
                        TensorMap& tensor_map) {
  const HloInstruction* conv =
          inst->to_apply()->root_instruction();

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = GetConvolutionPass(inst);

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(conv));

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(in, ShuffleConvolutionInputToPoplar(conv, in));

  TF_ASSIGN_OR_RETURN(kernel, ShuffleConvolutionWeightsToPoplar(conv, kernel,
                                                                true));

  kernel = AddGroupsDimensionToWeights(kernel);

  poplar::Tensor out = popconv::convolution(graph, in, kernel, params,
                                            true, prog, conv->name(), opts);

  TF_ASSIGN_OR_RETURN(out, ShuffleConvolutionOutputToTensorflow(conv, out));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

}
}
