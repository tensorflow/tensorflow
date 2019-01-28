#include <algorithm>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/conv_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static Window GetConvolutionWindow(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().window();
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution window on a non convolution "
                    "operation.";
    }
    return inst->window();
  }
}

static ConvolutionDimensionNumbers GetConvolutionDims(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().dimension_numbers();
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution dimension numbers on a non "
                    "convolution operation.";
    }
    return inst->convolution_dimension_numbers();
  }
}

static int64 GetFeatureGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().feature_group_count();
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution feature group count numbers "
                    "on a non convolution operation.";
    }
    return inst->feature_group_count();
  }
}

static int64 GetBatchGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().batch_group_count();
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution batch group count numbers on "
                    "a non convolution operation.";
    }
    return inst->batch_group_count();
  }
}

StatusOr<poplin::ConvParams> GetConvolutionParameters(
    const HloInstruction* inst, int64 input_index, int64 kernel_index) {
  const Shape& input = inst->operand(input_index)->shape();
  const Shape& kernel = inst->operand(kernel_index)->shape();
  const Shape& output = inst->shape();

  const Window& window = GetConvolutionWindow(inst);

  poplar::Type dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);
  std::vector<size_t> output_dims = PoplarShapeFromXlaShape(output);

  const auto& dims = GetConvolutionDims(inst);
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_j = kernel_dims[dims.kernel_input_feature_dimension()];
  unsigned int n_o = output_dims[dims.output_feature_dimension()];
  unsigned int n_p = kernel_dims[dims.kernel_output_feature_dimension()];

  unsigned int n_g = GetFeatureGroupCount(inst);

  if ((n_i >= n_j) && (n_o >= n_p)) {
    // Forward and backward passes
    if (n_g != (n_i / n_j) * (n_o / n_p)) {
      LOG(WARNING) << "Mismatch of the feature group for convolution "
                   << inst->name();
    }
    n_i = n_i / n_g;
    n_o = n_o / n_g;
  } else {
    // Weight update
    n_g = (n_j / n_i) * (n_p / n_o);
    n_b = n_b / n_g;
  }

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

  for (int64 i = 0; i < window.dimensions().size(); i++) {
    n_s.push_back(input_dims[dims.input_spatial_dimensions(i)]);
    f_s.push_back(kernel_dims[dims.kernel_spatial_dimensions(i)]);
    w_s.push_back(window.dimensions(i).stride());
    if (window.dimensions(i).padding_low() < 0) {
      unsigned int p = -window.dimensions(i).padding_low();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d - 1) / d;
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
      unsigned int trunc = (p + d - 1) / d;
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

  poplin::ConvParams params(dtype, n_b, n_s, f_s, n_i, n_o, n_g, t_l, t_u, d_i,
                            p_l, p_u, falses, zeros, zeros, d_w, zeros, zeros,
                            falses, zeros, zeros, w_s, zeros, zeros);

  return params;
}

poplar::Tensor ShuffleConvolutionInputToPoplar(const HloInstruction* inst,
                                               const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

  std::vector<unsigned int> shuffle(2 + d.input_spatial_dimensions_size());
  shuffle[0] = d.input_batch_dimension();
  shuffle[1] = d.input_feature_dimension();
  for (int64 i = 0; i < d.input_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = d.input_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionOutputToPoplar(const HloInstruction* inst,
                                                const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

  std::vector<unsigned int> shuffle(2 + d.output_spatial_dimensions().size());
  shuffle[0] = d.output_batch_dimension();
  shuffle[1] = d.output_feature_dimension();
  for (int64 i = 0; i < d.output_spatial_dimensions().size(); ++i) {
    shuffle[2 + i] = d.output_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionWeightsToPoplar(const HloInstruction* inst,
                                                 const poplar::Tensor& tensor,
                                                 bool swap_features) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

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

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

  std::vector<unsigned int> shuffle(2 + d.input_spatial_dimensions_size());
  shuffle[d.input_batch_dimension()] = 0;
  shuffle[d.input_feature_dimension()] = 1;
  for (int64 i = 0; i < d.input_spatial_dimensions_size(); i++) {
    shuffle[d.input_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

  std::vector<unsigned int> shuffle(2 + d.kernel_spatial_dimensions_size());
  shuffle[d.kernel_output_feature_dimension()] = 0;
  shuffle[d.kernel_input_feature_dimension()] = 1;
  for (int64 i = 0; i < d.kernel_spatial_dimensions_size(); i++) {
    shuffle[d.kernel_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));

  std::vector<unsigned int> shuffle(2 + d.output_spatial_dimensions_size());
  shuffle[d.output_batch_dimension()] = 0;
  shuffle[d.output_feature_dimension()] = 1;
  for (int64 i = 0; i < d.output_spatial_dimensions_size(); i++) {
    shuffle[d.output_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t,
                                                bool flipped) {
  poplar::Tensor out = t;
  return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped) {
  poplar::Tensor out = t;

  unsigned int out_dim = flipped ? 1 : 0;
  unsigned int in_dim = 1 - out_dim;

  if (p.getNumConvGroups() == 1) {
    // Non-grouped case
    return out.reshapePartial(0, 0, {1});
  } else {
    unsigned int chan_div[2];
    chan_div[in_dim] = out.dim(in_dim) / p.getNumInputChansPerConvGroup();
    chan_div[out_dim] = out.dim(out_dim) / p.getNumOutputChansPerConvGroup();

    // OI... ->(GO)(GI)...
    out = out.reshapePartial(0, 2, {chan_div[0], out.dim(0) / chan_div[0],
                                    chan_div[1], out.dim(1) / chan_div[1]});

    // (GO)(GI)... -> (GG)OI...
    out = out.dimShufflePartial({2}, {1});

    // (GG)OI... -> GOI...
    return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
  }
}

StatusOr<poplar::program::Program> CreateConv2D(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, res, inst, 0, prog));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst, 0, 1));

  in = ShuffleConvolutionInputToPoplar(inst, in);

  kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, false);

  kernel = AddGroupsDimensionToWeights(params, kernel, false);

  const auto conv_type = GetConvClassificationType(inst, res.annotations);

  auto out = conv_graph_caching::DoCachedConvolution(
      graph, res, in, kernel, params, conv_type, false,
      GetShardingDeviceId(inst), prog, GetDebugName(inst));

  out = ShuffleConvolutionOutputToTensorflow(inst, out);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

StatusOr<poplar::program::Program> Create2DConvWithReverse(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, res, inst, 0, prog));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst, 0, 1));

  in = ShuffleConvolutionInputToPoplar(inst, in);

  kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, true);

  kernel = AddGroupsDimensionToWeights(params, kernel, true);

  auto conv_type = GetConvClassificationType(inst, res.annotations);

  auto out = conv_graph_caching::DoCachedConvolution(
      graph, res, in, kernel, params, conv_type, true,
      GetShardingDeviceId(inst), prog, GetDebugName(inst));

  out = ShuffleConvolutionOutputToTensorflow(inst, out);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

StatusOr<poplar::program::Program> CreateDepthwiseBackpropFilter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, res, inst, 0, prog));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst, 0, 1));

  in = ShuffleConvolutionInputToPoplar(inst, in);

  // Move 'G' parts of the I to B (because B is the reducing dimension)
  unsigned n_g = params.getNumConvGroups();
  in = in.reshapePartial(0, 1, {n_g, in.dim(0) / n_g});
  in = in.dimShufflePartial({0}, {1});
  in = in.reshapePartial(1, 3, {in.dim(1) * in.dim(2)});

  kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, false);

  kernel = AddGroupsDimensionToWeights(params, kernel, false);

  auto conv_type = GetConvClassificationType(inst, res.annotations);

  poplar::Tensor out = conv_graph_caching::DoCachedConvolution(
      graph, res, in, kernel, params, conv_type, false,
      GetShardingDeviceId(inst), prog, GetDebugName(inst));

  // Move 'G' parts of the B back to I
  out = out.reshapePartial(1, 2, {n_g, out.dim(1) / n_g});
  out = out.dimShufflePartial({1}, {0});
  out = out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});

  out = ShuffleConvolutionOutputToTensorflow(inst, out);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

StatusOr<poplar::program::Program> CreateConvScaledInplace(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  // Find the weights tensor
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor w = inputs[0][0];

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, res, inst, 1, prog));

  // Find the deltas tensor
  poplar::Tensor deltas;
  TF_ASSIGN_OR_RETURN(deltas,
                      FindInstructionInput(tensor_map, res, inst, 2, prog));

  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst, 1, 2));

  TF_CHECK_OK(conv_graph_caching::DoCachedConvolutionScaledInplace(
      graph, res, w, in, deltas, params, GetShardingDeviceId(inst), prog, inst,
      tensor_map));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, w));

  return prog;
}

StatusOr<poplar::program::Program> CreateConvBiasAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor in = inputs[0][0];

  poplar::Tensor bias;
  TF_ASSIGN_OR_RETURN(
      bias, FindInstructionInput(tensor_map, res, inst, 1, prog, false));

  const auto* conv_op = inst->operand(0);

  poplar::Tensor shuffled_in = ShuffleConvolutionOutputToPoplar(conv_op, in);

  poplin::addBias(graph, shuffled_in, bias, prog, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

StatusOr<poplar::program::Program> ConvBiasApply(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;

  const HloInstruction* root =
      inst->fused_instructions_computation()->root_instruction();

  // Find the biases
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor biases = inputs[0][0];

  // Find the deltas
  TF_ASSIGN_OR_RETURN(poplar::Tensor deltas,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  // // Find reduction dimensions
  const auto* reduce = root->operand(1)->operand(0);
  std::vector<std::size_t> reduction_dims;
  for (auto d : reduce->dimensions()) {
    reduction_dims.push_back(d);
  }

  TF_CHECK_OK(conv_graph_caching::DoCachedBiasApply(
      graph, res, biases, deltas, reduction_dims, GetShardingDeviceId(inst),
      prog, inst, tensor_map));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, biases));

  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
