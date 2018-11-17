#include <algorithm>
#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/poplibs_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>

namespace xla {
namespace poplarplugin {
namespace {
static const size_t basic_lstm_cell_num_units = 4;

absl::flat_hash_map<std::string, CustomPoplibOpInfo> info_map = {
    {"lstm_layer_fwd", {AllocateLstmLayerFwdOp, CreateLstmLayerFwdOp}},
    {"lstm_layer_bwd", {AllocateLstmLayerBwdOp, CreateLstmLayerBwdOp}},
};

StatusOr<popnn::lstm::LstmParams> GetLstmParameters(
    const HloInstruction* inst,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  const auto input_shape = inst->operand(0)->shape();
  const auto time_steps = input_shape.dimensions(0);
  const auto batch_size = input_shape.dimensions(1);
  const auto input_size = input_shape.dimensions(2);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(input_shape));
  popnn::lstm::LstmParams lstm_params(type, batch_size, time_steps,
                                      {input_size, num_channels});

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));
  lstm_params.calcInputGradients = is_training;
  return lstm_params;
}

StatusOr<poplar::OptionFlags> GetLstmOpts(
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  poplar::OptionFlags lstm_opts;
  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));
  if (!is_training) {
    lstm_opts.set({{"inferenceOnly", "true"}});
  }
  // Get the partial type
  TF_ASSIGN_OR_RETURN(tensorflow::DataType partials_tf_type,
                      attribute_map.GetAttributeAsTFDataType("partials_dtype"));
  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_tf_type, &partials_xla_type));
  TF_ASSIGN_OR_RETURN(poplar::Type partials_poplar_type,
                      PoplarDataType(partials_xla_type));
  lstm_opts.set({{"partialsType", partials_poplar_type.toString()}});
  return lstm_opts;
}

poplar::Tensor UnflattenWeight(const poplar::Tensor& t) {
  return t
      .reshape({t.dim(0), basic_lstm_cell_num_units,
                t.dim(1) / basic_lstm_cell_num_units})
      .dimShuffle({1, 0, 2});
}

// The kernel is stored as:
// [input_size + output_size, basic_lstm_cell_num_units * output_size] tensor.
// This extracts the input and output weights.
std::pair<poplar::Tensor, poplar::Tensor> UnpackLstmKernel(
    poplar::Tensor kernel, const size_t input_size, const size_t output_size) {
  poplar::Tensor inputWeights = UnflattenWeight(kernel.slice(0, input_size));
  poplar::Tensor outputWeights =
      UnflattenWeight(kernel.slice(input_size, input_size + output_size));
  return {inputWeights, outputWeights};
}

poplar::Tensor FlattenWeight(const poplar::Tensor& t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// Reverse of UnpackLstmKernel
poplar::Tensor PackLstmKernel(poplar::Tensor input_weights,
                              poplar::Tensor output_weights) {
  return poplar::concat(FlattenWeight(input_weights),
                        FlattenWeight(output_weights));
}

}  // namespace

const absl::flat_hash_map<std::string, CustomPoplibOpInfo>&
GetPopnnOpInfoMap() {
  return info_map;
}

StatusOr<poplar::Tensor> AllocateLstmLayerFwdOp(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const HloInstruction* inst, const int64 target_idx,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                      GetLstmParameters(inst, attribute_map));
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts,
                      GetLstmOpts(attribute_map));
  switch (target_idx) {
    case 0: {
      // Allocate LSTM input tensor
      return popnn::lstm::createInput(graph, lstm_params, name, lstm_opts,
                                      &res.dot_cache);
    }
    case 1: {
      // Allocate initial output (h) tensor
      return popnn::lstm::createInitialOutput(graph, lstm_params, name,
                                              lstm_opts, &res.dot_cache);
    }
    case 2: {
      // Allocate initial cell state (c) tensor
      return popnn::lstm::createInitialCellState(graph, lstm_params, name,
                                                 lstm_opts, &res.dot_cache);
    }
    case 3: {
      // Allocate LSTM weights kernel
      poplar::Tensor input_weights;
      poplar::Tensor output_weights;
      std::tie(input_weights, output_weights) =
          popnn::lstm::createWeightsKernel(graph, lstm_params, name, lstm_opts,
                                           &res.dot_cache);
      return PackLstmKernel(input_weights, output_weights);
    }
    case 4: {
      // Allocate LSTM weights biases
      return popnn::lstm::createWeightsBiases(graph, lstm_params, name,
                                              lstm_opts, &res.dot_cache);
    }
    default: {
      return xla::FailedPrecondition(
          "Trying to allocate LstmLayerFwdOp tensor for an index out of range "
          "%d.",
          target_idx);
    }
  }
}

StatusOr<poplar::Tensor> AllocateLstmLayerBwdOp(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const HloInstruction* inst, const int64 target_idx,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  return xla::FailedPrecondition("LstmLayerBwdOp should not be allocating.");
}

StatusOr<poplar::program::Program> CreateLstmLayerFwdOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  VLOG(1) << "Processing " << inst->name() << " as CreateLstmLayerFwdOp.";
  poplar::program::Sequence seq;
  popnn::lstm::LstmWeights weights;

  TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_h_state,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_c_state,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(weights.biases,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));

  TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                      GetLstmParameters(inst, attribute_map));
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts,
                      GetLstmOpts(attribute_map));

  auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
  auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);
  std::tie(weights.inputWeights, weights.outputWeights) =
      UnpackLstmKernel(kernel, input_size, output_size);

  popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

  poplar::Tensor output, output_c_state, intermediates;

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));

  auto intermediates_ptr = is_training ? &intermediates : nullptr;
  std::tie(output, output_c_state) = popnn::lstm::lstmFwd(
      graph, lstm_params, init_state, input, weights, intermediates_ptr, seq,
      GetDebugName(inst), lstm_opts, &res.dot_cache);

  poplar::Tensor output_h_state = output[lstm_params.timeSteps - 1];

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, output_h_state));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, output_c_state));
  if (is_training) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 3, intermediates));
  }
  return seq;
}

StatusOr<poplar::program::Program> CreateLstmLayerBwdOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  VLOG(1) << "Processing " << inst->name() << " as CreateLstmLayerBwdOp.";
  poplar::program::Sequence seq;
  popnn::lstm::LstmWeights weights;

  TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_h_state,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_c_state,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(weights.biases,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                      FindInstructionInput(tensor_map, res, inst, 5, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_h_state,
                      FindInstructionInput(tensor_map, res, inst, 6, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_c_state,
                      FindInstructionInput(tensor_map, res, inst, 7, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor intermediates,
                      FindInstructionInput(tensor_map, res, inst, 8, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_backprop,
                      FindInstructionInput(tensor_map, res, inst, 9, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_h_state_backprop,
                      FindInstructionInput(tensor_map, res, inst, 10, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_c_state_backprop,
                      FindInstructionInput(tensor_map, res, inst, 11, seq));

  TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                      GetLstmParameters(inst, attribute_map));
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts,
                      GetLstmOpts(attribute_map));

  auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
  auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);
  std::tie(weights.inputWeights, weights.outputWeights) =
      UnpackLstmKernel(kernel, input_size, output_size);

  popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

  // TODO this could be inplace but we have no machanism for describing inplace
  // ops which are not an output.
  poplar::Tensor output_backprop_copy =
      graph.clone(output_backprop, "output_backprop.clone");
  seq.add(poplar::program::Copy(output_backprop, output_backprop_copy));
  popops::addInPlace(
      graph, output_backprop_copy[output_backprop_copy.dim(0) - 1],
      output_h_state_backprop, seq, GetDebugName(inst) + "/outputGradient");

  poplar::Tensor input_backprop;
  popnn::lstm::LstmWeights weights_backprop;
  popnn::lstm::LstmState init_state_backprop = popnn::lstm::lstmBwdWithWU(
      graph, lstm_params, seq, init_state, intermediates, weights, input,
      output, output_backprop_copy, &output_c_state_backprop, &input_backprop,
      weights_backprop, GetDebugName(inst), lstm_opts, &res.dot_cache);

  auto kernel_backprop = PackLstmKernel(weights_backprop.inputWeights,
                                        weights_backprop.outputWeights);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input_backprop));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, init_state_backprop.output));
  TF_CHECK_OK(
      AddOutputTensor(tensor_map, inst, 2, init_state_backprop.cellState));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 3, kernel_backprop));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 4, weights_backprop.biases));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
