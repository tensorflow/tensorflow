#include <algorithm>
#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/poplibs_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

absl::flat_hash_map<std::string, CustomPoplibsCallFn> call_map = {
    {"lstm_layer_fwd", CreateLstmLayerFwdOp},
    {"lstm_layer_bwd", CreateLstmLayerBwdOp},
};

StatusOr<popnn::lstm::LstmParams> GetLstmParameters(
    const HloInstruction* inst,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map,
    poplar::OptionFlags& lstm_opts, const bool is_bwd_pass) {
  const auto input_shape = inst->operand(0)->shape();
  const auto time_steps = input_shape.dimensions(0);
  const auto batch_size = input_shape.dimensions(1);
  const auto input_size = input_shape.dimensions(2);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(input_shape));
  popnn::lstm::LstmParams params(type, batch_size, time_steps,
                                 {input_size, num_channels});

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));
  params.calcInputGradients = is_bwd_pass && is_training;
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
  return params;
}
}

const absl::flat_hash_map<std::string, CustomPoplibsCallFn>& GetPopnnCallMap() {
  return call_map;
}

StatusOr<poplar::program::Program> CreateLstmLayerFwdOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  VLOG(1) << "Processing " << inst->name() << " as CreateLstmLayerFwdOp.";
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                      FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_h_state,
                      FindInstructionInput(tensor_map, inst, 1));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_c_state,
                      FindInstructionInput(tensor_map, inst, 2));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_weights,
                      FindInstructionInput(tensor_map, inst, 3));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_weights,
                      FindInstructionInput(tensor_map, inst, 4));
  TF_ASSIGN_OR_RETURN(poplar::Tensor biases,
                      FindInstructionInput(tensor_map, inst, 5));

  popnn::lstm::LstmWeights weights = {input_weights, output_weights, biases};

  poplar::OptionFlags lstm_opts;
  TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                      GetLstmParameters(inst, attribute_map, lstm_opts, false));
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

  TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                      FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_h_state,
                      FindInstructionInput(tensor_map, inst, 1));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_c_state,
                      FindInstructionInput(tensor_map, inst, 2));
  TF_ASSIGN_OR_RETURN(poplar::Tensor input_weights,
                      FindInstructionInput(tensor_map, inst, 3));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_weights,
                      FindInstructionInput(tensor_map, inst, 4));
  TF_ASSIGN_OR_RETURN(poplar::Tensor biases,
                      FindInstructionInput(tensor_map, inst, 5));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                      FindInstructionInput(tensor_map, inst, 6));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_h_state,
                      FindInstructionInput(tensor_map, inst, 7));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_c_state,
                      FindInstructionInput(tensor_map, inst, 8));
  TF_ASSIGN_OR_RETURN(poplar::Tensor intermediates,
                      FindInstructionInput(tensor_map, inst, 9));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_backprop,
                      FindInstructionInput(tensor_map, inst, 10));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_h_state_backprop,
                      FindInstructionInput(tensor_map, inst, 11));
  TF_ASSIGN_OR_RETURN(poplar::Tensor output_c_state_backprop,
                      FindInstructionInput(tensor_map, inst, 12));

  popnn::lstm::LstmWeights weights = {input_weights, output_weights, biases};

  poplar::OptionFlags lstm_opts;
  TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                      GetLstmParameters(inst, attribute_map, lstm_opts, true));

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

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input_backprop));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, init_state_backprop.output));
  TF_CHECK_OK(
      AddOutputTensor(tensor_map, inst, 2, init_state_backprop.cellState));
  TF_CHECK_OK(
      AddOutputTensor(tensor_map, inst, 3, weights_backprop.inputWeights));
  TF_CHECK_OK(
      AddOutputTensor(tensor_map, inst, 4, weights_backprop.outputWeights));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 5, weights_backprop.biases));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
