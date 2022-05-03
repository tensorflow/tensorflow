/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/timing_cache.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/strided_slice_op.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferPlugin.h"

// Check if the types are equal. Cast to int first so that failure log message
// would work!
#define TFTRT_CHECK_EQ_TYPE(val1, val2) CHECK_EQ((int)val1, (int)val2)

#define TFTRT_CHECK_INPUT_SIZE(size, exp_size, node_def)                 \
  if ((size) != (exp_size)) {                                            \
    TFTRT_ERROR(errors::InvalidArgument, node_def.op(), " got ", (size), \
                " inputs but expected ", (exp_size));                    \
  }

namespace tensorflow {
namespace tensorrt {
namespace convert {

using absl::StrAppend;
using absl::StrCat;

namespace {

#define ADD_LAYER(layer_name)              \
  case nvinfer1::LayerType::k##layer_name: \
    return #layer_name;

const char* LayerTypeToString(nvinfer1::LayerType layer_type) {
  switch (layer_type) {
    ADD_LAYER(CONVOLUTION)
    ADD_LAYER(FULLY_CONNECTED)
    ADD_LAYER(ACTIVATION)
    ADD_LAYER(POOLING)
    ADD_LAYER(LRN)
    ADD_LAYER(SCALE)
    ADD_LAYER(SOFTMAX)
    ADD_LAYER(DECONVOLUTION)
    ADD_LAYER(CONCATENATION)
    ADD_LAYER(ELEMENTWISE)
    ADD_LAYER(PLUGIN)
    ADD_LAYER(UNARY)
    ADD_LAYER(PADDING)
    ADD_LAYER(SHUFFLE)
    ADD_LAYER(REDUCE)
    ADD_LAYER(TOPK)
    ADD_LAYER(GATHER)
    ADD_LAYER(MATRIX_MULTIPLY)
    ADD_LAYER(RAGGED_SOFTMAX)
    ADD_LAYER(CONSTANT)
    ADD_LAYER(RNN_V2)
    ADD_LAYER(IDENTITY)
    ADD_LAYER(PLUGIN_V2)
    ADD_LAYER(SLICE)
    ADD_LAYER(SHAPE)
    ADD_LAYER(PARAMETRIC_RELU)
    ADD_LAYER(RESIZE)
    ADD_LAYER(TRIP_LIMIT)
    ADD_LAYER(RECURRENCE)
    ADD_LAYER(ITERATOR)
    ADD_LAYER(LOOP_OUTPUT)
    ADD_LAYER(SELECT)
    ADD_LAYER(FILL)
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    ADD_LAYER(QUANTIZE)
    ADD_LAYER(DEQUANTIZE)
#else
    // The TRT IRNNv2Layer has been deprecated in favor of the loop API.
    ADD_LAYER(RNN)
#endif
  }
  return "UNKNOWN_LAYER";
}

#undef ADD_LAYER

// Sets the ILayer name in the form of
// <engine_name>/<tf_related_part>:<trt_operation_name>.
void SetLayerNameHelper(nvinfer1::ILayer* layer, absl::string_view engine_name,
                        absl::string_view tf_name) {
  const char* trt_name = LayerTypeToString(layer->getType());
  layer->setName(
      absl::StrCat(engine_name, "/", tf_name, ":", trt_name).c_str());
}

// Returns a string in the form of <sub_op_name><sub_op_instance>.
std::string GetLayerNameSuffix(absl::string_view sub_op_name,
                               absl::optional<int> sub_op_instance) {
  std::string op_suffix(sub_op_name);
  if (sub_op_instance.has_value()) {
    op_suffix =
        absl::StrCat(op_suffix, "_", std::to_string(sub_op_instance.value()));
  }
  return op_suffix;
}

}  // namespace

bool IsEngineInput(absl::string_view name) {
  return absl::StartsWith(name, IONamePrefixes::kInputPHName);
}
bool IsEngineOutput(absl::string_view name) {
  return absl::StartsWith(name, IONamePrefixes::kOutputPHName);
}

void GetOutputProperties(const grappler::GraphProperties& graph_properties,
                         const Node* node, const int out_port,
                         PartialTensorShape* shape, DataType* dtype) {
  if (graph_properties.HasOutputProperties(node->name())) {
    auto output_params = graph_properties.GetOutputProperties(node->name());
    auto out_shape = output_params.at(out_port);
    *dtype = out_shape.dtype();
    *shape = out_shape.shape();
  } else {
    LOG(INFO) << "Unknown output shape" << node->name();
    *dtype = node->output_type(out_port);
  }
}

void GetInputProperties(const grappler::GraphProperties& graph_properties,
                        const Node* node, const int in_port,
                        PartialTensorShape* shape, DataType* dtype) {
  if (graph_properties.HasInputProperties(node->name())) {
    auto input_params = graph_properties.GetInputProperties(node->name());
    auto in_shape = input_params.at(in_port);
    *dtype = in_shape.dtype();
    *shape = in_shape.shape();
  } else {
    *dtype = node->input_type(in_port);
  }
}

// This function checks if a tensor is compatible with TRT.
//
// We check that the shape and datatype are compatible with TensorRT. We also
// return the corresponding trt_dtype, the trt_dims and the batch_size (latter
// is only needed in implicit batch mode).
//
// The return status indicates wether the tensor is compatible.
//
// For implicit batch mode, when validation_only == false, we also check that
// all input dimensions (besides the batch dimension) are known dimensions.
Status ValidateTensorProperties(const string& producer_node_type,
                                const DataType dtype,
                                const PartialTensorShape& shape,
                                const bool use_implicit_batch,
                                bool validation_only,
                                nvinfer1::DataType* trt_dtype,
                                nvinfer1::Dims* trt_dims, int* batch_size) {
  // Convert data type.
  TF_RETURN_IF_ERROR(TfTypeToTrtType(dtype, trt_dtype));

  // Convert shape.
  if (shape.dims() < 0) {
    return errors::InvalidArgument("Input tensor rank is unknown.");
  }
  // Add 1 to maximum rank for implicit batch dim.
  const int max_rank = nvinfer1::Dims::MAX_DIMS + (use_implicit_batch ? 1 : 0);
  if (shape.dims() > max_rank) {
    return errors::OutOfRange("Input tensor rank is greater than ", max_rank);
  }
  if (use_implicit_batch && (producer_node_type != "Const") &&
      (shape.dims() < 1)) {
    return errors::InvalidArgument(
        "Scalar input tensor is not supported since the first dimension "
        "is treated as batch dimension by TRT");
  }
  StatusOr<DimsAdapter> dims = DimsAdapter::Create(shape, use_implicit_batch);
  TRT_ENSURE_OK(dims);
  *trt_dims = dims->AsTrtDims();
  // Get batch size for tensor if it will not be included the shape.
  if (use_implicit_batch) {
    *batch_size = shape.dim_size(0);
  }

  // Don't convert empty tensors (dim value of 0).
  const int first_trt_dim = use_implicit_batch ? 1 : 0;
  for (int d = first_trt_dim; d < shape.dims(); ++d) {
    if (shape.dim_size(d) == 0) {
      return errors::Unimplemented(
          "Input tensor with shape ", shape.DebugString(),
          " is an empty tensor, which is not supported by TRT");
    }
  }

  if (validation_only) return Status::OK();

  // Following checks are only used during TRT engine creation time.
  if (use_implicit_batch) {
    for (int d = first_trt_dim; d < shape.dims(); ++d) {
      if (shape.dim_size(d) < 0) {
        return errors::InvalidArgument(
            "Input tensor with shape ", shape.DebugString(),
            " has an unknown non-batch dimension at dim ", d);
      }
    }
  }
  return Status::OK();
}

Status GetTrtBroadcastShape(const TRT_TensorOrWeights& operand_l,
                            const TRT_TensorOrWeights& operand_r,
                            const bool check_feasibility,
                            const bool use_implicit_batch,
                            nvinfer1::Dims* operand_l_new_dims,
                            nvinfer1::Dims* operand_r_new_dims) {
  // TensorRT Elementwise op supports broadcast but requires both tensor to be
  // of Identical rank
  //
  // We consider case of:
  //   1. operand_l to be a Tensor & operand_r to be a Const;
  //   2. operand_l to be a Tensor & operand_r to be a Tensor;
  // note: const op const (constant folding) should fallback to TensorFlow
  //
  // broadcast scheme:
  //       T:  1 3 5    (tensor would not have batch dimension)
  //       W:  1 1 3 1  (weight would have all explicit dimensions)
  // i. fill in explicit dimensions
  //    -> T: -1 1 3 5  (we put a -1 for batch dimension)
  //    -> W:  1 1 3 1
  // ii. compare broadcast feasibility
  //
  // We cannot support the following since TensorRT does not allow manipulation
  // on batch dimension, we cannot generate output with proper shape
  //    T: 3 5 1
  //    W: 1 1 1  1 3 5 1
  // -> T: 1 1 1 -1 3 5 1
  // -> W: 1 1 1  1 3 5 1
  // ***************************************************************************
  if (!operand_l.is_tensor() && !operand_r.is_tensor()) {
    return errors::InvalidArgument(
        "Broadcasting requires at least one of the operands be tensors");
  }

  constexpr int max_nb_dims = nvinfer1::Dims::MAX_DIMS + 1;
  auto compute_output_dims =
      [use_implicit_batch](const TRT_TensorOrWeights& input,
                           int broadcast_num_dims,
                           std::array<int32_t, max_nb_dims>* output_dims_array,
                           nvinfer1::Dims* output_dims) -> Status {
    const nvinfer1::Dims input_dims = input.GetTrtDims();
    absl::c_fill(*output_dims_array, 1);
    absl::c_copy(
        DimsAdapter(input_dims),
        output_dims_array->begin() + broadcast_num_dims - input_dims.nbDims);
    if (use_implicit_batch && input.is_tensor()) {
      const int true_input_dims = input_dims.nbDims + 1;
      if (true_input_dims < broadcast_num_dims) {
        return errors::InvalidArgument(
            "Broadcasting beyond batch dimension is not supported ",
            "(tensor #dims ", true_input_dims, " vs broadcast #dims ",
            broadcast_num_dims, ")");
      }
      // Set the batch dimension to -1, since batch size is not supposed to
      // be broadcasted.
      (*output_dims_array)[0] = -1;
    }
    // Copy to output dimensions
    auto offt = use_implicit_batch ? 1 : 0;
    output_dims->nbDims = broadcast_num_dims - offt;
    absl::c_copy(
        absl::MakeSpan(*output_dims_array).subspan(offt, broadcast_num_dims),
        output_dims->d);
    return Status::OK();
  };

  // Compute the output dimensions.
  const int broadcast_num_dims =
      std::max(operand_l.GetTrtDims().nbDims +
                   (use_implicit_batch && operand_l.is_tensor()),
               operand_r.GetTrtDims().nbDims +
                   (use_implicit_batch && operand_r.is_tensor()));
  std::array<int32_t, max_nb_dims> output_l, output_r;
  TF_RETURN_IF_ERROR(compute_output_dims(operand_l, broadcast_num_dims,
                                         &output_l, operand_l_new_dims));
  TF_RETURN_IF_ERROR(compute_output_dims(operand_r, broadcast_num_dims,
                                         &output_r, operand_r_new_dims));

  // Compare broadcast feasibility
  if (check_feasibility) {
    for (int i = 0; i < broadcast_num_dims; ++i) {
      if (!use_implicit_batch && (output_l[i] == -1 || output_r[i] == -1)) {
        // If the condition is true then we are in explicit batch mode and (at
        // least) one of the input dimensions are unknown. In other words we
        // are in dynamic shape mode. During conversion time we only see -1 for
        // the unknown shapes, therefore we cannot decide on the feasibility of
        // broadcast over the unknown dimensions. Therefore we just continue for
        // the next dimension. In dynamic shape mode TRT can only check the
        // feasibility of the broadcast when the actual input dimensions are
        // specified by SetTrtEngineInputs and the inference job is launched by
        // TrtEnque.
        continue;
      }
      if ((output_l[i] != output_r[i]) && (output_l[i] != 1) &&
          (output_r[i] != 1)) {
        return errors::InvalidArgument("Infeasible broadcast scheme (",
                                       "batch_dim: ", output_l[0], ", ",
                                       DebugString(*operand_l_new_dims), " vs ",
                                       "batch_dim: ", output_r[0], ", ",
                                       DebugString(*operand_r_new_dims), ")");
      }
    }
  }
  return Status::OK();
}

// Prepares a dynamic shape tensor for broadcast by adding leading 1 dimensions.
Status DynamicBroadcast(ITensorProxyPtr operand, OpConverterParams* params,
                        ITensorProxyPtr* output, int broadcasted_nbDims,
                        absl::optional<int> op_instance) {
  int operand_nbDims = operand->getDimensions().nbDims;
  if (broadcasted_nbDims > operand_nbDims) {
    if (params->validation_only) return Status::OK();
    int n_extra_dims = broadcasted_nbDims - operand_nbDims;
    VLOG(2) << "Dynamic broadcast adding " << n_extra_dims << " leading 1s";
    TF_RETURN_IF_ERROR(params->converter->DynamicReshape(
        /*input=*/operand,
        /*slices=*/{std::make_pair(0, operand_nbDims)},
        /*params=*/params,
        /*output=*/output,
        /*size_for_added_dims*/ {n_extra_dims},
        /*op_instance=*/op_instance));
  } else {
    *output = operand;
  }
  return Status::OK();
}

Status BroadcastWeights(std::unique_ptr<TRT_TensorOrWeights>& p,
                        const DimsAdapter& broadcasted_dims) {
  if (!p->is_weights()) return errors::Internal("Weight input expected");
  if (p->GetTrtDims().nbDims != broadcasted_dims.NumDims()) {
    TRT_ShapedWeights weights(p->weights());
    TF_RETURN_IF_ERROR(weights.SetShape(broadcasted_dims));
    p = std::make_unique<TRT_TensorOrWeights>(weights);
  }
  return Status::OK();
}

Status ApplyBroadcast(std::unique_ptr<TRT_TensorOrWeights>& operand,
                      const DimsAdapter& broadcasted_dims,
                      OpConverterParams* params,
                      absl::optional<int> op_instance) {
  if (operand->is_weights()) {
    TF_RETURN_IF_ERROR(BroadcastWeights(operand, broadcasted_dims));
  } else {
    ITensorProxyPtr tensor = nullptr;
    auto is_static_shuffle_compatible = [](const auto& dims) {
      return absl::c_count(dims, -1) <= 1;
    };
    if (is_static_shuffle_compatible(broadcasted_dims)) {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params->converter, *operand, broadcasted_dims,
          params->validation_only, &tensor, params->node_def));
    } else {
      TF_RETURN_IF_ERROR(DynamicBroadcast(
          /*operand=*/operand->tensor(),
          /*params=*/params,
          /*output=*/&tensor,
          /*broadcasted_nbDims*/ broadcasted_dims.NumDims(),
          /*op_instance=*/op_instance));
    }
    operand = std::make_unique<TRT_TensorOrWeights>(tensor);
  }
  return Status::OK();
}

// Inserts leading 1 dimensions so that both operands have the same rank.
// Note: In implicit batch mode, weights' shape can include an explicit 1 batch
// dimension. The broadcasted shape might loose this leading batch dim, because
// the broadcasted shape does not include the implicit batch dim.
// TODO(tfeher): Other code blocks that use GetTrtBroadcastShape need to be
// fixed to use this routine to handle dynamic inputs. Eventually,
// GetTrtBroadcastShape should only be used by this routine.
Status BroadcastTensors(std::unique_ptr<TRT_TensorOrWeights>& operand_l,
                        std::unique_ptr<TRT_TensorOrWeights>& operand_r,
                        bool check_feasibility, OpConverterParams* params) {
  nvinfer1::Dims broadcasted_dims_l, broadcasted_dims_r;
  TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
      *operand_l, *operand_r, check_feasibility, params->use_implicit_batch,
      &broadcasted_dims_l, &broadcasted_dims_r));

  if (params->validation_only) return Status::OK();

  TF_RETURN_IF_ERROR(ApplyBroadcast(
      /*operand=*/operand_l,
      /*broadcasted_dims=*/broadcasted_dims_l,
      /*params=*/params,
      /*op_instance=*/0));

  TF_RETURN_IF_ERROR(ApplyBroadcast(
      /*operand=*/operand_r,
      /*broadcasted_dims=*/broadcasted_dims_r,
      /*params=*/params,
      /*op_instance=*/1));

  return Status::OK();
}

ITensorProxyPtr Converter::CreateConstantLayer(const TRT_ShapedWeights& weights,
                                               const nvinfer1::Dims& dims) {
  nvinfer1::Weights trt_weights = weights.GetTrtWeights();
  nvinfer1::IConstantLayer* layer = network()->addConstant(dims, trt_weights);
  if (!layer) return nullptr;
  SetLayerName(layer, "_tftrt_constant_",
               std::to_string(next_constant_layer_id_));
  next_constant_layer_id_++;
  ITensorProxyPtr trt_tensor = layer->getOutput(0);
  return trt_tensor;
}

// Creates a scalar constant and fills with value.
template <typename T>
Status CreateScalarConstant(
    OpConverterParams* params, T value, ITensorProxyPtr* tensor,
    nvinfer1::DataType trt_type = nvinfer1::DataType::kINT32,
    const nvinfer1::Dims& dims = {1, {1}}) {
  StatusOr<TRT_ShapedWeights> weights =
      params->weight_store->GetTempWeights(trt_type, dims);
  TRT_ENSURE_OK(weights);
  TF_RETURN_IF_ERROR(weights->SetValues(value));
  *tensor = params->converter->CreateConstantLayer(*weights, dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, params->node_def.name());
  return Status::OK();
}

// Creates a constant with the same rank as dims, where each dimension has
// size = 1.
Status CreateBroadcastableScalarConstant(OpConverterParams* params, float value,
                                         const nvinfer1::Dims& dims,
                                         ITensorProxyPtr* tensor,
                                         const char* dtype_attr_name = "T") {
  nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT;  // Default to FP32.
  AttrSlice attrs(params->node_def);
  if (attrs.FindByString(dtype_attr_name) != nullptr) {
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, dtype_attr_name, &dtype));
    TF_RETURN_IF_ERROR(TfTypeToTrtType(dtype, &trt_type));
  }

  // In order to be broadcastable, the number of dims has to match.
  nvinfer1::Dims broadcastable_dims(dims);
  for (int i = 0; i < broadcastable_dims.nbDims; i++) {
    broadcastable_dims.d[i] = 1;
  }
  return CreateScalarConstant(params, value, tensor, trt_type,
                              broadcastable_dims);
}

// The function concatenates tensors on the first axis. This can be used to
// create a shape tensor from individual dimension sizes.
StatusOr<ITensorProxyPtr> ConcatenateTensors(
    OpConverterParams* params, const std::vector<ITensorProxyPtr> input_tensors,
    absl::optional<int> op_instance = absl::nullopt) {
  std::vector<nvinfer1::ITensor*> trt_input_tensors;
  for (const auto& t : input_tensors) {
    trt_input_tensors.push_back(t->trt_tensor());
  }
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          static_cast<nvinfer1::ITensor* const*>(trt_input_tensors.data()),
          input_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.op());
  params->converter->SetLayerName(layer, params->node_def.name(),
                                  "concat_shapes", op_instance);
  layer->setAxis(0);
  return ITensorProxyPtr(layer->getOutput(0));
}

// Convert an axis from TF format to TRT format while validating. TF format
// includes the batch dimension, while TRT does not if implicit batching is used
// (i.e. for tensors). TF can also use negative indices.
Status ConvertAxis(int tf_axis, int trt_nb_dims, absl::string_view node_name,
                   bool use_implicit_batch, int* trt_axis) {
  const int tf_nb_dims = trt_nb_dims + (use_implicit_batch ? 1 : 0);
  // Check bounds.
  if (tf_axis < -tf_nb_dims || tf_axis >= tf_nb_dims) {
    return errors::InvalidArgument(
        "Axis value of ", tf_axis, " is out of bounds, must be in range [",
        -tf_nb_dims, ", ", tf_nb_dims, "), at ", node_name);
  }
  // Make negative axis positive.
  if (tf_axis < 0) tf_axis += tf_nb_dims;
  // Don't allow axis to be the batch dimension.
  if (use_implicit_batch && tf_axis == 0) {
    return errors::Unimplemented(
        "TensorRT does not allow manipulation of the batch dimension");
  }
  // Remove batch dimension if it is implicit.
  *trt_axis = use_implicit_batch ? tf_axis - 1 : tf_axis;
  return Status::OK();
}

bool AllLengthsEqual(const std::vector<std::vector<int>>& inputs) {
  if (inputs.size() == 0) return true;
  int length = inputs.at(0).size();
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs.at(i).size() != length) return false;
  }
  return true;
}

bool DimsHaveSameSize(const DimsAdapter& lhs, const DimsAdapter& rhs) {
  return lhs.Volume() == rhs.Volume();
}

// Returns whether both dimensions are fully specified and the total number of
// elements equals.
bool AreDimsStaticWithSameSize(const DimsAdapter& lhs, const DimsAdapter& rhs) {
  if (!lhs.IsStatic() || !rhs.IsStatic()) return false;
  return DimsHaveSameSize(lhs, rhs);
}

bool AreDimsStaticWithDifferentSize(const DimsAdapter& lhs,
                                    const DimsAdapter& rhs) {
  if (!lhs.IsStatic() || !rhs.IsStatic()) return false;
  return !DimsHaveSameSize(lhs, rhs);
}

static std::vector<std::pair<int, int>> CreateSamePadding(
    const nvinfer1::Dims& stride, const nvinfer1::Dims& kernel,
    const std::vector<int64_t>& input_dims) {
  std::vector<std::pair<int, int>> padding(input_dims.size());
  CHECK_EQ(stride.nbDims, input_dims.size());  // TODO(jie): N+C? NC+?

  for (size_t i = 0; i < input_dims.size(); ++i) {
    // Formula to calculate the padding
    int p = ((input_dims[i] - 1) / stride.d[i]) * stride.d[i] + kernel.d[i] -
            input_dims[i];
    p = (p > 0) ? p : 0;

    // Right precedence padding, like in TensorFlow
    int left = p / 2;
    int right = p - left;

    VLOG(2) << "PADDING_" << i << " pre: " << left << ", post: " << right
            << "paras: " << input_dims[i] << ", " << stride.d[i] << ", "
            << "kernel: " << kernel.d[i];
    padding[i] = {left, right};
  }
  return padding;
}

string GetCommonNameScope(const string& op_name_a, const string& op_name_b) {
  size_t last_scope_separator = 0;
  const size_t min_size = std::min(op_name_a.size(), op_name_b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (op_name_a[i] != op_name_b[i]) break;
    if (op_name_a[i] == '/') last_scope_separator = i + 1;
  }
  return op_name_a.substr(0, last_scope_separator);
}

// Verifies that shapes of the given inputs match after masking the specified
// dimension.
Status VerifyShapesMatch(absl::Span<const TRT_TensorOrWeights> inputs,
                         int masked_dim, absl::string_view node_name) {
  size_t num_inputs = inputs.size();
  if (num_inputs <= 1) return Status::OK();

  const nvinfer1::Dims dims_0 = inputs.at(0).GetTrtDims();
  for (size_t i = 1; i < num_inputs; ++i) {
    const nvinfer1::Dims dim_i = inputs.at(i).GetTrtDims();
    if (dim_i.nbDims != dims_0.nbDims) {
      return errors::InvalidArgument(
          "Received inputs with inconsistent rank, at ", node_name);
    }
    for (size_t j = 0; j < dims_0.nbDims; ++j) {
      // Dynamic dimensions will be verified at runtime.
      if (dim_i.d[j] == -1 || dims_0.d[j] == -1) continue;
      if (dim_i.d[j] != dims_0.d[j] && j != masked_dim) {
        return errors::InvalidArgument(
            "Received inputs with inconsistent shape, at ", node_name);
      }
    }
  }
  return Status::OK();
}

// Perform 5 dimensional reorder of data on CPU
// This is done once at convert time and does not affect GPU inference perf
// Example: reorder NDHWC (Tensorflow) -> NCDHW (TensorRT)
template <typename T>
void Reorder5(const nvinfer1::Dims& shape, const T* idata,
              const nvinfer1::Dims& istrides, T* odata,
              const nvinfer1::Dims& ostrides) {
  for (int k = 0; k < shape.d[0]; ++k) {
    for (int c = 0; c < shape.d[1]; ++c) {
      for (int d = 0; d < shape.d[2]; ++d) {
        for (int r = 0; r < shape.d[3]; ++r) {
          for (int s = 0; s < shape.d[4]; ++s) {
            odata[k * ostrides.d[0] + c * ostrides.d[1] + d * ostrides.d[2] +
                  r * ostrides.d[3] + s * ostrides.d[4]] =
                idata[k * istrides.d[0] + c * istrides.d[1] +
                      d * istrides.d[2] + r * istrides.d[3] +
                      s * istrides.d[4]];
          }
        }
      }
    }
  }
}

// TODO(jie): reorder4 & reorder2 should be merged?
// TODO(aaroey): fix the order of parameters.
template <typename T>
void Reorder4(const nvinfer1::Dims4& shape, const T* idata,
              const nvinfer1::Dims4& istrides, T* odata,
              const nvinfer1::Dims4& ostrides) {
  for (int n = 0; n < shape.d[0]; ++n) {
    for (int c = 0; c < shape.d[1]; ++c) {
      for (int h = 0; h < shape.d[2]; ++h) {
        for (int w = 0; w < shape.d[3]; ++w) {
          odata[n * ostrides.d[0] + c * ostrides.d[1] + h * ostrides.d[2] +
                w * ostrides.d[3]] =
              idata[n * istrides.d[0] + c * istrides.d[1] + h * istrides.d[2] +
                    w * istrides.d[3]];
        }
      }
    }
  }
}

template <typename T>
void Reorder2(const nvinfer1::DimsHW& shape, const T* idata,
              const nvinfer1::DimsHW& istrides, T* odata,
              const nvinfer1::DimsHW& ostrides) {
  for (int h = 0; h < shape.h(); ++h) {
    for (int w = 0; w < shape.w(); ++w) {
      odata[h * ostrides.h() + w * ostrides.w()] =
          idata[h * istrides.h() + w * istrides.w()];
    }
  }
}

// TODO(jie): fallback to tensorflow!!
void ReorderCKtoKC(const TRT_ShapedWeights& iweights,
                   TRT_ShapedWeights* oweights) {
  const int c = iweights.Shape().dim(0);
  const int k = iweights.Shape().dim(1);
  oweights->Shape().dim(0) = k;
  oweights->Shape().dim(1) = c;
  const nvinfer1::DimsHW istrides = {1, k};
  const nvinfer1::DimsHW ostrides = {c, 1};
  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      Reorder2({k, c}, iweights.GetPointer<float>(), istrides,
               oweights->GetPointer<float>(), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder2({k, c}, iweights.GetPointer<Eigen::half>(), istrides,
               oweights->GetPointer<Eigen::half>(), ostrides);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type in reorder expected fp32 or fp16 but got "
                 << DebugString(iweights.TrtDType());
  }
}

void ReorderRSCKToKCRS(const TRT_ShapedWeights& iweights,
                       TRT_ShapedWeights* oweights, const int num_groups) {
  CHECK(iweights.TrtDType() == oweights->TrtDType());
  CHECK_EQ(iweights.size_bytes(), oweights->size_bytes());
  // K indexes over output channels, C over input channels, and R and S over the
  // height and width of the convolution
  const int r = iweights.Shape().dim(0);
  const int s = iweights.Shape().dim(1);
  // TRT requires GKcRS, while TF depthwise has RSCK where c=1, C=G
  const int c = iweights.Shape().dim(2) / num_groups;
  const int k = iweights.Shape().dim(3) * num_groups;
  VLOG(2) << "num_groups: " << num_groups << "c" << iweights.Shape().dim(2)
          << " then " << c << "k" << iweights.Shape().dim(3) << " then " << k
          << "r" << iweights.Shape().dim(0) << " then " << r << "s"
          << iweights.Shape().dim(1) << " then " << s;
  oweights->Shape().dim(0) = k / num_groups;
  oweights->Shape().dim(1) = c * num_groups;
  oweights->Shape().dim(2) = r;
  oweights->Shape().dim(3) = s;
  const nvinfer1::Dims4 istrides = {1, k, s * k * c, c * k};
  const nvinfer1::Dims4 ostrides = {c * r * s, r * s, s, 1};
  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      Reorder4({k, c, r, s}, iweights.GetPointer<float>(), istrides,
               oweights->GetPointer<float>(), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder4({k, c, r, s}, iweights.GetPointer<Eigen::half>(), istrides,
               oweights->GetPointer<Eigen::half>(), ostrides);
      break;
    }

    default:
      LOG(FATAL) << "Unsupported type, expected fp32 or fp16 but got "
                 << DebugString(iweights.TrtDType());
  }
}

// Initialize a Dims object with arbitrary dimension
nvinfer1::Dims InitDimsN(std::initializer_list<int> list) {
  nvinfer1::Dims dim;
  dim.nbDims = list.size();
  std::copy(list.begin(), list.end(), dim.d);
  return dim;
}

// Reorder 3D convolution weights from TF to TRT
void ReorderDRSCKToKCDRS(const TRT_ShapedWeights& iweights,
                         TRT_ShapedWeights* oweights, const int num_groups) {
  DCHECK(iweights.TrtDType() == oweights->TrtDType());
  CHECK_EQ(iweights.size_bytes(), oweights->size_bytes());
  // K indexes over output channels, C over input channels, and R, S, D over the
  // height, width, depth
  const int d = iweights.Shape().dim(0);
  const int r = iweights.Shape().dim(1);
  const int s = iweights.Shape().dim(2);
  // TRT requires GKcRS, while TF depthwise has RSCK where c=1, C=G
  const int c = iweights.Shape().dim(3) / num_groups;
  const int k = iweights.Shape().dim(4) * num_groups;

  VLOG(2) << "num_groups: " << num_groups << ", c: " << iweights.Shape().dim(3)
          << " becomes " << c << ", k: " << iweights.Shape().dim(4)
          << " becomes " << k << ", d: " << d << ", r: " << r << ", s: " << s;

  oweights->Shape().dim(0) = iweights.Shape().dim(4);  // k / num_groups;
  oweights->Shape().dim(1) = iweights.Shape().dim(3);  // c * num_groups;
  oweights->Shape().dim(2) = d;
  oweights->Shape().dim(3) = r;
  oweights->Shape().dim(4) = s;

  nvinfer1::Dims shape =
      InitDimsN({k, c, d, r, s});  // KCDRS shape (same as output)

  nvinfer1::Dims ostrides =
      InitDimsN({c * d * r * s, d * r * s, r * s, s,
                 1});  // Output = KCDRS = k*CDRS + c*DRS + d*RS + r*S + s

  nvinfer1::Dims istrides =
      InitDimsN({1, k, r * s * c * k, s * c * k,
                 c * k});  // Input = DRSCK = k*1 + c*K + d*RSCK + r*SCK + s*CK

  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      Reorder5(shape, iweights.GetPointer<float>(), istrides,
               oweights->GetPointer<float>(), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder5(shape, iweights.GetPointer<Eigen::half>(), istrides,
               oweights->GetPointer<Eigen::half>(), ostrides);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type, expected fp32 or fp16 but got "
                 << DebugString(iweights.TrtDType());
  }
}

OpConverterParams::OpConverterParams(
    const NodeDef& node_def, const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs, TrtWeightStore* weight_store,
    TrtPrecisionMode precision_mode, bool use_calibration,
    bool use_implicit_batch, bool use_explicit_precision)
    : node_def(node_def),
      inputs(inputs),
      outputs(outputs),
      validation_only(true),
      weight_store(weight_store),
      precision_mode(precision_mode),
      use_calibration(use_calibration),
      use_implicit_batch(use_implicit_batch),
      use_explicit_precision(use_explicit_precision) {}

OpConverterParams::OpConverterParams(
    Converter* converter, const NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs, TrtWeightStore* weight_store)
    : converter(converter),
      node_def(node_def),
      inputs(inputs),
      outputs(outputs),
      validation_only(false),
      weight_store(weight_store),
      precision_mode(converter->precision_mode()),
      use_calibration(converter->use_calibration()),
      use_implicit_batch(converter->use_implicit_batch()),
      use_explicit_precision(converter->UseExplicitPrecision()) {}

TrtNodeValidator::TrtNodeValidator(
    const grappler::GraphProperties& graph_properties,
    TrtPrecisionMode precision_mode, bool use_calibration,
    bool use_implicit_batch, bool use_explicit_precision)
    : graph_properties_(graph_properties),
      precision_mode_(precision_mode),
      use_calibration_(use_calibration),
      use_implicit_batch_(use_implicit_batch),
      use_explicit_precision_(use_explicit_precision) {}

StatusOr<OpConverter> TrtNodeValidator::GetValidator(const std::string& op) {
  return GetOpConverterRegistry()->LookUp(op);
}

Status TrtNodeValidator::ConvertToTensorOrWeights(
    const NodeDef& node_def, int output_port,
    TRT_TensorOrWeights* tensor_or_weights) {
  if (node_def.op() == "Const") {
    if (output_port != 0) {
      return errors::InvalidArgument("Const node should only have one output.");
    }
    // The output of the conversion will be used as input to other nodes to
    // determine whether TRT supports those nodes. If it cannot convert the
    // Const, it's very likely we cannot treat it as a tensor and make it an
    // input to the TRT network, since TRT removes the first dimension and
    // treats it as batch size. Also, it's not likely that the converter can
    // support the op, and performance may suffer even if it can, so we just
    // simply return error if the conversion fails.
    std::vector<TRT_TensorOrWeights> inputs;
    return ConvertConstToWeights(node_def, inputs, tensor_or_weights);
  }
  if (!graph_properties_.HasOutputProperties(node_def.name())) {
    return errors::InvalidArgument("Shape and data type are unknown");
  }

  // Validate and convert shape and dtype.
  const auto& output_params =
      graph_properties_.GetOutputProperties(node_def.name());
  const auto& tensor_properties = output_params.at(output_port);
  const DataType dtype = tensor_properties.dtype();
  const PartialTensorShape shape = tensor_properties.shape();
  nvinfer1::DataType trt_dtype;
  nvinfer1::Dims trt_dims;
  int batch_size = -1;
  TF_RETURN_IF_ERROR(ValidateTensorProperties(
      node_def.op(), dtype, shape, use_implicit_batch_,
      /*validation_only_=*/true, &trt_dtype, &trt_dims, &batch_size));

  // Adds a fake ITensor. This is fine since op converter operates in
  // validation-only mode and it won't (and shouldn't) use the tensor to do
  // any TRT network operations.
  *tensor_or_weights = TRT_TensorOrWeights(trt_dtype, trt_dims, batch_size);
  return Status::OK();
}

Status TrtNodeValidator::IsTensorRTCandidate(const Node* node) {
  const string& op = node->def().op();
  // In INT8 mode, we will always apply the quantization ranges provided by
  // these ops to the relevant tensors. This happens regardless of the value of
  // use_calibration.
  bool is_supported_op = false;
  if (absl::c_find(kQuantizationOpNames, op) != kQuantizationOpNames.end()) {
    is_supported_op = (precision_mode_ == TrtPrecisionMode::INT8);
  } else {
    is_supported_op = GetValidator(op).ok();
  }

  if (!is_supported_op) {
    return errors::Unimplemented("Op type ", op, " is not supported.");
  }

  // Convert input NodeDef and corresponding output ports to
  // TRT_TensorOrWeights.
  std::vector<TRT_TensorOrWeights> inputs;
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node->input_edges(&input_edges));
  for (const Edge* edge : input_edges) {
    TRT_TensorOrWeights tensor_or_weights;
    const NodeDef& src_def = edge->src()->def();
    Status status = ConvertToTensorOrWeights(src_def, edge->src_output(),
                                             &tensor_or_weights);
    if (!status.ok()) {
      VLOG(2) << "Failed to convert input `" << src_def.name() << "` to a "
              << "TRT_TensorOrWeights: " << status.error_message();

      return errors::Internal(
          "Failed to convert at least one input to a TRT_TensorOrWeights: ",
          status.error_message());
    }
    inputs.push_back(tensor_or_weights);
  }

  auto validator = GetValidator(op);
  TF_RETURN_IF_ERROR(validator.status());
  OpConverterParams params(node->def(), inputs, /*arg_outputs=*/nullptr,
                           &weight_store_, precision_mode_, use_calibration_,
                           use_implicit_batch_, use_explicit_precision_);
  return (*validator)(&params);
}

Status TrtNodeValidator::ConvertConstToWeights(
    const NodeDef& const_node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    TRT_TensorOrWeights* output) {
  std::vector<TRT_TensorOrWeights> outputs;
  OpConverterParams params(const_node_def, inputs, &outputs, &weight_store_,
                           precision_mode_, use_calibration_,
                           use_implicit_batch_, use_explicit_precision_);
  auto const_val = GetValidator("Const");
  TF_RETURN_IF_ERROR(const_val.status());
  Status status = (*const_val)(&params);
  if (status.ok() && (output != nullptr)) {
    *output = outputs[0];
  }
  return status;
}

// static
StatusOr<std::unique_ptr<Converter>> Converter::Create(
    TrtPrecisionMode precision_mode, bool use_calibration,
    nvinfer1::ILogger* trt_logger, const bool use_implicit_batch,
    absl::string_view engine_name, bool use_explicit_precision) {
  std::unique_ptr<Converter> converter = absl::WrapUnique(
      new Converter(precision_mode, use_calibration, trt_logger,
                    use_implicit_batch, engine_name, use_explicit_precision));
  TF_RETURN_IF_ERROR(converter->Init(trt_logger));
  return converter;
}

Converter::Converter(TrtPrecisionMode precision_mode, bool use_calibration,
                     nvinfer1::ILogger* trt_logger,
                     const bool use_implicit_batch,
                     absl::string_view engine_name, bool use_explicit_precision)
    : precision_mode_(precision_mode),
      use_calibration_(use_calibration),
      use_implicit_batch_(use_implicit_batch),
      engine_name_(engine_name),
      use_explicit_precision_(use_explicit_precision) {
  MaybeInitializeTrtPlugins(trt_logger);
}

Status Converter::Init(nvinfer1::ILogger* trt_logger) {
  VLOG(1) << "Creating TensorRT builder";
  trt_builder_.reset(nvinfer1::createInferBuilder(*trt_logger));

  VLOG(1) << "Creating TensorRT network";
  uint32_t flags =
      use_implicit_batch_
          ? 0U
          : (1U << static_cast<int>(
                 nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  if (use_explicit_precision_) {
    flags |=
        (1U << static_cast<int>(
             nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION));
  }
  trt_network_.reset(trt_builder_->createNetworkV2(flags));
  if (!trt_network_) {
    return errors::Internal("Failed to create TensorRT network object");
  }
  return Status::OK();
}

Status Converter::ConvertNode(const NodeDef& node_def) {
  std::vector<TRT_TensorOrWeights> inputs;
  std::vector<TRT_TensorOrWeights> outputs;
  TF_RETURN_IF_ERROR(this->GetInputs(node_def, &inputs));

  OpConverterParams params(this, node_def, inputs, &outputs, &weight_store_);
  const string& op = node_def.op();
  auto op_converter = GetOpConverterRegistry()->LookUp(op);
  TF_RETURN_IF_ERROR(op_converter.status());
  TF_RETURN_IF_ERROR((*op_converter)(&params));

  for (size_t i = 0; i < outputs.size(); ++i) {
    TRT_TensorOrWeights& output = outputs[i];
    string output_name = node_def.name();
    if (i != 0) {
      StrAppend(&output_name, ":", i);
    }
    // We need to check the name before setting it. If the input is one of the
    // engine input, setting the name here will overwrite engine input
    // bindings which will cause runtime error.
    // TODO(tmorris): Remove this work-around once we use TRT's IIdentityLayer
    // in ConvertIdentity.
    if (output.is_tensor()) {
      const char* tensor_name = output.tensor()->getName();
      if (!IsEngineInput(tensor_name)) {
        // TRT initializes tensor names as "(Unnamed ITensor* N)". We rename
        // them to match their corresponding TensorFlow name.
        // Note: ITensors that we create internally within TF-TRT which are
        // not inputs or outputs of a node will not be renamed. This is a
        // potential cause of confusion if an error message or warning
        // mentions the unnamed tensor.
        output.tensor()->setName(output_name.c_str());
      }
    }
    VLOG(2) << "Adding out tensor " << output_name << ": "
            << output.DebugString();
    Status status = AddTensorOrWeights(output_name, output);
    if (!status.ok()) {
      return errors::Create(
          status.code(),
          StrCat("Failed to add output for node: ", node_def.name(), ": ",
                 status.error_message()),
          errors::GetPayloads(status));
    }
  }
  return Status::OK();
}

Status Converter::AddInputTensor(const string& name, nvinfer1::DataType dtype,
                                 const nvinfer1::Dims& dims, int batch_size) {
  // We verify the batch size only for the input nodes, and rely on individual
  // op converter to ensure the batch size of the outputs is not changed.
  // TODO(laigd): we need to test this properties.
  Status status;
  if (use_implicit_batch_) {
    status = MaybeUpdateBatchSize(batch_size);
    if (!status.ok()) {
      return errors::CreateWithUpdatedMessage(
          status, StrCat("Batch size doesn't match for tensor ", name, ": ",
                         status.error_message()));
    }
  }
  ITensorProxyPtr tensor = network()->addInput(name.c_str(), dtype, dims);
  if (*tensor == nullptr) {
    return errors::InvalidArgument("Failed to create Input layer tensor ", name,
                                   " rank=", dims.nbDims);
  }
  status = AddTensorOrWeights(name, TRT_TensorOrWeights(tensor));
  if (!status.ok()) {
    return errors::CreateWithUpdatedMessage(
        status, StrCat("Failed to add input tensor ", name, ": ",
                       status.error_message()));
  }
  return Status::OK();
}

Status Converter::RenameAndMarkOutputTensors(
    const std::vector<Converter::EngineOutputInfo>& output_tensors) {
  int output_index = 0;
  for (const auto& output : output_tensors) {
    TRT_TensorOrWeights tensor_or_weights;
    TF_RETURN_IF_ERROR(
        GetTensorOrWeights(output.source_tensor_name, &tensor_or_weights));
    if (!tensor_or_weights.is_tensor()) {
      return errors::InvalidArgument("Output ", output.source_tensor_name,
                                     " is weights not tensor");
    }
    ITensorProxyPtr tensor = tensor_or_weights.tensor();
    if (*tensor == nullptr) {
      return errors::NotFound("Output tensor not found: ",
                              output.source_tensor_name);
    }
    // Check if this tensor has already been marked as an input or output.
    //
    // ConvertIdentity can cause the same tensor to be repeated in
    // output_tensors, which can cause us to overwrite the name of the output
    // tensor binding. For example, if we rename OutputPH_0 to OutputPH_1 then
    // we won't be able to locate OutputPH_0 during runtime. To fix this,
    // duplicate the tensor using no-op shuffle.
    //
    // TODO(tmorris): Remove this work-around once we use TRT's IIdentityLayer
    // in ConvertIdentity.
    if (IsEngineInput(tensor->getName()) || IsEngineOutput(tensor->getName())) {
      // Using shuffle layer for identity by not setting reshape or transpose.
      nvinfer1::IShuffleLayer* layer =
          network()->addShuffle(*tensor->trt_tensor());
      TFTRT_RETURN_ERROR_IF_NULLPTR(
          layer, StrCat("Output Copy for ", tensor->getName()));
      SetLayerName(layer, tensor->getName(), "shuffle", output_index);
      tensor = layer->getOutput(0);
    }
    tensor->setName(output.dest_node_name.c_str());
    network()->markOutput(*tensor->trt_tensor());
    // Set type after marking as output. TRT only supports setType for engine
    // outputs and inputs (type is inferred otherwise).
    tensor->setType(output.trt_dtype);
    output_index++;
    VLOG(1) << "Marking output TRT tensor " << output.source_tensor_name
            << " with data type " << DebugString(output.trt_dtype)
            << ", which feeds TF node " << output.dest_node_name;
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Created TensorRT network with the following layers:";
    for (int i = 0; i < network()->getNbLayers(); i++) {
      auto layer = network()->getLayer(i);
      VLOG(2) << "    " << layer->getName() << " ("
              << "type: " << static_cast<int>(layer->getType())
              << ", precision: " << static_cast<int>(layer->getPrecision())
              << ")";
    }
  }
  return Status::OK();
}

// Returns the value of TF_TRT_ABORT_CUDA_ENGINE_BUILD environment variable.
// This variable can be used to abort CUDA engine construction, therefore it
// provides a way to test and debug the native segment fallback of TF-TRT.
bool AbortCudaEngineBuild() {
  bool value;
  Status status = ReadBoolFromEnvVar("TF_TRT_ABORT_CUDA_ENGINE_BUILD",
                                     /*default_value=*/false, &value);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return value;
}

Status Converter::BuildCudaEngine(
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, int max_batch_size,
    size_t max_workspace_size_bytes, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator, TrtShapeOptimizationProfile* profiles) {
  tensorflow::profiler::AnnotatedTraceMe activity(
      [&]() {
        return tensorflow::profiler::TraceMeOpOverride("TRTEngineOp",
                                                       "BuildEngine");
      },
      tensorflow::profiler::TraceMeLevel::kInfo);

  if (AbortCudaEngineBuild()) {
    return errors::Aborted(
        "Engine creation aborted by TF_TRT_ABORT_CUDA_ENGINE_BUILD variable");
  }

  VLOG(1) << "Configuring TensorRT builder";
  trt_builder_->setMaxBatchSize(max_batch_size);
  trt_builder_->setGpuAllocator(allocator);

  // Create a network configuration and use it to build a TRT engine.
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
      trt_builder_->createBuilderConfig());
  builder_config->setMaxWorkspaceSize(max_workspace_size_bytes);

  // Create the algorithm selector. For TensorRT 7.x, the algorithm selector
  // cannot be used when building with INT8 calibration.
  std::unique_ptr<nvinfer1::IAlgorithmSelector> trt_algorithm_selector{nullptr};
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
    if (!use_calibration_ || precision_mode_ != TrtPrecisionMode::INT8) {
      trt_algorithm_selector = MaybeCreateAlgorithmSelector();
    }
  } else {
    trt_algorithm_selector = MaybeCreateAlgorithmSelector();
  }

  if (trt_algorithm_selector != nullptr) {
    builder_config->setAlgorithmSelector(trt_algorithm_selector.get());
  }

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  builder_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  VLOG(1) << "Setting sparsity for TensorRT8!";
#endif

  if (tensorflow::tensor_float_32_execution_enabled()) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kTF32);
  } else {
    builder_config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  }

  if (precision_mode_ == TrtPrecisionMode::FP16) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (precision_mode_ == TrtPrecisionMode::INT8) {
    // FP16 is not available in Explicit Precision mode with TensorRT 7.
    if (IS_TRT_VERSION_GE(8, 0, 0, 0) || !use_explicit_precision_) {
      builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
      LOG_WARNING_WITH_PREFIX << "With explicit precision mode, FP16 is not "
                                 "allowed before TensorRT 8. TRT will consider "
                                 "INT8 and FP32 tactics.";
    }
    builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  if (!use_implicit_batch_ && profiles) {
    TF_RETURN_IF_ERROR(profiles->ConfigureBuilder(
        trt_builder_.get(), builder_config.get(), network()));
  }
  if (precision_mode_ == TrtPrecisionMode::INT8) {
    builder_config->setInt8Calibrator(use_calibration_ ? calibrator : nullptr);
  }

  std::unique_ptr<TimingCacheRegistry::TimingCache> timing_cache = nullptr;
  // We only use a timing cache if the algorithm selector is not used. If we
  // are using TRT version >= 8.0, then we can try to deserialize an existing
  // cache.
  if (trt_algorithm_selector == nullptr) {
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    TimingCacheRegistry* registry = GetTimingCacheRegistry();

    auto cache = registry->LookUp("default_cache", builder_config.get());
    if (!cache.ok()) {
      LOG(WARNING) << "failed to create a timing cache: "
                   << cache.status().error_message();
    } else {
      timing_cache = std::move(*cache);
      builder_config->setTimingCache(*timing_cache, /*ignoreMismatch*/ false);
    }
#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
  } else {
    // Disabling the timing cache is recommended when using the algorithm
    // selector.
    builder_config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
  }

  string precision_mode_str;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(precision_mode_, &precision_mode_str));
  string trt_network_name = StrCat(
      "TF:", TF_VERSION_STRING, ", ",
      "TRT:", absl::StrJoin(GetLoadedTensorRTVersion(), "."), "-",
      "Precision:", precision_mode_str, ", ", "Calibration:", use_calibration_,
      ", ", "Max-Batch-Size:", max_batch_size, ", ",
      "Max-Workspace-Size:", max_workspace_size_bytes);
  VLOG(1) << "Setting TensorRT network name to " << trt_network_name;
  network()->setName(trt_network_name.c_str());

  VLOG(1) << "Building TensorRT engine";
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Network inputs";
    int n_inputs = network()->getNbInputs();
    for (int i = 0; i < n_inputs; i++) {
      const ITensorProxyPtr input = network()->getInput(i);
      if (*input) {
        VLOG(2) << "  " << i << " " << input->getName();
      } else {
        VLOG(2) << "Could not find input " << i;
      }
    }
  }
  engine->reset(
      trt_builder_->buildEngineWithConfig(*network(), *builder_config));
  if (engine->get() == nullptr) {
    return errors::Internal("Failed to build TensorRT engine");
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "TRT engine created";
    int nbBindings = (*engine)->getNbBindings();
    VLOG(2) << "Number of engine bindings: " << nbBindings;
    for (int i = 0; i < nbBindings; i++) {
      auto get_location_string = [&engine](int i) {
        if ((*engine)->getLocation(i) == nvinfer1::TensorLocation::kDEVICE)
          return " on device";
        else
          return " on host";
      };
      VLOG(2) << "Binding " << i << " name: " << (*engine)->getBindingName(i)
              << get_location_string(i);
    }
  }

  // Write back the new timing cache results to the registry.
  if (timing_cache) {
    GetTimingCacheRegistry()->Upsert("default_cache", timing_cache.get());
  }

  return Status::OK();
}

Status Converter::MaybeUpdateBatchSize(int batch_size) {
  // OK iff either is unknown or they equal to each other.
  if (this->batch_size_ < 0 || batch_size < 0 ||
      this->batch_size_ == batch_size) {
    if (this->batch_size_ < 0 && batch_size >= 0) {
      this->batch_size_ = batch_size;
    }
    return Status::OK();
  }
  return errors::InvalidArgument(
      "Provided batch size does not match converter batch size: ", batch_size,
      " vs ", batch_size_);
}

Status Converter::AddTensorOrWeights(const string& name,
                                     TRT_TensorOrWeights input) {
  // Set the batch size of the tensor, using batch size collected from the
  // input tensors to the TRT subgraph at the beginning of the conversion.
  // We rely on the individual op converter to understand the semantics of the
  // TF node, and make sure it doesn't change the batch size nor introduce
  // intra-element dependency inside the batch.
  if (use_implicit_batch_ && input.is_tensor()) {
    input.set_batch_size(batch_size_);
  }
  if (trt_tensors_.insert({name, std::move(input)}).second) return Status::OK();
  return errors::AlreadyExists("tensor/weights ", name, " already exist.");
}

Status Converter::GetTensorOrWeights(const string& name,
                                     TRT_TensorOrWeights* output) {
  if (!trt_tensors_.count(name)) {
    return errors::NotFound("Tensor or weights with name ", name,
                            " could not be found.");
  }
  *output = trt_tensors_.at(name);
  return Status::OK();
}

Status Converter::TransposeTensor(ITensorProxyPtr input_tensor,
                                  const std::vector<int>& order_with_batch_dim,
                                  ITensorProxyPtr* output_tensor,
                                  const NodeDef& node_def,
                                  absl::string_view sub_op_name) {
  const auto dims = input_tensor->getDimensions();
  const int order_size = use_implicit_batch_ ? order_with_batch_dim.size() - 1
                                             : order_with_batch_dim.size();
  if (order_size != size_t(dims.nbDims)) {
    return errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
  }
  if (use_implicit_batch_ && order_with_batch_dim[0] != 0) {
    return errors::Unimplemented(
        "Transpose at batch dimension is not supported.");
  }

  nvinfer1::IShuffleLayer* layer =
      this->network()->addShuffle(*input_tensor->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Transpose");
  SetLayerName(layer, node_def, sub_op_name);

  nvinfer1::Permutation permutation;
  if (use_implicit_batch_) {
    for (int32_t i = 0; i < dims.nbDims; ++i) {
      permutation.order[i] = order_with_batch_dim[i + 1] - 1;
    }
  } else {
    std::copy(order_with_batch_dim.begin(), order_with_batch_dim.end(),
              permutation.order);
  }
  VLOG(1) << "TransposeTensor permutation: "
          << DebugString(permutation, dims.nbDims);
  layer->setFirstTranspose(permutation);

  nvinfer1::Dims reshape_dims;
  reshape_dims.nbDims = dims.nbDims;
  for (int32_t i = 0; i < reshape_dims.nbDims; ++i) {
    reshape_dims.d[i] = 0;
  }
  layer->setReshapeDimensions(reshape_dims);

  *output_tensor = layer->getOutput(0);
  return Status::OK();
}

Status Converter::GetWeightRange(const TRT_ShapedWeights& weights,
                                 float* out_min, float* out_max) const {
  switch (weights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      auto inp = weights.GetPointer<float>();
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = *result.first;
      *out_max = *result.second;
      break;
    }
    case nvinfer1::DataType::kHALF: {
      auto inp = weights.GetPointer<Eigen::half>();
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = static_cast<float>(*result.first);
      *out_max = static_cast<float>(*result.second);
      break;
    }
    case nvinfer1::DataType::kINT32: {
      auto inp = weights.GetPointer<int>();
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = static_cast<float>(*result.first);
      *out_max = static_cast<float>(*result.second);
      break;
    }
    default:
      return errors::Unimplemented(
          "Data type not supported for GetWeightRange: ",
          DebugString(weights.TrtDType()));
  }
  return Status::OK();
}

// Constructs <tf_related_part> for the ILayer name as
// <tf_node_def_name>_<sub_op_name>_<sub_op_instance> and callSetLayerNameHelper
// to set the name for the ILayer.
//
// If the operation represented by the ILayer is generated by the converter to
// support the conversion of node_def, callers need to specify a non-empty
// sub_op_name to be appended to the name of node_def to avoid layer name
// conflicts. If the operation is generated multiple times, callers also need
// to specify sub_op_instance to be appended to the name of the layers to avoid
// layer name conflicts.
void Converter::SetLayerName(nvinfer1::ILayer* layer, const NodeDef& node_def,
                             absl::string_view sub_op_name,
                             absl::optional<int> sub_op_instance,
                             absl::optional<std::string> origin_node_name) {
  std::string sub_op_suffix = GetLayerNameSuffix(sub_op_name, sub_op_instance);
  if (sub_op_suffix.empty()) {
    SetLayerNameHelper(layer, engine_name_, node_def.name());
  } else if (origin_node_name.has_value()) {
    auto layer_name = absl::StrCat(node_def.name(), "-",
                                   absl::string_view(origin_node_name.value()),
                                   "-", sub_op_suffix);
    SetLayerNameHelper(layer, engine_name_, layer_name);
  } else {
    SetLayerNameHelper(layer, engine_name_,
                       absl::StrCat(node_def.name(), "-", sub_op_suffix));
  }
}

// Constructs <tf_related_part> for the ILayer name as
// <main_op_name>_<sub_op_name>_<sub_op_instance> and callSetLayerNameHelper to
// set the name for the ILayer.
void Converter::SetLayerName(nvinfer1::ILayer* layer,
                             absl::string_view main_op_name,
                             absl::string_view sub_op_name,
                             absl::optional<int> sub_op_instance) {
  std::string layer_name_suffix =
      GetLayerNameSuffix(sub_op_name, sub_op_instance);
  SetLayerNameHelper(layer, engine_name_,
                     absl::StrCat(main_op_name, "-", layer_name_suffix));
}

// Converts 'input' of 'node_def' into 'tensor' with shape specified by 'dims'
// (which doesn't contain the batch dimension).
//
// If validation_only is true, it doesn't do the conversion but only do some
// minimum validation for the eligibility of the conversion, and *tensor will
// be set to nullptr.
Status PrepareTensorForShape(Converter* converter,
                             const TRT_TensorOrWeights& input,
                             const DimsAdapter& dims,
                             const bool validation_only,
                             ITensorProxyPtr* tensor, const NodeDef& node_def,
                             absl::optional<int> op_instance,
                             absl::optional<std::string> origin_node_name) {
  DimsAdapter input_dims(input.GetTrtDims());
  // The input shape may have -1s for dynamic shape. The target shape may have
  // 0s representing copy over the corresponding input dimensions. It may also
  // have at most one -1 representing a dimension value that needs to be
  // inferred. If none of those special values present, we verify that the total
  // sizes of the input and output shape are the same.
  // TODO(tfeher): Verify that the total sizes of the input and output shape are
  // the same in the present of 0s but no -1 in the target shape.
  // If an input is a weight, it is going to become a tensor via
  // CreateConstantLayer. So we can treat it as a tensor for
  // AreDimsStaticWithDifferentSize(). This really only matters for 0-D tensors.
  if (dims.Volume() > 0 && AreDimsStaticWithDifferentSize(input_dims, dims)) {
    return errors::InvalidArgument(
        "Incompatible shapes: ", input_dims.DebugString(), " vs. ",
        dims.DebugString());
  }
  // ConstantLayer requires static shapes (cannot infer -1).
  if (input.is_weights() && !dims.IsStatic()) {
    return errors::InvalidArgument("Shape is not fully defined: ",
                                   dims.DebugString());
  }
  if (validation_only) {
    *tensor = nullptr;
    return Status::OK();
  }

  TFTRT_RETURN_ERROR_IF_NULLPTR(converter, "converter is nullptr");
  if (input.is_tensor()) {
    if (input_dims == dims) {
      *tensor = input.tensor();
    } else {
      nvinfer1::IShuffleLayer* layer =
          converter->network()->addShuffle(*input.tensor()->trt_tensor());
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Reshape");
      converter->SetLayerName(layer, node_def, "shuffle", op_instance,
                              origin_node_name);
      layer->setReshapeDimensions(dims.AsTrtDims());
      *tensor = layer->getOutput(0);
    }
  } else {
    *tensor = converter->CreateConstantLayer(input.weights(), dims.AsTrtDims());
    TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, "TF-TRT Internal Reshape");
  }
  return Status::OK();
}

void Converter::ProvideQuantizationRange(ITensorProxyPtr* tensor,
                                         float min_range, float max_range) {
  float symmetric_range = std::max(std::abs(min_range), std::abs(max_range));
  if ((*tensor)->is_trt_tensor()) {
    quantization_ranges_[(*tensor)->trt_tensor()] = symmetric_range;
  } else if ((*tensor)->is_simple_tensor()) {
    quantization_ranges_proxy_[tensor] = symmetric_range;
  }
}

void Converter::MaybeApplyQuantizationRanges() {
  if (precision_mode() != TrtPrecisionMode::INT8) return;

  // Apply ranges.
  for (auto pair : quantization_ranges_) {
    nvinfer1::ITensor* tensor = pair.first;
    const float range = pair.second;
    VLOG(1) << "Setting range for: " << tensor->getName() << ": " << range;
    // TODO(laigd): if 'tensor' already has a range set which doesn't match
    // 'range', it should report error.
    tensor->setDynamicRange(-range, range);
  }
  for (auto pair : quantization_ranges_proxy_) {
    ITensorProxyPtr tensor = *pair.first;
    const float range = pair.second;
    VLOG(1) << "Setting range for: " << tensor->getName() << ": " << range;
    // TODO(laigd): if 'tensor' already has a range set which doesn't match
    // 'range', it should report error.
    tensor->setDynamicRange(-range, range);
  }
}

Status Converter::GetInputs(const NodeDef& node_def,
                            std::vector<TRT_TensorOrWeights>* inputs) const {
  for (auto const& input_name : node_def.input()) {
    /*************************************************************************
     * TODO(jie): handle case 1) here.
     * Normalizes the inputs and extracts associated metadata:
     * 1) Inputs can contain a colon followed by a suffix of characters.
     *    That suffix may be a single number (e.g. inputName:1) or several
     *    word characters separated from a number by a colon
     *    (e.g. inputName:foo:1). The
     *    latter case is used to denote inputs and outputs of functions.
     * 2) Control dependency inputs contain caret at the beginning and we
     *    remove this and annotate the edge as a control dependency.
     ************************************************************************/
    // skip control nodes
    if (input_name[0] == '^') continue;
    string name = input_name;
    auto last = name.find_last_of(':');
    // TODO(aaroey): use TensorId
    if (last != string::npos && last + 2 == name.size() &&
        name[last + 1] == '0') {
      name.erase(last);
    }

    if (trt_tensors_.count(name)) {
      TRT_TensorOrWeights input = trt_tensors_.at(name);
      inputs->push_back(input);
      VLOG(2) << "Retrieved input " << name << ": " << input.DebugString();
    } else {
      // TODO(aaroey): this should not happen, make it a CHECK.
      // TODO(aaroey): use StrCat for pattern like this.
      string msg("Node ");
      StrAppend(&msg, node_def.name(), " should have an input named '", name,
                "' but it is not available");
      LOG(ERROR) << msg;
      return errors::InvalidArgument(msg);
    }
  }
  return Status::OK();
}

// Checks that the number of inputs match, and enforces that the inputs marked
// as weights are constant. Inputs are allowed to be both weight and tensor.
Status CheckInputsWeights(
    const OpConverterParams& params,
    const std::vector<std::pair<string, TrtInputArg>>& expected_inputs) {
  const auto& inputs = params.inputs;
  const auto& node_def = params.node_def;
  TFTRT_CHECK_INPUT_SIZE(inputs.size(), expected_inputs.size(), node_def);
  for (int i = 0; i < inputs.size(); i++) {
    if (expected_inputs[i].second == TrtInputArg::kWeight &&
        inputs.at(i).is_tensor()) {
      return errors::Unimplemented("The input \"", expected_inputs[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a constant");
    }
    // TODO(tfeher): Remove this check and provide a method to automatically
    // retrieve an input as a tensor, converting via CreateConstantLayer if it
    // was originally a weight. We will want a caching mechanism to prevent many
    // duplicate constants from being created.
    if (expected_inputs[i].second == TrtInputArg::kTensor &&
        inputs.at(i).is_weights()) {
      return errors::Unimplemented("The input \"", expected_inputs[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a tensor");
    }
  }
  return Status::OK();
}

// Checks that the number of inputs match, and enforces that the inputs marked
// as true are constant weights. true means that the input must be a weight,
// while false means the input must be a tensor.
Status CheckInputsWeights(
    const OpConverterParams& params,
    const std::vector<std::pair<string, bool>>& inputs_is_weight) {
  std::vector<std::pair<string, TrtInputArg>> expected_inputs;
  expected_inputs.reserve(inputs_is_weight.size());
  std::transform(
      inputs_is_weight.begin(), inputs_is_weight.end(),
      std::back_inserter(expected_inputs), [](std::pair<string, bool> x) {
        return std::make_pair(
            x.first, x.second ? TrtInputArg::kWeight : TrtInputArg::kTensor);
      });
  return CheckInputsWeights(params, expected_inputs);
}

Status GetNodeDefTfType(const NodeDef& node_def, DataType* tf_type,
                        const char* type_attr_name) {
  AttrSlice attrs(node_def);
  if (attrs.FindByString(type_attr_name) == nullptr) {
    return errors::InvalidArgument("Attribute with name ", type_attr_name,
                                   " not found.");
  }
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, type_attr_name, tf_type));
  return Status::OK();
}

Status GetInputTfType(const OpConverterParams& params, DataType* tf_type,
                      int pos) {
  const std::vector<TRT_TensorOrWeights>& inputs = params.inputs;
  if (inputs.size() <= pos) {
    return errors::Internal("Invalid input position");
  }

  return inputs[pos].GetTfType(tf_type);
}

constexpr const char kOutputTypeAttrName[] = "T";

Status GetOutputTfType(const OpConverterParams& params, DataType* tf_type) {
  return GetNodeDefTfType(params.node_def, tf_type, kOutputTypeAttrName);
}

Status AllowDataTypes(const OpConverterParams& params,
                      const std::set<DataType>& allowed_types,
                      const char* type_attr_name = kOutputTypeAttrName) {
  const auto& node_def = params.node_def;
  DataType tf_type;
  TF_RETURN_IF_ERROR(GetNodeDefTfType(node_def, &tf_type, type_attr_name));
  if (!allowed_types.count(tf_type)) {
    string allowed_types_string = absl::StrJoin(
        allowed_types, ", ", [](string* out, const DataType& type) {
          absl::StrAppendFormat(out, "%s", DataTypeString(type));
        });
    return errors::Unimplemented(
        "Data type ", DataTypeString(tf_type), " is not supported for ",
        node_def.op(), ", must be one of [", allowed_types_string, "]");
  }
  return Status::OK();
}

namespace {
// Extracts the spatial dimensions from `output_sizes` and returns them as a
// vector of size 2.
std::vector<int64_t> GetSpatialDimsFromOutputSizes(
    const TRT_TensorOrWeights& output_sizes, const int h_index,
    const int w_index) {
  // We use h_index and w_index instead of 1 and 2 because we haven't
  // transposed output_sizes along with the input.
  const TRT_ShapedWeights& weights = output_sizes.weights();
  const int output_sizes_length = weights.count();
  auto output_sizes_values = weights.GetPointer<int>();
  // The length of output_sizes can be 2 or 4. When the length is 4,
  // output_sizes represents <height,width>.
  return {output_sizes_values[output_sizes_length == 4 ? h_index : 0],
          output_sizes_values[output_sizes_length == 4 ? w_index : 1]};
}
}  // namespace

Status ConvertConv2DHelper(OpConverterParams* params, int group,
                           bool is_conv2d_backprop_input) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TRT_TensorOrWeights backprop_output_size;
  ITensorProxyPtr tensor = nullptr;
  if (is_conv2d_backprop_input) {
    // In the case when Conv2dBackpropInput is used for conv2d_transpose, these
    // inputs correspond to: output size, filter, and input.
    // TODO(cbate): refine this check when moving to structured op converter.
    if (!params->use_explicit_precision) {
      TF_RETURN_IF_ERROR(CheckInputsWeights(
          *params,
          {{"input_sizes", true}, {"filter", true}, {"out_backprop", false}}));
    }

    backprop_output_size = inputs.at(0);
    tensor = inputs.at(2).tensor();
    bool has_dynamic_hw_shape{false};
    int start_idx{0};
    auto dims = tensor->getDimensions();
    if (params->use_implicit_batch) {
      if (dims.nbDims != 3) {
        return errors::Internal(
            "In implicit batch mode, input nbDims should be 3");
      }
      start_idx = 1;
    } else {
      if (dims.nbDims != 4) {
        return errors::Internal(
            "In explicit batch mode, input nbDims should be 4");
      }
      start_idx = 2;
    }
    for (int i = start_idx; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0) {
        has_dynamic_hw_shape = true;
      }
    }
    if (has_dynamic_hw_shape) {
      return errors::Unimplemented(
          "Conv2dBackpropInput does not support input with unknown spatial "
          "shape");
    }
  } else {
    TF_RETURN_IF_ERROR(CheckInputsWeights(
        *params,
        {{"input", false}, {"filter", !params->use_explicit_precision}}));
    tensor = inputs.at(0).tensor();
  }
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  if (inputs.at(1).GetTrtDims().nbDims != 4) {
    return errors::InvalidArgument("Conv2D expects kernel of dimension 4");
  }

  string data_format, padding_type;
  std::vector<int64_t> tf_dilations, tf_stride;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &tf_stride));

  int c_index = (data_format == "NHWC") ? 3 : 1;
  int h_index = (data_format == "NHWC") ? 1 : 2;
  int w_index = (data_format == "NHWC") ? 2 : 3;

  if (tf_dilations.size() != 4) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 4 dimensions");
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions");
  }
  const nvinfer1::DimsHW dilation(tf_dilations[h_index], tf_dilations[w_index]);
  if (is_conv2d_backprop_input && (dilation.d[0] != 1 || dilation.d[1] != 1)) {
    return errors::Unimplemented(
        "Dilation with Conv2DBackpropInput (conv2d_transpose) is not"
        " supported");
  }

  if (tf_stride.size() != 4) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 4 dimensions");
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions");
  }
  // Channel dim must be static for DepthwiseConv2dNative since we use that
  // value for num_groups at build time.
  if (!params->use_implicit_batch && tensor->getDimensions().d[c_index] == -1) {
    return errors::InvalidArgument("Channel dimension must be static");
  }

  if (padding_type != "SAME" && padding_type != "VALID") {
    return errors::Unimplemented(padding_type +
                                 " padding type not implemented, "
                                 "only VALID and SAME are supported");
  }
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  if (params->validation_only) return Status::OK();

  // Transpose to NCHW (NCHW is required for IConvLayer).
  const bool need_transpose = (data_format == "NHWC");
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 3, 1, 2}, &tensor, node_def, "to_NCHW"));
  }
  // Dimensions of transposed tensor.
  const auto tensor_dim = tensor->getDimensions();
  const int c_dim_size = tensor_dim.d[params->use_implicit_batch ? 0 : 1];

  // group == 0 signifies that this is a depthwise convolution, so set
  // num_groups to size of input's channel dim. For a non-depthwise conv,
  // num_groups will be 1.
  const int num_groups = (group == 0) ? c_dim_size : group;

  // For conv, TF weights are RSCK, and TRT expects KCRS.
  // For backprop, TF weights are RSKC, and TRT expects CKRS.
  // Therefore, this reorder will work for both cases.
  const int output_axis = is_conv2d_backprop_input ? 2 : 3;
  auto weights_shape = inputs.at(1).GetTrtDims();
  const int noutput = weights_shape.d[output_axis] * num_groups;
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = weights_shape.d[0];
  kernel_size.w() = weights_shape.d[1];

  TRT_ShapedWeights weights_rsck;
  if (inputs.at(1).is_weights()) {
    weights_rsck = inputs.at(1).weights();
  } else {
    StatusOr<TRT_ShapedWeights> tmp = params->weight_store->GetTempWeights(
        nvinfer1::DataType::kFLOAT, weights_shape);
    TRT_ENSURE_OK(tmp);
    weights_rsck = tmp.ConsumeValueOrDie();
  }

  // In explcit precision mode, trace the input back to the constant while also
  // verifying that QDQ scale layers are present.
  if (!inputs.at(1).is_weights()) {
    TRT_ENSURE(params->use_explicit_precision);
    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params->converter->network(), params->weight_store);
    TRT_ENSURE_OK(builder);
    auto dequant_layer =
        builder->FindProducerOf(inputs.at(1).tensor()->trt_tensor());
    TRT_ENSURE_PTR_OK(dequant_layer);

    // TODO(cbate): corresponding TRT layer name check
    if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
      TRT_ENSURE((*dequant_layer)->getType() == nvinfer1::LayerType::kSCALE);
    }

    auto quant_layer = builder->UniqueParentOf(*dequant_layer, 0);
    TRT_ENSURE_PTR_OK(quant_layer);

    // TODO(cbate): corresponding TRT layer name check
    if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
      TRT_ENSURE((*quant_layer)->getType() == nvinfer1::LayerType::kSCALE);
    }

    auto weights_layer = builder->UniqueParentOf(*quant_layer, 0);
    TRT_ENSURE_PTR_OK(weights_layer);
    TRT_ENSURE((*weights_layer)->getType() == nvinfer1::LayerType::kCONSTANT);
    auto const_weights_rsck =
        reinterpret_cast<nvinfer1::IConstantLayer*>(*weights_layer)
            ->getWeights();

    TRT_ENSURE(weights_rsck.count() == weights_rsck.count());
    const auto* weights_ptr =
        static_cast<const float*>(const_weights_rsck.values);
    std::copy_n(weights_ptr, const_weights_rsck.count,
                weights_rsck.GetPointer<float>());
  }

  StatusOr<TRT_ShapedWeights> weights =
      params->weight_store->GetTempWeights(weights_rsck);
  TRT_ENSURE_OK(weights);
  StatusOr<TRT_ShapedWeights> biases = params->weight_store->GetTempWeights(
      nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {noutput}});
  TRT_ENSURE_OK(biases);
  std::fill_n(biases->GetPointer<float>(), noutput, 0.0f);
  ReorderRSCKToKCRS(weights_rsck, &*weights, num_groups);

  // Add convolution.
  nvinfer1::ILayer* conv_layer = nullptr;
  if (is_conv2d_backprop_input) {
    nvinfer1::IDeconvolutionLayer* layer =
        params->converter->network()->addDeconvolution(
            *tensor->trt_tensor(), noutput, kernel_size,
            weights->GetTrtWeights(), biases->GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStride(stride);
    // VALID padding is the default TRT behavior.
    if (padding_type == "SAME") {
      // SAME_UPPER means that post padding is preferred.
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
    layer->setNbGroups(num_groups);
    conv_layer = layer;
  } else {
    const nvinfer1::Weights empty_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                          0};
    nvinfer1::IConvolutionLayer* layer =
        params->converter->network()->addConvolution(
            *tensor->trt_tensor(), noutput, kernel_size,
            params->use_explicit_precision ? empty_weights
                                           : weights->GetTrtWeights(),
            empty_weights);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStride(stride);
    if (padding_type == "SAME") {
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
    layer->setNbGroups(num_groups);
    layer->setDilation(dilation);
    conv_layer = layer;
  }

  // After creating the conv layer, if we are in explicit precision mode and the
  // weights input is a tensor, then we need to override the weights input by
  // calling setInput() on the layer.
  if (params->use_explicit_precision) {
    TRT_ENSURE(inputs.at(1).is_tensor());

    conv_layer->setInput(1, *inputs.at(1).tensor()->trt_tensor());
  }

  params->converter->SetLayerName(conv_layer, node_def, "conv");
  ITensorProxyPtr output_tensor = conv_layer->getOutput(0);
  // Add an extra padding for Deconv because TRT doesn't accept the
  // argument output_shape and thus the TRT output shape could be wrong
  // in case of strides>1.
  if (is_conv2d_backprop_input) {
    std::vector<int64_t> output_spatial_dims =
        GetSpatialDimsFromOutputSizes(backprop_output_size, h_index, w_index);
    const int output_height = output_spatial_dims[0];
    const int output_width = output_spatial_dims[1];
    nvinfer1::Dims trt_output_shape = output_tensor->getDimensions();
    // What determines the padding size is the difference between the given
    // input_sizes (tf_output_shape) and TRT computed size.
    int out_h_idx = params->use_implicit_batch ? 1 : 2;
    int out_w_idx = params->use_implicit_batch ? 2 : 3;
    const int height_diff = output_height - trt_output_shape.d[out_h_idx];
    const int width_diff = output_width - trt_output_shape.d[out_w_idx];
    if ((height_diff < 0) || (width_diff < 0)) {
      return errors::InvalidArgument(
          "input_sizes argument of Conv2DBackprop (i.e. output_shape argument "
          "of conv2d_transpose) ",
          "is too small for the given out_backprop argument of Conv2DBackprop "
          "(i.e. input argument of conv2d_transpose). Expect: ",
          "(", output_height, ", ", output_width, ") >= ", "(",
          trt_output_shape.d[out_h_idx], ", ", trt_output_shape.d[out_w_idx],
          ")");
    }
    // Only add a padding layer if padding sizes are larger than 0
    if ((height_diff > 0) || (width_diff > 0)) {
      nvinfer1::DimsHW pre_padding(0, 0);
      nvinfer1::DimsHW post_padding(height_diff, width_diff);
      nvinfer1::IPaddingLayer* padding_layer =
          params->converter->network()->addPadding(*output_tensor->trt_tensor(),
                                                   pre_padding, post_padding);
      output_tensor = padding_layer->getOutput(0);
      params->converter->SetLayerName(padding_layer, node_def, "pad");
    }
  }
  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor, node_def, "to_NHWC"));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

bool AllowInefficientTranspose() {
  static bool result = [] {
    bool value;
    Status status =
        ReadBoolFromEnvVar("TF_DEBUG_TRT_ALLOW_INEFFICIENT_TRANSPOSE",
                           /*default_value=*/false, &value);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    return value;
  }();

  return result;
}

Status ConvertTranspose(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"x", false}, {"perm", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  // Get the permutation from weights.
  TRT_ShapedWeights weights = inputs.at(1).weights();
  const int* weights_ptr = weights.GetPointer<int>();
  std::vector<int> perm(weights_ptr, weights_ptr + weights.count());

  // Verify the permutation.
  ITensorProxyPtr input_tensor = inputs.at(0).tensor();
  const int perm_size =
      params->use_implicit_batch ? perm.size() - 1 : perm.size();
  if (perm_size != size_t(input_tensor->getDimensions().nbDims)) {
    return errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
  }
  if (params->use_implicit_batch && perm[0] != 0) {
    return errors::Unimplemented(
        "Transpose at batch dimension is not supported.");
  }

  if (!IS_TRT_VERSION_GE(7, 1, 3, 4)) {
    // TensorRT versions before 7.1.3.4 is slow transposing large tensors.
    // So check tensor size, and don't convert if it is too large.
    constexpr int64_t kMaxEfficientTranspose = 2500000;
    int64_t tensor_size = DimsAdapter(input_tensor->getDimensions()).Volume();
    if (!AllowInefficientTranspose() && tensor_size > kMaxEfficientTranspose) {
      return errors::Unimplemented(StrCat("Transpose too large:", tensor_size));
    }
  }

  if (params->validation_only) return Status::OK();

  // Start conversion.
  ITensorProxyPtr output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
      input_tensor, perm, &output_tensor, params->node_def));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertShape(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", TrtInputArg::kBoth}}));
  if (params->use_implicit_batch) {
    return errors::Unimplemented(
        "Shape is only supported for explicit batch mode.");
  }
  DimsAdapter input_dims(inputs.at(0).GetTrtDims());
  if (params->validation_only) return Status::OK();

  StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
      params->converter->network(), params->weight_store);
  TRT_ENSURE_OK(builder);
  if (input_dims.IsStatic()) {
    // Create a const node with the value of the shape.
    StatusOr<nvinfer1::IConstantLayer*> const_layer =
        builder->ConstantShape(input_dims);
    TRT_ENSURE_PTR_OK(const_layer);
    params->outputs->push_back(
        TRT_TensorOrWeights((*const_layer)->getOutput(0)));
    return Status::OK();
  }
  StatusOr<nvinfer1::IShapeLayer*> shape_layer =
      builder->Shape(inputs.at(0).tensor()->trt_tensor());
  TRT_ENSURE_PTR_OK(shape_layer);
  params->converter->SetLayerName(*shape_layer, params->node_def, "shape");
  params->outputs->push_back(TRT_TensorOrWeights((*shape_layer)->getOutput(0)));
  return Status::OK();
}

Status ExpectShapeTensor(const TRT_TensorOrWeights& tensor) {
  if (tensor.tensor()->getType() != nvinfer1::DataType::kINT32) {
    return errors::InvalidArgument("Expected a shape tensor with INT32 type");
  }
  if (tensor.GetTrtDims().nbDims > 1) {
    return errors::InvalidArgument("Expected a 0D or 1D shape tensor");
  }
  return Status::OK();
}

// Converts Reshape op if the input has dynamic (unknown) dims.
Status ConvertDynamicReshape(OpConverterParams* params) {
  if (params->use_implicit_batch) {
    return errors::InvalidArgument(
        "The input \"shape\" for Reshape must be a constant in implicit batch"
        " mode.");
  }
  if (!IS_TRT_VERSION_GE(7, 1, 3, 0)) {
    // While officially TRT supports shape value input , there are problems with
    // shape input handling that cause networks converted with
    // ConvertDynamicReshape fail. Here we conservatively switch off the
    // converter before TRT 7.1.3.
    return errors::InvalidArgument(
        "Non constant shape input tensor for Reshape requires minimum TRT "
        "7.1.3");
  }
  const auto& inputs = params->inputs;
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);

  // If the input is a tensor it must be a shape tensor.
  TF_RETURN_IF_ERROR(ExpectShapeTensor(inputs.at(1)));
  if (inputs.at(1).tensor()->getDimensions().nbDims == 0) {
    // Dynamic reshape requires a 1D shape tensor.
    return errors::Unimplemented(
        "Reshape with dynamic input requires 1D input tensor");
  }
  if (params->validation_only) return Status::OK();
  nvinfer1::IShuffleLayer* layer = params->converter->network()->addShuffle(
      *input_tensor.tensor()->trt_tensor());
  VLOG(2) << "ConvertReshape setInput (1) "
          << DebugString(inputs.at(1).tensor()->getDimensions());
  layer->setInput(1, *inputs.at(1).tensor()->trt_tensor());
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

// Converts Reshape in explicit batch mode if the input has static (known) dims.
Status ConvertStaticReshapeForExplicitBatchMode(
    OpConverterParams* params, DimsAdapter output_dims,
    ITensorProxyPtr* output_tensor) {
  return PrepareTensorForShape(params->converter, params->inputs.at(0),
                               output_dims, params->validation_only,
                               output_tensor, params->node_def);
}

// Converts Reshape in implicit batch mode. The input has static (known) dims.
Status ConvertStaticReshapeForImplicitBatchMode(
    OpConverterParams* params, DimsAdapter output_dims,
    ITensorProxyPtr* output_tensor) {
  const auto& inputs = params->inputs;
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);
  const int input_batch_dim = input_tensor.batch_size();
  const int64_t output_batch_dim = output_dims.dim(0);

  DimsAdapter input_nonbatch_dims(input_tensor.GetTrtDims());
  DimsAdapter output_nonbatch_dims(output_dims);
  TF_RETURN_IF_ERROR(output_nonbatch_dims.RemoveBatchDimension());

  VLOG(1) << "input_batch_dim=" << input_batch_dim
          << ", input_nonbatch_dims=" << input_nonbatch_dims.DebugString()
          << "\nresult_batch_dim=" << output_batch_dim
          << ", result_nonbatch_dims=" << output_nonbatch_dims.DebugString();

  // Check whether input_batch_dim and output_batch_dim will have the same
  // static value.
  bool reshape_may_change_batch_dim = false;
  if (input_batch_dim != -1 && output_batch_dim != -1) {
    reshape_may_change_batch_dim = (input_batch_dim != output_batch_dim);
  } else {
    reshape_may_change_batch_dim =
        !AreDimsStaticWithSameSize(input_nonbatch_dims, output_nonbatch_dims);
  }
  if (reshape_may_change_batch_dim) {
    return errors::Unimplemented("Reshape on batch dimension is not supported");
  }
  // Perform the conversion.
  return PrepareTensorForShape(params->converter, input_tensor,
                               output_nonbatch_dims, params->validation_only,
                               output_tensor, params->node_def);
}

Status ConvertReshape(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"tensor", TrtInputArg::kTensor}, {"shape", TrtInputArg::kBoth}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  if (inputs.at(1).is_tensor()) {
    return ConvertDynamicReshape(params);
  }

  // TODO(bixia): we can't use inputs.at(1).weights().ToVector<int>() for two
  // reasons: (1) When weights.count()==0, TRT_ShapedWeights::tensor_ dtype is
  // not properly set to INT32. (2) I tried a fix for the first problem, I got
  // shared pointer related error in convert_nodes_test. We should fix the
  // problems and switch to use inputs.at(1).weights().ToVector<int>(), a type
  // safe method to access the content of the tensor.
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.count() == 0 && params->use_implicit_batch) {
    return errors::Unimplemented("Reshape to shape=[] is not supported");
  }

  DimsAdapter output_shape_dims(
      absl::MakeSpan(weights.GetPointer<int>(), weights.count()));
  ITensorProxyPtr output_tensor = nullptr;

  if (!params->use_implicit_batch) {
    TF_RETURN_IF_ERROR(ConvertStaticReshapeForExplicitBatchMode(
        params, output_shape_dims, &output_tensor));
  } else {
    TF_RETURN_IF_ERROR(ConvertStaticReshapeForImplicitBatchMode(
        params, output_shape_dims, &output_tensor));
  }
  if (params->validation_only) return Status::OK();

  // Record the conversion result.
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertExpandDims(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"axis", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  // Get input shape as vector.
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);
  const nvinfer1::Dims dims = input_tensor.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  // Get axis to expand on.
  auto axis = inputs.at(1).weights().GetSpan<int>();
  if (axis.size() != 1) {
    return errors::InvalidArgument("ExpandDims axis must be a scalar");
  }
  // Use rank = nbDims + 1 for ConvertAxis's bounds checking to account for
  // ExpandDim's ability to add an axis at end of the shape.
  int trt_axis;
  TF_RETURN_IF_ERROR(ConvertAxis(axis[0], dims.nbDims + 1, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));
  if (params->validation_only) return Status::OK();
  ITensorProxyPtr output_tensor = nullptr;

  if (!params->use_implicit_batch && !HasStaticShape(input_dims)) {
    TF_RETURN_IF_ERROR(params->converter->DynamicExpandDims(
        /*input=*/input_tensor.tensor(),
        /*dims=*/dims,
        /*axis=*/trt_axis,
        /*params=*/params,
        /*output=*/&output_tensor));
  } else {
    // ExpandDims: Insert new dim of size 1.
    input_dims.insert(input_dims.begin() + trt_axis, 1);
    // Reshape tensor.
    DimsAdapter dims(input_dims);
    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, input_tensor, dims,
        /*validation_only=*/false, &output_tensor, params->node_def));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status Converter::DynamicReshape(ITensorProxyPtr input,
                                 std::vector<std::pair<int, int>> slices,
                                 OpConverterParams* params,
                                 ITensorProxyPtr* output,
                                 std::vector<int> size_for_added_dims,
                                 absl::optional<int> op_instance) {
  *output = nullptr;
  // DynamicReshape relies on INetworkDefinition::addShape
  if (params->validation_only) {
    return errors::Internal(
        "DynamicReshape should not be used during validation");
  }
  ITensorProxyPtr shape =
      network()->addShape(*input->trt_tensor())->getOutput(0);
  // Build new shape = shape[:trt_axis] + [1] + shape[trt_axis:]
  std::vector<ITensorProxyPtr> concat_inputs;
  int max_num_slices = std::max(slices.size(), size_for_added_dims.size());
  int op_instance_value = op_instance.has_value() ? op_instance.value() : 0;

  for (int i = 0; i < max_num_slices; i++) {
    ITensorProxyPtr tensor;
    // maybe_add_a_dimension(i);
    if (i < size_for_added_dims.size() && size_for_added_dims[i] >= 0) {
      nvinfer1::Dims dims{1, {1}};
      if (size_for_added_dims[i] > 0) {
        dims.d[0] = size_for_added_dims[i];
      }
      TF_RETURN_IF_ERROR(
          CreateScalarConstant(params, std::min(size_for_added_dims[i], 1),
                               &tensor, nvinfer1::DataType::kINT32, dims));
      concat_inputs.push_back(tensor);
    }
    if (i < slices.size()) {
      nvinfer1::ISliceLayer* slice_layer = network()->addSlice(
          *shape->trt_tensor(), {1, {slices[i].first}},
          {1, {slices[i].second - slices[i].first}}, {1, {1}});
      concat_inputs.push_back(slice_layer->getOutput(0));
      string slice_name = StrCat("slice_", op_instance_value);
      SetLayerName(slice_layer, params->node_def, slice_name,
                   /*op_instance=*/i);
    }
  }
  std::vector<nvinfer1::ITensor*> trt_concat_inputs;
  for (const auto& t : concat_inputs) {
    trt_concat_inputs.push_back(t->trt_tensor());
  }
  nvinfer1::IConcatenationLayer* concat_layer = network()->addConcatenation(
      static_cast<nvinfer1::ITensor* const*>(trt_concat_inputs.data()),
      concat_inputs.size());
  SetLayerName(concat_layer, params->node_def, "concat", op_instance);
  concat_layer->setAxis(0);
  ITensorProxyPtr new_shape = concat_layer->getOutput(0);
  // Reshape input using new shape
  nvinfer1::IShuffleLayer* shuffle =
      network()->addShuffle(*input->trt_tensor());
  SetLayerName(shuffle, params->node_def, "shuffle", op_instance);
  shuffle->setInput(1, *new_shape->trt_tensor());
  *output = shuffle->getOutput(0);
  return Status::OK();
}

Status Converter::DynamicExpandDims(ITensorProxyPtr input,
                                    const nvinfer1::Dims& dims, int axis,
                                    OpConverterParams* params,
                                    ITensorProxyPtr* output,
                                    absl::optional<int> op_instance) {
  if (params->validation_only) {
    *output = nullptr;
    return errors::Internal(
        "DynamicExpandDims should not be used during validation");
  }
  std::vector<std::pair<int, int>> slices;
  std::vector<int> extra_dims;
  if (axis != 0) {
    slices.push_back(std::pair<int, int>{0, axis});
    extra_dims.push_back(-1);
  }
  extra_dims.push_back(1);
  if (axis != dims.nbDims) {
    slices.push_back(std::pair<int, int>{axis, dims.nbDims});
  }
  return DynamicReshape(
      /*input=*/input,
      /*slices=*/slices,
      /*params=*/params,
      /*output=*/output,
      /*size_for_added_dims=*/extra_dims,
      /*op_instance=*/op_instance);
}

Status Converter::SqueezeTensor(ITensorProxyPtr input,
                                std::vector<int>* input_dims,
                                OpConverterParams* params,
                                ITensorProxyPtr* output,
                                absl::optional<int> op_instance) {
  // If the remaining dimensions of a squeeze operation have dynamic sizes, we
  // need to use TRT ops to build the result shape for the squeeze operation.
  // This is because IShuffleLayer::setReshapeDimensions treats -1 as a special
  // value.
  if (!params->use_implicit_batch && !HasStaticShape(*input_dims)) {
    std::vector<std::pair<int, int>> slices;
    for (int i = 0; i < input_dims->size(); i++) {
      if (input_dims->at(i) != 0) {
        slices.push_back(std::pair<int, int>(i, i + 1));
      }
    }
    return DynamicReshape(
        /*input=*/input,
        /*slices=*/slices,
        /*params=*/params,
        /*output=*/output,
        /*size_for_added_dims=*/{},
        /*op_instance=*/op_instance);
  }
  // Remove all dims which are equal to 0.
  input_dims->erase(std::remove(input_dims->begin(), input_dims->end(), 0),
                    input_dims->end());
  // Reshape tensor.
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, TRT_TensorOrWeights(input), DimsAdapter(*input_dims),
      /*validation_only=*/false, output, params->node_def));
  return Status::OK();
}

Status ConvertSqueeze(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  // Get input shape.
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);
  const nvinfer1::Dims dims = input_tensor.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  std::vector<int64_t> squeeze_dims;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(node_def), "squeeze_dims", &squeeze_dims));
  if (squeeze_dims.empty()) {
    if (params->use_implicit_batch || !HasStaticShape(dims)) {
      return errors::Unimplemented(
          "Squeeze is not implemented for empty squeeze_dims");
    } else {
      // explicit batch mode with static input shape we squeeze all singleton
      // dimensions
      for (int& dim : input_dims) {
        if (dim == 1) {
          // Mark it for removal by setting it to 0
          dim = 0;
        }
      }
    }
  } else {
    std::vector<int> trt_axes;
    trt_axes.reserve(squeeze_dims.size());
    for (int tf_axis : squeeze_dims) {
      // If the axis is valid, then convert it to TRT axis, otherwise abort
      // conversion.
      int trt_axis;
      TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims, node_def.name(),
                                     params->use_implicit_batch, &trt_axis));
      // Make sure target dimension is size 1 or unknown size (-1)
      if (input_dims[trt_axis] != -1 && input_dims[trt_axis] != 1) {
        return errors::InvalidArgument(
            "Dimension ", tf_axis, " with size ", input_dims[trt_axis],
            " cannot be squeezed because it must be size 1");
      }
      trt_axes.push_back(trt_axis);
    }
    // Mark axes to remove by setting them to 0.
    for (int axis : trt_axes) {
      input_dims[axis] = 0;
    }
  }
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
      /*input=*/input_tensor.tensor(),
      /*input_dims=*/&input_dims,
      /*params=*/params,
      /*output=*/&output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params, {{"input", false}, {"begin", true}, {"size", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  const TRT_ShapedWeights& begin_weights = inputs.at(1).weights();
  const TRT_ShapedWeights& size_weights = inputs.at(2).weights();

  // Check that "begin" is not negative.
  if (absl::c_any_of(begin_weights.GetSpan<int32>(),
                     [](const int32 val) { return val < 0; })) {
    return errors::InvalidArgument("\"begin\" in Slice is out of range");
  }

  // Check that "size" is not less than -1.
  if (absl::c_any_of(size_weights.GetSpan<int32>(),
                     [](const int32 val) { return val < -1; })) {
    return errors::InvalidArgument("\"size\" in Slice is out of range");
  }

  // Get the input dims and add batch dimension so that indexes line up
  // properly.
  PartialTensorShape input_shape;
  TF_RETURN_IF_ERROR(
      DimsAdapter(inputs.at(0).GetTrtDims())
          .PartialTensorShape(
              &input_shape, params->use_implicit_batch
                                ? absl::optional<int>(inputs.at(0).batch_size())
                                : absl::nullopt));

  if (static_cast<int64>(input_shape.dims()) !=
          begin_weights.GetTensor().NumElements() ||
      static_cast<int64>(input_shape.dims()) !=
          size_weights.GetTensor().NumElements()) {
    return errors::InvalidArgument(
        "Length of begin and size arguments must equal rank of input for "
        "Slice");
  }

  // Check that batch dimension is unmodified.
  if (params->use_implicit_batch) {
    auto begin_v = begin_weights.GetSpan<int32>();
    auto size_v = size_weights.GetSpan<int32>();

    // The batch dimension is modified if begin doesn't start from 0 or slice
    // size on d0 is not equal to input size on d0. Slice size -1 means slices
    // to the end of the dimension.
    if (begin_v[0] != 0 ||
        (size_v[0] != -1 && size_v[0] != input_shape.dim_size(0))) {
      return errors::Unimplemented(
          "TensorRT does not allow modifications to the batch dimension in "
          "implicit batch mode");
    }
  }

  PartialTensorShape processing_shape;
  PartialTensorShape final_shape;
  bool is_identity;
  bool is_simple_slice;
  bool slice_dim0;
  absl::InlinedVector<int64, 4> begin;
  absl::InlinedVector<int64, 4> end;
  absl::InlinedVector<int64, 4> strides;
  StridedSliceShapeSpec strided_slice_spec;
  std::bitset<32> begin_mask(0);
  std::bitset<32> end_mask(0);
  std::bitset<32> ellipsis_mask(0);
  std::bitset<32> new_axis_mask(0);
  std::bitset<32> shrink_axis_mask(0);
  Tensor strides_tensor = tensor::DeepCopy(begin_weights.GetTensor());
  Tensor end_tensor = tensor::DeepCopy(size_weights.GetTensor());
  Tensor size_tensor = tensor::DeepCopy(size_weights.GetTensor());

  // Use the content in begin_weights and size_tensor to setup begin_mask,
  // end_mask, end_tensor, strides_tensor, and end_tensor.
  auto strides_vec = strides_tensor.flat<int32>();
  auto end_vec = end_tensor.flat<int32>();
  auto size_vec = size_tensor.flat<int32>();
  auto begin_vec = begin_weights.GetTensor().flat<int32>();

  for (int i = 0; i < input_shape.dims(); i++) {
    strides_vec(i) = 1;
    begin_mask[i] = false;
    if (size_vec(i) == -1) {
      end_mask[i] = true;
      end_vec(i) = 0;
      size_vec(i) = 0;
    } else {
      end_mask[i] = false;
      end_vec(i) = begin_vec(i) + size_vec(i);
      if (end_vec(i) > input_shape.dim_size(i) && input_shape.dim_size(i) > 0) {
        return errors::InvalidArgument("\"begin\" + \"size\" for dimension ", i,
                                       " in Slice is out of range");
      }
    }
  }

  auto bitset_to_int32 = [](const std::bitset<32>& bs) {
    return static_cast<int32_t>(bs.to_ulong());
  };

  TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
      &begin_weights.GetTensor(), &end_tensor, strides_tensor, input_shape,
      bitset_to_int32(begin_mask), bitset_to_int32(end_mask),
      bitset_to_int32(ellipsis_mask), bitset_to_int32(new_axis_mask),
      bitset_to_int32(shrink_axis_mask), &processing_shape, &final_shape,
      &is_identity, &is_simple_slice, &slice_dim0, &begin, &end, &strides,
      &strided_slice_spec));

  VLOG(2) << "ConvertSlice: "
          << "\n input_shape: " << input_shape
          << "\n procesing_shape: " << processing_shape
          << "\n final_shape: " << final_shape
          << "\n  begin: " << DebugString(begin)
          << "\n  stride: " << DebugString(strides)
          << "\n  end: " << DebugString(end)
          << "\n is identity: " << is_identity
          << "\n is simple_slice: " << is_simple_slice
          << "\n slice dim0: " << slice_dim0 << " StridedSliceShapeSpec:"
          << "\n   begin_dense_mask: "
          << std::bitset<32>(strided_slice_spec.begin_dense_mask)
          << "\n   end_dense_mask: "
          << std::bitset<32>(strided_slice_spec.end_dense_mask)
          << "\n   shrink_dense_mask: "
          << std::bitset<32>(strided_slice_spec.shrink_axis_dense_mask);

  return ConvertStridedSliceHelper(params, inputs.at(0), input_shape, begin,
                                   strides, end, absl::nullopt, absl::nullopt,
                                   strided_slice_spec);
}

Status ConvertStridedSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"input", false}, {"begin", true}, {"end", true}, {"strides", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  int32 begin_mask, end_mask, ellipsis_mask, shrink_axis_mask, new_axis_mask;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ellipsis_mask", &ellipsis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "shrink_axis_mask", &shrink_axis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "new_axis_mask", &new_axis_mask));

  // New_axis_mask is not supported. TODO(tfeher): Support this by expanddims.
  if (new_axis_mask != 0) {
    return errors::Unimplemented(
        "new_axis_mask is not supported for StridedSlice");
  }

  // Shrinking axis on batch dimension is not allowed in implicit batch mode.
  if (params->use_implicit_batch && shrink_axis_mask & 1) {
    return errors::Unimplemented(
        "TensorRT does not allow modifications to the batch dimension");
  }

  // Convert TensorRT dimensions to TensorFlow shape. Implicit batch is added to
  // the TensorFlow shape, to be consistent with the weights and masks and to
  // support the use of TensorFlow slice op validator.
  PartialTensorShape input_shape;
  TF_RETURN_IF_ERROR(
      DimsAdapter(inputs.at(0).GetTrtDims())
          .PartialTensorShape(
              &input_shape, params->use_implicit_batch
                                ? absl::optional<int>(inputs.at(0).batch_size())
                                : absl::nullopt));

  const TRT_ShapedWeights& begin_weights = inputs.at(1).weights();
  const TRT_ShapedWeights& end_weights = inputs.at(2).weights();
  const TRT_ShapedWeights& stride_weights = inputs.at(3).weights();
  if (!AllLengthsEqual({begin_weights.ToVector<int>(),
                        end_weights.ToVector<int>(),
                        stride_weights.ToVector<int>()})) {
    return errors::InvalidArgument(
        "Length of begin, end, and stride must be equal");
  }

  PartialTensorShape processing_shape;
  PartialTensorShape final_shape;
  bool is_identity;
  bool is_simple_slice;
  bool slice_dim0;
  absl::InlinedVector<int64, 4> begin;
  absl::InlinedVector<int64, 4> end;
  absl::InlinedVector<int64, 4> strides;
  StridedSliceShapeSpec strided_slice_spec;

  TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
      &begin_weights.GetTensor(), &end_weights.GetTensor(),
      stride_weights.GetTensor(), input_shape, begin_mask, end_mask,
      ellipsis_mask, new_axis_mask, shrink_axis_mask, &processing_shape,
      &final_shape, &is_identity, &is_simple_slice, &slice_dim0, &begin, &end,
      &strides, &strided_slice_spec));

  if (!params->validation_only) {
    VLOG(2) << "After ValidateStridedSliceOp:"
            << "\n input_shape: " << input_shape
            << "\n procesing_shape: " << processing_shape
            << "\n final_shape: " << final_shape
            << "\n  begin: " << DebugString(begin)
            << "\n  stride: " << DebugString(strides)
            << "\n  end: " << DebugString(end)
            << " is identity: " << is_identity
            << "\n is simple_slice: " << is_simple_slice
            << "\n slice dim0: " << slice_dim0 << " StridedSliceShapeSpec:"
            << "\n   begin_dense_mask: "
            << std::bitset<32>(strided_slice_spec.begin_dense_mask)
            << "\n   end_dense_mask: "
            << std::bitset<32>(strided_slice_spec.end_dense_mask)
            << "\n   shrink_dense_mask: "
            << std::bitset<32>(strided_slice_spec.shrink_axis_dense_mask);
  }

  // If the first dimension of the ellepsis_mask is set, and fewer dimensions
  // are specified than the number of input dimensions, then the batch dimension
  // is not modified Otherwise we must check whether the batch dimension is
  // modified.
  if (params->use_implicit_batch &&
      !((ellipsis_mask & 1) &&
        begin_weights.Shape().NumDims() < input_shape.dims())) {
    // Check that batch dimension is unmodified. We need to use the expanded
    // begin/end/strides array since the original array may be incorrect when
    // (ellipsis_mask&1)==1.
    const bool begin_is_modified = !(begin_mask & 1) && (begin[0] != 0);
    const bool stride_is_modified = (strides[0] != 1);
    // If the batch size is -1 and the end mask is not set, we can only know if
    // the batch dimension is unmodified when the batch size is defined. When
    // the batch size is undefined, we don't convert to be safe.
    const bool batch_size_is_defined = (input_shape.dim_size(0) > 0);
    const bool end_is_modified =
        !(end_mask & 1) &&
        (!batch_size_is_defined || (end[0] != input_shape.dim_size(0)));
    if (begin_is_modified || stride_is_modified || end_is_modified) {
      return errors::Unimplemented(
          "TensorRT does not allow modifications to the batch dimension");
    }
  }

  // shrink_axis_mask requires a reshape after the slice.
  absl::optional<nvinfer1::Dims> final_shape_dims = absl::nullopt;
  if (shrink_axis_mask) {
    final_shape_dims.emplace();
    auto dims_adap =
        DimsAdapter::Create(final_shape, params->use_implicit_batch);
    TRT_ENSURE_OK(dims_adap);
    *final_shape_dims = dims_adap->AsTrtDims();
  }

  return ConvertStridedSliceHelper(params, inputs.at(0), input_shape, begin,
                                   strides, end, final_shape_dims, 0,
                                   strided_slice_spec);
}

Status ConvertConv2D(OpConverterParams* params) {
  return ConvertConv2DHelper(params, 1, /*is_conv2d_backprop_input=*/false);
}

Status ConvertConv2DDepthwise(OpConverterParams* params) {
  return ConvertConv2DHelper(params, 0, /*is_conv2d_backprop_input=*/false);
}

Status ConvertConv2DBackpropInput(OpConverterParams* params) {
  return ConvertConv2DHelper(params, 1, /*is_conv2d_backprop_input=*/true);
}

Status ConvertConv3DHelper(OpConverterParams* params, int group,
                           bool is_conv3d_backprop_input = false) {
  const int kNumDims = 5;
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TRT_TensorOrWeights backprop_output_size;
  ITensorProxyPtr tensor = nullptr;
  if (is_conv3d_backprop_input) {
    // In the case when Conv3dBackpropInput is used for conv3d_transpose, these
    // inputs correspond to: output size, filter, and input.
    TF_RETURN_IF_ERROR(CheckInputsWeights(
        *params,
        {{"input_sizes", true}, {"filter", true}, {"out_backprop", false}}));
    backprop_output_size = inputs.at(0);
    tensor = inputs.at(2).tensor();
  } else {
    TF_RETURN_IF_ERROR(
        CheckInputsWeights(*params, {{"input", false}, {"filter", true}}));
    tensor = inputs.at(0).tensor();
  }
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  const TRT_ShapedWeights weights_drsck = inputs.at(1).weights();
  if (weights_drsck.Shape().NumDims() != kNumDims) {
    return errors::InvalidArgument("Conv3D expects kernel of dimension 5");
  }

  string data_format, padding_type;
  std::vector<int64_t> tf_dilations, tf_stride;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &tf_stride));

  const bool is_ndhwc = (data_format == "NDHWC");  // Or NCDHW 01234 - > 02341
  const int d_index = is_ndhwc ? 1 : 2;
  const int h_index = is_ndhwc ? 2 : 3;
  const int w_index = is_ndhwc ? 3 : 4;
  const int c_index = is_ndhwc ? 4 : 1;
  if (tf_dilations.size() != kNumDims) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 5 dimensions");
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions");
  }

  const nvinfer1::Dims3 dilation_dhw(
      tf_dilations[d_index], tf_dilations[h_index], tf_dilations[w_index]);
  if (is_conv3d_backprop_input &&
      (dilation_dhw.d[0] != 1 || dilation_dhw.d[1] != 1 ||
       dilation_dhw.d[2] != 1)) {
    return errors::Unimplemented(
        "Dilation with Conv3DBackpropInputV2 (conv3d_transpose) is not "
        "supported");
  }

  if (tf_stride.size() != kNumDims) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 5 dimensions");
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions");
  }

  const nvinfer1::Dims3 stride_dhw(tf_stride[d_index], tf_stride[h_index],
                                   tf_stride[w_index]);
  const auto tensor_dim = tensor->getDimensions();

  // Asymmetric padding on Deconv not supported for now
  if (is_conv3d_backprop_input && padding_type == "SAME") {
    StatusOr<TRT_ShapedWeights> weights =
        params->weight_store->GetTempWeights(weights_drsck);
    TRT_ENSURE_OK(weights);
    nvinfer1::Dims3 effective_kernel_size(
        weights->Shape().dim(0) +
            (weights->Shape().dim(0) - 1) * (dilation_dhw.d[0] - 1),  // D
        weights->Shape().dim(1) +
            (weights->Shape().dim(1) - 1) * (dilation_dhw.d[1] - 1),  // R
        weights->Shape().dim(2) +
            (weights->Shape().dim(2) - 1) * (dilation_dhw.d[2] - 1)  // S
    );

    const auto output_size_weights =
        backprop_output_size.weights().GetPointer<int>();
    const std::vector<int64_t> input_dims = {output_size_weights[d_index],
                                             output_size_weights[h_index],
                                             output_size_weights[w_index]};

    const std::vector<std::pair<int, int>> padding =
        CreateSamePadding(stride_dhw, effective_kernel_size, input_dims);

    if (padding[0].first != padding[0].second ||
        padding[1].first != padding[1].second ||
        padding[2].first != padding[2].second) {
      return errors::Unimplemented(
          "Asymmetric padding with Conv3DBackpropInputV2 (conv3d_transpose) is "
          "not supported");
    }
  }

  // Channel dim must be static for Conv3D since we use that value for
  // num_groups at build time.
  // TODO: Allow conversion if kImplicitBatchModeCompatible||kOptimal is used.
  int implicit_batch_offset = params->use_implicit_batch ? -1 : 0;
  if (tensor->getDimensions().d[c_index + implicit_batch_offset] == -1) {
    return errors::InvalidArgument("Channel dimension must be static");
  }

  // Finished validation checks
  if (params->validation_only) return Status::OK();

  // Transpose to NCDHW (NCDHW is required for IConvLayer).
  const bool need_transpose = is_ndhwc;
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 4, 1, 2, 3}, &tensor, node_def, "to_NCDHW"));
  }

  // group == 0 signifies that this is a depthwise convolution, so set
  // num_groups to size of input's channel dim. For a non-depthwise conv,
  // num_groups will be 1.
  const int num_groups = (group == 0) ? tensor_dim.d[0] : group;

  // For conv, TF weights are DRSCK, and TRT expects KCDRS.
  // For backprop, TF weights are DRSKC, and TRT expects KCDRS.
  // Therefore, this reorder will work for both cases.
  StatusOr<TRT_ShapedWeights> weights =
      params->weight_store->GetTempWeights(weights_drsck);
  TRT_ENSURE_OK(weights);
  ReorderDRSCKToKCDRS(weights_drsck, &*weights, num_groups);
  TRT_ShapedWeights biases(weights->TrtDType());
  const int output_axis = is_conv3d_backprop_input ? 1 : 0;
  const int noutput = weights->Shape().dim(output_axis) * num_groups;
  nvinfer1::Dims3 kernel_size_drs(weights->Shape().dim(2),  // D
                                  weights->Shape().dim(3),  // R
                                  weights->Shape().dim(4)   // S
  );

  // Add convolution.
  nvinfer1::ILayer* conv_layer = nullptr;
  if (is_conv3d_backprop_input) {
    nvinfer1::IDeconvolutionLayer* layer =
        params->converter->network()->addDeconvolutionNd(
            *tensor->trt_tensor(), noutput, kernel_size_drs,
            weights->GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);  // change to nd set stride

    if (padding_type == "SAME") {
      VLOG(2) << "Using SAME padding";
      // SAME_UPPER means that post padding is preferred.
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    layer->setNbGroups(num_groups);
    conv_layer = layer;
  } else {
    nvinfer1::IConvolutionLayer* layer =
        params->converter->network()->addConvolutionNd(
            *tensor->trt_tensor(), noutput, kernel_size_drs,
            weights->GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);

    if (padding_type == "SAME") {
      VLOG(2) << "Using SAME padding";
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    layer->setNbGroups(num_groups);
    layer->setDilationNd(dilation_dhw);
    conv_layer = layer;
  }
  params->converter->SetLayerName(conv_layer, node_def, "conv");
  ITensorProxyPtr output_tensor = conv_layer->getOutput(0);

  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 4, 1}, &output_tensor, node_def, "to_NDHWC"));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertConv3D(OpConverterParams* params) {
  return ConvertConv3DHelper(params, 1, /*is_conv3d_backprop_input=*/false);
}

Status ConvertConv3DBackpropInputV2(OpConverterParams* params) {
  return ConvertConv3DHelper(params, 1, /*is_conv3d_backprop_input=*/true);
}

Status ConvertPool3D(OpConverterParams* params) {
  const int kNumDims = 5;
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  nvinfer1::PoolingType type;
  if (node_def.op() == "MaxPool3D") {
    type = nvinfer1::PoolingType::kMAX;
  } else if (node_def.op() == "AvgPool3D") {
    type = nvinfer1::PoolingType::kAVERAGE;
  } else {
    return errors::Unimplemented("Unsupported pooling type: ", node_def.op());
  }

  string data_format, padding_type;
  std::vector<int64_t> tf_stride, tf_kernel;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &tf_stride));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &tf_kernel));

  if ((padding_type != "SAME") && (padding_type != "VALID")) {
    return errors::Unimplemented("Unsupported padding type: ", padding_type);
  }

  const bool is_ndhwc = (data_format == "NDHWC");
  const int c_index = is_ndhwc ? 4 : 1;
  const int d_index = is_ndhwc ? 1 : 2;
  const int h_index = is_ndhwc ? 2 : 3;
  const int w_index = is_ndhwc ? 3 : 4;

  if (tf_stride.size() != kNumDims) {
    return errors::InvalidArgument(
        "Pooling strides field must specify 5 dimensions");
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "stride must be 1 for batch and channel dimensions");
  }

  if (tf_kernel.size() != kNumDims) {
    return errors::InvalidArgument(
        "Pooling ksize field must specify 5 dimensions");
  }
  if (tf_kernel[0] != 1 || tf_kernel[c_index] != 1) {
    return errors::Unimplemented(
        "ksize must be 1 for batch and channel dimensions");
  }
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  if (data_format == "NDHWC") {
    // NDHWC => NCDHW
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 4, 1, 2, 3}, &tensor, node_def, "to_NCDHW"));
  }

  const nvinfer1::Dims3 stride(tf_stride[d_index], tf_stride[h_index],
                               tf_stride[w_index]);
  const nvinfer1::Dims3 ksize(tf_kernel[d_index], tf_kernel[h_index],
                              tf_kernel[w_index]);

  nvinfer1::IPoolingLayer* layer = params->converter->network()->addPoolingNd(
      *tensor->trt_tensor(), type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  layer->setStrideNd(stride);
  // VALID padding is the default TRT behavior.
  if (padding_type == "SAME") {
    // SAME_UPPER means that post padding is preferred.
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  params->converter->SetLayerName(layer, node_def, "pooling");

  ITensorProxyPtr output_tensor = layer->getOutput(0);
  if (data_format == "NDHWC") {
    // NCDHW => NDHWC
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 4, 1}, &output_tensor, node_def, "to_NDHWC"));
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertFusedConv2DBiasActivation(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false},
                                                  {"filter", true},
                                                  {"bias", true},
                                                  {"side_input", true},
                                                  {"conv_input_scale", true},
                                                  {"side_input_scale", true}}));
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.Shape().NumDims() != 4) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation expects kernel of dimension 4");
  }

  string data_format, filter_format, activation_mode, padding_type;
  std::vector<int64_t> tf_dilations, tf_stride;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "filter_format", &filter_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "activation_mode", &activation_mode));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &tf_stride));

  if (data_format != "NHWC" && data_format != "NCHW") {
    return errors::InvalidArgument("Unsupported data_format:", data_format);
  }
  int c_index = (data_format == "NHWC") ? 3 : 1;
  int h_index = (data_format == "NHWC") ? 1 : 2;
  int w_index = (data_format == "NHWC") ? 2 : 3;

  if (tf_dilations.size() != 4) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 4 dimensions");
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions");
  }
  const nvinfer1::DimsHW dilation(tf_dilations[h_index], tf_dilations[w_index]);

  if (tf_stride.size() != 4) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 4 dimensions");
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions");
  }
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  auto op_pair = ActivationTypeMap()->find(activation_mode);
  if (op_pair == ActivationTypeMap()->end() && activation_mode != "None") {
    return errors::Unimplemented("Activation mode not supported: ",
                                 activation_mode);
  }

  if (filter_format != "HWIO" && filter_format != "OIHW") {
    return errors::InvalidArgument("Unsupported filter_format:", filter_format);
  }
  // Check that there's no side_input or conv_input_scale.
  TRT_ShapedWeights side_input = inputs.at(3).weights();
  if (side_input.count() != 0) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation doesn't yet support side_input");
  }
  TRT_ShapedWeights conv_input_scale = inputs.at(4).weights();
  if (conv_input_scale.count() != 1 ||
      conv_input_scale.TrtDType() != nvinfer1::DataType::kFLOAT ||
      conv_input_scale.GetSpan<float>()[0] != 1.0) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation doesn't yet support conv_input_scale");
  }
  if (params->validation_only) return Status::OK();

  // Transpose to NCHW (NCHW is required for IConvLayer).
  const bool need_transpose = (data_format == "NHWC");
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 3, 1, 2}, &tensor, node_def, "to_NCHW"));
  }

  nvinfer1::DimsHW kernel_size;
  if (filter_format == "OIHW") {
    kernel_size.h() = weights.Shape().dim(2);
    kernel_size.w() = weights.Shape().dim(3);
  } else {
    // HWIO.
    DCHECK_EQ(filter_format, "HWIO");
    kernel_size.h() = weights.Shape().dim(0);
    kernel_size.w() = weights.Shape().dim(1);
  }

  // Add convolution.
  TRT_ShapedWeights biases = inputs.at(2).weights();
  nvinfer1::IConvolutionLayer* conv_layer = nullptr;
  if (filter_format == "OIHW") {
    // Weights are already in the right order.
    conv_layer = params->converter->network()->addConvolution(
        *tensor->trt_tensor(), weights.Shape().dim(0), kernel_size,
        weights.GetTrtWeights(), biases.GetTrtWeights());
  } else {
    // For conv, TF weights are RSCK, and TRT expects KCRS.
    TRT_ENSURE(filter_format == "HWIO");
    StatusOr<TRT_ShapedWeights> weights_kcrs =
        params->weight_store->GetTempWeights(weights);
    TRT_ENSURE_OK(weights_kcrs);
    ReorderRSCKToKCRS(weights, &*weights_kcrs, 1);
    conv_layer = params->converter->network()->addConvolution(
        *tensor->trt_tensor(), weights.Shape().dim(3), kernel_size,
        weights_kcrs->GetTrtWeights(), biases.GetTrtWeights());
  }
  TFTRT_RETURN_ERROR_IF_NULLPTR(conv_layer, node_def.name());
  conv_layer->setStride(stride);
  if (padding_type == "SAME") {
    conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  params->converter->SetLayerName(conv_layer, node_def, "conv");
  conv_layer->setNbGroups(1);
  conv_layer->setDilation(dilation);
  ITensorProxyPtr output_tensor = conv_layer->getOutput(0);

  // Add activation if there is one.
  if (op_pair != ActivationTypeMap()->end()) {
    nvinfer1::IActivationLayer* activation_layer =
        params->converter->network()->addActivation(
            *output_tensor->trt_tensor(), op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(activation_layer, node_def.name());
    params->converter->SetLayerName(activation_layer, node_def, "activation");
    output_tensor = activation_layer->getOutput(0);
  }
  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor, node_def, "to_NHWC"));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertPool(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT8};
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));
  nvinfer1::PoolingType type;
  if (node_def.op() == "MaxPool") {
    type = nvinfer1::PoolingType::kMAX;
  } else if (node_def.op() == "AvgPool") {
    type = nvinfer1::PoolingType::kAVERAGE;
  } else {
    return errors::Unimplemented("Unsupported pooling type: ", node_def.op());
  }

  string data_format, padding_type;
  std::vector<int64_t> tf_stride, tf_kernel;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &tf_stride));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &tf_kernel));

  if ((padding_type != "SAME") && (padding_type != "VALID")) {
    return errors::Unimplemented("Unsupported padding type: ", padding_type);
  }
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  int h_index = 2;
  int w_index = 3;
  if (data_format == "NHWC") {
    h_index = 1;
    w_index = 2;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 3, 1, 2}, &tensor, node_def, "to_NCHW"));
  }

  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  const nvinfer1::DimsHW ksize(tf_kernel[h_index], tf_kernel[w_index]);

  nvinfer1::IPoolingLayer* layer = params->converter->network()->addPooling(
      *tensor->trt_tensor(), type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  layer->setStride(stride);
  // VALID padding is the default TRT behavior.
  if (padding_type == "SAME") {
    // SAME_UPPER means that post padding is preferred.
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  params->converter->SetLayerName(layer, node_def, "pooling");
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  if (data_format == "NHWC") {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor, node_def, "to_NHWC"));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertLeakyRelu(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  float alpha{0.f};
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "alpha", &alpha));

  // Use IActivationLayer when available.
  if (params->validation_only) return Status::OK();

  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(),
          nvinfer1::ActivationType::kLEAKY_RELU);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "activation");
  layer->setAlpha(alpha);
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertClipByValue(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // TODO(tmorris): We can also allow the case where min and max are tensors by
  // using elementwise min and max layers.
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"t", false}, {"clip_value_min", true}, {"clip_value_max", true}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "T", &dtype));

  float clip_value_min = 0.0f;
  float clip_value_max = 0.0f;
  // TODO(tmorris): Add a templated helper function to get scalar weights of
  // InType casted to OutType.
  if (dtype == DataType::DT_FLOAT) {
    clip_value_min = inputs.at(1).weights().GetSpan<float>()[0];
    clip_value_max = inputs.at(2).weights().GetSpan<float>()[0];
  } else if (dtype == DataType::DT_HALF) {
    clip_value_min =
        static_cast<float>(inputs.at(1).weights().GetSpan<Eigen::half>()[0]);
    clip_value_max =
        static_cast<float>(inputs.at(2).weights().GetSpan<Eigen::half>()[0]);
  }

  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(),
          nvinfer1::ActivationType::kCLIP);
  layer->setAlpha(clip_value_min);
  layer->setBeta(clip_value_max);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "activation");
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

const operationMap<nvinfer1::ActivationType>* ActivationTypeMap() {
  static auto* const m =
      new std::unordered_map<string, nvinfer1::ActivationType>({
          {"Relu", nvinfer1::ActivationType::kRELU},
          {"Sigmoid", nvinfer1::ActivationType::kSIGMOID},
          {"Tanh", nvinfer1::ActivationType::kTANH},
          {"Elu", nvinfer1::ActivationType::kELU},
          {"Selu", nvinfer1::ActivationType::kSELU},
          {"Softsign", nvinfer1::ActivationType::kSOFTSIGN},
          {"Softplus", nvinfer1::ActivationType::kSOFTPLUS},
      });
  return m;
}

Status ConvertActivation(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  auto op_pair = ActivationTypeMap()->find(node_def.op());
  if (op_pair == ActivationTypeMap()->end()) {
    return errors::Unimplemented("Activation op: ", node_def.op(),
                                 " not supported");
  }
  if (params->validation_only) return Status::OK();

  // Start conversion.
  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "activation");
  // Set parameters.
  if (node_def.op() == "Elu") {
    layer->setAlpha(1.0f);
  } else if (node_def.op() == "Selu") {
    // From tensorflow/core/kernels/relu_op_functor.h
    layer->setAlpha(1.7580993408473768599402175208123f);
    layer->setBeta(1.0507009873554804934193349852946f);
  } else if (node_def.op() == "Softplus") {
    layer->setAlpha(1.0f);
    layer->setBeta(1.0f);
  }
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertRelu6(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(),
          nvinfer1::ActivationType::kCLIP);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  layer->setAlpha(0.0f);
  layer->setBeta(6.0f);
  ITensorProxyPtr output_tensor = layer->getOutput(0);
  params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, 6.0f);
  params->converter->SetLayerName(layer, node_def, "activation");
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertBiasAdd(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TFTRT_CHECK_INPUT_SIZE(inputs.size(), 2, node_def);

  if (inputs[0].is_weights() && inputs[1].is_weights()) {
    return errors::InvalidArgument(
        "All inputs are weights, but Grappler is expected to fold them.");
  }

  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  string data_format;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(node_def), "data_format", &data_format));

  nvinfer1::Dims input_shape = inputs.at(0).GetTrtDims();
  nvinfer1::Dims bias_shape = inputs.at(1).GetTrtDims();
  // The bias input arg is a 1-D tensor with length C. If the input is NCHW,
  // then we need to unsqueeze the bias such that its shape is [1, C, 1, 1].
  if (data_format == "NCHW") {
    if (params->use_implicit_batch) {
      // The batch dim is not included in implicit batch mode, so the shape of
      // the bias tensor is [C, 1, 1].
      bias_shape.nbDims = input_shape.nbDims;
      std::fill(bias_shape.d + 1, bias_shape.d + bias_shape.nbDims, 1);
    } else {
      // In explicit batch mode we create a tensor with shape [1, C, 1, 1].
      std::vector<int> bias_shape_vec(bias_shape.d,
                                      bias_shape.d + bias_shape.nbDims);
      // Insert 1 before for batch dim
      bias_shape_vec.insert(bias_shape_vec.begin(), 1);
      // Trail with 1s to match input_shape size
      bias_shape_vec.insert(bias_shape_vec.end(),
                            input_shape.nbDims - bias_shape_vec.size(), 1);
      DimsAdapter(bias_shape_vec).TrtDims(&bias_shape);
    }
  } else {
    // Next, broadcast the bias across the input.
    TF_RETURN_IF_ERROR(GetTrtBroadcastShape(inputs.at(0), inputs.at(1),
                                            /*check_feasibility=*/true,
                                            params->use_implicit_batch,
                                            &input_shape, &bias_shape));
  }

  // Convert input to a TRT tensor
  ITensorProxyPtr input_tensor{nullptr};
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, inputs.at(0), DimsAdapter(input_shape),
      params->validation_only, &input_tensor, node_def,
      /*op_instance=*/0));

  // Finally, reshape bias. Since the bias is usually a constant, this will
  // normally happen at conversion-time.
  ITensorProxyPtr bias_tensor{nullptr};
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, inputs.at(1), DimsAdapter(bias_shape),
      params->validation_only, &bias_tensor, node_def,
      /*op_instance=*/1));
  VLOG(2) << "Bias shape adjusted to " << DebugString(bias_shape);

  if (params->validation_only) return Status::OK();

  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *input_tensor->trt_tensor(), *bias_tensor->trt_tensor(),
          nvinfer1::ElementWiseOperation::kSUM);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "sum");
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

template <typename Input>
inline bool IsIntegerInInt32Bounds(const Input& inp) {
  static_assert(std::is_integral<Input>::value,
                "This function is only implemented for integral types.");
  // If Input is always within the range of int32, return true.
  if (sizeof(Input) < sizeof(int32) || std::is_same<Input, int32>::value) {
    return true;
  }
  // Otherwise, we need to check the value of the input. If the input is
  // unsigned, we only check the upper bound.
  if (!std::numeric_limits<Input>::is_signed) {
    return inp <= static_cast<Input>(std::numeric_limits<int32>::max());
  }
  // We can safely cast lowest() here since we now know that Input is signed and
  // sizeof(Input) >= sizeof(int32)
  return (inp >= static_cast<Input>(std::numeric_limits<int32>::lowest()) &&
          inp <= static_cast<Input>(std::numeric_limits<int32>::max()));
}

template <DataType dtype>
Status CopyToTrtInt32Array(const Tensor& tensor, int32* dst) {
  typedef typename EnumToDataType<dtype>::Type CType;
  const CType* src = tensor.flat<CType>().data();
  for (int i = 0; i < tensor.NumElements(); ++i) {
    // This becomes a no-op if CType is within bounds of int32
    if (!IsIntegerInInt32Bounds(src[i])) {
      return errors::InvalidArgument("Value at index ", i,
                                     " is outside the range of int32");
    }
    dst[i] = static_cast<int32>(src[i]);
  }
  return Status::OK();
}

Status TfTensorToTrtWeights(const Tensor& tensor, TrtWeightStore* weight_store,
                            TRT_ShapedWeights* weights) {
  const DataType dtype = tensor.dtype();

  // We always convert the integer constants to INT32.
  //
  // TODO(aaroey): FP16 will remain in half format and is not converted to
  // FP32, but the converter currently uses all float weights as FP32. Fix
  // this.
  DataType converted_dtype = DataTypeIsInteger(dtype) ? DT_INT32 : dtype;

  // Verify that the dtype is supported by TensorRT. Otherwise, return an error.
  nvinfer1::DataType trt_dtype;
  TF_RETURN_IF_ERROR(TfTypeToTrtType(converted_dtype, &trt_dtype));

  if (tensor.NumElements() == 0) {
    // Return empty weights.
    *weights = TRT_ShapedWeights(trt_dtype);
    return Status::OK();
  }

  StatusOr<DimsAdapter> weight_dims = DimsAdapter::Create(tensor.shape());
  TRT_ENSURE_OK(weight_dims);

  auto tmp = weight_store->GetTempWeights(trt_dtype, weight_dims->AsTrtDims());
  TRT_ENSURE_OK(tmp);
  *weights = tmp.ConsumeValueOrDie();

  // Copy the tensor directly if the tensor does not require cast to the
  // supported type.
  if (converted_dtype == dtype) {
    std::copy_n(tensor.tensor_data().data(), tensor.TotalBytes(),
                weights->GetPointer<int8>());
    return Status::OK();
  }

  Status status = Status::OK();
  // Copy tensor elements after casting them to the converted DataType.
  int32* dst = weights->GetPointer<int32>();
  switch (dtype) {
    case DT_INT8:
      status = CopyToTrtInt32Array<DT_INT8>(tensor, dst);
      break;
    case DT_UINT8:
      status = CopyToTrtInt32Array<DT_UINT8>(tensor, dst);
      break;
    case DT_INT16:
      status = CopyToTrtInt32Array<DT_INT16>(tensor, dst);
      break;
    case DT_UINT16:
      status = CopyToTrtInt32Array<DT_UINT16>(tensor, dst);
      break;
    case DT_UINT32:
      status = CopyToTrtInt32Array<DT_UINT32>(tensor, dst);
      break;
    case DT_INT64:
      status = CopyToTrtInt32Array<DT_INT64>(tensor, dst);
      break;
    case DT_UINT64:
      status = CopyToTrtInt32Array<DT_UINT64>(tensor, dst);
      break;
    default:
      return errors::Internal("Unexpected DataType: ", DataTypeString(dtype));
  }
  return status;
}

// Convert a Const NodeDef to TRT_ShapedWeights. This is a special converter, it
// always ignores the params->validation_only parameter but adds the converted
// weights to params->outputs. We did this since TrtNodeValidator needs the
// weights as input to other nodes, and use it to determine whether those nodes
// are supported by TRT.
Status ConvertConst(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (!inputs.empty()) {
    return errors::InvalidArgument(
        "Constant node is expected to have empty input list");
  }

  // Create shaped weights as output
  const auto& tensor_proto = node_def.attr().at("value").tensor();
  Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return errors::Internal("Cannot parse weight tensor proto: ",
                            node_def.name());
  }

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "dtype", &dtype));

  if (dtype != tensor.dtype()) {
    return errors::InvalidArgument("DataType mismatch between attr (",
                                   DataTypeString(dtype), ") and tensor (",
                                   DataTypeString(tensor.dtype()), ")");
  }

  TRT_ShapedWeights weights;
  TF_RETURN_IF_ERROR(
      TfTensorToTrtWeights(tensor, params->weight_store, &weights));

  if (params->outputs != nullptr) {
    params->outputs->push_back(TRT_TensorOrWeights(weights));
  }
  return Status::OK();
}

Status ConvertIdentity(OpConverterParams* params) {
  // TODO(tmorris): TRT's Identity layer does not get optimized away as of TRT
  // 5.0, however once we know that it does it would be nice to use that
  // instead.
  if (params->validation_only) return Status::OK();

  for (int i = 0; i < params->inputs.size(); i++) {
    params->outputs->push_back(params->inputs.at(i));
  }
  return Status::OK();
}

Status ConvertSquare(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  if (params->validation_only) return Status::OK();

  // Constant 2 with same rank as input
  ITensorProxyPtr const2_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, 2.0f, inputs.at(0).GetTrtDims(), &const2_tensor));

  // ElementWise Pow Operation
  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *inputs.at(0).tensor()->trt_tensor(), *const2_tensor->trt_tensor(),
          nvinfer1::ElementWiseOperation::kPOW);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertReduce(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"axis", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  auto tf_axes_list = inputs.at(1).weights().GetSpan<int>();

  DataType idx_dtype{DataType::DT_INT32};
  bool keep_dims{false};
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Tidx", &idx_dtype));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "keep_dims", &keep_dims));

  // Only expect to handle INT32 as attributes for now
  if (idx_dtype != DataType::DT_INT32) {
    return errors::Unimplemented("Tidx supports only DT_INT32");
  }

  int axes = 0;
  if (tf_axes_list.size() == 0) {
    return errors::InvalidArgument(
        "TRT cannot support reduce on all (batch) dimensions");
  }
  for (int i = 0; i < tf_axes_list.size(); i++) {
    int trt_axis;
    TF_RETURN_IF_ERROR(
        ConvertAxis(tf_axes_list[i], tensor->getDimensions().nbDims,
                    node_def.name(), params->use_implicit_batch, &trt_axis));
    axes |= (1 << trt_axis);
  }

  nvinfer1::ReduceOperation reduce_operation;
  if (node_def.op() == "Sum") {
    reduce_operation = nvinfer1::ReduceOperation::kSUM;
  } else if (node_def.op() == "Prod") {
    reduce_operation = nvinfer1::ReduceOperation::kPROD;
  } else if (node_def.op() == "Max") {
    reduce_operation = nvinfer1::ReduceOperation::kMAX;
  } else if (node_def.op() == "Min") {
    reduce_operation = nvinfer1::ReduceOperation::kMIN;
  } else if (node_def.op() == "Mean") {
    reduce_operation = nvinfer1::ReduceOperation::kAVG;
  } else {
    return errors::Unimplemented("Op not supported ", node_def.op());
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::ILayer* layer = params->converter->network()->addReduce(
      *tensor->trt_tensor(), reduce_operation, axes, keep_dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

// TensorRT does not support the Pack op natively. Therefore, Pack op is
// converted by first expanding input tensors by adding a new dimension of size
// one at the specified axis and then concatenating the tensors at the same
// axis.
Status ConvertPack(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  int num_inputs{0};
  int64_t tf_axis{0};
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &num_inputs));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "axis", &tf_axis));

  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "Number of inputs for Pack is inconsistent with N attribute");
  }

  // In implicit batch mode we do not allow weight input. An input tensor with
  // dims NCHW is represented with dims CHW during conversion time, and N is
  // defined only during runtime. A weight is represented with dims NCHW. We
  // cannot be sure that the runtime N will agree with the conversion time N,
  // therefore we do not convert the pack op if it has both tensor and weight
  // inputs. This restriction does not apply in explicit batch mode, in that
  // case the input tensors are also represented with full dims that include the
  // batch size.
  TrtInputArg expected_arg =
      params->use_implicit_batch ? TrtInputArg::kTensor : TrtInputArg::kBoth;

  std::vector<std::pair<string, TrtInputArg>> inputs_is_weight;
  inputs_is_weight.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs_is_weight.push_back({StrCat("values_", i), expected_arg});
  }
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, inputs_is_weight));

  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT32};
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));
  if (num_inputs > 1) {
    // Verify that inputs are compatible for concatenation after the expansion.
    TF_RETURN_IF_ERROR(
        VerifyShapesMatch(inputs, /*masked_dim=*/-1, node_def.name()));
  }

  // Find the dimension of the inputs. In general inputs can have dynamic shape,
  // in that case we have to use DynamicExpandDims to calculate the expanded
  // dimensions. To avoid that, we try to find a weight input which is
  // guaranteed to have known static shape.
  int idx = 0;
  for (int i = 1; i < inputs.size(); i++) {
    if (HasStaticShape(inputs.at(i).GetTrtDims())) {
      idx = i;
    }
  }
  DimsAdapter dims(inputs.at(idx).GetTrtDims());
  // Convert axis from the TensorFlow format to TensorRT format.
  int trt_axis{0};
  TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.NumDims() + 1, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));

  // Compute expanded dimensions and then reshape input tensors.
  std::vector<int64_t> tensor_dims(dims.begin(), dims.end());
  tensor_dims.insert(tensor_dims.begin() + trt_axis, 1);
  std::vector<ITensorProxyPtr> expanded_tensors;

  int input_index = 0;
  for (const TRT_TensorOrWeights& input : inputs) {
    ITensorProxyPtr expanded_tensor = nullptr;
    if (input.is_tensor() && !params->use_implicit_batch &&
        !HasStaticShape(dims)) {
      if (!params->validation_only) {
        TF_RETURN_IF_ERROR(params->converter->DynamicExpandDims(
            /*input=*/input.tensor(),
            /*dims=*/dims.AsTrtDims(),
            /*axis=*/trt_axis,
            /*params=*/params,
            /*output=*/&expanded_tensor,
            /*op_instance=*/input_index));
      }
    } else {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          /*converter=*/params->converter,
          /*input=*/input,
          /*dims=*/DimsAdapter(tensor_dims),
          /*validation_only=*/params->validation_only,
          /*tensor=*/&expanded_tensor,
          /*node_def=*/node_def,
          /*op_instance=*/input_index));
    }
    if (!params->validation_only) {
      expanded_tensors.push_back(expanded_tensor);
    }
    input_index++;
  }
  if (params->validation_only) return Status::OK();

  // If there is only one tensor in the input, return the expanded tensor.
  if (num_inputs == 1) {
    params->outputs->push_back(TRT_TensorOrWeights(expanded_tensors[0]));
    return Status::OK();
  }

  // Otherwise, concatenate expanded tensors.
  std::vector<nvinfer1::ITensor*> trt_expanded_tensors;
  for (const auto& t : expanded_tensors) {
    trt_expanded_tensors.push_back(t->trt_tensor());
  }
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          static_cast<nvinfer1::ITensor* const*>(trt_expanded_tensors.data()),
          expanded_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "concat");
  // Note that trt_axis stays the same even after expanding tensors at the axis.
  layer->setAxis(trt_axis);
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertPad(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"tensor", false}, {"paddings", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT8}));

  // Implement tensor binaryOp weight [channel wise] for now;
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  const auto dims = tensor->getDimensions();
  // Restore implicit batch dimension
  const int nb_dims =
      params->use_implicit_batch ? dims.nbDims + 1 : dims.nbDims;

  // TODO(tfeher): Support nb_dims < 4 by inserting extra dimensions to the
  // original input.
  if (nb_dims < 4) {
    return errors::InvalidArgument("Convertpad requires at least 4D input");
  }
  TRT_ShapedWeights pads = inputs.at(1).weights();

  // TODO(jie): handle data type conversion for TRT?
  DataType padding_dtype{DataType::DT_INT32};
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(node_def), "Tpaddings", &padding_dtype));

  if (pads.Shape().dim(0) != nb_dims || pads.Shape().dim(1) != 2) {
    return errors::InvalidArgument("Paddings must be a weight with shape ",
                                   "[n, 2], where n is the rank of input ",
                                   "tensor");
  }

  // Only expect to handle INT32 as attributes for now
  if (padding_dtype != DataType::DT_INT32) {
    return errors::Unimplemented("Tpaddings supports only DT_INT32");
  }
  auto pad_data = pads.GetPointer<int>();

  std::vector<int32_t> tf_pad_index;
  for (int i = 0; i < nb_dims; i++) {
    if (pad_data[2 * i] != 0 || pad_data[2 * i + 1] != 0) {
      tf_pad_index.push_back(i);
    }
  }

  // No padding at all, we should exit
  if (tf_pad_index.empty()) {
    params->outputs->push_back(inputs.at(0));
    return Status::OK();
  }

  // TRT pad layer can only support padding on up to 2 dimensions (TRT-2579).
  // TODO(tfeher): Use multiple TRT pad layers to support padding on more than 2
  // dimensions.
  if (tf_pad_index.size() > 2) {
    return errors::InvalidArgument(
        "Padding layer does not support padding on > 2");
  }

  // Padding on batch dimension is not supported
  if (params->use_implicit_batch && tf_pad_index[0] == 0) {
    return errors::InvalidArgument(
        "Padding layer does not support padding on batch dimension");
  }

  if (params->validation_only) return Status::OK();

  // TRT can only do the padding at the last two dimensions. We transpose the
  // input tensor if needed.
  bool transposed_pad = false;
  std::vector<int> transpose_idx(nb_dims);
  std::iota(transpose_idx.begin(), transpose_idx.end(), 0);

  // trt_pad_index denotes the actual idx where the padding is performed by TRT.
  std::vector<int> trt_pad_index{nb_dims - 2, nb_dims - 1};

  // How many zeros are padded at the last two dimensions.
  nvinfer1::DimsHW pre_padding(0, 0);
  nvinfer1::DimsHW post_padding(0, 0);

  // Dimension to set in the pre_padding and post_padding array.
  std::vector<int> trt_pre_post_padding_index{0, 1};

  // Two special cases where we can avoid permutations.
  if (tf_pad_index.size() == 1 && tf_pad_index[0] == nb_dims - 1) {
    // Only one dimension needs to be padded. We store its index at
    // trt_pad_index[0]. We ignore trt_pad_index[1].
    trt_pad_index[0] = nb_dims - 1;
    trt_pre_post_padding_index[0] = 1;
  }
  if (tf_pad_index.size() == 2 && tf_pad_index[1] == nb_dims - 2) {
    // tf_pad_index only has two values that are in ascending order. If
    // tf_pad_index[1] is nb_dims-2, then swapping the two values in
    // trt_pad_index here makes it possible to only swap one pair of dimensions
    // (swap tf_pad_index[0] with nb_dims-1) in the input tensor. Otherwise, we
    // would have to swap two pairs of dimensions in the input tensor:
    // (tf_pad_index[0] with nb_dims-2) and (tf_pad_index[1], with nb_dims-1).
    // Here is an example for a 4D input tensor:
    // tf_pad_index = [1, 2]
    // trt_pad_index = [3, 2]
    // transpose_idx = [0, 3, 2, 1]
    std::swap(trt_pad_index[0], trt_pad_index[1]);
    std::swap(trt_pre_post_padding_index[0], trt_pre_post_padding_index[1]);
  }

  for (int i = 0; i < tf_pad_index.size(); i++) {
    const int tf_index = tf_pad_index[i];
    const int trt_index = trt_pad_index[i];
    const int k = trt_pre_post_padding_index[i];
    pre_padding.d[k] = pad_data[tf_index * 2];
    post_padding.d[k] = pad_data[tf_index * 2 + 1];
    if (tf_index != trt_index) {
      transposed_pad = true;
      std::swap(transpose_idx[tf_index], transpose_idx[trt_index]);
    }
  }

  if (transposed_pad) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, transpose_idx, &tensor, node_def, "to_pad"));
  }

  nvinfer1::IPaddingLayer* layer = params->converter->network()->addPadding(
      *tensor->trt_tensor(), pre_padding, post_padding);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  if (transposed_pad) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, transpose_idx, &output_tensor, node_def, "from_pad"));
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertSplitHelper(OpConverterParams* params,
                          const TRT_TensorOrWeights& input, int tf_axis,
                          int num_splits, bool squeeze_after) {
  const auto& node_def = params->node_def;
  const nvinfer1::Dims dims = input.GetTrtDims();
  // Convert axis.
  int trt_axis;
  TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));

  if (dims.d[trt_axis] < 0) {
    return errors::InvalidArgument("Dimension ", tf_axis,
                                   " must have statically defined dimensions");
  }

  // Dimension must equal num_splits for Unstack (when squeeze_after is true)
  if (squeeze_after && dims.d[trt_axis] != num_splits) {
    return errors::InvalidArgument(
        "Dimension ", tf_axis, " has size ", dims.d[trt_axis],
        " which is not equal to num of ", num_splits);
  }
  // Dimension must be evenly divisible by num_splits.
  if (dims.d[trt_axis] % num_splits != 0) {
    return errors::InvalidArgument("Dimension ", tf_axis, " of size ",
                                   dims.d[trt_axis],
                                   " is not evenly divisible by ", num_splits);
  }

  // Create parameters for StridedSliceHelper.
  // Slice will begin on zero for all dims, except the one being split which
  // will change.
  std::vector<int> begin(dims.nbDims, 0);
  std::vector<int64> input_dims(dims.d, dims.d + dims.nbDims);

  // Determine size of split. Slice will get the full length of all dims, except
  // the one being split. Undefined dims (-1) will translate to a size of -1
  // which will tell StridedSlice to take full length of that dim.
  std::vector<int> size(dims.d, dims.d + dims.nbDims);
  const int split_size_on_axis = dims.d[trt_axis] / num_splits;
  size[trt_axis] = split_size_on_axis;
  // Stride will always be 1
  std::vector<int> stride(dims.nbDims, 1);
  // Add dummy batch dimension
  if (params->use_implicit_batch) {
    begin.insert(begin.begin(), 0);
    size.insert(size.begin(), 1);
    stride.insert(stride.begin(), 1);
    input_dims.insert(input_dims.begin(), std::max(-1, input.batch_size()));
  }
  PartialTensorShape input_shape(input_dims);

  // Create final shape for Unpack/Unstack, where split axis is squeezed.
  absl::optional<nvinfer1::Dims> final_shape_for_unpack = absl::nullopt;

  // We can't use final_shape_for_unpack_ptr when input dimensions are not
  // fully defined.
  const bool is_dynamic_shape = !HasStaticShape(dims);
  if (squeeze_after && !is_dynamic_shape) {
    std::vector<int> size_after_squeeze(size);
    const int tf_axis = trt_axis + (params->use_implicit_batch ? 1 : 0);
    size_after_squeeze.erase(size_after_squeeze.begin() + tf_axis);
    DimsAdapter adap(size_after_squeeze);
    if (params->use_implicit_batch)
      TF_RETURN_IF_ERROR(adap.RemoveBatchDimension());
    final_shape_for_unpack = adap.AsTrtDims();
  }

  // Slice the input. ConvertStridedSliceHelper will push the outputs onto
  // params->outputs.
  for (int i = 0; i < num_splits; ++i) {
    const int tf_axis = trt_axis + (params->use_implicit_batch ? 1 : 0);
    begin[tf_axis] = i * split_size_on_axis;

    // Stride is 1 for all dims.
    absl::InlinedVector<int64, 4> stride_v(begin.size(), 1);
    absl::InlinedVector<int64, 4> begin_v;
    absl::InlinedVector<int64, 4> end_v;
    for (int i = 0; i < begin.size(); i++) {
      end_v.push_back(begin[i] + size[i]);
      begin_v.push_back(begin[i]);
    }

    TF_RETURN_IF_ERROR(ConvertStridedSliceHelper(
        params, input, input_shape, begin_v, stride_v, end_v,
        final_shape_for_unpack,
        /*op_instance=*/i, /*strided_slice_spec=*/absl::nullopt));
  }
  if (params->validation_only) return Status::OK();

  // Squeeze for dynamic shapes
  if (squeeze_after && is_dynamic_shape) {
    for (int i = 0; i < params->outputs->size(); i++) {
      ITensorProxyPtr output_tensor = nullptr;
      std::vector<int> in_dims(dims.d, dims.d + dims.nbDims);
      input_dims[trt_axis] = 0;
      TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
          /*input=*/params->outputs->at(i).tensor(),
          /*input_dims=*/&in_dims,
          /*params=*/params,
          /*output=*/&output_tensor,
          /*op_instance=*/i));
      (*params->outputs)[i] = TRT_TensorOrWeights(output_tensor);
    }
  }
  return Status::OK();
}

Status ConvertSplit(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"axis", true}, {"value", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, {
                                                 DataType::DT_FLOAT,
                                                 DataType::DT_HALF,
                                                 DataType::DT_INT32,
                                             }));
  int tf_axis = inputs.at(0).weights().GetSpan<int>()[0];

  int num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "num_split", &num_split));

  return ConvertSplitHelper(params, inputs.at(1), tf_axis, num_split, false);
}

Status ConvertUnpack(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"value", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, {
                                                 DataType::DT_FLOAT,
                                                 DataType::DT_HALF,
                                                 DataType::DT_INT32,
                                             }));
  // Input must be rank 1 or higher, since we can't unpack on axis 0.
  if (inputs.at(0).GetTrtDims().nbDims == 0) {
    return errors::Unimplemented(
        "Input \"value\" for Unpack must be rank 2 or greater");
  }

  int tf_axis = 0, num = 0;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "axis", &tf_axis));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "num", &num));

  return ConvertSplitHelper(params, inputs.at(0), tf_axis, num, true);
}

// Supports cast fp16=>fp32 through IIdentityLayer.
Status ConvertCast(OpConverterParams* params) {
  const NodeDef& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  auto unsupport_cast_error = [&](string msg) {
    return errors::Unimplemented("Cast op is not supported - ", msg);
  };

  DataType input_type;
  TF_RETURN_IF_ERROR(GetInputTfType(*params, &input_type, 0));
  if (input_type != DataType::DT_HALF) {
    return unsupport_cast_error(
        StrCat("input dtype != ", DataTypeString(DataType::DT_HALF),
               ", received: ", DataTypeString(input_type)));
  }

  DataType output_type;
  TF_RETURN_IF_ERROR(GetNodeDefTfType(params->node_def, &output_type,
                                      kCastOutputTypeAttrName));

  if (output_type != DataType::DT_FLOAT) {
    return unsupport_cast_error(
        StrCat("output dtype != ", DataTypeString(DataType::DT_FLOAT),
               ", received: ", DataTypeString(output_type)));
  }

  if (params->validation_only) return Status::OK();

  ITensorProxyPtr input = params->inputs.at(0).tensor();
  nvinfer1::IIdentityLayer* layer =
      params->converter->network()->addIdentity(*input->trt_tensor());
  params->converter->SetLayerName(layer, node_def);
  layer->setPrecision(nvinfer1::DataType::kFLOAT);

  if (layer->getOutput(0)->getType() != nvinfer1::DataType::kFLOAT) {
    return errors::Internal("IIdentityLayer doesn't work as expected");
  }

  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertConcat(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  int num_inputs{0};
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "N", &num_inputs));

  if (num_inputs != static_cast<int>(inputs.size()) - 1) {
    return errors::InvalidArgument(
        "Number of inputs for ConcatV2 is inconsistent with N attributes.");
  }
  // Validate inputs.
  std::vector<std::pair<string, TrtInputArg>> inputs_kinds;
  TrtInputArg expected_input =
      params->use_implicit_batch ? TrtInputArg::kTensor : TrtInputArg::kBoth;

  inputs_kinds.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs_kinds.push_back({StrCat("values_", i), expected_input});
  }
  inputs_kinds.push_back({"axis", TrtInputArg::kWeight});
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, inputs_kinds));

  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT32};

  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));
  const auto axis = inputs.at(num_inputs).weights().GetSpan<int>();
  if (axis.size() != 1) {
    return errors::InvalidArgument("Axis for ConcatV2 must be a scalar");
  }
  int trt_axis = 0;
  const auto dim = inputs.at(0).GetTrtDims();
  TF_RETURN_IF_ERROR(ConvertAxis(axis[0], dim.nbDims, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));
  // Check that dimensions match on non-concatenate axis.
  TF_RETURN_IF_ERROR(VerifyShapesMatch(
      absl::Span<const TRT_TensorOrWeights>(inputs).first(num_inputs), trt_axis,
      node_def.name()));
  if (params->validation_only) return Status::OK();

  // Gather inputs as tensors
  std::vector<ITensorProxyPtr> input_tensors;
  input_tensors.reserve(num_inputs);

  for (int i = 0; i < num_inputs; i++) {
    if (inputs.at(i).is_tensor()) {
      input_tensors.push_back(inputs.at(i).tensor());
    } else {
      input_tensors.push_back(params->converter->CreateConstantLayer(
          inputs.at(i).weights(), inputs.at(i).GetTrtDims()));
    }
  }
  std::vector<nvinfer1::ITensor*> trt_input_tensors;
  for (const auto& t : input_tensors) {
    trt_input_tensors.push_back(t->trt_tensor());
  }
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          static_cast<nvinfer1::ITensor* const*>(trt_input_tensors.data()),
          input_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  layer->setAxis(trt_axis);
  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

Status ConvertFusedBatchNorm(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false},
                                                  {"scale", true},
                                                  {"offset", true},
                                                  {"mean", true},
                                                  {"variance", true}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  float epsilon{0.1f};
  string data_format;
  bool is_training{false};
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "epsilon", &epsilon));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "is_training", &is_training));

  if (is_training) {
    // Trying to use batchnorm in training mode is a very common problem.
    // Because the error message will only be printed in VLOG(1) by the
    // segmenter, we issue a special warning so that users will actually see it.
    LOG_WARNING_WITH_PREFIX
        << node_def.op() << " only supports is_training=false. If you "
        << "are using Keras, please call "
        << "keras.backend.set_learning_phase(0) before constructing "
        << "your model. At " << node_def.name();
    return errors::Unimplemented(node_def.op(),
                                 " only supports is_training=false");
  }
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  if (!params->use_implicit_batch) {
    // This check is to make sure that channel dimension is known during
    // conversion.
    //
    // We check this only in explicit batch mode and reject an op with unknown
    // channel dimension during segmentation. In implicit batch mode we have
    // known shapes during conversion even though the shapes may not be known
    // during segmentation (see the actual argument for input_shapes when
    // ConvertGraphDefToEngine is called from TRTEngineOp::BuildEngine).
    int channel_dim = (data_format == "NCHW" ? 1 : 3);
    if (tensor->getDimensions().d[channel_dim] == -1) {
      return errors::InvalidArgument("Channel dimension must be static");
    }
  }
  //  Check parameter types
  auto parameter_type = inputs.at(1).weights().TrtDType();
  if ((parameter_type != nvinfer1::DataType::kFLOAT) &&
      (parameter_type != nvinfer1::DataType::kHALF)) {
    return errors::Unimplemented(
        "Only float32 or float16 weight data type is supported,", " got ",
        DebugString(parameter_type));
  }
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).weights().TrtDType() != parameter_type) {
      return errors::Unimplemented(
          "Inconsistent parameter type for batchnorm is not supported");
    }
  }

  TRT_ShapedWeights dummy_power_weights(parameter_type);
  size_t nweight = 0;
  for (int i = 1; i < 5; i++) {
    nweight = std::max<size_t>(nweight, inputs.at(i).weights().count());
  }
  const TRT_ShapedWeights* ptr_shape_weights = nullptr;
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).weights().count() == nweight) {
      ptr_shape_weights = &(inputs.at(i).weights());
    } else if (inputs.at(i).weights().count() != 1) {
      return errors::InvalidArgument("Inconsistent batchnorm parameter count");
    }
  }
  if (params->validation_only) return Status::OK();

  //  We could technically have two weights with different shape.
  //  that requires two addScale op, arguably less performant
  StatusOr<TRT_ShapedWeights> combined_scale_weights =
      params->weight_store->GetTempWeights(*ptr_shape_weights);
  TRT_ENSURE_OK(combined_scale_weights);
  StatusOr<TRT_ShapedWeights> combined_offset_weights =
      params->weight_store->GetTempWeights(*ptr_shape_weights);
  TRT_ENSURE_OK(combined_offset_weights);

  const Eigen::half* cast_vals_array[4];
  const float* vals_array[4];
  for (int j = 0; j < 4; j++) {
    cast_vals_array[j] = inputs.at(j + 1).weights().GetPointer<Eigen::half>();
    vals_array[j] = inputs.at(j + 1).weights().GetPointer<float>();
  }
  Eigen::half* cast_combined_scale_vals =
      combined_scale_weights->GetPointer<Eigen::half>();
  Eigen::half* cast_combined_offset_vals =
      combined_offset_weights->GetPointer<Eigen::half>();
  float* combined_scale_vals = combined_scale_weights->GetPointer<float>();
  float* combined_offset_vals = combined_offset_weights->GetPointer<float>();

  for (size_t i = 0; i < nweight; ++i) {
    float batchnorm_data[4];
    for (int j = 0; j < 4; j++) {
      if (inputs.at(j + 1).weights().count() != 1) {
        if (parameter_type == nvinfer1::DataType::kFLOAT) {
          batchnorm_data[j] = vals_array[j][i];
        } else if (parameter_type == nvinfer1::DataType::kHALF) {
          batchnorm_data[j] = static_cast<float>(cast_vals_array[j][i]);
        }
      } else {
        if (parameter_type == nvinfer1::DataType::kFLOAT) {
          batchnorm_data[j] = vals_array[j][0];
        } else if (parameter_type == nvinfer1::DataType::kHALF) {
          batchnorm_data[j] = static_cast<float>(cast_vals_array[j][0]);
        }
      }
    }
    float scale = batchnorm_data[0];
    float offset = batchnorm_data[1];
    float mean = batchnorm_data[2];
    float variance = batchnorm_data[3];
    float combined_scale_val = scale / sqrtf(variance + epsilon);
    float combined_offset_val = offset - mean * combined_scale_val;
    if (parameter_type == nvinfer1::DataType::kFLOAT) {
      combined_scale_vals[i] = combined_scale_val;
      combined_offset_vals[i] = combined_offset_val;
    } else if (parameter_type == nvinfer1::DataType::kHALF) {
      cast_combined_scale_vals[i] = Eigen::half(combined_scale_val);
      cast_combined_offset_vals[i] = Eigen::half(combined_offset_val);
    }
  }

  ITensorProxyPtr output_tensor;

  if (data_format == "NCHW") {
    // IScaleLayer CHANNEL mode requires NCHW format.
    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
    nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
        *tensor->trt_tensor(), mode, combined_offset_weights->GetTrtWeights(),
        combined_scale_weights->GetTrtWeights(),
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0});
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    params->converter->SetLayerName(layer, node_def);
    output_tensor = layer->getOutput(0);
  }
  if (data_format == "NHWC") {
    // nweight is the number of channels. TensorRT IElementWiseLayer supports
    // implicit broadcasting for dimensions of size 1.
    nvinfer1::Dims dims = tensor->getDimensions();
    for (int i = 0; i < dims.nbDims - 1; i++) {
      dims.d[i] = 1;
    }
    dims.d[dims.nbDims - 1] = nweight;
    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params->converter->network(), params->weight_store);
    TRT_ENSURE_OK(builder);
    auto scale_constant_layer = builder->WeightsToConstant(
        combined_scale_weights->GetTrtWeights(), dims);
    ITensorProxyPtr scale_constant = (*scale_constant_layer)->getOutput(0);
    auto scale_layer =
        builder->Mul(tensor->trt_tensor(), scale_constant->trt_tensor());
    auto offset_constant_layer = builder->WeightsToConstant(
        combined_offset_weights->GetTrtWeights(), dims);
    ITensorProxyPtr offset_constant = (*offset_constant_layer)->getOutput(0);
    auto offset_layer = builder->Add((*scale_layer)->getOutput(0),
                                     offset_constant->trt_tensor());
    output_tensor = (*offset_layer)->getOutput(0);
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertGather(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // TODO(tmorris): Use CheckInputsWeights by changing bool to enum with an
  // option for an input to be either tensor or weight.
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"params", TrtInputArg::kBoth},
                                   {"indices", TrtInputArg::kBoth},
                                   {"axis", TrtInputArg::kWeight}}));

  const auto& params_input = inputs.at(0);
  const auto& indices_input = inputs.at(1);
  const auto& axis_input = inputs.at(2);

  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32},
      /*dtype_attr_name=*/"Tparams"));
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, {DataType::DT_INT32},
                                    /*dtype_attr_name=*/"Tindices"));

  absl::Span<const int> axis = axis_input.weights().GetSpan<int>();
  if (axis.size() != 1) {
    return errors::InvalidArgument("Axis for GatherV2 must be a scalar");
  }
  int trt_axis = 0;
  TF_RETURN_IF_ERROR(ConvertAxis(
      axis[0], params_input.GetTrtDims().nbDims, node_def.name(),
      params->use_implicit_batch && params_input.is_tensor(), &trt_axis));
  if (params->use_implicit_batch && params_input.is_weights() &&
      trt_axis != 0) {
    return errors::Unimplemented(
        "The input axis must be zero when params is a weight.");
  }
  if (params->use_implicit_batch &&
      (params_input.is_tensor() == indices_input.is_tensor()) &&
      (indices_input.batch_size() != 1 || params_input.batch_size() != 1)) {
    return errors::Unimplemented(
        "Params and indices must have a batch size of 1 when params and indices"
        " are both tensors or both constants.");
  }

  auto get_rank = [params](const auto& input) {
    return input.GetTrtDims().nbDims +
           (params->use_implicit_batch && input.is_tensor() ? 1 : 0);
  };
  // Both input are tensors, and the TF gather result will have rank:
  // (params.nbDims + 1) + (indices.nbDims + 1) - 1,
  // where "+ 1" adds the batch dim. If params is a weight, the TRT rank matches
  // the TF rank so we don't have to add + 1.
  const int params_tf_rank = get_rank(params_input);
  const int indices_tf_rank = get_rank(indices_input);
  const int tf_gather_output_rank = params_tf_rank + indices_tf_rank - 1;
  if (tf_gather_output_rank >
      nvinfer1::Dims::MAX_DIMS + (params->use_implicit_batch ? 1 : 0)) {
    return errors::InvalidArgument(
        "Result of gather has dimension greater than ",
        nvinfer1::Dims::MAX_DIMS + 1);
  }
  if (params->validation_only) return Status::OK();

  // Convert input or indices to tensor if it is a constant.
  auto populate_tensor = [params](const auto& input) -> ITensorProxyPtr {
    ITensorProxyPtr result_tensor = nullptr;

    if (input.is_weights()) {
      result_tensor = params->converter->CreateConstantLayer(
          input.weights(), input.GetTrtDims());
    } else {
      result_tensor = input.tensor();
    }

    return result_tensor;
  };

  ITensorProxyPtr params_tensor = populate_tensor(params_input);
  ITensorProxyPtr indices_tensor = populate_tensor(indices_input);

  // Note on how IGatherLayer works: if both the data and indices tensors have
  // a batch size dimension of size N, it performs:
  // for batchid in xrange(N):
  //   output[batchid, a0, ..., an, i, ..., j, b0, ..., bn] = (
  //       data[batchid, a0, ..., an, indices[batchid, i, ..., j] b0, ..., bn])
  nvinfer1::IGatherLayer* layer = params->converter->network()->addGather(
      *params_tensor->trt_tensor(), *indices_tensor->trt_tensor(), trt_axis);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  ITensorProxyPtr output_tensor = layer->getOutput(0);
  nvinfer1::Dims trt_gather_output_dims = output_tensor->getDimensions();

  if (params->use_implicit_batch) {
    // Note for the "- 2": one is for the output batch dim encapsulated by
    // TF-TRT, and the other is for the output dimension that is squeezed by
    // IGatherLayer because of the implicit batch dim in the indices (see the
    // above note).
    const int expected_trt_output_rank = tf_gather_output_rank -
                                         (params_input.is_tensor() ? 1 : 0) -
                                         (indices_input.is_tensor() ? 1 : 0);

    if (trt_gather_output_dims.nbDims != expected_trt_output_rank) {
      return errors::Internal(
          "Get unexpected output dimensions of IGatherLayer. Expect nbDims: ",
          expected_trt_output_rank,
          ", actual nbDims: ", trt_gather_output_dims.nbDims);
    }
  }
  // Reshape the output so after adding the implicit batch dim it'll match the
  // output shape of TF GatherV2.
  if (params->use_implicit_batch && params_input.is_tensor() &&
      indices_input.is_tensor()) {
    for (int i = trt_gather_output_dims.nbDims; i > trt_axis; --i) {
      trt_gather_output_dims.d[i] = trt_gather_output_dims.d[i - 1];
    }
    trt_gather_output_dims.d[trt_axis] = 1;
    ++trt_gather_output_dims.nbDims;

    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, TRT_TensorOrWeights(output_tensor),
        trt_gather_output_dims,
        /*validation_only=*/false, &output_tensor, node_def));
  }

  // When input and indices are both constants, for the supported cases, reshape
  // output so that after removing the implicit batch dim it will match the
  // output shape of TF GatherV2 op.
  if (params->use_implicit_batch && params_input.is_weights() &&
      indices_input.is_weights()) {
    for (int i = trt_axis; i < trt_gather_output_dims.nbDims - 1; ++i) {
      trt_gather_output_dims.d[i] = trt_gather_output_dims.d[i + 1];
    }

    // Squeeze the implicit batch dimension out. Note: this works only
    // when batch size for both inputs and indices are 1.
    --trt_gather_output_dims.nbDims;

    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, TRT_TensorOrWeights(output_tensor),
        trt_gather_output_dims,
        /*validation_only=*/false, &output_tensor, node_def));
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

// Converts the input matrix multiplication node to a fully connected (FC) layer
// if possible, as the FC layer has more tactics and INT implementations.
// Returns the output ITensor* if the node is converted or nullptr if conversion
// is not possible. An error status indicates internal problems during
// conversion.
StatusOr<ITensorProxyPtr> ConvertFullyConnectedImpl(OpConverterParams* params,
                                                    TRT_TensorOrWeights input_a,
                                                    TRT_TensorOrWeights input_b,
                                                    bool transpose_a,
                                                    bool transpose_b) {
  if (!(!transpose_a && input_a.is_tensor() && input_b.is_weights())) {
    VLOG(2) << "Not FC compatible, A must be non transposed tensor, and B "
               "must be constant.";
    return ITensorProxyPtr(nullptr);
  }

  if (!params->use_implicit_batch && input_b.GetTrtDims().nbDims > 2 &&
      input_b.GetTrtDims().d[0] != 1) {
    // Implicit broadcasting, if needed, has already been considered to
    // transform the inputs and ensure the two operands have the same rank here.
    // If the inputs have rank >= 3, then d[0] is the explicit batch dimension.
    // The weight (input_b) must have batch size 1 in implicit batch mode.
    VLOG(2) << "Not FC compatible, if B has an explicit batch dimension, then "
               "it must be 1.";
    return ITensorProxyPtr(nullptr);
  }

  nvinfer1::Dims input_dim = input_a.GetTrtDims();
  if (input_dim.d[input_dim.nbDims - 1] == -1) {
    VLOG(2) << "Not FC compatible, last dim of A must be static.";
    return ITensorProxyPtr(nullptr);
  }

  if (input_dim.nbDims + 2 > nvinfer1::Dims::MAX_DIMS) {
    VLOG(2) << "Not FC compatible, cannot expand A's shape.";
    return ITensorProxyPtr(nullptr);
  }

  // Add two trailing 1's because FC layer combines the last three dims.
  ITensorProxyPtr tensor_a = nullptr;

  // Initialize the elements of reshap_dim to 0. A value 0 in
  // reshape_dim(i) will preserve the i-th dimension value from the shape of
  // input_a. Add two trailing dimensions of size 1.
  auto reshape_dim = DimsAdapter(input_dim.nbDims,
                                 DimsAdapter::StorageType(input_dim.nbDims, 0))
                         .Append(1)
                         .Append(1);

  const NodeDef& node_def = params->node_def;
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, input_a, reshape_dim,
      /*validation_only=*/false, &tensor_a, node_def, /*op_instance=*/0,
      /*origin_node_name=*/"FULLY_CONNECTED"));

  VLOG(2) << "New shape of A " << DebugString(tensor_a->getDimensions());

  TRT_ShapedWeights weights_b = input_b.weights();
  TRT_ShapedWeights weights_2D(weights_b);
  if (weights_b.Shape().NumDims() > 2) {
    // Combine first nbDims-1 dims into a single dim, e.g. for a 4D tensor we
    // transform [N, H, W, C] -> [N*H*W, C]. This is only valid if all batch
    // dimensions are 1.
    if (std::any_of(weights_b.Shape().begin(),
                    weights_b.Shape().begin() + weights_b.Shape().NumDims() - 2,
                    [](int d) { return d != 1; })) {
      VLOG(2) << "Not FC compatible, B has a batch dim larger than 1";
      return ITensorProxyPtr(nullptr);
    }
    int k = weights_b.Shape().dim(weights_b.Shape().NumDims() - 1);
    nvinfer1::Dims dims{2, {static_cast<int>(weights_b.count() / k), k}};
    TF_RETURN_IF_ERROR(weights_2D.SetShape(dims));
  }

  // FC layer will transpose weights, so we need to pre-transpose.
  TRT_ShapedWeights weights(weights_2D.TrtDType());
  if (!transpose_b) {
    auto tmp = params->weight_store->GetTempWeights(weights_2D);
    TRT_ENSURE_OK(tmp);
    weights = tmp.ConsumeValueOrDie();
    ReorderCKtoKC(weights_2D, &weights);
  } else {
    weights = weights_2D;
  }
  TRT_ShapedWeights biases(weights.TrtDType());
  int k = weights.Shape().dim(weights.Shape().NumDims() - 1);
  const int noutput = weights.count() / k;
  VLOG(2) << "Using fully connected layer with k=" << k
          << ", n_output=" << noutput
          << " weights shape: " << weights.Shape().DebugString()
          << " to convert " << node_def.op();
  nvinfer1::IFullyConnectedLayer* layer =
      params->converter->network()->addFullyConnected(
          *tensor_a->trt_tensor(), noutput, weights.GetTrtWeights(),
          biases.GetTrtWeights());

  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  // A fully connected layer produces output with two trailing singleton
  // dimensions. We remove these.
  auto output_dim = output_tensor->getDimensions();
  output_dim.nbDims -= 2;
  // A zero in output_dim indicates copying the corresponding input dimension
  // value during reshape.
  std::fill(output_dim.d, output_dim.d + output_dim.nbDims, 0);
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, TRT_TensorOrWeights(output_tensor), output_dim,
      /*validation_only=*/false, &output_tensor, node_def,
      /*op_instance=*/1, /*origin_node_name=*/"FULLY_CONNECTED"));
  return output_tensor;
}

StatusOr<ITensorProxyPtr> ConvertMatMulImpl(OpConverterParams* params,
                                            TRT_TensorOrWeights input_a,
                                            TRT_TensorOrWeights input_b,
                                            bool transpose_a,
                                            bool transpose_b) {
  if (params->use_implicit_batch) {
    // In implicit batch mode we are very limited when can we multiply 2D
    // matrices. If input_A is a 2D tensor, then nbDims==1 (implicit batch dim
    // not counted). If A is not transposed and B is weight, then we can convert
    // this treating A as a batch of vectors. This is the only possibility
    // to implement MatMul with 2D input in implicit batch mode.
    if ((input_a.GetTrtDims().nbDims < 2 &&
         (transpose_a || !input_b.is_weights())) ||
        (input_b.GetTrtDims().nbDims < 2)) {
      return errors::InvalidArgument(
          "MatMul with 2D tensors requires explicit batch mode, or that tensor"
          " A is not transposed and B is a constant tensor.");
    }
  }

  if (params->validation_only) return ITensorProxyPtr(nullptr);

  StatusOr<ITensorProxyPtr> result = ConvertFullyConnectedImpl(
      params, input_a, input_b, transpose_a, transpose_b);
  TF_RETURN_IF_ERROR(result.status());
  ITensorProxyPtr output = result.ValueOrDie();
  if (*output) {
    // FC conversion was successful, we can return.
    return output;
  }
  const auto convert_to_itensor =
      [&params](TRT_TensorOrWeights operand) -> ITensorProxyPtr {
    if (operand.is_tensor()) {
      return operand.tensor();
    } else {
      return params->converter->CreateConstantLayer(operand.weights(),
                                                    operand.GetTrtDims());
    }
  };

  ITensorProxyPtr tensor_a = convert_to_itensor(input_a);
  ITensorProxyPtr tensor_b = convert_to_itensor(input_b);

  const auto get_matrix_op = [](ITensorProxyPtr in,
                                bool transpose) -> nvinfer1::MatrixOperation {
    return (transpose) ? nvinfer1::MatrixOperation::kTRANSPOSE
                       : nvinfer1::MatrixOperation::kNONE;
  };
  nvinfer1::MatrixOperation op_a, op_b;
  // Note: In implicit batch mode kTRANSPOSE and kNONE are only valid if the
  // matrix has at least 2 non-batch dimension. In implicit batch mode, if a has
  // 1 dim (excluding batch dim), then we can only use kVECTOR, which will treat
  // matrix A as a batch of vectors.
  op_a = (tensor_a->getDimensions().nbDims < 2)
             ? nvinfer1::MatrixOperation::kVECTOR
             : get_matrix_op(tensor_a, transpose_a);
  // In implicit batch mode, if B has only 1 dims (excluding batch dim) then we
  // already reject the case and don't convert. One could consider using the
  // kVECTOR flag to express C = MatMul(A, B.T) if A is weight, but the result
  // will not have the correct shape: in TRT's implicit batch implementation,
  // the result is a batch of vectors D_ji = A_ik * B_jk, where j is the batch
  // dimension. In contrast, the TF MatMul op produces C = D.T, and we cannot
  // transpose over the batch dimension (implicit batch mode).
  op_b = get_matrix_op(tensor_b, transpose_b);

  nvinfer1::IMatrixMultiplyLayer* layer =
      params->converter->network()->addMatrixMultiply(
          *tensor_a->trt_tensor(), op_a, *tensor_b->trt_tensor(), op_b);

  const auto& node_def = params->node_def;
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  return ITensorProxyPtr(layer->getOutput(0));
}

Status ConvertMatMulHelper(OpConverterParams* params,
                           TRT_TensorOrWeights input_a,
                           TRT_TensorOrWeights input_b, bool transpose_a,
                           bool transpose_b) {
  StatusOr<ITensorProxyPtr> result =
      ConvertMatMulImpl(params, input_a, input_b, transpose_a, transpose_b);
  TF_RETURN_IF_ERROR(result.status());
  if (!params->validation_only) {
    params->outputs->push_back(TRT_TensorOrWeights(result.ValueOrDie()));
  }
  return Status::OK();
}

// inputs are both two dimensional (ops::MatMul)
Status ConvertMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TFTRT_CHECK_INPUT_SIZE(inputs.size(), 2, node_def);

  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  bool transpose_a = false, transpose_b = false;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "transpose_b", &transpose_b));

  return ConvertMatMulHelper(params, inputs.at(0), inputs.at(1), transpose_a,
                             transpose_b);
}

Status ConvertBatchMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TFTRT_CHECK_INPUT_SIZE(inputs.size(), 2, node_def);

  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params, {{"x", TrtInputArg::kBoth}, {"y", TrtInputArg::kBoth}}));
  // TODO(tfeher): Consider adding INT8 type because FC layer can support it.
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
    return errors::InvalidArgument(
        "All inputs are weights, but Grappler is expected to fold them.");
  }

  bool transpose_a = false, transpose_b = false;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "adj_x", &transpose_a));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "adj_y", &transpose_b));

  // In case input_l is weight, check whether input_l has implicit batch mode
  // compatible batch dim.
  const auto check_weight_is_not_batched =
      [](const TRT_TensorOrWeights& input_l,
         const TRT_TensorOrWeights& input_r) {
        // There is no way to batch constants in TRT using implicit batch mode.
        // Example:
        // Tensor with TF Dims: 12 5 3 -> TRT Dims: 5 3
        // Weight with TF Dims: 12 3 6 -> TRT Dims: 12 3 6
        // It is not possible to treat the weight input as a batched [3, 6]
        // tensor. Batched weight tensors must have batch dim = 1 (after the
        // broadcast).
        if (input_l.is_weights() &&
            input_l.GetTrtDims().nbDims > input_r.GetTrtDims().nbDims &&
            input_l.GetTrtDims().d[0] != 1) {
          return errors::Unimplemented(
              "TensorRT does not support batched constants in implicit batch "
              "mode.");
        }
        return Status::OK();
      };
  if (params->use_implicit_batch) {
    TF_RETURN_IF_ERROR(check_weight_is_not_batched(inputs.at(0), inputs.at(1)));
    TF_RETURN_IF_ERROR(check_weight_is_not_batched(inputs.at(1), inputs.at(0)));
  }

  // Broadcast inputs. We don't check feasibility since the dimensions in a
  // MatMul don't need to match. For example, consider a valid set of inputs
  // which would produce an output of shape [N, T, K]:
  // input 0: [N, T, C]
  // input 1: [1, C, K]
  // Since C != K and T != C, check feasiblity would fail.
  auto input_l = std::make_unique<TRT_TensorOrWeights>(inputs.at(0));
  auto input_r = std::make_unique<TRT_TensorOrWeights>(inputs.at(1));
  TF_RETURN_IF_ERROR(BroadcastTensors(input_l, input_r,
                                      /*check_feasibility=*/false, params));

  if (params->validation_only) return Status::OK();

  return ConvertMatMulHelper(params, *input_l, *input_r, transpose_a,
                             transpose_b);
}

Status ConvertSoftmax(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"logits", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  ITensorProxyPtr tensor = inputs.at(0).tensor();

  const int num_trt_dims = tensor->getDimensions().nbDims;
  if (num_trt_dims == 0 && params->use_implicit_batch) {
    return errors::InvalidArgument(
        "TensorRT Softmax cannot apply on batch dimension");
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::ISoftMaxLayer* layer =
      params->converter->network()->addSoftMax(*tensor->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  // Tensorflow SoftMax assumes applying softmax on the last dimension.
  layer->setAxes(1 << (num_trt_dims - 1));

  ITensorProxyPtr output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertArgMinMax(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"dimension", true}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  DataType output_dtype{DataType::DT_INT32};
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(node_def), "output_type", &output_dtype));

  if (output_dtype != DataType::DT_INT32) {
    return errors::Unimplemented("Output type ", DataTypeString(output_dtype),
                                 " is not supported");
  }
  int tf_axis = inputs.at(1).weights().GetSpan<int>()[0];
  int trt_axis;
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));
  nvinfer1::TopKOperation topk_op;
  if (node_def.op() == "ArgMin") {
    topk_op = nvinfer1::TopKOperation::kMIN;
  } else if (node_def.op() == "ArgMax") {
    topk_op = nvinfer1::TopKOperation::kMAX;
  } else {
    return errors::InvalidArgument("Unsupported ArgMin/Max operation");
  }

#if !IS_TRT_VERSION_GE(7, 0, 0, 11)
  const nvinfer1::Dims trt_dims = params->inputs.at(0).GetTrtDims();
  if (trt_dims.nbDims >= 4) {
    string trt_dim_str = DebugString(trt_dims);

    return errors::Unimplemented(node_def.op(), "op is not able to support",
                                 " tensors with 4+ dimensions (excluding batch",
                                 " size). Received: ", trt_dim_str);
  }
#endif

  if (params->validation_only) return Status::OK();

  // Use TopK with k = 1. Only indices output is needed (output 1).
  const uint32_t reduce_axes = 1 << trt_axis;
  nvinfer1::ITopKLayer* layer = params->converter->network()->addTopK(
      *inputs.at(0).tensor()->trt_tensor(), topk_op, 1, reduce_axes);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "topk");
  ITensorProxyPtr output_indices_tensor = layer->getOutput(1);

  // Squeeze on axis.
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  input_dims[trt_axis] = 0;
  ITensorProxyPtr output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
      /*input=*/output_indices_tensor,
      /*input_dims=*/&input_dims,
      /*params=*/params,
      /*output=*/&output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));

  return Status::OK();
}

Status ConvertTopK(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"k", true}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  bool sorted{false};
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "sorted", &sorted));

  if (!sorted) {
    // TensorRT only supports sorted output. Although TensorFlow API
    // doesn't specify the order of output elements in case sorted=false,
    // but it's safer to not convert because the output of TensorRT might
    // be different with TensorFlow which can cause confusion.
    return errors::InvalidArgument("Only sorted=True is supported");
  }

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  const int num_dims = tensor->getDimensions().nbDims;
  if (num_dims == 0) {
    return errors::InvalidArgument(
        "TensorRT TopK cannot apply on batch dimension");
  }

  TRT_ShapedWeights k_w = inputs.at(1).weights();
  if (k_w.count() != 1) {
    return errors::InvalidArgument("k value of TopK should be a scalar");
  }
  // Note that ITopKLayer always have sorted outputs, so we don't need to handle
  // the 'sorted' attribute of the node.
  if (params->validation_only) return Status::OK();

  const nvinfer1::TopKOperation op = nvinfer1::TopKOperation::kMAX;
  const int k = *(k_w.GetPointer<int>());
  const uint32_t reduce_axes = 1 << (num_dims - 1);
  nvinfer1::ITopKLayer* layer = params->converter->network()->addTopK(
      *tensor->trt_tensor(), op, k, reduce_axes);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  ITensorProxyPtr output_value_tensor = layer->getOutput(0);
  ITensorProxyPtr output_indices_tensor = layer->getOutput(1);
  params->outputs->push_back(TRT_TensorOrWeights(output_value_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_indices_tensor));
  return Status::OK();
}

StatusOr<std::pair<ITensorProxyPtr, ITensorProxyPtr>>
CalcDepthSpaceDynamicShape(OpConverterParams* params, int block_size,
                           string data_format) {
  // Instead we use a shape layer and shape arithmetic to calculate the reshape
  // dimensions.
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  const int channels_axis = data_format == "NCHW" ? 1 : 3;
  const int h_axis = data_format == "NCHW" ? 2 : 1;
  const int w_axis = data_format == "NCHW" ? 3 : 2;

  // Get shapes.
  ITensorProxyPtr shape = params->converter->network()
                              ->addShape(*inputs.at(0).tensor()->trt_tensor())
                              ->getOutput(0);
  ITensorProxyPtr batch_size =
      params->converter->network()
          ->addSlice(*shape->trt_tensor(), {1, {0}}, {1, {1}}, {1, {1}})
          ->getOutput(0);
  ITensorProxyPtr num_channels =
      params->converter->network()
          ->addSlice(*shape->trt_tensor(), {1, {channels_axis}}, {1, {1}},
                     {1, {1}})
          ->getOutput(0);
  ITensorProxyPtr h =
      params->converter->network()
          ->addSlice(*shape->trt_tensor(), {1, {h_axis}}, {1, {1}}, {1, {1}})
          ->getOutput(0);
  ITensorProxyPtr w =
      params->converter->network()
          ->addSlice(*shape->trt_tensor(), {1, {w_axis}}, {1, {1}}, {1, {1}})
          ->getOutput(0);
  ITensorProxyPtr r;
  TF_RETURN_IF_ERROR(CreateScalarConstant(params, block_size, &r));
  ITensorProxyPtr r_squared;
  TF_RETURN_IF_ERROR(
      CreateScalarConstant(params, block_size * block_size, &r_squared));
  // Get shuffle parameters.
  std::vector<ITensorProxyPtr> first_shuffle_tensors(6, nullptr);
  std::vector<ITensorProxyPtr> second_shuffle_tensors(4, nullptr);
  if (node_def.op() == "DepthToSpace") {
    // First Reshape [N, C, H, W] - > [N, r, r, C/(r*r), H, W].
    first_shuffle_tensors[0] = batch_size;
    first_shuffle_tensors[1] = r;
    first_shuffle_tensors[2] = r;
    first_shuffle_tensors[3] =
        params->converter->network()
            ->addElementWise(*num_channels->trt_tensor(),
                             *r_squared->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    first_shuffle_tensors[4] = h;
    first_shuffle_tensors[5] = w;
    // Second Reshape [N, C/(r*r), H, r, W, r] -> [N, C/(r*r), H * r, W * r].
    second_shuffle_tensors[0] = batch_size;
    second_shuffle_tensors[1] =
        params->converter->network()
            ->addElementWise(*num_channels->trt_tensor(),
                             *r_squared->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    second_shuffle_tensors[2] =
        params->converter->network()
            ->addElementWise(*h->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    second_shuffle_tensors[3] =
        params->converter->network()
            ->addElementWise(*w->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
  } else if (node_def.op() == "SpaceToDepth") {
    // First Reshape [N, C, H, W] -> [N, C, H/r, r, W/r, r].
    first_shuffle_tensors[0] = batch_size;
    first_shuffle_tensors[1] = num_channels;
    first_shuffle_tensors[2] =
        params->converter->network()
            ->addElementWise(*h->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    first_shuffle_tensors[3] = r;
    first_shuffle_tensors[4] =
        params->converter->network()
            ->addElementWise(*w->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    first_shuffle_tensors[5] = r;

    // Second Reshape  [N, r, r, C, H/r, W/r] -> [N, C*r*r, H/r, W/r].
    second_shuffle_tensors[0] = batch_size;
    second_shuffle_tensors[1] =
        params->converter->network()
            ->addElementWise(*num_channels->trt_tensor(),
                             *r_squared->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    second_shuffle_tensors[2] =
        params->converter->network()
            ->addElementWise(*h->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    second_shuffle_tensors[3] =
        params->converter->network()
            ->addElementWise(*w->trt_tensor(), *r->trt_tensor(),
                             nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
  }

  StatusOr<ITensorProxyPtr> result =
      ConcatenateTensors(params, first_shuffle_tensors, 0);
  TF_RETURN_IF_ERROR(result.status());
  ITensorProxyPtr first_shuffle_shape = result.ValueOrDie();

  result = ConcatenateTensors(params, second_shuffle_tensors, 1);
  TF_RETURN_IF_ERROR(result.status());
  ITensorProxyPtr second_shuffle_shape = result.ValueOrDie();

  return std::make_pair(first_shuffle_shape, second_shuffle_shape);
}

Status ConvertDepthSpaceShuffle(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  string data_format;
  int block_size;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "block_size", &block_size));

  if (block_size < 2) {
    return errors::InvalidArgument("Block size must be 2 or greater");
  }

  if (data_format != "NCHW" && data_format != "NHWC") {
    return errors::Unimplemented("Data format ", data_format,
                                 " is not supported");
  }
  int idx_offset = params->use_implicit_batch ? 0 : 1;
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  const int required_rank = 3 + idx_offset;
  if (dims.nbDims != required_rank) {
    return errors::InvalidArgument("The input to ", node_def.op(),
                                   " must be rank 4");
  }
  const int num_channels =
      data_format == "NCHW" ? dims.d[0 + idx_offset] : dims.d[2 + idx_offset];
  const int h =
      data_format == "NCHW" ? dims.d[1 + idx_offset] : dims.d[0 + idx_offset];
  const int w =
      data_format == "NCHW" ? dims.d[2 + idx_offset] : dims.d[1 + idx_offset];
  // Get shuffle parameters.
  nvinfer1::Dims first_shuffle_shape;
  nvinfer1::Permutation transpose_perm;
  nvinfer1::Dims second_shuffle_shape;

  // We define all the shuffle and transpose dimensions assuming implicit batch
  // mode. Afterwards we will update them to explicit batch mode if needed.
  // Additionally, an NCHW layout is assumed, and this assumption is corrected
  // afterwards with an initial transpose op. TODO(tfeher): Get rid of the
  // layout_transpose ops by defining shuffle shape specifically for NCHW and
  // NHCW.
  if (node_def.op() == "DepthToSpace") {
    if (num_channels != -1 && num_channels % (block_size * block_size) != 0) {
      return errors::InvalidArgument(
          "Number of channels must be divisible by block_size*block_size");
    }
    // First Reshape [C, H, W] - > [r, r, C/(r*r), H, W]
    first_shuffle_shape = {
        /*nbDims=*/5,
        /*d=*/{block_size, block_size, num_channels / (block_size * block_size),
               h, w}};
    // Transpose [r, r, C/(r*r), H, W] -> [C/(r*r), H, r, W, r]
    transpose_perm = {2, 3, 0, 4, 1};
    // Second Reshape [C/(r*r), H, r, W, r] -> [C/(r*r), H * r, W * r]
    second_shuffle_shape =
        nvinfer1::Dims3(num_channels / (block_size * block_size),
                        h * block_size, w * block_size);
  } else {
    if (node_def.op() != "SpaceToDepth")
      return errors::InvalidArgument("Incorrect op type ", node_def.op());
    if ((h != -1 && h % block_size != 0) || (w != -1 && w % block_size != 0)) {
      return errors::InvalidArgument(
          "Width and height must be divisible by block_size");
    }
    // First Reshape [C, H, W] -> [C, H/r, r, W/r, r]
    first_shuffle_shape = {/*nbDims=*/5,
                           /*d=*/{num_channels, h / block_size, block_size,
                                  w / block_size, block_size}};
    // Transpose [C, H/r, r, W/r, r] -> [r, r, C, H/r, W/r]
    transpose_perm = {2, 4, 0, 1, 3};
    // Second Reshape  [r, r, C, H/r, W/r] -> [C*r*r, H/r, W/r]
    second_shuffle_shape = nvinfer1::Dims3(
        num_channels * block_size * block_size, h / block_size, w / block_size);
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::IShuffleLayer* first_shuffle =
      params->converter->network()->addShuffle(
          *inputs.at(0).tensor()->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(first_shuffle, node_def.name());
  params->converter->SetLayerName(first_shuffle, node_def, "shuffle",
                                  /*op_instance=*/0);

  ITensorProxyPtr second_shuffle_shape_tensor;

  if (HasStaticShape(inputs.at(0).GetTrtDims())) {
    // Adjust a reshape constructed at implicit batch mode for explicit batch
    // mode. In particular, we need to insert the batch dimension size to the
    // beginning of all the dimension sizes. Example: reshape {20,10,30} for
    // implicit batch mode becomes reshape {N,20,10,30} for explicit batch mode.
    auto adjust_reshape = [](int N, nvinfer1::Dims dims,
                             bool use_implicit_batch) {
      if (use_implicit_batch) return dims;
      for (int i = dims.nbDims; i > 0; i--) {
        dims.d[i] = dims.d[i - 1];
      }
      dims.d[0] = N;
      dims.nbDims++;
      return dims;
    };

    first_shuffle_shape = adjust_reshape(dims.d[0], first_shuffle_shape,
                                         params->use_implicit_batch);
    second_shuffle_shape = adjust_reshape(dims.d[0], second_shuffle_shape,
                                          params->use_implicit_batch);

    first_shuffle->setReshapeDimensions(first_shuffle_shape);
  } else {
    StatusOr<std::pair<ITensorProxyPtr, ITensorProxyPtr>> result =
        CalcDepthSpaceDynamicShape(params, block_size, data_format);
    TF_RETURN_IF_ERROR(result.status());
    first_shuffle->setInput(1, *result.ValueOrDie().first->trt_tensor());
    second_shuffle_shape_tensor = result.ValueOrDie().second;
  }

  // Adjust a transpose constructed assuming implicit batch mode for explicit
  // batch mode. In particular, we need to add the batch dimension to d0 and
  // add 1 to all the dimension id in the transpose. Example: permutation
  // for implicit batch mode becomes permutation {0,3,2,1} for explicit batch
  // mode.
  auto adjust_perm = [](int n, nvinfer1::Permutation perm,
                        bool use_implicit_batch) {
    if (use_implicit_batch) return perm;
    for (int i = n; i > 0; i--) {
      perm.order[i] = perm.order[i - 1] + 1;
    }
    perm.order[0] = 0;
    return perm;
  };
  transpose_perm = adjust_perm(5, transpose_perm, params->use_implicit_batch);

  if (data_format == "NHWC") {
    nvinfer1::Permutation layout_transpose =
        adjust_perm(3, {2, 0, 1}, params->use_implicit_batch);
    first_shuffle->setFirstTranspose(layout_transpose);
  }
  first_shuffle->setSecondTranspose(transpose_perm);

  nvinfer1::IShuffleLayer* second_shuffle =
      params->converter->network()->addShuffle(*first_shuffle->getOutput(0));
  TFTRT_RETURN_ERROR_IF_NULLPTR(second_shuffle, node_def.name());
  params->converter->SetLayerName(second_shuffle, node_def, "shuffle",
                                  /*op_instance=*/1);

  if (HasStaticShape(inputs.at(0).GetTrtDims())) {
    second_shuffle->setReshapeDimensions(second_shuffle_shape);
  } else {
    second_shuffle->setInput(1, *second_shuffle_shape_tensor->trt_tensor());
  }
  if (data_format == "NHWC") {
    nvinfer1::Permutation layout_transpose =
        adjust_perm(3, {1, 2, 0}, params->use_implicit_batch);
    second_shuffle->setSecondTranspose(layout_transpose);
  }

  params->outputs->push_back(TRT_TensorOrWeights(second_shuffle->getOutput(0)));
  return Status::OK();
}

Status ConvertSquaredDifference(OpConverterParams* params) {
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}, {"y", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // Broadcast inputs.
  nvinfer1::Dims broadcasted_dims_l, broadcasted_dims_r;
  TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
      inputs.at(0), inputs.at(1), /*check_feasibility=*/true,
      params->use_implicit_batch, &broadcasted_dims_l, &broadcasted_dims_r));
  ITensorProxyPtr tensor_l = nullptr;
  ITensorProxyPtr tensor_r = nullptr;
  TF_RETURN_IF_ERROR(
      PrepareTensorForShape(params->converter, inputs.at(0), broadcasted_dims_l,
                            params->validation_only, &tensor_l, node_def));
  TF_RETURN_IF_ERROR(
      PrepareTensorForShape(params->converter, inputs.at(1), broadcasted_dims_r,
                            params->validation_only, &tensor_r, node_def));
  if (params->validation_only) return Status::OK();

  // Subtract x - y.
  nvinfer1::IElementWiseLayer* sub =
      params->converter->network()->addElementWise(
          *tensor_l->trt_tensor(), *tensor_r->trt_tensor(),
          nvinfer1::ElementWiseOperation::kSUB);
  TFTRT_RETURN_ERROR_IF_NULLPTR(sub, node_def.name());
  params->converter->SetLayerName(sub, node_def, "sub");

  // Multiply (x - y) * (x - y).
  nvinfer1::IElementWiseLayer* mul =
      params->converter->network()->addElementWise(
          *sub->getOutput(0), *sub->getOutput(0),
          nvinfer1::ElementWiseOperation::kPROD);
  TFTRT_RETURN_ERROR_IF_NULLPTR(mul, node_def.name());
  params->converter->SetLayerName(mul, node_def, "mul");

  params->outputs->push_back(TRT_TensorOrWeights(mul->getOutput(0)));
  return Status::OK();
}

#if IS_TRT_VERSION_GE(8, 2, 1, 6) || defined(TF_TRT_USE_EFFICIENT_NMS_PLUGIN)

Status ConvertCombinedNMS(OpConverterParams* params) {
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params, {{"boxes", TrtInputArg::kTensor},
                {"scores", TrtInputArg::kTensor},
                {"max_output_size_per_class", TrtInputArg::kWeight},
                {"max_total_size", TrtInputArg::kWeight},
                {"iou_threshold", TrtInputArg::kWeight},
                {"score_threshold", TrtInputArg::kWeight}}));
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  ITensorProxyPtr boxes_tensor = inputs.at(0).tensor();
  ITensorProxyPtr scores_tensor = inputs.at(1).tensor();
  if (params->use_implicit_batch) {
    return errors::Unimplemented(
        "Implict batch mode not supported with CombinedNMS", node_def.name());
  }

  TRT_ShapedWeights output_size_per_class = inputs.at(2).weights();
  TRT_ShapedWeights total_size = inputs.at(3).weights();
  TRT_ShapedWeights iou_threshold = inputs.at(4).weights();
  TRT_ShapedWeights score_threshold = inputs.at(5).weights();
  const int max_size_per_class = *(output_size_per_class.GetPointer<int>());
  int max_total_size = *(total_size.GetPointer<int>());
  const float iou_thresh = *(iou_threshold.GetPointer<float>());
  const float score_thresh = *(score_threshold.GetPointer<float>());

  AttrSlice attrs(node_def);
  bool clip_boxes = false, pad_per_class = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "clip_boxes", &clip_boxes));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "pad_per_class", &pad_per_class));

  // Validate tensors and weights
  const auto boxes_dims = boxes_tensor->getDimensions();
  const auto scores_dims = scores_tensor->getDimensions();
  if (boxes_dims.nbDims != 4) {
    return errors::InvalidArgument(
        "NMS TRT Plugin input boxes must be 4-D including batch ",
        node_def.name());
  }
  const int num_classes = scores_dims.d[2];
  bool box_check = boxes_dims.d[2] == 1 || boxes_dims.d[2] == num_classes;
  if (!box_check) {
    return errors::InvalidArgument(
        "NMS TRT Plugin third dimension of boxes must be either 1 "
        "or match the num_classes dimension of scores ",
        node_def.name());
  }

  if (output_size_per_class.count() != 1) {
    return errors::InvalidArgument(
        "NMS TRT Plugin max_output_size_per_class must be scalar ",
        node_def.name());
  }
  if (max_size_per_class <= 0) {
    return errors::InvalidArgument(
        "NMS TRT Plugin max_output_size_per_class should be > 0",
        node_def.name());
  }
  if (total_size.count() != 1) {
    return errors::InvalidArgument(
        "NMS TRT Plugin max_total_size must be scalar ", node_def.name());
  }
  if (max_total_size <= 0) {
    return errors::InvalidArgument(
        "NMS TRT Plugin max_total_size should be > 0", node_def.name());
  }
  if (iou_threshold.count() != 1) {
    return errors::InvalidArgument(
        "NMS TRT Plugin iou_threshold must be scalar ", node_def.name());
  }
  if (iou_thresh < 0.0 || iou_thresh > 1.0) {
    return errors::InvalidArgument(
        "NMS TRT Plugin iou_threshold must be in [0, 1]", node_def.name());
  }
  if (score_threshold.count() != 1) {
    return errors::InvalidArgument(
        "NMS TRT Plugin score_threshold must be scalar ", node_def.name());
  }

  if (params->validation_only) return Status::OK();

  // Create plugin
  nvinfer1::PluginField fields[6] = {
      nvinfer1::PluginField{"max_output_size_per_class", &max_size_per_class,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"max_total_size", &max_total_size,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"iou_threshold", &iou_thresh,
                            nvinfer1::PluginFieldType::kFLOAT32, 1},
      nvinfer1::PluginField{"score_threshold", &score_thresh,
                            nvinfer1::PluginFieldType::kFLOAT32, 1},
      nvinfer1::PluginField{"pad_per_class", &pad_per_class,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"clip_boxes", &clip_boxes,
                            nvinfer1::PluginFieldType::kINT32, 1},
  };
  nvinfer1::PluginFieldCollection fc{6, fields};

  auto creator =
      getPluginRegistry()->getPluginCreator("EfficientNMS_TFTRT_TRT", "1", "");
  TFTRT_RETURN_ERROR_IF_NULLPTR(creator, node_def.name());

  TrtUniquePtrType<nvinfer1::IPluginV2> plugin(
      creator->createPlugin(node_def.name().c_str(), &fc));
  TFTRT_RETURN_ERROR_IF_NULLPTR(plugin, node_def.name());

  // Set plugin inputs
  std::vector<nvinfer1::ITensor*> trt_plugin_inputs;
  trt_plugin_inputs.push_back(boxes_tensor->trt_tensor());
  trt_plugin_inputs.push_back(scores_tensor->trt_tensor());

  // Add plugin to network
  nvinfer1::IPluginV2Layer* layer = params->converter->network()->addPluginV2(
      &trt_plugin_inputs[0], static_cast<int>(trt_plugin_inputs.size()),
      *plugin);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "plugin");

  // Set plugin outputs
  ITensorProxyPtr output_num_detections = layer->getOutput(0);
  ITensorProxyPtr output_detection_boxes = layer->getOutput(1);
  ITensorProxyPtr output_detection_scores = layer->getOutput(2);
  ITensorProxyPtr output_detection_classes = layer->getOutput(3);

  // Cast the classes output from int32 to float32
  nvinfer1::IIdentityLayer* layer_detection_classes =
      params->converter->network()->addIdentity(
          *output_detection_classes->trt_tensor());
  layer_detection_classes->setOutputType(0, nvinfer1::DataType::kFLOAT);
  output_detection_classes = layer_detection_classes->getOutput(0);

  // The plugin produces a [N, 1] tensor for the num output, squeeze it to [N]
  std::vector<int> input_dims{output_num_detections->getDimensions().d[0], 0};
  TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
      /*input=*/output_num_detections,
      /*input_dims=*/&input_dims,
      /*params=*/params,
      /*output=*/&output_num_detections));

  // Final outputs
  params->outputs->push_back(TRT_TensorOrWeights(output_detection_boxes));
  params->outputs->push_back(TRT_TensorOrWeights(output_detection_scores));
  params->outputs->push_back(TRT_TensorOrWeights(output_detection_classes));
  params->outputs->push_back(TRT_TensorOrWeights(output_num_detections));

  return Status::OK();
}

#elif IS_TRT_VERSION_GE(7, 1, 3, 0)

bool AllowNmsTopkOverride() {
  static bool result = [] {
    bool value;
    Status status = ReadBoolFromEnvVar("TF_TRT_ALLOW_NMS_TOPK_OVERRIDE",
                                       /*default_value=*/false, &value);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    return value;
  }();
  return result;
}

Status ConvertCombinedNMS(OpConverterParams* params) {
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"boxes", false},
                                   {"scores", false},
                                   {"max_output_size_per_class", true},
                                   {"max_total_size", true},
                                   {"iou_threshold", true},
                                   {"score_threshold", true}}));
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  ITensorProxyPtr boxes_tensor = inputs.at(0).tensor();
  ITensorProxyPtr scores_tensor = inputs.at(1).tensor();
  TRT_ShapedWeights output_size_per_class = inputs.at(2).weights();
  TRT_ShapedWeights total_size = inputs.at(3).weights();
  TRT_ShapedWeights iou_threshold = inputs.at(4).weights();
  TRT_ShapedWeights score_threshold = inputs.at(5).weights();

  // Validate tensors and weights (also set some of the needed plugin fields)
  const auto boxes_dims = boxes_tensor->getDimensions();
  const auto scores_dims = scores_tensor->getDimensions();
  if (!params->use_implicit_batch &&
      (!HasStaticShape(boxes_dims) || !HasStaticShape(scores_dims))) {
    return errors::Unimplemented(
        "TensorRT BatchedNMS Plugin requires input with static shape");
  }
  const int offset = params->use_implicit_batch ? 0 : 1;
  if (boxes_dims.nbDims != 3 + offset) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin input boxes must be 4-D including batch");
  }
  const int class_idx = 1 + offset;
  const int num_classes = scores_dims.d[class_idx];
  const int num_boxes = boxes_dims.d[0 + offset];
  bool box_check =
      boxes_dims.d[class_idx] == 1 || boxes_dims.d[class_idx] == num_classes;
  if (!box_check) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin third dimension of boxes must be either 1 "
        "or num_classes");
  }

  if (output_size_per_class.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class must be scalar");
  }
  int max_size_per_class = *(output_size_per_class.GetPointer<int>());
  if (max_size_per_class <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class should be > 0");
  }
  if (total_size.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size must be scalar");
  }
  int max_total_size = *(total_size.GetPointer<int>());
  if (max_total_size <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size should be > 0");
  }
  if (iou_threshold.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be scalar");
  }
  float iou_thresh = *(iou_threshold.GetPointer<float>());
  if (iou_thresh < 0.0 || iou_thresh > 1.0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be in [0, 1]");
  }
  if (score_threshold.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin score_threshold must be scalar");
  }

  bool pad_per_class = false, clip_boxes = false;
  AttrSlice attrs(node_def);
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "pad_per_class", &pad_per_class));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "clip_boxes", &clip_boxes));

  // TRT op is_normalized=False treats input corrdinates as pixels and
  // calculates width/height as (max - min + 1).
  //
  // TF op CombinedNonMaxSuppression doesn't care about the normalization and
  // calculates width/height  as (max-min).
  //
  // We set is_normalized = true to be consistent with TF IOU calculaton.
  const bool is_normalized = true;

  bool share_location = (boxes_dims.d[class_idx] == 1);
  int keep_top_k = 0;
  if (pad_per_class) {
    keep_top_k = std::min(max_size_per_class * num_classes, max_total_size);
  } else {
    keep_top_k = max_total_size;
  }

  // According to the batchedNMS plugin description we need to set top_k so that
  // keep_top_k <= top_k
  // https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
  // Before the NMS step, TRT selects top_k candidate from each class and
  // discards the rest. The NMS step is performed only among the top_k
  // candidates. To be strictly compatible with the TF op, we need that top_k is
  // greater equal to num_boxes.
  int top_k = std::max(num_boxes, keep_top_k);
  // TRT has a limitation: top_k <=4096.
  if (top_k > 4096) {
    if (AllowNmsTopkOverride()) {
      top_k = 4096;
      keep_top_k = std::min(top_k, keep_top_k);
    } else {
      return errors::InvalidArgument(
          "TRT NMS plugin allow top_k<=4096, where top_k = max(num_boxes, "
          "max_total_size). You can override this by setting "
          "TF_TRT_ALLOW_NMS_TOPK_OVERRIDE=1 environment variable, but this can "
          "result in a loss of accuracy.");
    }
  }

  if (params->validation_only) return Status::OK();
  float score_thresh = *(score_threshold.GetPointer<float>());
  const int background_id = -1;
  nvinfer1::PluginField fields[9] = {
      nvinfer1::PluginField{"shareLocation", &share_location,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"backgroundLabelId", &background_id,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"numClasses", &num_classes,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"topK", &top_k, nvinfer1::PluginFieldType::kINT32,
                            1},
      nvinfer1::PluginField{"keepTopK", &keep_top_k,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"scoreThreshold", &score_thresh,
                            nvinfer1::PluginFieldType::kFLOAT32, 1},
      nvinfer1::PluginField{"iouThreshold", &iou_thresh,
                            nvinfer1::PluginFieldType::kFLOAT32, 1},
      nvinfer1::PluginField{"isNormalized", &is_normalized,
                            nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"clipBoxes", &clip_boxes,
                            nvinfer1::PluginFieldType::kINT32, 1}};
  nvinfer1::PluginFieldCollection fc{9, fields};

  // Get plugin creator
  auto creator =
      getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1", "");
  TFTRT_RETURN_ERROR_IF_NULLPTR(creator, node_def.name());

  // Create plugin
  TrtUniquePtrType<nvinfer1::IPluginV2> plugin(
      creator->createPlugin(node_def.name().c_str(), &fc));
  TFTRT_RETURN_ERROR_IF_NULLPTR(plugin, node_def.name());

  // Set plugin inputs
  std::vector<nvinfer1::ITensor*> trt_plugin_inputs;
  trt_plugin_inputs.push_back(boxes_tensor->trt_tensor());
  trt_plugin_inputs.push_back(scores_tensor->trt_tensor());

  // Add plugin to network
  nvinfer1::IPluginV2Layer* layer = params->converter->network()->addPluginV2(
      &trt_plugin_inputs[0], static_cast<int>(trt_plugin_inputs.size()),
      *plugin);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "plugin");

  // Set plugin outputs
  ITensorProxyPtr output_nmsed_boxes = layer->getOutput(1);

  // TensorRT fixes (removes) the extra last dimension in CombinedNMS outputs
  ITensorProxyPtr output_num_detections = layer->getOutput(0);
  ITensorProxyPtr output_nmsed_scores = layer->getOutput(2);
  ITensorProxyPtr output_nmsed_classes = layer->getOutput(3);

  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_boxes));
  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_scores));
  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_classes));
  params->outputs->push_back(TRT_TensorOrWeights(output_num_detections));

  return Status::OK();
}

#endif  // IS_TRT_VERSION_GE(7, 1, 3, 0)

Status ConvertResize(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"input", TrtInputArg::kTensor}, {"size", TrtInputArg::kBoth}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  // Get input tensor.
  ITensorProxyPtr inputs_tensor = inputs.at(0).tensor();
  TFTRT_RETURN_ERROR_IF_NULLPTR(inputs_tensor, params->node_def.name());

  // Check output size. It must constain two values i.e. [H_out, W_out]
  const bool const_output_size = inputs.at(1).is_weights();
  if (const_output_size) {
    // Output size is given as a constant.
    if (inputs.at(1).weights().count() != 2) {
      return errors::Unimplemented("Resize requires 2D values for the size");
    }
  } else {
    // Output size is given as a tensor, possibly as the result of shape
    // calculation ops in the graph.
    if (params->use_implicit_batch) {
      return errors::Unimplemented(
          "Resize requires constant size in implicit batch mode");
    }
    TF_RETURN_IF_ERROR(ExpectShapeTensor(inputs.at(1)));
    if (inputs.at(1).tensor()->getDimensions().d[0] != 2) {
      return errors::Unimplemented("Resize requires 2D values for the size");
    }
  }

  // Verify and consume node attributes.
  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(AttrSlice(node_def), "align_corners", &align_corners));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  // Verify resize mode. Initialize resize mode if supported.
  nvinfer1::ResizeMode resize_mode;
  if (node_def.op() == "ResizeBilinear") {
#if IS_TRT_VERSION_GE(7, 1, 0, 0)
    if (!align_corners) {
      return errors::InvalidArgument(
          "Cannot Convert Bilinear Resize when align_corners=False");
    }
#endif
    resize_mode = nvinfer1::ResizeMode::kLINEAR;
  } else if (node_def.op() == "ResizeNearestNeighbor") {
    resize_mode = nvinfer1::ResizeMode::kNEAREST;
  } else {
    return errors::Unimplemented(node_def.op(), " is not yet implemented");
  }

  // return after validation if only validation is requested.
  if (params->validation_only) return Status::OK();

  // Transpose tensor from NHWC to NCHW format.
  TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
      inputs_tensor, {0, 3, 1, 2}, &inputs_tensor, node_def, "to_NCHW"));

  // Calculate the output shape as static dimensions or a shape tensor:
  // Given input shape [N, C, H, W] and output size [H_out, W_out],
  // output shape equals [N, C, H_out, W_out].
  nvinfer1::Dims output_shape_dims;
  ITensorProxyPtr output_shape_tensor;
  const bool static_output_shape =
      HasStaticShape(inputs_tensor->getDimensions()) && const_output_size;
  if (static_output_shape) {
    // If the output shape can be fully determined at build time, calculate it
    // as a set of dimensions.
    output_shape_dims.nbDims = inputs_tensor->getDimensions().nbDims;
    for (int i = 0; i < output_shape_dims.nbDims; ++i) {
      output_shape_dims.d[i] = inputs_tensor->getDimensions().d[i];
    }
    const int* weights_ptr = inputs.at(1).weights().GetPointer<int>();
    output_shape_dims.d[output_shape_dims.nbDims - 2] = weights_ptr[0];
    output_shape_dims.d[output_shape_dims.nbDims - 1] = weights_ptr[1];
  } else {
    // Otherwise, build the output shape as a shape tensor that will be computed
    // at run time.
    // The batch size and num of channels will be copied from the input shape.
    ITensorProxyPtr shape = params->converter->network()
                                ->addShape(*inputs_tensor->trt_tensor())
                                ->getOutput(0);
    ITensorProxyPtr batch_size =
        params->converter->network()
            ->addSlice(*shape->trt_tensor(), {1, {0}}, {1, {1}}, {1, {1}})
            ->getOutput(0);
    ITensorProxyPtr num_channels =
        params->converter->network()
            ->addSlice(*shape->trt_tensor(), {1, {1}}, {1, {1}}, {1, {1}})
            ->getOutput(0);

    // The height and width will be obtained from the requested output size.
    ITensorProxyPtr height, width;
    if (const_output_size) {
      // If the output size is constant, the height and width dimensions can be
      // created as constants from the size values.
      const int* weights_ptr = inputs.at(1).weights().GetPointer<int>();
      TF_RETURN_IF_ERROR(CreateScalarConstant(params, weights_ptr[0], &height));
      TF_RETURN_IF_ERROR(CreateScalarConstant(params, weights_ptr[1], &width));
    } else {
      // Otherwise, the size is a tensor which can be sliced, and each element
      // used directly as the output height and width dimensions.
      ITensorProxyPtr size = inputs.at(1).tensor();
      height = params->converter->network()
                   ->addSlice(*size->trt_tensor(), {1, {0}}, {1, {1}}, {1, {1}})
                   ->getOutput(0);
      width = params->converter->network()
                  ->addSlice(*size->trt_tensor(), {1, {1}}, {1, {1}}, {1, {1}})
                  ->getOutput(0);
    }

    StatusOr<ITensorProxyPtr> result = ConcatenateTensors(
        params, {batch_size, num_channels, height, width}, 0);
    TF_RETURN_IF_ERROR(result.status());
    output_shape_tensor = result.ValueOrDie();
  }

  // Add resize layer.
  nvinfer1::IResizeLayer* layer =
      params->converter->network()->addResize(*inputs_tensor->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  // Set layer parameters.
  layer->setResizeMode(resize_mode);
  layer->setAlignCorners(align_corners);

  // Set output shape.
  if (static_output_shape) {
    // If the shapes are fully known at build time, pass the static output shape
    // to the resize layer as expected output dimensions.
    layer->setOutputDimensions(output_shape_dims);
  } else {
    // Otherwise, pass the output shape tensor to the resize layer as an input.
    layer->setInput(1, *output_shape_tensor->trt_tensor());
  }

  // Get output tensor. Transpose it from NCHW to NHWC.
  ITensorProxyPtr output = layer->getOutput(0);

  TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
      output, {0, 2, 3, 1}, &output, node_def, "to_NHWC"));
  params->outputs->push_back(TRT_TensorOrWeights(output));
  // Success
  return Status::OK();
}  // ConvertResize

Status ConvertAddN(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  int num_inputs;
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "N", &num_inputs));

  if (num_inputs < 2) {
    return errors::InvalidArgument("AddN requires at least two inputs");
  }

  TFTRT_CHECK_INPUT_SIZE(inputs.size(), num_inputs, node_def);

  for (const auto& input : inputs) {
    if (!input.is_tensor() && input.weights().Shape().dim(0) != 1) {
      return errors::InvalidArgument(
          "Weights input to AddN is required to have batch dimension 1.");
    }
  }
  if (params->validation_only) return Status::OK();

  // AddN doesn't support broadcast.
  std::vector<ITensorProxyPtr> tensor_inputs;
  tensor_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (input.is_tensor()) {
      tensor_inputs.push_back(input.tensor());
    } else {
      auto dims = input.weights().Shape();
      TF_RETURN_IF_ERROR(dims.RemoveBatchDimension());
      tensor_inputs.push_back(params->converter->CreateConstantLayer(
          input.weights(), dims.AsTrtDims()));
    }
  }
  ITensorProxyPtr lhs = tensor_inputs[0];
  for (int i = 1; i < num_inputs; ++i) {
    ITensorProxyPtr rhs = tensor_inputs[i];
    nvinfer1::ILayer* layer = params->converter->network()->addElementWise(
        *lhs->trt_tensor(), *rhs->trt_tensor(),
        nvinfer1::ElementWiseOperation::kSUM);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    params->converter->SetLayerName(layer, node_def, std::to_string(i));
    lhs = layer->getOutput(0);
  }
  params->outputs->push_back(TRT_TensorOrWeights(lhs));
  return Status::OK();
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertBiasAdd, "BiasAdd");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertClipByValue, "ClipByValue");
#if IS_TRT_VERSION_GE(7, 1, 3, 0) || defined(TF_TRT_USE_EFFICIENT_NMS_PLUGIN)
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertCombinedNMS,
                                  "CombinedNonMaxSuppression");
#endif
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertAddN, "AddN");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertCast, "Cast");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConcat, "ConcatV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConst, "Const");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConv2D, "Conv2D");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConv2DBackpropInput,
                                  "Conv2DBackpropInput");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertDepthSpaceShuffle, "DepthToSpace");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConv2DDepthwise,
                                  "DepthwiseConv2dNative");

REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertExpandDims, "ExpandDims");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertFusedConv2DBiasActivation,
                                  "FusedConv2DBiasActivation");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertGather, "GatherV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertLeakyRelu, "LeakyRelu");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertMatMul, "MatMul");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertPack, "Pack");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertPad, "Pad");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertRelu6, "Relu6");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertReshape, "Reshape");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConv3D, "Conv3D");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertConv3DBackpropInputV2,
                                  "Conv3DBackpropInputV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertResize, "ResizeBilinear");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertResize, "ResizeNearestNeighbor");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertPool3D, "AvgPool3D");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertPool3D, "MaxPool3D");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertShape, "Shape");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSlice, "Slice");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSoftmax, "Softmax");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertDepthSpaceShuffle, "SpaceToDepth");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSplit, "Split");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSquare, "Square");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSquaredDifference,
                                  "SquaredDifference");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertSqueeze, "Squeeze");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertStridedSlice, "StridedSlice");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertTopK, "TopKV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertTranspose, "Transpose");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertUnpack, "Unpack");
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertActivation,
                                  GetOperationNames(*ActivationTypeMap()));
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertPool, {"MaxPool", "AvgPool"});
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertFusedBatchNorm,
                                  {"FusedBatchNorm", "FusedBatchNormV2",
                                   "FusedBatchNormV3"});
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertReduce,
                                  {"Sum", "Prod", "Max", "Min", "Mean"});
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertArgMinMax, {"ArgMin", "ArgMax"});
// The following are no-ops during inference and will not be mapped to any
// TRT layer.
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertIdentity,
                                  {"Identity", "IdentityN", "Snapshot",
                                   "StopGradient", "_CopyFromHostToGpu"});
REGISTER_DEFAULT_TRT_OP_CONVERTER(ConvertBatchMatMul,
                                  {"BatchMatMul", "BatchMatMulV2"});

Status ConvertGraphDefToEngine(
    const GraphDef& gdef, TrtPrecisionMode precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<PartialTensorShape>& input_shapes,
    nvinfer1::ILogger* trt_logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, bool use_calibration,
    const bool use_implicit_batch, bool* convert_successfully,
    TrtShapeOptimizationProfile* profiles, absl::string_view engine_name,
    bool use_explicit_precision, tensorflow::grappler::Cluster* cluster) {
  engine->reset();
  if (convert_successfully) *convert_successfully = false;

  // Creating converter, TensorRT builder and network
  auto statusor = Converter::Create(precision_mode, use_calibration, trt_logger,
                                    use_implicit_batch, engine_name,
                                    use_explicit_precision);

  TF_RETURN_IF_ERROR(statusor.status());
  std::unique_ptr<Converter> converter = std::move(statusor.ValueOrDie());

  GraphDef graph = gdef;
  if (cluster != nullptr) {
    bool apply_layout_optim;
    Status status =
        ReadBoolFromEnvVar("TF_TRT_ENABLE_LAYOUT_OPTIMIZER",
                           /*default_value=*/true, &apply_layout_optim);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    if (apply_layout_optim) {
      tensorflow::grappler::GrapplerItem grappler_item;
      grappler_item.graph = gdef;
      // TensorRT API requires the input for convolution to be in NCHW.
      tensorflow::grappler::GenericLayoutOptimizer layout_optimizer("NCHW");
      TF_RETURN_IF_ERROR(
          layout_optimizer.Optimize(cluster, grappler_item, &graph));

      grappler_item.graph = graph;

      tensorflow::grappler::ConstantFolding const_optimizer(
          nullptr,
          /*disable_compressed_tensor_optimization=*/false,
          /*fold_quantization_emulation=*/false);
      TF_RETURN_IF_ERROR(
          const_optimizer.Optimize(cluster, grappler_item, &graph));

      // The optimizers may break the topological order
      // so we need these steps to restore it
      Graph g(OpRegistry::Global());
      TF_RETURN_IF_ERROR(
          ConvertGraphDefToGraph(GraphConstructorOptions(), graph, &g));
      g.ToGraphDef(&graph);
    }
  }
  VLOG(1) << "Starting to convert TensorFlow ops to TensorRT layers";
  std::vector<Converter::EngineOutputInfo> output_tensors;
  int num_layers = converter->network()->getNbLayers();
  absl::flat_hash_set<const char*> layer_names;
  // Graph nodes are already topologically sorted during construction
  for (const auto& node_def : graph.node()) {
    const string& node_name = node_def.name();
    VLOG(2) << "Converting node " << node_name << ", op=" << node_def.op();
    if (IsEngineInput(node_name)) {
      int32 slot_number = -1;
      string type_key;
      if (node_def.op() == "Placeholder") {
        if (!strings::safe_strto32(  // non-absl ok
                node_name.c_str() + strlen(IONamePrefixes::kInputPHName),
                &slot_number)) {
          return errors::InvalidArgument("Failed to parse slot number from ",
                                         node_name);
        }
        type_key = "dtype";
      } else if (tensorflow::grappler::IsArg(node_def)) {
        // Maybe remove the dependence on grappler and re-implement IsArg,
        // which is pretty simple (but could change if new Arg nodes are added)
        slot_number = node_def.attr().at("index").i();
        type_key = "T";
      } else {
        return errors::InvalidArgument(
            "Node ", node_name,
            " with is neither Placeholder nor Arg, instead ", node_def.op());
      }
      nvinfer1::DataType trt_dtype;
      nvinfer1::Dims trt_dims;
      int batch_size = -1;
      auto shape = input_shapes.at(slot_number);
      auto status = ValidateTensorProperties(
          node_def.op(), node_def.attr().at(type_key).type(), shape,
          use_implicit_batch, /*validation_only=*/false, &trt_dtype, &trt_dims,
          &batch_size);
      if (!status.ok()) {
        const string error_message =
            StrCat("Validation failed for ", node_name, " and input slot ",
                   slot_number, ": ", status.error_message());
        LOG_WARNING_WITH_PREFIX << error_message;
        return errors::CreateWithUpdatedMessage(status, error_message);
      }
      VLOG(2) << "Adding engine input tensor " << node_name << " with shape "
              << DebugString(trt_dims);
      // TODO(laigd): the conversion should always happen at runtime where all
      // the shapes are known, and we can provide a mode to generate the
      // engines offline, by calling sess.run() and cache/serialize the engines.
      TF_RETURN_IF_ERROR(converter->AddInputTensor(node_name, trt_dtype,
                                                   trt_dims, batch_size));
    } else if (IsEngineOutput(node_name)) {
      int32 slot_number = -1;
      if (node_def.op() == "Identity") {
        if (!strings::safe_strto32(  // non-absl ok
                node_name.c_str() + strlen(IONamePrefixes::kOutputPHName),
                &slot_number)) {
          return errors::InvalidArgument("Failed to parse slot number from ",
                                         node_name);
        }
      } else if (tensorflow::grappler::IsRetval(node_def)) {
        slot_number = node_def.attr().at("index").i();
      } else {
        return errors::InvalidArgument(
            "Node with name ", node_name,
            " starting with IONamePrefixes::kOutputPHName is "
            "neither Identity nor Retval, instead ",
            node_def.op());
      }
      // Get output type that TensorFlow expects
      DataType tf_dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node_def), "T", &tf_dtype));
      nvinfer1::DataType trt_dtype;
      TF_RETURN_IF_ERROR(TfTypeToTrtType(tf_dtype, &trt_dtype));
      if (output_tensors.size() <= slot_number) {
        output_tensors.resize(slot_number + 1);
      }
      output_tensors.at(slot_number) = {node_def.input(0), node_name,
                                        trt_dtype};
    } else {
      TF_RETURN_IF_ERROR(converter->ConvertNode(node_def));
    }

    // To support TF-TRT profiling, we ensure each ILayer has a non-empty name.
    // BuildCudaEngine returns an error if there is any ILayer name collision.
    // We want to report the error here before BuildCudaEngine in a more
    // meaningful way.
    int new_num_layers = converter->network()->getNbLayers();
    for (int i = num_layers; i < new_num_layers; i++) {
      auto layer = converter->network()->getLayer(i);
      if (layer->getName() == nullptr ||
          !layer_names.insert(layer->getName()).second) {
        std::string error_message = absl::StrCat(
            "Converting node ", node_name, ", op=", node_def.op(),
            layer->getName() ? " creates a layer with name collision"
                             : " creates a layer without a name");
        LOG_WARNING_WITH_PREFIX << error_message;
        return errors::Internal(error_message);
      }
    }
    num_layers = new_num_layers;
  }
  TF_RETURN_IF_ERROR(converter->RenameAndMarkOutputTensors(output_tensors));
  if (convert_successfully) *convert_successfully = true;

  // Apply user provided quantization ranges to tensors
  if (!use_explicit_precision) {
    converter->MaybeApplyQuantizationRanges();
  }

  // Build the engine.
  TF_RETURN_IF_ERROR(converter->BuildCudaEngine(
      engine, max_batch_size, max_workspace_size_bytes, allocator, calibrator,
      profiles));

  VLOG(1) << "Finished conversion";
  return Status::OK();
}

Status ConvertSegmentToGraphDef(
    const Graph* graph, const grappler::GraphProperties& graph_properties,
    const std::vector<const Node*>& subgraph_nodes,  // In topological order
    EngineInfo* engine_info) {
  std::vector<EngineConnection>* connections = &engine_info->connections;
  GraphDef* segment_def = &engine_info->segment_graph_def;
  std::set<string> marker_nodes;
  // Update connection shapes/data types and add corresponding input/output
  // nodes in the segment graphdef.
  for (size_t i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    if (connection.is_control_edge()) continue;
    auto outside_node = graph->FindNodeId(connection.outside_id);
    if (!outside_node) {
      // This should never happen, unless the original graph is problematic.
      return errors::NotFound("Cannot find node with id ",
                              connection.outside_id, " in the graph.");
    }
    // Updates the shape and data types of input/output connections.
    DataType dtype;
    PartialTensorShape partial_shape;
    if (connection.is_input_edge) {
      GetOutputProperties(graph_properties,
                          graph->FindNodeId(connection.outside_id),
                          connection.outside_port, &partial_shape, &dtype);
      connection.outside_shape = partial_shape;
    } else {
      GetInputProperties(graph_properties,
                         graph->FindNodeId(connection.outside_id),
                         connection.outside_port, &partial_shape, &dtype);
      connection.inside_shape = partial_shape;
    }
    connection.connection_type = dtype;

    // Add dummy input/output nodes to the segment graphdef.
    if (connection.is_input_edge) {
      const string node_name =
          StrCat(IONamePrefixes::kInputPHName, connection.port_number);
      if (marker_nodes.count(node_name)) {
        VLOG(1) << "Reusing input " << node_name << " for the edge "
                << connection.outside_node_name << ":"
                << connection.outside_port << " -> "
                << connection.inside_node_name << ":" << connection.inside_port;
        continue;
      }
      marker_nodes.insert(node_name);
      auto seg_node = segment_def->add_node();
      NodeDefBuilder builder(node_name, "_Arg");
      auto status = builder.Attr("shape", partial_shape)
                        .Attr("T", dtype)
                        .Attr("index", connection.port_number)
                        .Finalize(seg_node);
      VLOG(1) << "Constructing input " << node_name << " for the edge "
              << connection.outside_node_name << ":" << connection.outside_port
              << " -> " << connection.inside_node_name << ":"
              << connection.inside_port;
    } else {
      const string node_name =
          StrCat(IONamePrefixes::kOutputPHName, connection.port_number);
      if (marker_nodes.count(node_name)) {
        VLOG(1) << "Reusing output " << node_name << " for the edge "
                << connection.inside_node_name << ":" << connection.inside_port
                << " -> " << connection.outside_node_name << ":"
                << connection.outside_port;
        continue;
      }
      marker_nodes.insert(node_name);
      auto seg_node = segment_def->add_node();
      NodeDefBuilder builder(node_name, "_Retval");
      auto status =
          builder.Attr("T", dtype)
              .Attr("index", connection.port_number)
              .Input(connection.inside_node_name, connection.inside_port, dtype)
              .Finalize(seg_node);
      VLOG(1) << "Constructing output " << node_name << " for the edge "
              << connection.inside_node_name << ":" << connection.inside_port
              << " -> " << connection.outside_node_name << ":"
              << connection.outside_port;
    }
  }  // for each connection.

  std::unordered_map<int, int> old_to_new_id_map;
  // Copy internal nodes to new graphdef
  string local_scope = subgraph_nodes.front()->name();
  for (const Node* node : subgraph_nodes) {
    local_scope = GetCommonNameScope(local_scope, node->name());
    old_to_new_id_map[node->id()] = segment_def->node_size();
    auto snode = segment_def->add_node();
    *snode = node->def();
    VLOG(2) << "Copying " << snode->name() << " to subgraph";
  }
  // Update the inputs of the new input nodes to point to placeholder nodes.
  for (int i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    if (connection.is_control_edge() || !connection.is_input_edge) continue;
    auto snode =
        segment_def->mutable_node(old_to_new_id_map[connection.inside_id]);
    const string arg_name =
        StrCat(IONamePrefixes::kInputPHName, connection.port_number);
    VLOG(1) << "Updating " << snode->name() << ":" << connection.inside_port
            << " from " << snode->input(connection.inside_port) << " to "
            << arg_name;
    snode->set_input(connection.inside_port, arg_name);
  }
  std::set<string> subgraph_node_names;
  for (const Node* node : subgraph_nodes) {
    subgraph_node_names.insert(node->name());
  }

  // Remove control inputs that are not inside the segment.
  for (int i = 0; i < segment_def->node_size(); ++i) {
    auto snode = segment_def->mutable_node(i);
    const int input_size = snode->input_size();
    int input_idx = 0;
    int actual_input_idx = 0;
    while (input_idx < input_size) {
      TensorId input = ParseTensorName(snode->input(input_idx));
      if (!subgraph_node_names.count(
              string(input.first.data(), input.first.size())) &&
          !IsEngineInput(input.first)) {
        if (input.second == Graph::kControlSlot) {
          VLOG(1) << "... removing control inputs " << input.first
                  << " from subgraph.";
          ++input_idx;
          continue;
        } else {
          return errors::InvalidArgument(
              "Found non control input outside the segment that is not an "
              "engine connection to ",
              snode->name(), ": ", input.first);
        }
      }
      if (actual_input_idx != input_idx) {
        snode->set_input(actual_input_idx, snode->input(input_idx));
      }
      ++input_idx;
      ++actual_input_idx;
    }
    for (int remove = input_size - actual_input_idx; remove > 0; --remove) {
      snode->mutable_input()->RemoveLast();
    }
  }
  return Status::OK();
}

bool OutputEdgeValidator::operator()(const Edge* out_edge) const {
  if (out_edge->IsControlEdge()) return true;
  if (out_edge->src()->type_string() == "Const") {
    VLOG(1) << "--> Need to remove output node " << out_edge->src()->name()
            << " which is a Const.";
    return false;
  }
  return true;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
