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
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/kernels/linalg/einsum_op_impl.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
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

#define TFTRT_INTERNAL_ERROR_AT_NODE(node)                           \
  do {                                                               \
    return errors::Internal("TFTRT::", __FUNCTION__, ":", __LINE__,  \
                            " failed to add TRT layer, at: ", node); \
  } while (0)

#define TFTRT_RETURN_ERROR_IF_NULLPTR(ptr, node) \
  do {                                           \
    if (ptr == nullptr) {                        \
      TFTRT_INTERNAL_ERROR_AT_NODE(node);        \
    }                                            \
  } while (0)

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
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    ADD_LAYER(SHAPE)
    ADD_LAYER(PARAMETRIC_RELU)
    ADD_LAYER(RESIZE)
#endif
#if IS_TRT_VERSION_GE(7, 0, 0, 0)
    ADD_LAYER(TRIP_LIMIT)
    ADD_LAYER(RECURRENCE)
    ADD_LAYER(ITERATOR)
    ADD_LAYER(LOOP_OUTPUT)
    ADD_LAYER(SELECT)
    ADD_LAYER(FILL)
#endif
#if !IS_TRT_VERSION_GE(8, 0, 0, 0)
    // The TRT IRNNv2Layer has been deprecated in favor of the loop API.
    ADD_LAYER(RNN)
#else
    ADD_LAYER(QUANTIZE)
    ADD_LAYER(DEQUANTIZE)
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

class TFAttrs {
 public:
  explicit TFAttrs(const NodeDef& tf_node) {
    for (const auto& attr : tf_node.attr()) {
      attrs_.insert({attr.first, &attr.second});
    }
  }

  bool count(const string& key) const { return attrs_.count(key); }

  AttrValue const* at(const string& key) const {
    if (!attrs_.count(key)) {
      LOG(FATAL) << "Attribute not found: " << key;
    }
    return attrs_.at(key);
  }

  template <typename T>
  T get(const string& key) const;

  template <typename T>
  T get(const string& key, const T& default_value) const {
    return attrs_.count(key) ? this->get<T>(key) : default_value;
  }

 private:
  std::map<string, AttrValue const*> attrs_;
};

template <>
string TFAttrs::get<string>(const string& key) const {
  return this->at(key)->s();
}

template <>
std::vector<int64> TFAttrs::get<std::vector<int64>>(const string& key) const {
  auto attr = this->at(key)->list().i();
  return std::vector<int64>(attr.begin(), attr.end());
}

template <>
std::vector<float> TFAttrs::get<std::vector<float>>(const string& key) const {
  auto attr = this->at(key)->list().f();
  return std::vector<float>(attr.begin(), attr.end());
}

template <>
nvinfer1::DataType TFAttrs::get<nvinfer1::DataType>(const string& key) const {
  nvinfer1::DataType trt_dtype(nvinfer1::DataType::kFLOAT);
  TF_CHECK_OK(TfTypeToTrtType(this->at(key)->type(), &trt_dtype));
  return trt_dtype;
}

template <>
DataType TFAttrs::get<DataType>(const string& key) const {
  return this->at(key)->type();
}

template <>
float TFAttrs::get<float>(const string& key) const {
  return this->at(key)->f();
}

template <>
bool TFAttrs::get<bool>(const string& key) const {
  return this->at(key)->b();
}

template <>
int64 TFAttrs::get<int64>(const string& key) const {
  return this->at(key)->i();
}

// TODO(laigd): use this utility function in more places.
Status RemoveBatchDimension(nvinfer1::Dims* dims) {
  if (dims->nbDims < 2) {
    return errors::InvalidArgument(
        "Dropping batch dimension requires dims with rank>=2.");
  }
  std::copy(dims->d + 1, dims->d + dims->nbDims, dims->d);
  dims->nbDims--;
  return Status::OK();
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
  TF_RETURN_IF_ERROR(
      TensorShapeToTrtDims(shape,
                           /*ignore_first_dim=*/use_implicit_batch, trt_dims));
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

  const int max_nb_dims = nvinfer1::Dims::MAX_DIMS + 1;
  auto compute_output_dims = [use_implicit_batch](
                                 const TRT_TensorOrWeights& input,
                                 int broadcast_num_dims, int* output_dims_array,
                                 nvinfer1::Dims* output_dims) {
    const nvinfer1::Dims input_dims = input.GetTrtDims();
    std::fill(output_dims_array, output_dims_array + max_nb_dims, 1);
    std::copy(input_dims.d, input_dims.d + input_dims.nbDims,
              output_dims_array + broadcast_num_dims - input_dims.nbDims);
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
      output_dims_array[0] = -1;
    }
    // Copy to output dimensions
    if (use_implicit_batch) {
      // Strip batch dimension while copying
      output_dims->nbDims = broadcast_num_dims - 1;
      std::copy(output_dims_array + 1, output_dims_array + broadcast_num_dims,
                output_dims->d);
    } else {
      output_dims->nbDims = broadcast_num_dims;
      std::copy(output_dims_array, output_dims_array + broadcast_num_dims,
                output_dims->d);
    }

    return Status::OK();
  };

  // Compute the output dimensions.
  const int broadcast_num_dims =
      std::max(operand_l.GetTrtDims().nbDims +
                   (use_implicit_batch && operand_l.is_tensor()),
               operand_r.GetTrtDims().nbDims +
                   (use_implicit_batch && operand_r.is_tensor()));
  int output_l[max_nb_dims], output_r[max_nb_dims];
  TF_RETURN_IF_ERROR(compute_output_dims(operand_l, broadcast_num_dims,
                                         output_l, operand_l_new_dims));
  TF_RETURN_IF_ERROR(compute_output_dims(operand_r, broadcast_num_dims,
                                         output_r, operand_r_new_dims));

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
                        ITensorProxyPtr* output, int broadcasted_nbDims) {
  int operand_nbDims = operand->getDimensions().nbDims;
  if (broadcasted_nbDims > operand_nbDims) {
    if (params->validation_only) return Status::OK();
    int n_extra_dims = broadcasted_nbDims - operand_nbDims;
    VLOG(2) << "Dynamic broadcast adding " << n_extra_dims << " leading 1s";
    TF_RETURN_IF_ERROR(params->converter->DynamicReshape(
        operand, {std::make_pair(0, operand_nbDims)}, params, output,
        {n_extra_dims}));
  } else {
    *output = operand;
  }
  return Status::OK();
}

Status BroadcastWeights(std::unique_ptr<TRT_TensorOrWeights>& p,
                        nvinfer1::Dims broadcasted_dims) {
  if (!p->is_weights()) return errors::Internal("Weight input expected");
  if (p->GetTrtDims().nbDims != broadcasted_dims.nbDims) {
    TRT_ShapedWeights weights(p->weights());
    TF_RETURN_IF_ERROR(weights.SetShape(broadcasted_dims));
    p = std::make_unique<TRT_TensorOrWeights>(weights);
  }
  return Status::OK();
}

Status ApplyBroadcast(std::unique_ptr<TRT_TensorOrWeights>& operand,
                      nvinfer1::Dims broadcasted_dims,
                      OpConverterParams* params) {
  if (operand->is_weights()) {
    TF_RETURN_IF_ERROR(BroadcastWeights(operand, broadcasted_dims));
  } else {
    ITensorProxyPtr tensor = nullptr;
    auto is_static_shuffle_compatible = [](nvinfer1::Dims dims) {
      return std::count(dims.d, dims.d + dims.nbDims, -1) <= 1;
    };
    if (is_static_shuffle_compatible(broadcasted_dims)) {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params->converter, *operand, broadcasted_dims,
          params->validation_only, &tensor, params->node_def));
    } else {
      TF_RETURN_IF_ERROR(DynamicBroadcast(operand->tensor(), params, &tensor,
                                          broadcasted_dims.nbDims));
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

  TF_RETURN_IF_ERROR(ApplyBroadcast(operand_l, broadcasted_dims_l, params));
  TF_RETURN_IF_ERROR(ApplyBroadcast(operand_r, broadcasted_dims_r, params));

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
#if !IS_TRT_VERSION_GE(5, 1, 3, 0)
  // TODO(laigd): there is a bug in TensorRT 5.0 library that, if we don't set
  // the data type below, it will always be kFLOAT regardless what the data type
  // of the weights is. Once NVIDIA fixes this bug, we should remove the data
  // type setting logic below and test should still pass.
  trt_tensor->setType(trt_weights.type);
#endif
  return trt_tensor;
}

// Creates a scalar constant and fills with value.
template <typename T>
Status CreateScalarConstant(
    OpConverterParams* params, T value, ITensorProxyPtr* tensor,
    nvinfer1::DataType trt_type = nvinfer1::DataType::kINT32,
    const nvinfer1::Dims& dims = {1, {1}}) {
  TRT_ShapedWeights weights =
      params->weight_store->GetTempWeights(trt_type, dims);
  TF_RETURN_IF_ERROR(weights.SetValues(value));
  *tensor = params->converter->CreateConstantLayer(weights, dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, params->node_def.name());
  params->converter->ProvideQuantizationRange(tensor, value, value);
  return Status::OK();
}

// Creates a constant with the same rank as dims, where each dimension has
// size = 1.
Status CreateBroadcastableScalarConstant(OpConverterParams* params, float value,
                                         const nvinfer1::Dims& dims,
                                         ITensorProxyPtr* tensor,
                                         const char* dtype_attr_name = "T") {
  nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT;  // Default to FP32.
  TFAttrs attrs(params->node_def);
  if (attrs.count(dtype_attr_name)) {
    DataType dtype = attrs.get<DataType>(dtype_attr_name);
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
        "TensorRT does not allow manipulation of the batch dimension, at ",
        node_name);
  }
  // Remove batch dimension if it is implicit.
  *trt_axis = use_implicit_batch ? tf_axis - 1 : tf_axis;
  return Status::OK();
}

inline bool DimsEqual(const nvinfer1::Dims& dim_l,
                      const nvinfer1::Dims& dim_r) {
  if (dim_l.nbDims != dim_r.nbDims) {
    return false;
  }
  for (int i = 0; i < dim_l.nbDims; i++) {
    if (dim_l.d[i] != dim_r.d[i]) {
      return false;
    }
  }
  return true;
}

bool AllLengthsEqual(const std::vector<std::vector<int>>& inputs) {
  if (inputs.size() == 0) return true;
  int length = inputs.at(0).size();
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs.at(i).size() != length) return false;
  }
  return true;
}

inline nvinfer1::Dims GetTrtDimsForTensor(const Tensor& tensor) {
  nvinfer1::Dims dims;
  dims.nbDims = tensor.dims();
  for (int i = 0; i < dims.nbDims; i++) {
    dims.d[i] = tensor.dim_size(i);
  }
  return dims;
}

int64_t Prod(const nvinfer1::Dims& dims) {
  int64_t count = 1;
  for (int d = 0; d < dims.nbDims; ++d) {
    count *= dims.d[d];
  }
  return count;
}

// Returns total number of elements in an ITensor dimension.
// Returns 1 if the number of dims is 0 (the total number is fully determined by
// the batch size).
// Returns -1 if any dimension is known.
int64_t TrtTensorDimsNumElements(const nvinfer1::Dims& dims) {
  if (!HasStaticShape(dims)) return -1;
  return Prod(dims);
}

bool DimsHaveSameSize(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
  return TrtTensorDimsNumElements(lhs) == TrtTensorDimsNumElements(rhs);
}

// Returns whether both dimensions are fully specified and the total number of
// elements equals.
bool AreDimsStaticWithSameSize(const nvinfer1::Dims& lhs,
                               const nvinfer1::Dims& rhs) {
  if (!HasStaticShape(lhs) || !HasStaticShape(rhs)) return false;
  return DimsHaveSameSize(lhs, rhs);
}

bool AreDimsStaticWithDifferentSize(const nvinfer1::Dims& lhs,
                                    const nvinfer1::Dims& rhs) {
  if (!HasStaticShape(lhs) || !HasStaticShape(rhs)) return false;
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

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type) : type_(type) {
  shape_.nbDims = 0;
  shape_.d[0] = 0;
}

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type,
                                     nvinfer1::Dims dims, Tensor tensor)
    : shape_(dims), type_(type), tensor_(tensor) {
  if (dims.nbDims == 0) {
    DCHECK(dims.d[0] == 0 || dims.d[0] == 1);
  }
}

TRT_ShapedWeights::TRT_ShapedWeights(const TRT_ShapedWeights& rhs)
    : shape_(rhs.shape_), type_(rhs.type_), tensor_(rhs.tensor_) {}

int64_t TRT_ShapedWeights::count(nvinfer1::Dims dims) {
  if (dims.nbDims == 0) {
    assert(dims.d[0] == 0 || dims.d[0] == 1);
    return dims.d[0];
  }
  return Prod(dims);
}

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
  return nvinfer1::Weights{type_, GetValues(), count()};
}

template <typename T>
Status TRT_ShapedWeights::SetValues(T value) {
  switch (type_) {
    case nvinfer1::DataType::kFLOAT: {
      float* ptr = tensor_.flat<float>().data();
      std::fill(ptr, ptr + count(), value);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Eigen::half* ptr = tensor_.flat<Eigen::half>().data();
      std::fill(ptr, ptr + count(), Eigen::half(value));
      break;
    }
    case nvinfer1::DataType::kINT32: {
      int32* ptr = tensor_.flat<int32>().data();
      std::fill(ptr, ptr + count(), value);
      break;
    }
    default:
      return errors::InvalidArgument("Unsupported data type ",
                                     tensorflow::tensorrt::DebugString(type_));
  }
  return Status::OK();
}

Status TRT_ShapedWeights::SetShape(nvinfer1::Dims dims) {
  if (this->count() != TRT_ShapedWeights::count(dims)) {
    VLOG(2) << "Changing shape from "
            << tensorflow::tensorrt::DebugString(shape_) << ", to "
            << tensorflow::tensorrt::DebugString(dims);
    return errors::Internal("SetShape would change number of elements");
  }
  shape_ = dims;
  return Status::OK();
}

size_t TRT_ShapedWeights::size_bytes() const {
  size_t data_type_size = -1;
  switch (type_) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      data_type_size = 4;
      break;
    case nvinfer1::DataType::kHALF:
      data_type_size = 2;
      break;
    case nvinfer1::DataType::kINT8:
#if IS_TRT_VERSION_GE(7, 0, 0, 0)
    case nvinfer1::DataType::kBOOL:
#endif
      data_type_size = 1;
      break;
  }
  return this->count() * data_type_size;
}

string TRT_ShapedWeights::DebugString() const {
  return StrCat(
      "TRT_ShapedWeights(shape=", tensorflow::tensorrt::DebugString(shape_),
      ", type=", tensorflow::tensorrt::DebugString(type_),
      ", values=", reinterpret_cast<uintptr_t>(GetValues()), ")");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor)
    : tensor_proxy_ptr_(tensor), initialized_(true), is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::ITensor* tensor,
                                         int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                                         const nvinfer1::Dims& trt_dims,
                                         int batch_size)
    : tensor_proxy_ptr_(new SimpleITensor(trt_dtype, trt_dims)),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
    : weights_(weights), initialized_(true), is_tensor_(false) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
    : tensor_proxy_ptr_(rhs.tensor_proxy_ptr_),
      batch_size_(rhs.batch_size_),
      weights_(rhs.weights_),
      initialized_(rhs.initialized_),
      is_tensor_(rhs.is_tensor_) {}

void TRT_TensorOrWeights::operator=(const TRT_TensorOrWeights& rhs) {
  tensor_proxy_ptr_ = rhs.tensor_proxy_ptr_;
  batch_size_ = rhs.batch_size_;
  weights_ = rhs.weights_;
  initialized_ = rhs.initialized_;
  is_tensor_ = rhs.is_tensor_;
}

ITensorProxyPtr TRT_TensorOrWeights::tensor() const {
  CHECK(is_tensor());
  return tensor_proxy_ptr_;
}

nvinfer1::Dims TRT_TensorOrWeights::GetTrtDims() const {
  if (is_tensor()) {
    return tensor()->getDimensions();
  } else {
    return weights().shape_;
  }
}

Status TRT_TensorOrWeights::GetTfType(DataType* tf_type) const {
  if (is_tensor()) {
    nvinfer1::DataType trt_type = tensor()->getType();
    return TrtTypeToTfType(trt_type, tf_type);
  }
  if (is_weights()) {
    *tf_type = weights().GetTensor().dtype();
    return Status::OK();
  }
  return errors::Internal("The object is probably not initialized");
}

string TRT_TensorOrWeights::DebugString() const {
  string output = "TRT_TensorOrWeights(type=";
  if (is_tensor()) {
    StrAppend(&output, "tensor=", tensorflow::tensorrt::DebugString(tensor()),
              ", batch_size=", batch_size_);
  } else {
    StrAppend(&output, "weights=", weights_.DebugString());
  }
  StrAppend(&output, ")");
  return output;
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
  const int c = iweights.shape_.d[0];
  const int k = iweights.shape_.d[1];
  oweights->shape_.d[0] = k;
  oweights->shape_.d[1] = c;
  const nvinfer1::DimsHW istrides = {1, k};
  const nvinfer1::DimsHW ostrides = {c, 1};
  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      Reorder2({k, c}, static_cast<float const*>(iweights.GetValues()),
               istrides, static_cast<float*>(oweights->GetValues()), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder2({k, c}, static_cast<Eigen::half const*>(iweights.GetValues()),
               istrides, static_cast<Eigen::half*>(oweights->GetValues()),
               ostrides);
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
  const int r = iweights.shape_.d[0];
  const int s = iweights.shape_.d[1];
  // TRT requires GKcRS, while TF depthwise has RSCK where c=1, C=G
  const int c = iweights.shape_.d[2] / num_groups;
  const int k = iweights.shape_.d[3] * num_groups;
  VLOG(2) << "num_groups: " << num_groups << "c" << iweights.shape_.d[2]
          << " then " << c << "k" << iweights.shape_.d[3] << " then " << k
          << "r" << iweights.shape_.d[0] << " then " << r << "s"
          << iweights.shape_.d[1] << " then " << s;
  oweights->shape_.d[0] = k / num_groups;
  oweights->shape_.d[1] = c * num_groups;
  oweights->shape_.d[2] = r;
  oweights->shape_.d[3] = s;
  const nvinfer1::Dims4 istrides = {1, k, s * k * c, c * k};
  const nvinfer1::Dims4 ostrides = {c * r * s, r * s, s, 1};
  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      Reorder4({k, c, r, s}, static_cast<float const*>(iweights.GetValues()),
               istrides, static_cast<float*>(oweights->GetValues()), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder4({k, c, r, s},
               static_cast<Eigen::half const*>(iweights.GetValues()), istrides,
               static_cast<Eigen::half*>(oweights->GetValues()), ostrides);
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
  const int d = iweights.shape_.d[0];
  const int r = iweights.shape_.d[1];
  const int s = iweights.shape_.d[2];
  // TRT requires GKcRS, while TF depthwise has RSCK where c=1, C=G
  const int c = iweights.shape_.d[3] / num_groups;
  const int k = iweights.shape_.d[4] * num_groups;

  VLOG(2) << "num_groups: " << num_groups << ", c: " << iweights.shape_.d[3]
          << " becomes " << c << ", k: " << iweights.shape_.d[4] << " becomes "
          << k << ", d: " << d << ", r: " << r << ", s: " << s;

  oweights->shape_.d[0] = iweights.shape_.d[4];  // k / num_groups;
  oweights->shape_.d[1] = iweights.shape_.d[3];  // c * num_groups;
  oweights->shape_.d[2] = d;
  oweights->shape_.d[3] = r;
  oweights->shape_.d[4] = s;

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
      Reorder5(shape, static_cast<float const*>(iweights.GetValues()), istrides,
               static_cast<float*>(oweights->GetValues()), ostrides);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      Reorder5(shape, static_cast<Eigen::half const*>(iweights.GetValues()),
               istrides, static_cast<Eigen::half*>(oweights->GetValues()),
               ostrides);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type, expected fp32 or fp16 but got "
                 << DebugString(iweights.TrtDType());
  }
}

TRT_ShapedWeights TrtWeightStore::GetTempWeights(nvinfer1::DataType trt_dtype,
                                                 const nvinfer1::Dims& dims) {
  TensorShape shape;
  DataType tf_dtype;
  // TODO(laigd): make it return a status.
  TF_CHECK_OK(TensorShapeUtils::MakeShape(dims.d, dims.nbDims, &shape));
  TF_CHECK_OK(TrtTypeToTfType(trt_dtype, &tf_dtype));
  // TODO(jie): check weights size_bytes. 0 means type error
  Tensor tensor(tf_dtype, shape);
  TRT_ShapedWeights weights(trt_dtype, dims, tensor);
  store_.emplace_back(std::move(tensor));
  return weights;
}

OpConverterParams::OpConverterParams(
    const NodeDef& node_def, const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs, TrtWeightStore* weight_store,
    TrtPrecisionMode precision_mode, bool use_calibration,
    bool use_implicit_batch)
    : node_def(node_def),
      inputs(inputs),
      outputs(outputs),
      validation_only(true),
      weight_store(weight_store),
      precision_mode(precision_mode),
      use_calibration(use_calibration),
      use_implicit_batch(use_implicit_batch) {}

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
      use_implicit_batch(converter->use_implicit_batch()) {}

const std::set<string>* TrtNodeValidator::quantize_ops = new std::set<string>{
    "QuantizeAndDequantizeV2",
    "QuantizeAndDequantizeV3",
    "FakeQuantWithMinMaxVars",
    "FakeQuantWithMinMaxArgs",
};

TrtNodeValidator::TrtNodeValidator(
    const grappler::GraphProperties& graph_properties,
    TrtPrecisionMode precision_mode, bool use_calibration,
    bool use_implicit_batch)
    : graph_properties_(graph_properties),
      precision_mode_(precision_mode),
      use_calibration_(use_calibration),
      use_implicit_batch_(use_implicit_batch) {
  RegisterOpValidators();
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
  if (quantize_ops->count(op)) {
    is_supported_op = (precision_mode_ == TrtPrecisionMode::INT8);
  } else {
    is_supported_op = op_validators_.count(op);
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
      return errors::Internal(
          "Failed to convert input ", src_def.name(),
          " to a TRT_TensorOrWeights: ", status.error_message());
    }
    inputs.push_back(tensor_or_weights);
  }

  OpConverter validator = op_validators_[op];
  OpConverterParams params(node->def(), inputs, /*arg_outputs=*/nullptr,
                           &weight_store_, precision_mode_, use_calibration_,
                           use_implicit_batch_);
  return validator(&params);
}

Status TrtNodeValidator::ConvertConstToWeights(
    const NodeDef& const_node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    TRT_TensorOrWeights* output) {
  std::vector<TRT_TensorOrWeights> outputs;
  OpConverterParams params(const_node_def, inputs, &outputs, &weight_store_,
                           precision_mode_, use_calibration_,
                           use_implicit_batch_);
  Status status = op_validators_["Const"](&params);
  if (status.ok() && output) *output = outputs[0];
  return status;
}

// static
StatusOr<std::unique_ptr<Converter>> Converter::Create(
    TrtPrecisionMode precision_mode, bool use_calibration,
    nvinfer1::ILogger* trt_logger, const bool use_implicit_batch,
    absl::string_view engine_name) {
  std::unique_ptr<Converter> converter = absl::WrapUnique(
      new Converter(precision_mode, use_calibration, trt_logger,
                    use_implicit_batch, engine_name));
  TF_RETURN_IF_ERROR(converter->Init(trt_logger));
  return converter;
}

Converter::Converter(TrtPrecisionMode precision_mode, bool use_calibration,
                     nvinfer1::ILogger* trt_logger,
                     const bool use_implicit_batch,
                     absl::string_view engine_name)
    : precision_mode_(precision_mode),
      use_calibration_(use_calibration),
      use_implicit_batch_(use_implicit_batch),
      engine_name_(engine_name) {
  MaybeInitializeTrtPlugins(trt_logger);
  this->RegisterOpConverters();
}

Status Converter::Init(nvinfer1::ILogger* trt_logger) {
  VLOG(1) << "Creating TensorRT builder";
  trt_builder_.reset(nvinfer1::createInferBuilder(*trt_logger));

  VLOG(1) << "Creating TensorRT network";
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  const uint32_t flags =
      use_implicit_batch_
          ? 0U
          : (1U << static_cast<int>(
                 nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  trt_network_.reset(trt_builder_->createNetworkV2(flags));
#else
  trt_network_.reset(trt_builder_->createNetwork());
#endif
  if (!trt_network_) {
    return errors::Internal("Failed to create TensorRT network object");
  }
  return Status::OK();
}

Status Converter::ConvertNode(const NodeDef& node_def) {
  std::vector<TRT_TensorOrWeights> inputs, outputs;
  TF_RETURN_IF_ERROR(this->GetInputs(node_def, &inputs));

  OpConverterParams params(this, node_def, inputs, &outputs, &weight_store_);
  const string& op = node_def.op();
  auto itr = op_registry_.find(op);
  if (itr == op_registry_.end()) {
    return errors::Unimplemented("No converter registered for op: ", op);
  }
  OpConverter op_converter = itr->second;
  TF_RETURN_IF_ERROR(op_converter(&params));

  for (size_t i = 0; i < outputs.size(); ++i) {
    TRT_TensorOrWeights& output = outputs[i];
    string output_name = node_def.name();
    if (i != 0) absl::StrAppend(&output_name, ":", i);
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
      return Status(status.code(),
                    StrCat("Failed to add output for node ", node_def.name(),
                           ": ", status.error_message()));
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
      return Status(status.code(),
                    StrCat("Batch size doesn't match for tensor ", name, ": ",
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
    return Status(status.code(), StrCat("Failed to add input tensor ", name,
                                        ": ", status.error_message()));
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
      ITensorProxyPtr output_tensor = layer->getOutput(0);
      MarkQuantizationRangesAsInferrable(&tensor, &output_tensor);
      tensor = output_tensor;
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

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
// An algorithm selector that always returns a specific ID for selectAlgorithms.
// This is used to support the implementation of using environment variable
// `TF_TRT_FIXED_ALGORITHM_ID` for debugging TensorRT.
class StaticAlgorithmSelector : public nvinfer1::IAlgorithmSelector {
 private:
  int32_t algorithm_id_;

 public:
  StaticAlgorithmSelector(int32_t algorithm_id) : algorithm_id_(algorithm_id) {}

  // Returns value in [0, nbChoices] for a valid algorithm.
  int32_t selectAlgorithms(const nvinfer1::IAlgorithmContext& algoContext,
                           const nvinfer1::IAlgorithm* const* algoChoices,
                           int32_t nbChoices,
                           int32_t* selection) noexcept override {
    // TensorRT always provides more than zero number of algorithms
    // in selectAlgorithms.
    assert(nbChoices > 0);

    // making sure that the requested TRT algorithm ID doesn't go above the
    // max value accepted.
    selection[0] = std::min(algorithm_id_, nbChoices);
    return 1;
  }

  // Called by TensorRT to report choices it made.
  void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
                        const nvinfer1::IAlgorithm* const* algoChoices,
                        int32_t nbAlgorithms) noexcept override {
  }  // do nothing
};
#endif

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

  VLOG(1) << "Configuring TensorRT builder";
  trt_builder_->setMaxBatchSize(max_batch_size);
  trt_builder_->setGpuAllocator(allocator);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Create a network configuration and use it to build a TRT engine.
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
      trt_builder_->createBuilderConfig());
  builder_config->setMaxWorkspaceSize(max_workspace_size_bytes);

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
  static int32_t trt_algorithm_id = [] {
    int64 trt_algorithm_id;
    TF_CHECK_OK(tensorflow::ReadInt64FromEnvVar("TF_TRT_FIXED_ALGORITHM_ID",
                                                /*default_val=*/-1,
                                                &trt_algorithm_id));
    return static_cast<int32_t>(trt_algorithm_id);
  }();

  if (trt_algorithm_id >= 0) {
    VLOG(1) << "Forcing TRT algorithm selection to: ID=" << trt_algorithm_id;
    StaticAlgorithmSelector trt_algorithm_selector(trt_algorithm_id);
    builder_config->setAlgorithmSelector(&trt_algorithm_selector);
  }
#endif

  if (precision_mode_ == TrtPrecisionMode::FP16) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (precision_mode_ == TrtPrecisionMode::INT8) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (use_calibration_) {
      builder_config->setInt8Calibrator(calibrator);
    } else {
      builder_config->setInt8Calibrator(nullptr);
    }
  }
  if (!use_implicit_batch_ && profiles) {
    TF_RETURN_IF_ERROR(profiles->ConfigureBuilder(
        trt_builder_.get(), builder_config.get(), network()));
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
#else
  trt_builder_->setMaxWorkspaceSize(max_workspace_size_bytes);
  if (precision_mode_ == TrtPrecisionMode::FP16) {
    trt_builder_->setFp16Mode(true);
  } else if (precision_mode_ == TrtPrecisionMode::INT8) {
    // Setting FP16 mode as well allows TRT to also consider FP16 kernels and
    // use them in situations where they are faster than INT8 or where INT8 is
    // not supported for a given layer.
    trt_builder_->setFp16Mode(true);
    trt_builder_->setInt8Mode(true);
    if (use_calibration_) {
      trt_builder_->setInt8Calibrator(calibrator);
    } else {
      trt_builder_->setInt8Calibrator(nullptr);
    }
  }

  VLOG(1) << "Building TensorRT engine";
  engine->reset(trt_builder_->buildCudaEngine(*network()));
#endif
  if (engine->get() == nullptr) {
    return errors::Internal("Failed to build TensorRT engine");
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "TRT engine created";
    int nbBindings = (*engine)->getNbBindings();
    VLOG(2) << "Number of engine bindings: " << nbBindings;
    for (int i = 0; i < nbBindings; i++) {
      VLOG(2) << "Binding " << i << " name: " << (*engine)->getBindingName(i);
    }
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

  ITensorProxyPtr shuffle_tensor = layer->getOutput(0);
  MarkQuantizationRangesAsInferrable(&input_tensor, &shuffle_tensor);
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
      auto inp = static_cast<float const*>(weights.GetValues());
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = *result.first;
      *out_max = *result.second;
      break;
    }
    case nvinfer1::DataType::kHALF: {
      auto inp = static_cast<Eigen::half const*>(weights.GetValues());
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = static_cast<float>(*result.first);
      *out_max = static_cast<float>(*result.second);
      break;
    }
    case nvinfer1::DataType::kINT32: {
      auto inp = static_cast<int const*>(weights.GetValues());
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
    SetLayerNameHelper(layer, engine_name_,
                       absl::StrCat(node_def.name(), "-",
                                    absl::string_view(origin_node_name.value()),
                                    "-", sub_op_suffix));
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
                             const nvinfer1::Dims& dims,
                             const bool validation_only,
                             ITensorProxyPtr* tensor, const NodeDef& node_def,
                             absl::optional<int> op_instance,
                             absl::optional<std::string> origin_node_name) {
  const nvinfer1::Dims input_dims = input.GetTrtDims();
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
  if (Prod(dims) > 0 && AreDimsStaticWithDifferentSize(input_dims, dims)) {
    return errors::InvalidArgument(
        "Incompatible shapes: ", DebugString(input_dims), " vs. ",
        DebugString(dims));
  }
  // ConstantLayer requires static shapes (cannot infer -1).
  if (input.is_weights() && !HasStaticShape(dims)) {
    return errors::InvalidArgument("Shape is not fully defined: ",
                                   DebugString(dims));
  }
  if (validation_only) {
    *tensor = nullptr;
    return Status::OK();
  }

  TFTRT_RETURN_ERROR_IF_NULLPTR(converter, "converter is nullptr");
  if (input.is_tensor()) {
    if (DimsEqual(input_dims, dims)) {
      *tensor = input.tensor();
    } else {
      nvinfer1::IShuffleLayer* layer =
          converter->network()->addShuffle(*input.tensor()->trt_tensor());
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Reshape");
      converter->SetLayerName(layer, node_def, "shuffle", op_instance,
                              origin_node_name);
      layer->setReshapeDimensions(dims);
      ITensorProxyPtr input_tensor = input.tensor();
      ITensorProxyPtr output_tensor = layer->getOutput(0);
      converter->MarkQuantizationRangesAsInferrable(&input_tensor,
                                                    &output_tensor);
      *tensor = output_tensor;
    }
  } else {
    *tensor = converter->CreateConstantLayer(input.weights(), dims);
    TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, "TF-TRT Internal Reshape");
    if (converter->precision_mode() == TrtPrecisionMode::INT8 &&
        !converter->use_calibration()) {
      // If we are in int8 mode and not calibrating, we need to explicitly set a
      // quantization range for the output tensor of the IConstantLayer. Here we
      // set the range to [min(weights), max(weights)].
      float min_range = 0.0f;
      float max_range = 0.0f;
      TF_RETURN_IF_ERROR(
          converter->GetWeightRange(input.weights(), &min_range, &max_range));
      // Avoid setting range to 0 because TRT will throw an error. If the
      // weights are zero then the range doesn't matter: using 127.0f should
      // ensure the quantized weight will be exactly zero.
      if (min_range == 0.0f && max_range == 0.0f) {
        min_range = -127.0f;
        max_range = 127.0f;
      }
      converter->ProvideQuantizationRange(tensor, min_range, max_range);
    }
  }
  return Status::OK();
}

void Converter::MarkQuantizationRangesAsInferrable(ITensorProxyPtr* input,
                                                   ITensorProxyPtr* output) {
  if ((*input)->is_trt_tensor()) {
    quantization_infer_.push_back(
        {(*input)->trt_tensor(), (*output)->trt_tensor()});
    quantization_infer_.push_back(
        {(*output)->trt_tensor(), (*input)->trt_tensor()});
  } else if ((*input)->is_simple_tensor()) {
    quantization_infer_proxy_.push_back({input, output});
    quantization_infer_proxy_.push_back({output, input});
  }
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

namespace {

bool IsConvolution(const nvinfer1::ILayer* layer) {
  return layer->getType() == nvinfer1::LayerType::kCONVOLUTION;
}

bool IsScale(const nvinfer1::ILayer* layer) {
  return layer->getType() == nvinfer1::LayerType::kSCALE;
}

bool IsClipOrRelu(const nvinfer1::ILayer* layer) {
  if (layer->getType() != nvinfer1::LayerType::kACTIVATION) {
    return false;
  }
  auto activation_type = static_cast<const nvinfer1::IActivationLayer*>(layer)
                             ->getActivationType();
#if IS_TRT_VERSION_GE(5, 1, 2, 0)
  return activation_type == nvinfer1::ActivationType::kRELU ||
         activation_type == nvinfer1::ActivationType::kCLIP;
#else
  return activation_type == nvinfer1::ActivationType::kRELU;
#endif
}

bool IsAdd(const nvinfer1::ILayer* layer) {
  if (layer->getType() != nvinfer1::LayerType::kELEMENTWISE) {
    return false;
  }
  auto operation =
      static_cast<const nvinfer1::IElementWiseLayer*>(layer)->getOperation();
  return operation == nvinfer1::ElementWiseOperation::kSUM;
}

}  // namespace

void Converter::MaybeApplyQuantizationRanges() {
  if (precision_mode() != TrtPrecisionMode::INT8) return;

  // Infer ranges across marked ops.
  PropagateQuantizationRanges();
  // Apply ranges.
#if IS_TRT_VERSION_GE(5, 0, 0, 0)
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
#endif

  if (use_calibration()) return;
#if !IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Attempt to find tensors that are missing ranges, and set the corresponding
  // layer's precision to FP16 to avoid Builder::buildCudaEngine() failing.
  // This is only needed for TensorRT 5 and before because
  // TensorRT6 falls to FP16 internally.
  // TensorRT doesn't need ranges for intermediate tensors when layers are fused
  // so find fused layers first.
  // Get all tensors from network and deduce fused ops.
  std::map<nvinfer1::ILayer*, std::vector<nvinfer1::ILayer*>> layer_consumers;
  std::map<ITensorProxyPtr*, nvinfer1::ILayer*> tensor_layer;
  std::set<ITensorProxyPtr*> all_tensors;
  for (int i = 0; i < this->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = this->network()->getLayer(i);
    layer_consumers[layer] = {};
    for (int j = 0; j < layer->getNbInputs(); j++) {
      ITensorProxyPtr input_tensor = layer->getInput(j);
      all_tensors.insert(&input_tensor);
    }
    for (int j = 0; j < layer->getNbOutputs(); j++) {
      ITensorProxyPtr output_tensor = layer->getOutput(j);
      tensor_layer[&output_tensor] = layer;
      all_tensors.insert(&output_tensor);
    }
  }
  for (int i = 0; i < this->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = this->network()->getLayer(i);
    layer_consumers[layer] = {};
    for (int j = 0; j < layer->getNbInputs(); j++) {
      ITensorProxyPtr input_tensor = layer->getInput(j);
      auto input_layer = tensor_layer.find(&input_tensor);
      if (input_layer != tensor_layer.end()) {
        auto consumed_layer = layer_consumers.find(input_layer->second);
        if (consumed_layer != layer_consumers.end()) {
          consumed_layer->second.push_back(layer);
        }
      }
      all_tensors.insert(&input_tensor);
    }
  }
  // Identify fused tensors.
  // Conv+BiasAdd+Add+Activation(Clip or Relu), Conv+BiasAdd+Add,
  // Conv+BiasAdd+Activation(Clip or Relu), Conv+BiasAdd,
  // Conv+Activation(Clip or Relu) are fused.
  std::set<ITensorProxyPtr*> fused_tensors;
  typedef std::function<bool(const nvinfer1::ILayer*)> matcher;
  const std::vector<std::pair<string, std::vector<matcher>>> fused_patterns = {
      {"Fused Conv+Bias+Add+Activation",
       {
           IsConvolution,
           IsScale,
           IsAdd,
           IsClipOrRelu,
       }},
      {"Fused Conv+Bias+Add",
       {
           IsConvolution,
           IsScale,
           IsAdd,
       }},
      {"Fused Conv+Bias+Activation",
       {
           IsConvolution,
           IsScale,
           IsClipOrRelu,
       }},
      {"Fused Conv+Bias",
       {
           IsConvolution,
           IsScale,
       }},
      {"Fused Conv+Activation",
       {
           IsConvolution,
           IsClipOrRelu,
       }},
  };
  for (int i = 0; i < this->network()->getNbLayers(); i++) {
    for (const auto& pattern : fused_patterns) {
      size_t last_matcher = pattern.second.size() - 1;
      nvinfer1::ILayer* layer = this->network()->getLayer(i);
      // We should skip this layer if its outputs are already marked as fused,
      // but all the current patterns start with a convolution and are ordered
      // in decreasing pattern length, so that is not necessary (yet).
      std::vector<nvinfer1::ILayer*> fused_candidates;
      for (size_t index = 0; index <= last_matcher; ++index) {
        if ((!pattern.second[index](layer)) ||
            (index < last_matcher && layer_consumers[layer].size() != 1)) {
          fused_candidates.clear();
          break;
        }
        if (index < last_matcher) {
          fused_candidates.push_back(layer);
          layer = layer_consumers[layer].front();
        }
      }
      if (!fused_candidates.empty()) {
        VLOG(1) << pattern.first;
        for (const auto& fused_layer : fused_candidates) {
          for (int i = 0; i < fused_layer->getNbOutputs(); i++) {
            VLOG(1) << "  Fused output tensor:"
                    << fused_layer->getOutput(i)->getName();
            ITensorProxyPtr output_tensor = fused_layer->getOutput(i);
            fused_tensors.insert(&output_tensor);
          }
        }
        break;  // Don't try other patterns on this layer.
      }
    }
  }
  // Find tensors with no ranges that are not fused and force their layers to
  // not be quantized.
  for (auto tensor : all_tensors) {
    if (!quantization_ranges_.count(tensor) &&
        fused_tensors.find(tensor) == fused_tensors.end()) {
      // Note: there may be some warnings for "(Unnamed ITensor* N)". These
      // are tensors which are created internally by TF-TRT. The ranges for
      // these unnamed ITensors are always inferred from user provided ranges,
      // thus there will also be a warning for the range(s) the user missed.
      LOG_WARNING_WITH_PREFIX << "Quantization range was not found for "
                              << (*tensor)->getName() << ". "
                              << "Setting invalid quantization range.";
      // Set the range to something unusable so the engine will fail if it
      // tries to actually use the tensor's range.
      (*tensor)->setDynamicRange(0, 0);
      auto layer = tensor_layer.find(tensor);
      // If the tensor is the output of a layer, set the layer's precision
      // to fp16 so that it isn't quantized.
      // Shuffle doesn't support setting precision.
      if (layer != tensor_layer.end() &&
          layer->second->getType() != nvinfer1::LayerType::kSHUFFLE) {
        VLOG(1) << "And setting layer " << layer->second->getName()
                << " precision to fp16.";
        layer->second->setPrecision(nvinfer1::DataType::kHALF);
      }
    }
  }
#endif
}

void Converter::PropagateQuantizationRanges() {
  // Propagate ranges across edges in quantization_infer_ until no new
  // information is added.
  // Note: this function modifies quantization_infer_, it might be better to
  // modify a copy instead if we for some reason need quantization_infer_
  // later.
  bool information_added = true;
  while (information_added) {
    // Propogate for real tensors.
    information_added = false;
    for (auto it = quantization_infer_.begin();
         it != quantization_infer_.end();) {
      auto input_tensor_range = quantization_ranges_.find(it->first);
      auto output_tensor_range = quantization_ranges_.find(it->second);
      if (input_tensor_range != quantization_ranges_.end() &&
          output_tensor_range == quantization_ranges_.end()) {
        // Input has range but output doesn't: copy range
        // TODO(laigd): consider reporting error if it a different range is
        // already set.
        quantization_ranges_[it->second] = input_tensor_range->second;
        information_added = true;
        VLOG(1) << "Copy quantization range: " << it->first->getName() << " -> "
                << it->second->getName();
      }
      // We can remove edges when the output range is known
      if (quantization_ranges_.find(it->second) != quantization_ranges_.end()) {
        it = quantization_infer_.erase(it);
      } else {
        ++it;
      }
    }
    // Propogate for proxy.
    information_added = false;
    for (auto it = quantization_infer_proxy_.begin();
         it != quantization_infer_proxy_.end();) {
      auto input_tensor_range = quantization_ranges_proxy_.find(it->first);
      auto output_tensor_range = quantization_ranges_proxy_.find(it->second);
      if (input_tensor_range != quantization_ranges_proxy_.end() &&
          output_tensor_range == quantization_ranges_proxy_.end()) {
        // Input has range but output doesn't: copy range
        // TODO(laigd): consider reporting error if it a different range is
        // already set.
        quantization_ranges_proxy_[it->second] = input_tensor_range->second;
        information_added = true;
        VLOG(1) << "Copy quantization range: " << (*it->first)->getName()
                << " -> " << (*it->second)->getName();
        std::cout << "Copy quantization range: " << (*it->first)->getName()
                  << " -> " << (*it->second)->getName();
      }
      // We can remove edges when the output range is known
      if (quantization_ranges_proxy_.find(it->second) !=
          quantization_ranges_proxy_.end()) {
        it = quantization_infer_proxy_.erase(it);
      } else {
        ++it;
      }
    }
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

enum class TrtInputArg { kTensor = 1, kWeight = 2, kBoth = 3 };

// Checks that the number of inputs match, and enforces that the inputs marked
// as weights are constant. Inputs are allowed to be both weight and tensor.
Status CheckInputsWeights(
    const OpConverterParams& params,
    const std::vector<std::pair<string, TrtInputArg>>& expected_inputs) {
  const auto& inputs = params.inputs;
  const auto& node_def = params.node_def;
  if (inputs.size() != expected_inputs.size()) {
    return errors::InvalidArgument(
        node_def.op(), " got ", inputs.size(), " inputs but expected ",
        expected_inputs.size(), ", at ", node_def.name());
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (expected_inputs[i].second == TrtInputArg::kWeight &&
        inputs.at(i).is_tensor()) {
      return errors::Unimplemented("The input \"", expected_inputs[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a constant, at ", node_def.name());
    }
    // TODO(tfeher): Remove this check and provide a method to automatically
    // retrieve an input as a tensor, converting via CreateConstantLayer if it
    // was originally a weight. We will want a caching mechanism to prevent many
    // duplicate constants from being created.
    if (expected_inputs[i].second == TrtInputArg::kTensor &&
        inputs.at(i).is_weights()) {
      return errors::Unimplemented("The input \"", expected_inputs[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a tensor, at ", node_def.name());
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
  TFAttrs attrs(node_def);
  if (!attrs.count(type_attr_name)) {
    return errors::InvalidArgument("Attribute with name ", type_attr_name,
                                   " not found.");
  }
  *tf_type = attrs.get<DataType>(type_attr_name);
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
    return errors::Unimplemented("Data type ", DataTypeString(tf_type),
                                 " is not supported for ", node_def.op(),
                                 ", must be one of [", allowed_types_string,
                                 "], at ", node_def.name());
  }
  return Status::OK();
}

// ****************************************************************************
// Constant folding functions for weights.
// TODO(laigd): we should probably use eigen directly.
// *****************************************************************************
struct LambdaFactory {
  enum class OP_CATEGORY : int { RSQRT = 0, NEG, RECIP };
  OP_CATEGORY op;

  template <typename T>
  std::function<T(T)> unary() {
    switch (op) {
      case OP_CATEGORY::RSQRT: {
        VLOG(2) << "RSQRT GETS DONE";
        return [](T t) -> T { return 1.0 / std::sqrt(t); };
      }
      case OP_CATEGORY::NEG:
        return [](T t) -> T { return -t; };
      case OP_CATEGORY::RECIP:
        return [](T t) -> T { return 1.0 / t; };
      default:
        LOG(ERROR) << "Not supported op for unary: " << static_cast<int>(op);
        return nullptr;
    }
  }
};

template <>
std::function<Eigen::half(Eigen::half)> LambdaFactory::unary<Eigen::half>() {
  switch (op) {
    case OP_CATEGORY::RSQRT: {
      VLOG(2) << "RSQRT GETS DONE";
      return [](Eigen::half t) {
        return Eigen::half(1.0 / std::sqrt(static_cast<float>(t)));
      };
    }
    case OP_CATEGORY::NEG:
      return [](Eigen::half t) { return -t; };
    case OP_CATEGORY::RECIP:
      return [](Eigen::half t) {
        return Eigen::half(1.0 / static_cast<float>(t));
      };
    default:
      LOG(ERROR) << "Not supported op for unary: " << static_cast<int>(op);
      return nullptr;
  }
}

Status UnaryCompute(const TRT_ShapedWeights& iweights,
                    TRT_ShapedWeights* oweights, LambdaFactory unary_op) {
  CHECK(iweights.TrtDType() == oweights->TrtDType());
  switch (iweights.TrtDType()) {
    case nvinfer1::DataType::kFLOAT: {
      auto inp = static_cast<float const*>(iweights.GetValues());
      auto oup = static_cast<float*>(oweights->GetValues());
      std::transform(inp, inp + iweights.count(), oup, unary_op.unary<float>());
      break;
    }
    case nvinfer1::DataType::kHALF: {
      auto inp = static_cast<Eigen::half const*>(iweights.GetValues());
      auto oup = static_cast<Eigen::half*>(oweights->GetValues());
      std::transform(inp, inp + iweights.count(), oup,
                     unary_op.unary<Eigen::half>());
      break;
    }
    default:
      return errors::Unimplemented("Data type not supported: ",
                                   DebugString(iweights.TrtDType()));
  }
  return Status::OK();
}

// Before TRT 5.1.3, we have to calculate padding for convolutions ourselves.
Status Conv2DPaddingHelper(OpConverterParams* params, const TFAttrs& attrs,
                           const nvinfer1::DimsHW& kernel_size,
                           const nvinfer1::DimsHW& dilation,
                           const nvinfer1::DimsHW& stride,
                           const std::vector<int64_t>& input_dims,
                           ITensorProxyPtr tensor,
                           std::vector<std::pair<int, int>>* padding,
                           ITensorProxyPtr* padded_tensor) {
  if (attrs.get<string>("padding") == "SAME") {
    nvinfer1::DimsHW effective_kernel_size = kernel_size;
    effective_kernel_size.h() += (kernel_size.h() - 1) * (dilation.h() - 1);
    effective_kernel_size.w() += (kernel_size.w() - 1) * (dilation.w() - 1);
    *padding = CreateSamePadding(stride, effective_kernel_size, input_dims);
  } else {
    *padding = {{0, 0}, {0, 0}};
  }

  // Handle asymmetric padding. TensorRT 5.1 added support for asymmetric
  // padding via setPrePadding and setPostPadding. Due to a bug in 5.1.2, we can
  // only use asymmetric padding in convolutions with 5.1.3+. But in 5.1.3, we
  // will always use setPaddingMode for simplicity.
  if ((*padding)[0].first != (*padding)[0].second ||
      (*padding)[1].first != (*padding)[1].second) {
    auto pad_layer = params->converter->network()->addPadding(
        *tensor->trt_tensor(),
        nvinfer1::DimsHW((*padding)[0].first, (*padding)[1].first),
        nvinfer1::DimsHW((*padding)[0].second, (*padding)[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, params->node_def.name());
    params->converter->SetLayerName(pad_layer, params->node_def, "pad");
    ITensorProxyPtr output_tensor = pad_layer->getOutput(0);
    params->converter->MarkQuantizationRangesAsInferrable(&tensor,
                                                          &output_tensor);
    *padding = {{0, 0}, {0, 0}};
    tensor = output_tensor;
  }
  *padded_tensor = tensor;
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
  auto output_sizes_values = static_cast<int*>(weights.GetValues());
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
    TF_RETURN_IF_ERROR(CheckInputsWeights(
        *params,
        {{"input_sizes", true}, {"filter", true}, {"out_backprop", false}}));
    backprop_output_size = inputs.at(0);
    tensor = inputs.at(2).tensor();
    if (!HasStaticShape(tensor->getDimensions())) {
      // TODO(tfeher): Allow dynamic input. We need to implement padding
      // correction for dynamic shapes in this case.
      return errors::Unimplemented(
          "Conv2dBackpropInput does not support input with unknown shape, at ",
          node_def.name());
    }
  } else {
    TF_RETURN_IF_ERROR(
        CheckInputsWeights(*params, {{"input", false}, {"filter", true}}));
    tensor = inputs.at(0).tensor();
  }
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  TRT_ShapedWeights weights_rsck = inputs.at(1).weights();
  if (weights_rsck.shape_.nbDims != 4) {
    return errors::InvalidArgument("Conv2D expects kernel of dimension 4, at " +
                                   node_def.name());
  }
  TFAttrs attrs(node_def);
  auto data_format = attrs.get<string>("data_format");
  int c_index = (data_format == "NHWC") ? 3 : 1;
  int h_index = (data_format == "NHWC") ? 1 : 2;
  int w_index = (data_format == "NHWC") ? 2 : 3;
  auto tf_dilations = attrs.get<std::vector<int64>>("dilations");
  if (tf_dilations.size() != 4) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const nvinfer1::DimsHW dilation(tf_dilations[h_index], tf_dilations[w_index]);
  if (is_conv2d_backprop_input && (dilation.d[0] != 1 || dilation.d[1] != 1)) {
    return errors::Unimplemented(
        "Dilation with Conv2DBackpropInput (conv2d_transpose) is not supported",
        ", at ", node_def.name());
  }

  const auto tf_stride = attrs.get<std::vector<int64>>("strides");
  if (tf_stride.size() != 4) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  // Channel dim must be static for DepthwiseConv2dNative since we use that
  // value for num_groups at build time.
  if (!params->use_implicit_batch && tensor->getDimensions().d[c_index] == -1) {
    return errors::InvalidArgument("Channel dimension must be static, at ",
                                   node_def.name());
  }
  string padding = attrs.get<string>("padding");
  if (padding != "SAME" && padding != "VALID") {
    return errors::Unimplemented(padding +
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
  TRT_ShapedWeights weights =
      params->weight_store->GetTempWeights(weights_rsck);
  ReorderRSCKToKCRS(weights_rsck, &weights, num_groups);
  TRT_ShapedWeights biases(weights.TrtDType());
  const int output_axis = is_conv2d_backprop_input ? 1 : 0;
  const int noutput = weights.shape_.d[output_axis] * num_groups;
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = weights.shape_.d[2];
  kernel_size.w() = weights.shape_.d[3];

// Before TRT 5.1.3, we have to calculate padding ourselves.
#if !IS_TRT_VERSION_GE(5, 1, 3, 0)
  std::vector<std::pair<int, int>> padding;
  std::vector<int64_t> input_dims;
  if (is_conv2d_backprop_input) {
    // For backprop, calculate padding based on "input_sizes" input, which
    // actually corresponds to output size. ("input_sizes" makes sense in the
    // context of Conv2DBackpropInput).
    input_dims =
        GetSpatialDimsFromOutputSizes(backprop_output_size, h_index, w_index);
  } else {
    // Use 1 and 2 because tensor_dim has the dimensions of the transposed
    // input.
    input_dims = {static_cast<int>(tensor_dim.d[1]),
                  static_cast<int>(tensor_dim.d[2])};
  }
  ITensorProxyPtr padded_tensor = nullptr;
  TF_RETURN_IF_ERROR(Conv2DPaddingHelper(params, attrs, kernel_size, dilation,
                                         stride, input_dims, tensor, &padding,
                                         &padded_tensor));
  tensor = padded_tensor;
#endif

  // Add convolution.
  nvinfer1::ILayer* conv_layer = nullptr;
  if (is_conv2d_backprop_input) {
    nvinfer1::IDeconvolutionLayer* layer =
        params->converter->network()->addDeconvolution(
            *tensor->trt_tensor(), noutput, kernel_size,
            weights.GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStride(stride);
// TensorRT 5.1.3 added support for padding modes.
#if IS_TRT_VERSION_GE(5, 1, 3, 0)
    // VALID padding is the default TRT behavior.
    if (attrs.get<string>("padding") == "SAME") {
      // SAME_UPPER means that post padding is preferred.
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
#else
    layer->setPadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
#endif
    layer->setNbGroups(num_groups);
    conv_layer = layer;
  } else {
    nvinfer1::IConvolutionLayer* layer =
        params->converter->network()->addConvolution(
            *tensor->trt_tensor(), noutput, kernel_size,
            weights.GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStride(stride);
#if IS_TRT_VERSION_GE(5, 1, 3, 0)
    if (attrs.get<string>("padding") == "SAME") {
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
#else
    layer->setPadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
#endif
    layer->setNbGroups(num_groups);
    layer->setDilation(dilation);
    conv_layer = layer;
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
          ") for op ", node_def.name());
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
  const int* weights_ptr = static_cast<int*>(weights.GetValues());
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

#if !IS_TRT_VERSION_GE(7, 1, 3, 4)
  // TensorRT versions before 7.1.3.4 is slow transposing large tensors.
  // So check tensor size, and don't convert if it is too large.
  constexpr int64_t kMaxEfficientTranspose = 2500000;
  int64_t tensor_size = TrtTensorDimsNumElements(input_tensor->getDimensions());
  if (!AllowInefficientTranspose() && tensor_size > kMaxEfficientTranspose) {
    return errors::Unimplemented(StrCat("Transpose too large:", tensor_size));
  }
#endif

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
  if (HasStaticShape(inputs.at(0).GetTrtDims())) {
    if (params->validation_only) return Status::OK();
    nvinfer1::Dims input_dims = inputs.at(0).GetTrtDims();
    nvinfer1::Dims output_dims{1, {input_dims.nbDims}};
    // Create a const node with the values of output_dims
    TRT_ShapedWeights weight = params->weight_store->GetTempWeights(
        nvinfer1::DataType::kINT32, output_dims);
    int32* values_ptr = static_cast<int32*>(weight.GetValues());
    std::copy(input_dims.d, input_dims.d + input_dims.nbDims, values_ptr);
    auto output = params->converter->CreateConstantLayer(weight, output_dims);
    params->outputs->push_back(TRT_TensorOrWeights(output));
    return Status::OK();
  }
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  if (params->validation_only) return Status::OK();
  nvinfer1::IShapeLayer* shape_layer = params->converter->network()->addShape(
      *inputs.at(0).tensor()->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(shape_layer, params->node_def.name());
  params->converter->SetLayerName(shape_layer, params->node_def, "shape");
  params->outputs->push_back(TRT_TensorOrWeights(shape_layer->getOutput(0)));
  return Status::OK();
#else
  return errors::Unavailable(
      "Shape op conversion requires TensorRT 6 or above");
#endif
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
        " mode, at ",
        params->node_def.name());
  }
  if (!IS_TRT_VERSION_GE(7, 1, 3, 0)) {
    // While officially TRT supports shape value input as of TRT 6, there are
    // problems with shape input handling that cause networks converted with
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
        "Reshape with dynamic input requires 1D input tensor, at ",
        params->node_def.name());
  }
  if (params->validation_only) return Status::OK();
  nvinfer1::IShuffleLayer* layer = params->converter->network()->addShuffle(
      *input_tensor.tensor()->trt_tensor());
  VLOG(2) << "ConvertReshape setInput (1) "
          << DebugString(inputs.at(1).tensor()->getDimensions());
  layer->setInput(1, *inputs.at(1).tensor()->trt_tensor());
  ITensorProxyPtr in_tensor = input_tensor.tensor();
  ITensorProxyPtr out_tensor = layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(&in_tensor,
                                                        &out_tensor);
  params->outputs->push_back(TRT_TensorOrWeights(out_tensor));
  return Status::OK();
}

// Converts Reshape in explicit batch mode if the input has static (known) dims.
Status ConvertStaticReshapeForExplicitBatchMode(
    OpConverterParams* params, const int* output_dims, int num_dims,
    ITensorProxyPtr* output_tensor) {
  nvinfer1::Dims dims;
  dims.nbDims = num_dims;
  std::copy(output_dims, output_dims + num_dims, dims.d);
  return PrepareTensorForShape(params->converter, params->inputs.at(0), dims,
                               params->validation_only, output_tensor,
                               params->node_def);
}

// Converts Reshape in implicit batch mode. The input has static (known) dims.
Status ConvertStaticReshapeForImplicitBatchMode(
    OpConverterParams* params, const int* output_shape_dims,
    int output_shape_dims_count, ITensorProxyPtr* output_tensor) {
  const auto& inputs = params->inputs;
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);
  const int input_batch_dim = input_tensor.batch_size();
  const int output_batch_dim =
      (output_shape_dims_count > 0) ? output_shape_dims[0] : 0;

  const nvinfer1::Dims input_nonbatch_dims = input_tensor.GetTrtDims();
  nvinfer1::Dims output_nonbatch_dims;
  output_nonbatch_dims.nbDims = output_shape_dims_count - 1;
  for (int i = 1; i < output_shape_dims_count; i++) {
    output_nonbatch_dims.d[i - 1] = output_shape_dims[i];
  }

  VLOG(1) << "input_batch_dim=" << input_batch_dim
          << ", input_nonbatch_dims=" << DebugString(input_nonbatch_dims)
          << "\nresult_batch_dim=" << output_batch_dim
          << ", result_nonbatch_dims=" << DebugString(output_nonbatch_dims);

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
    const string msg =
        StrCat("Reshape on batch dimension is not supported, at ",
               params->node_def.name(), ". input_batch_dim=", input_batch_dim,
               ", ", DebugString(input_nonbatch_dims),
               "; output_batch_dim=", output_batch_dim, ", ",
               DebugString(output_nonbatch_dims));
    return errors::Unimplemented(msg);
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
    return errors::Unimplemented("Reshape to shape=[] is not supported, at ",
                                 params->node_def.name());
  }

  const int* output_shape_dims = static_cast<int*>(weights.GetValues());
  size_t output_shape_dims_count = weights.count();
  ITensorProxyPtr output_tensor = nullptr;

  if (!params->use_implicit_batch) {
    TF_RETURN_IF_ERROR(ConvertStaticReshapeForExplicitBatchMode(
        params, output_shape_dims, output_shape_dims_count, &output_tensor));
  } else {
    TF_RETURN_IF_ERROR(ConvertStaticReshapeForImplicitBatchMode(
        params, output_shape_dims, output_shape_dims_count, &output_tensor));
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
    return errors::InvalidArgument("ExpandDims axis must be a scalar, at ",
                                   node_def.name());
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
        input_tensor.tensor(), dims, trt_axis, params, &output_tensor));
  } else {
    // ExpandDims: Insert new dim of size 1.
    input_dims.insert(input_dims.begin() + trt_axis, 1);
    // Reshape tensor.
    nvinfer1::Dims new_dims;
    TF_RETURN_IF_ERROR(ContainerToTrtDims(input_dims, &new_dims));
    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, input_tensor, new_dims, /*validation_only=*/false,
        &output_tensor, params->node_def));
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
  // DynamicReshape relies on INetworkDefinition::addShape that was introduced
  // in TensorRT 6.
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
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
    int slice_instance = i * max_num_slices + op_instance_value;
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
      SetLayerName(slice_layer, params->node_def, "slice", slice_instance);
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
#else
  return errors::Unavailable(
      "Dynamic shape input requires TensorRT 6 or above");
#endif
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
  return DynamicReshape(input, slices, params, output, extra_dims, op_instance);
}

Status Converter::SqueezeTensor(ITensorProxyPtr input,
                                std::vector<int>* input_dims,
                                OpConverterParams* params,
                                ITensorProxyPtr* output) {
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
    return DynamicReshape(input, slices, params, output);
  }
  // Remove all dims which are equal to 0.
  input_dims->erase(std::remove(input_dims->begin(), input_dims->end(), 0),
                    input_dims->end());
  // Reshape tensor.
  nvinfer1::Dims new_dims;
  VLOG(2) << "input_dims: " << input_dims;
  TF_RETURN_IF_ERROR(ContainerToTrtDims(*input_dims, &new_dims));
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, TRT_TensorOrWeights(input), new_dims,
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
  TFAttrs attrs(node_def);
  auto squeeze_dims = attrs.get<std::vector<int64>>("squeeze_dims");
  if (squeeze_dims.empty()) {
    if (params->use_implicit_batch || !HasStaticShape(dims)) {
      return errors::Unimplemented(
          "Squeeze is not implemented for empty squeeze_dims, at ",
          node_def.name());
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
            " cannot be squeezed because it must be size 1, at ",
            node_def.name());
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
      input_tensor.tensor(), &input_dims, params, &output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

template <typename Container>
Status ConvertStridedSliceHelper(
    OpConverterParams* params, const TRT_TensorOrWeights& input,
    Container begin, Container size, const Container& stride,
    const nvinfer1::Dims* final_shape = nullptr,
    absl::optional<int> op_instance = absl::nullopt) {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  if (!params->use_implicit_batch &&
      (!HasStaticShape(begin) || !HasStaticShape(size))) {
    return errors::Unimplemented(
        "Strided slice op not implemented for dynamic shape input");
  }
#endif
  const auto& node_def = params->node_def;
  // Get input dims.
  nvinfer1::Dims dims = input.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  if (params->use_implicit_batch) {
    // Begin, size and stride does include explicit batch dim. Add batch
    // dimension to input_dims so that indexes line up properly.
    input_dims.insert(input_dims.begin(), -1);
  }
  // Check bounds.
  for (int i = 1; i < input_dims.size(); i++) {
    if (input_dims[i] < 0 || size[i] < 0) continue;
    if (begin[i] < 0 || begin[i] > input_dims[i]) {
      return errors::InvalidArgument("\"begin\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
    int end = begin[i];
    if (size[i] > 0) end += (size[i] - 1) * stride[i];
    if (end < 0 || end > input_dims[i]) {
      return errors::InvalidArgument("\"begin\" + \"size\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
  }

  // We will use ISliceLayer, which is only available in TRT 5.1+.
  if (!IS_TRT_VERSION_GE(5, 1, 3, 1)) {
    return errors::Unimplemented("Strided slice conversion requires TRT 5.1.3");
  }
  nvinfer1::Dims begin_dims, size_dims, stride_dims;
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(begin, &begin_dims,
                         /*ignore_first_dim=*/params->use_implicit_batch));
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(size, &size_dims,
                         /*ignore_first_dim=*/params->use_implicit_batch));
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(stride, &stride_dims, params->use_implicit_batch));
  if (params->validation_only) return Status::OK();

  VLOG(2) << "Adding slice layer with begin=" << DebugString(begin_dims)
          << ", size=" << DebugString(size_dims)
          << ", stride=" << DebugString(stride_dims);
  nvinfer1::ISliceLayer* layer = params->converter->network()->addSlice(
      *input.tensor()->trt_tensor(), begin_dims, size_dims, stride_dims);
  params->converter->SetLayerName(layer, params->node_def, "slice",
                                  op_instance);
  ITensorProxyPtr tensor = layer->getOutput(0);
  // Reshape for shrink_axis.
  if (final_shape) {
    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, TRT_TensorOrWeights(tensor), *final_shape,
        /*validation_only=*/false, &tensor, node_def, op_instance));
  }
  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return Status::OK();
}

Status ConvertSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params, {{"input", false}, {"begin", true}, {"size", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  std::vector<int> begin = inputs.at(1).weights().ToVector<int>();
  std::vector<int> size = inputs.at(2).weights().ToVector<int>();
  // Get input dims.
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  // Add batch dimension so that indexes line up properly.
  if (params->use_implicit_batch) {
    input_dims.insert(input_dims.begin(), inputs.at(0).batch_size());
  }
  if (!AllLengthsEqual({input_dims, begin, size})) {
    return errors::InvalidArgument(
        "Length of begin and size arguments must equal rank of input for "
        "Slice, at ",
        node_def.name());
  }
  // Check that batch dimension is unmodified.
  if (params->use_implicit_batch) {
    const bool begin_is_modified = begin[0] != 0;
    // If size[0]s is not -1, we can only know if the batch dimension is
    // unmodified when the batch size is defined. When the batch size is
    // undefined, we don't convert to be safe.
    const bool size_is_unchanged = size[0] == -1 || size[0] == input_dims[0];
    if (begin_is_modified || !size_is_unchanged) {
      return errors::Unimplemented(
          "TensorRT does not allow modifications to the batch dimension, at ",
          node_def.name());
    }
  }
  // Size of -1 signifies to take all remaining elements.
  for (int i = 0; i < input_dims.size(); i++) {
    if (size[i] == -1) {
      if (input_dims[i] == -1) {
        return errors::Unimplemented(
            "Input dims must be defined for size = -1, at ", node_def.name());
      }
      size[i] = input_dims[i] - begin[i];
    } else if (size[i] < -1) {
      return errors::InvalidArgument("Invalid size value at ", node_def.name());
    }
    if (input_dims[i] != -1 && (begin[i] < 0 || begin[i] > input_dims[i])) {
      return errors::InvalidArgument("\"begin\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
    const int end = begin[i] + size[i];
    if (input_dims[i] != -1 && (end < 0 || end > input_dims[i])) {
      return errors::InvalidArgument("\"begin\" + \"size\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
  }
  // Stride is 1 for all dims.
  std::vector<int> stride(begin.size(), 1);
  return ConvertStridedSliceHelper(params, inputs.at(0), begin, size, stride);
}

Status ConvertStridedSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // The TF op allows negative begin/end indices while TRT requires values
  // within bounds. This is because we use the the default slice mode
  // (see ISliceLayer::SetMode) with TRT: "Fail with error when the coordinates
  // are out of bounds". If begin/end tensors have negative values then we map
  // them to positive vales. The way this is currently implemented requires that
  // begin / end are constants, therefore we allow only weighs for begin / end.
  //
  // The output size is determined by begin, end and strides. For shape tensors
  // TRT requires that the output size is known at engine construction time. To
  // reduce complexity of the converter, we also require constant size for non
  // shape input. This implies that the stride input also have to be a constant
  // (weights).
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"input", false}, {"begin", true}, {"end", true}, {"strides", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  // TODO(tfeher): Enable dynamic shape input.
  if (!HasStaticShape(inputs.at(0).GetTrtDims())) {
    return errors::Unimplemented(
        "Strided slice op not implemented for dynamic shape input");
  }
  TFAttrs attrs(node_def);
  // New_axis_mask is not supported. TODO(tfeher): Support this by expanddims.
  const int32 new_axis_mask = attrs.get<int64>("new_axis_mask");
  if (new_axis_mask != 0) {
    return errors::Unimplemented(
        "new_axis_mask is not supported for StridedSlice, at ",
        node_def.name());
  }
  const int32 begin_mask = attrs.get<int64>("begin_mask");
  const int32 end_mask = attrs.get<int64>("end_mask");
  const int32 ellipsis_mask = attrs.get<int64>("ellipsis_mask");
  const int32 shrink_axis_mask = attrs.get<int64>("shrink_axis_mask");

  // Get input dims.
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  std::vector<int64> input_dims(dims.d, dims.d + dims.nbDims);
  // Add batch dimension so that indexes line up properly. Set it to -1 if it's
  // unknown, so ValidateStridedSliceOp() can handle it correctly below.
  if (params->use_implicit_batch) {
    input_dims.insert(input_dims.begin(),
                      std::max(-1, inputs.at(0).batch_size()));
  }

  const TRT_ShapedWeights& begin_weights = inputs.at(1).weights();
  const TRT_ShapedWeights& end_weights = inputs.at(2).weights();
  const TRT_ShapedWeights& stride_weights = inputs.at(3).weights();
  if (!AllLengthsEqual({begin_weights.ToVector<int>(),
                        end_weights.ToVector<int>(),
                        stride_weights.ToVector<int>()})) {
    return errors::InvalidArgument(
        "Length of begin, end, and stride must be equal, at ", node_def.name());
  }

  // The slice op has many ways to define the actual operation that needs to be
  // performed. We use ValidateStridedSliceOp to map the input parameters to
  // begin, end, & strides. ValidateStridedSliceOp makes an effort to set known
  // (static) begin/end/strides parameters. On return, begin, end, stride,
  // processing_shape has the same rank as input. final_shape has extra dims
  // added/removed. Negative values in begin/end/stride are converted to
  // positive values to produce a known processing_shape if the input shape is
  // static. Otherwise, processing_shape and final_shape may contain unknown
  // dimension values.
  PartialTensorShape input_shape(input_dims);
  PartialTensorShape processing_shape;
  PartialTensorShape final_shape;
  bool is_identity;
  bool is_simple_slice;
  bool slice_dim0;
  absl::InlinedVector<int64, 4> begin;
  absl::InlinedVector<int64, 4> end;
  absl::InlinedVector<int64, 4> strides;
  TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
      &begin_weights.GetTensor(), &end_weights.GetTensor(),
      stride_weights.GetTensor(), input_shape, begin_mask, end_mask,
      ellipsis_mask, new_axis_mask, shrink_axis_mask, &processing_shape,
      &final_shape, &is_identity, &is_simple_slice, &slice_dim0, &begin, &end,
      &strides));

  // If batch dimension is covered by the ellipsis mask, it means it's left
  // untouched. Otherwise we check whether it modifies the batch dimension here.
  if (params->use_implicit_batch &&
      (!(ellipsis_mask & 1) ||
       begin_weights.shape_.nbDims >= input_dims.size())) {
    // Check that batch dimension is unmodified. We need to use the expanded
    // begin/end/strides array since the original array may be incorrect when
    // (ellipsis_mask&1)==1.
    const bool begin_is_modified = !(begin_mask & 1) && (begin[0] != 0);
    const bool stride_is_modified = (strides[0] != 1);
    // If the batch size is -1 and the end mask is not set, we can only know if
    // the batch dimension is unmodified when the batch size is defined. When
    // the batch size is undefined, we don't convert to be safe.
    const bool batch_size_is_defined = (input_dims[0] > 0);
    const bool end_is_modified =
        !(end_mask & 1) && (!batch_size_is_defined ||
                            (batch_size_is_defined && end[0] != input_dims[0]));
    if (begin_is_modified || stride_is_modified || end_is_modified) {
      return errors::Unimplemented(
          "TensorRT does not allow modifications to the batch dimension, at ",
          node_def.name());
    }
  }
  // Can't shrink axis on batch dimension.
  if (params->use_implicit_batch && shrink_axis_mask & 1) {
    return errors::Unimplemented(
        "TensorRT does not allow modifications to the batch dimension, at ",
        node_def.name());
  }

  // TRT Slice layer uses (begin, size) instead of (begin, end). We calculate
  // the size if possible, otherwise we set it to -1.
  absl::InlinedVector<int64, 4> size(input_dims.size());
  for (int i = 0; i < input_dims.size(); i++) {
    if (input_dims[i] < 0) {
      // Often begin[i] and end[i] could be used to calculate the size.
      // (Although the presence of begin/end manks make it non-trivial beacues
      // 0 value might indicate that a mask was used). But the size has to be
      // clamped to match the array size, for which we need to use the dynamic
      // version of the helper routines. Therefore we set size to -1,
      // which will select the dynamic shape helper (to be implemented).
      size[i] = -1;
      continue;
    }
    // Divide by stride (round up).
    size[i] = strides[i] > 0
                  ? (end[i] - begin[i] + strides[i] - 1) / strides[i]
                  : (begin[i] - end[i] + abs(strides[i]) - 1) / abs(strides[i]);
    if (size[i] < 0) {
      return errors::InvalidArgument(
          "\"size\" cannot be negative for StridedSlice");
    }
  }

  // shrink_axis_mask requires a reshape after the slice.
  nvinfer1::Dims final_shape_dims;
  nvinfer1::Dims* final_shape_dims_ptr = nullptr;
  if (shrink_axis_mask) {
    TF_RETURN_IF_ERROR(TensorShapeToTrtDims(
        final_shape, /*ignore_first_dim=*/params->use_implicit_batch,
        &final_shape_dims));
    final_shape_dims_ptr = &final_shape_dims;
  }

  return ConvertStridedSliceHelper(params, inputs.at(0), begin, size, strides,
                                   final_shape_dims_ptr, 0);
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

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
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
  if (weights_drsck.shape_.nbDims != kNumDims) {
    return errors::InvalidArgument("Conv3D expects kernel of dimension 5, at ",
                                   node_def.name());
  }
  TFAttrs attrs(node_def);
  auto data_format = attrs.get<string>("data_format");
  const bool is_ndhwc = (data_format == "NDHWC");  // Or NCDHW 01234 - > 02341
  const int d_index = is_ndhwc ? 1 : 2;
  const int h_index = is_ndhwc ? 2 : 3;
  const int w_index = is_ndhwc ? 3 : 4;
  const int c_index = is_ndhwc ? 4 : 1;
  auto tf_dilations = attrs.get<std::vector<int64>>("dilations");
  if (tf_dilations.size() != kNumDims) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 5 dimensions, at ",
        node_def.name());
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }

  const nvinfer1::Dims3 dilation_dhw(
      tf_dilations[d_index], tf_dilations[h_index], tf_dilations[w_index]);
  if (is_conv3d_backprop_input &&
      (dilation_dhw.d[0] != 1 || dilation_dhw.d[1] != 1 ||
       dilation_dhw.d[2] != 1)) {
    return errors::Unimplemented(
        "Dilation with Conv3DBackpropInputV2 (conv3d_transpose) is not "
        "supported",
        ", at ", node_def.name());
  }

  const auto tf_stride = attrs.get<std::vector<int64>>("strides");
  if (tf_stride.size() != kNumDims) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 5 dimensions, at ",
        node_def.name());
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }

  const nvinfer1::Dims3 stride_dhw(tf_stride[d_index], tf_stride[h_index],
                                   tf_stride[w_index]);
  const auto tensor_dim = tensor->getDimensions();

  // Asymmetric padding on Deconv not supported for now
  if (is_conv3d_backprop_input && attrs.get<string>("padding") == "SAME") {
    TRT_ShapedWeights weights =
        params->weight_store->GetTempWeights(weights_drsck);

    nvinfer1::Dims3 effective_kernel_size(
        weights.shape_.d[0] +
            (weights.shape_.d[0] - 1) * (dilation_dhw.d[0] - 1),  // D
        weights.shape_.d[1] +
            (weights.shape_.d[1] - 1) * (dilation_dhw.d[1] - 1),  // R
        weights.shape_.d[2] +
            (weights.shape_.d[2] - 1) * (dilation_dhw.d[2] - 1)  // S
    );

    const auto output_size_weights =
        static_cast<int*>(backprop_output_size.weights().GetValues());
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
          "not supported, at ",
          node_def.name());
    }
  }

  // Channel dim must be static for Conv3D since we use that value for
  // num_groups at build time.
  // TODO: Allow conversion if kImplicitBatchModeCompatible||kOptimal is used.
  int implicit_batch_offset = params->use_implicit_batch ? -1 : 0;
  if (tensor->getDimensions().d[c_index + implicit_batch_offset] == -1) {
    return errors::InvalidArgument("Channel dimension must be static, at ",
                                   node_def.name());
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
  TRT_ShapedWeights weights =
      params->weight_store->GetTempWeights(weights_drsck);
  ReorderDRSCKToKCDRS(weights_drsck, &weights, num_groups);
  TRT_ShapedWeights biases(weights.TrtDType());
  const int output_axis = is_conv3d_backprop_input ? 1 : 0;
  const int noutput = weights.shape_.d[output_axis] * num_groups;
  nvinfer1::Dims3 kernel_size_drs(weights.shape_.d[2],  // D
                                  weights.shape_.d[3],  // R
                                  weights.shape_.d[4]   // S
  );

  // Add convolution.
  nvinfer1::ILayer* conv_layer = nullptr;
  if (is_conv3d_backprop_input) {
    nvinfer1::IDeconvolutionLayer* layer =
        params->converter->network()->addDeconvolutionNd(
            *tensor->trt_tensor(), noutput, kernel_size_drs,
            weights.GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);  // change to nd set stride

    // TensorRT 5.1.3 added support for padding modes.
    if (attrs.get<string>("padding") == "SAME") {
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
            weights.GetTrtWeights(), biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);

    if (attrs.get<string>("padding") == "SAME") {
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
    return errors::Unimplemented("Unsupported pooling type: ", node_def.op(),
                                 ", at ", node_def.name());
  }
  TFAttrs attrs(node_def);
  const string padding_type = attrs.get<string>("padding");
  if ((padding_type != "SAME") && (padding_type != "VALID")) {
    return errors::Unimplemented("Unsupported padding type: ", padding_type,
                                 ", at ", node_def.name());
  }
  const auto data_format = attrs.get<string>("data_format");
  const bool is_ndhwc = (data_format == "NDHWC");
  const int c_index = is_ndhwc ? 4 : 1;
  const int d_index = is_ndhwc ? 1 : 2;
  const int h_index = is_ndhwc ? 2 : 3;
  const int w_index = is_ndhwc ? 3 : 4;
  const auto tf_stride = attrs.get<std::vector<int64>>("strides");
  if (tf_stride.size() != kNumDims) {
    return errors::InvalidArgument(
        "Pooling strides field must specify 5 dimensions, at ",
        node_def.name());
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "stride must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const auto tf_kernel = attrs.get<std::vector<int64>>("ksize");
  if (tf_kernel.size() != kNumDims) {
    return errors::InvalidArgument(
        "Pooling ksize field must specify 5 dimensions, at ", node_def.name());
  }
  if (tf_kernel[0] != 1 || tf_kernel[c_index] != 1) {
    return errors::Unimplemented(
        "ksize must be 1 for batch and channel dimensions, at ",
        node_def.name());
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
  ITensorProxyPtr output_tensor = layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(&tensor,
                                                        &output_tensor);

  layer->setStrideNd(stride);
  // VALID padding is the default TRT behavior.
  if (padding_type == "SAME") {
    // SAME_UPPER means that post padding is preferred.
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  params->converter->SetLayerName(layer, node_def, "pooling");

  if (data_format == "NDHWC") {
    // NCDHW => NDHWC
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 4, 1}, &output_tensor, node_def, "to_NDHWC"));
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}
#endif  // #if IS_TRT_VERSION_GE(6, 0, 0, 0)

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
  if (weights.shape_.nbDims != 4) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation expects kernel of dimension 4, at " +
        node_def.name());
  }
  TFAttrs attrs(node_def);
  auto data_format = attrs.get<string>("data_format");
  if (data_format != "NHWC" && data_format != "NCHW") {
    return errors::InvalidArgument("Unsupported data_format:", data_format,
                                   " at ", node_def.name());
  }

  int c_index = (data_format == "NHWC") ? 3 : 1;
  int h_index = (data_format == "NHWC") ? 1 : 2;
  int w_index = (data_format == "NHWC") ? 2 : 3;
  auto tf_dilations = attrs.get<std::vector<int64>>("dilations");
  if (tf_dilations.size() != 4) {
    return errors::InvalidArgument(
        "Convolution dilations field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const nvinfer1::DimsHW dilation(tf_dilations[h_index], tf_dilations[w_index]);

  const auto tf_stride = attrs.get<std::vector<int64>>("strides");
  if (tf_stride.size() != 4) {
    return errors::InvalidArgument(
        "Convolution strides field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  const auto activation_mode = attrs.get<string>("activation_mode");
  auto op_pair = ActivationTypeMap()->find(activation_mode);
  if (op_pair == ActivationTypeMap()->end() && activation_mode != "None") {
    return errors::Unimplemented("Activation mode: ", activation_mode,
                                 " not supported at: ", node_def.name());
  }

  const auto filter_format = attrs.get<string>("filter_format");
  if (filter_format != "HWIO" && filter_format != "OIHW") {
    return errors::InvalidArgument("Unsupported filter_format:", filter_format,
                                   " at ", node_def.name());
  }
  // Check that there's no side_input or conv_input_scale.
  TRT_ShapedWeights side_input = inputs.at(3).weights();
  if (side_input.count() != 0) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation doesn't yet support side_input, at " +
        node_def.name());
  }
  TRT_ShapedWeights conv_input_scale = inputs.at(4).weights();
  if (conv_input_scale.count() != 1 ||
      conv_input_scale.TrtDType() != nvinfer1::DataType::kFLOAT ||
      conv_input_scale.GetSpan<float>()[0] != 1.0) {
    return errors::InvalidArgument(
        "FusedConv2DBiasActivation doesn't yet support conv_input_scale, at " +
        node_def.name());
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
    kernel_size.h() = weights.shape_.d[2];
    kernel_size.w() = weights.shape_.d[3];
  } else {
    // HWIO.
    DCHECK_EQ(filter_format, "HWIO");
    kernel_size.h() = weights.shape_.d[0];
    kernel_size.w() = weights.shape_.d[1];
  }
// Before TRT 5.1.3, we have to calculate padding ourselves.
#if !IS_TRT_VERSION_GE(5, 1, 3, 0)
  const auto tensor_dim = tensor->getDimensions();
  std::vector<int64_t> input_dims;
  // Use 1 and 2 because tensor_dim has the dimensions of the transposed
  // input.
  input_dims = {static_cast<int>(tensor_dim.d[1]),
                static_cast<int>(tensor_dim.d[2])};
  std::vector<std::pair<int, int>> padding;
  ITensorProxyPtr padded_tensor = nullptr;
  TF_RETURN_IF_ERROR(Conv2DPaddingHelper(params, attrs, kernel_size, dilation,
                                         stride, input_dims, tensor, &padding,
                                         &padded_tensor));
  tensor = padded_tensor;
#endif

  // Add convolution.
  TRT_ShapedWeights biases = inputs.at(2).weights();
  nvinfer1::IConvolutionLayer* conv_layer = nullptr;
  if (filter_format == "OIHW") {
    // Weights are already in the right order.
    conv_layer = params->converter->network()->addConvolution(
        *tensor->trt_tensor(), weights.shape_.d[0], kernel_size,
        weights.GetTrtWeights(), biases.GetTrtWeights());
  } else {
    // For conv, TF weights are RSCK, and TRT expects KCRS.
    DCHECK_EQ(filter_format, "HWIO");
    TRT_ShapedWeights weights_kcrs =
        params->weight_store->GetTempWeights(weights);
    ReorderRSCKToKCRS(weights, &weights_kcrs, 1);
    conv_layer = params->converter->network()->addConvolution(
        *tensor->trt_tensor(), weights.shape_.d[3], kernel_size,
        weights_kcrs.GetTrtWeights(), biases.GetTrtWeights());
  }
  TFTRT_RETURN_ERROR_IF_NULLPTR(conv_layer, node_def.name());
  conv_layer->setStride(stride);
#if IS_TRT_VERSION_GE(5, 1, 3, 0)
  if (attrs.get<string>("padding") == "SAME") {
    conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
#else
  conv_layer->setPadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
#endif
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
#if IS_TRT_VERSION_GE(5, 1, 0, 0)
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT8};
#else
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF};
#endif
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));
  nvinfer1::PoolingType type;
  if (node_def.op() == "MaxPool") {
    type = nvinfer1::PoolingType::kMAX;
  } else if (node_def.op() == "AvgPool") {
    type = nvinfer1::PoolingType::kAVERAGE;
  } else {
    return errors::Unimplemented("Unsupported pooling type: ", node_def.op(),
                                 ", at ", node_def.name());
  }
  TFAttrs attrs(node_def);
  const string padding_type = attrs.get<string>("padding");
  if ((padding_type != "SAME") && (padding_type != "VALID")) {
    return errors::Unimplemented("Unsupported padding type: ", padding_type,
                                 ", at ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  int h_index = 2;
  int w_index = 3;
  const auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    h_index = 1;
    w_index = 2;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, {0, 3, 1, 2}, &tensor, node_def, "to_NCHW"));
  }

  const auto tf_stride = attrs.get<std::vector<int64>>("strides");
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);

  const auto tf_kernel = attrs.get<std::vector<int64>>("ksize");
  const nvinfer1::DimsHW ksize(tf_kernel[h_index], tf_kernel[w_index]);

// Before TRT 5.1.3, we have to calculate padding ourselves.
#if !IS_TRT_VERSION_GE(5, 1, 3, 0)
  auto tensor_dim = tensor->getDimensions();
  std::vector<std::pair<int, int>> padding;
  if (padding_type == "SAME") {
    // This is NCHW tensor with no batch dimension.
    //  1 -> h
    //  2 -> w
    padding = CreateSamePadding(
        stride, ksize,
        {static_cast<int>(tensor_dim.d[1]), static_cast<int>(tensor_dim.d[2])});
  } else if (padding_type == "VALID") {
    padding = {{0, 0}, {0, 0}};
  }
#endif
// TensorRT 5.1 added support for asymmetric padding. Before that, we need an
// extra padding layer.
#if !IS_TRT_VERSION_GE(5, 1, 0, 0)
  // Asymmetric padding case.
  if (padding[0].first != padding[0].second ||
      padding[1].first != padding[1].second) {
    auto pad_layer = params->converter->network()->addPadding(
        *tensor->trt_tensor(),
        nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, node_def.name());
    params->converter->SetLayerName(pad_layer, node_def, "pad");
    ITensorProxyPtr out_tensor = pad_layer->getOutput(0);
    params->converter->MarkQuantizationRangesAsInferrable(&tensor, &out_tensor);
    padding = {{0, 0}, {0, 0}};
    tensor = out_tensor;
  }
#endif

  nvinfer1::IPoolingLayer* layer = params->converter->network()->addPooling(
      *tensor->trt_tensor(), type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  // TODO(tmorris): Average pooling may not be entirely safe to infer
  // quantization range through (at least forwards - backwards should be fine).
  // Max pooling is okay.
  ITensorProxyPtr out_tensor = layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(&tensor, &out_tensor);

  layer->setStride(stride);
#if IS_TRT_VERSION_GE(5, 1, 3, 0)
  // VALID padding is the default TRT behavior.
  if (attrs.get<string>("padding") == "SAME") {
    // SAME_UPPER means that post padding is preferred.
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
#elif IS_TRT_VERSION_GE(5, 1, 0, 0)
  layer->setPrePadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
  layer->setPostPadding(nvinfer1::DimsHW{padding[0].second, padding[1].second});
#else
  layer->setPadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
#endif
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
  TFAttrs attrs(node_def);
  const float alpha = attrs.get<float>("alpha");

#if IS_TRT_VERSION_GE(5, 1, 2, 0)
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
#else
  // Use elementwise ops when IActivationLayer is not available.
  if (alpha < 0.0f || alpha > 1.0f) {
    return errors::Unimplemented(
        "Alpha value for LeakyRelu must be between 0 and 1, at ",
        node_def.name());
  }
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  // Create const for alpha.
  ITensorProxyPtr const_alpha_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, alpha, tensor->getDimensions(), &const_alpha_tensor));
  // alpha * x
  nvinfer1::IElementWiseLayer* mul_layer =
      params->converter->network()->addElementWise(
          *tensor->trt_tensor(), *const_alpha_tensor->trt_tensor(),
          nvinfer1::ElementWiseOperation::kPROD);
  TFTRT_RETURN_ERROR_IF_NULLPTR(mul_layer, node_def.name());
  params->converter->SetLayerName(mul_layer, node_def, "mul");
  // max(x, alpha * x)
  nvinfer1::IElementWiseLayer* max_layer =
      params->converter->network()->addElementWise(
          *tensor->trt_tensor(), *mul_layer->getOutput(0),
          nvinfer1::ElementWiseOperation::kMAX);
  TFTRT_RETURN_ERROR_IF_NULLPTR(max_layer, node_def.name());
  params->converter->SetLayerName(mul_layer, node_def, "max");
  ITensorProxyPtr max_tensor = max_layer->getOutput(0);
  ITensorProxyPtr mul_tensor = mul_layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(&max_tensor,
                                                        &mul_tensor);

  params->outputs->push_back(TRT_TensorOrWeights(max_tensor));
  return Status::OK();
#endif
}

#if IS_TRT_VERSION_GE(5, 1, 2, 0)
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

  TFAttrs attrs(node_def);
  const DataType dtype = attrs.get<DataType>("T");
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
  ITensorProxyPtr output_tensor = layer->getOutput(0);
  params->converter->ProvideQuantizationRange(&output_tensor, clip_value_min,
                                              clip_value_max);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}
#endif

const std::unordered_map<string, nvinfer1::ActivationType>*
ActivationTypeMap() {
  static auto* const m =
      new std::unordered_map<string, nvinfer1::ActivationType>({
        {"Relu", nvinfer1::ActivationType::kRELU},
            {"Sigmoid", nvinfer1::ActivationType::kSIGMOID},
            {"Tanh", nvinfer1::ActivationType::kTANH},
#if IS_TRT_VERSION_GE(5, 1, 2, 0)
            {"Elu", nvinfer1::ActivationType::kELU},
            {"Selu", nvinfer1::ActivationType::kSELU},
            {"Softsign", nvinfer1::ActivationType::kSOFTSIGN},
            {"Softplus", nvinfer1::ActivationType::kSOFTPLUS},
#endif
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
                                 " not supported at: ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  // Start conversion.
  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "activation");
  // Set parameters.
#if IS_TRT_VERSION_GE(5, 1, 2, 0)
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
#endif
  ITensorProxyPtr output_tensor = layer->getOutput(0);
  // Set quantization range for output when known.
  if (node_def.op() == "Sigmoid") {
    params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, 1.0f);
  } else if (node_def.op() == "Tanh") {
    params->converter->ProvideQuantizationRange(&output_tensor, -1.0f, 1.0f);
  } else if (node_def.op() == "Softsign") {
    params->converter->ProvideQuantizationRange(&output_tensor, -1.0f, 1.0f);
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertQuantize(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (node_def.op() == "FakeQuantWithMinMaxArgs") {
    TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  } else if (node_def.op() == "FakeQuantWithMinMaxVars") {
    TF_RETURN_IF_ERROR(CheckInputsWeights(
        *params, {{"input", false}, {"min", true}, {"max", true}}));
  } else if (node_def.op() == "QuantizeAndDequantizeV2") {
    TF_RETURN_IF_ERROR(CheckInputsWeights(
        *params, {{"input", false}, {"input_min", true}, {"input_max", true}}));
  } else if (node_def.op() == "QuantizeAndDequantizeV3") {
    TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false},
                                                    {"input_min", true},
                                                    {"input_max", true},
                                                    {"num_bits", true}}));
  }
  float min_range = 0.0f;
  float max_range = 0.0f;
  if (node_def.op() == "FakeQuantWithMinMaxArgs") {
    // Get ranges via node attributes.
    TFAttrs attrs(node_def);
    if (attrs.count("min") == 0 || attrs.count("max") == 0) {
      return errors::InvalidArgument("Min or max attribute not found for ",
                                     node_def.op(), " at ", node_def.name());
    }
    min_range = attrs.get<float>("min");
    max_range = attrs.get<float>("max");
  } else if (node_def.op() == "FakeQuantWithMinMaxVars" ||
             node_def.op() == "QuantizeAndDequantizeV2" ||
             node_def.op() == "QuantizeAndDequantizeV3") {
    // Get ranges via inputs.
    auto get_weights_value = [&inputs](int index) {
      auto raw_weights =
          static_cast<float*>(inputs.at(index).weights().GetValues());
      return raw_weights[0];
    };
    min_range = get_weights_value(1);
    max_range = get_weights_value(2);
  } else {
    return errors::InvalidArgument("Unknown quantization op ", node_def.op(),
                                   ", at ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  // Store ranges for tensor
  ITensorProxyPtr input0 = inputs.at(0).tensor();
  params->converter->ProvideQuantizationRange(&input0, min_range, max_range);
  // Sometimes, TRT may not quantize a tensor, either because it chooses to
  // execute a higher precision kernel or because of op fusion. In these cases,
  // accuracy will suffer if the model was trained to expect quantization at
  // that tensor. We should consider adding a clip(tensor, min_range, max_range)
  // operation here to ensure that any arbitrarily placed quantize node will
  // execute as expected. However, this will negatively affect performance. If
  // users train their models in a way which models inference as close as
  // possible (i.e. not quantizing in place where fusion will occur), then there
  // is no problem with the current implementation.
  params->outputs->push_back(inputs.at(0));
  return Status::OK();
}

Status ConvertRelu6(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

#if IS_TRT_VERSION_GE(5, 1, 2, 0)
  // Use IActivationLayer for TRT >= 5.1
  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor()->trt_tensor(),
          nvinfer1::ActivationType::kCLIP);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  layer->setAlpha(0.0f);
  layer->setBeta(6.0f);
  params->converter->SetLayerName(layer, node_def, "activation");
  ITensorProxyPtr output_tensor = layer->getOutput(0);
  params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, 6.0f);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
#else
  // Convert using min(Relu(x), 6) before TRT 5.1
  // Input Tensor
  ITensorProxyPtr tensor = inputs.at(0).tensor();

  // Relu operation i.e. Relu(x) = max(0, x)
  nvinfer1::IActivationLayer* relu_layer =
      params->converter->network()->addActivation(
          *tensor->trt_tensor(), nvinfer1::ActivationType::kRELU);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu_layer, node_def.name());
  params->converter->SetLayerName(relu_layer, node_def, "activation");

  // Large range of relu is problematic during quantization in INT8 precision
  // mode. Setting dynamic range of relu = [0.f, 6.0f] helps with quantization.
  // TRT only uses dynamic ranges in INT8 precision mode,
  // and this does not affect the FP32 path.
  params->converter->ProvideQuantizationRange(&relu_layer->getOutput(0), 0.0f,
                                              6.0f);

  // Create a constant layer to store the floating point weight i.e. 6.0f
  ITensorProxyPtr const6_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, 6.0f, relu_layer->getOutput(0)->getDimensions(), &const6_tensor));

  // ElementWise Min Operation
  // Min op is a nop for INT8 execution path, as the input tensor
  // to this layer will only have values in range [0.f, 6.0f].
  nvinfer1::IElementWiseLayer* relu6_layer =
      params->converter->network()->addElementWise(
          *relu_layer->getOutput(0), *const6_tensor->trt_tensor(),
          nvinfer1::ElementWiseOperation::kMIN);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu6_layer, node_def.name());
  params->converter->SetLayerName(relu6_layer, node_def, "min");
  ITensorProxyPtr output_tensor = relu6_layer->getOutput(0);
  params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, 6.0f);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
#endif
}

Status ConvertBiasAddInt8WithoutCalibration(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"value", false}, {"bias", true}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  const nvinfer1::Dims original_dims = tensor->getDimensions();
  TFAttrs attrs(node_def);
  const string data_format = attrs.get<string>("data_format");
  const int channel_index =
      (data_format == "NHWC" ? original_dims.nbDims - 1 : 0);

  nvinfer1::Permutation permutation;
  if (channel_index != 0) {
    // Permute the dimensions so that the channel dimension is the first
    // dimension.
    for (int i = 0; i < original_dims.nbDims; ++i) {
      permutation.order[i] = i;
    }
    permutation.order[0] = channel_index;
    permutation.order[channel_index] = 0;
    VLOG(1) << "ConvertBiasAdd permutation: "
            << DebugString(permutation, original_dims.nbDims);
  }

  // TensorRT addScale requires input to be of rank 3, we need to apply
  // transpose as well as reshape.
  // TODO(laigd): this doesn't match what the TRT doc says, fix the doc?
  if (channel_index != 0 || original_dims.nbDims != 3) {
    nvinfer1::IShuffleLayer* shuffle_layer =
        params->converter->network()->addShuffle(*tensor->trt_tensor());
    TFTRT_RETURN_ERROR_IF_NULLPTR(shuffle_layer, node_def.name());
    params->converter->SetLayerName(shuffle_layer, node_def, "shuffle",
                                    /*op_instance=*/0);
    ITensorProxyPtr out_tensor = shuffle_layer->getOutput(0);
    params->converter->MarkQuantizationRangesAsInferrable(&tensor, &out_tensor);

    // NOTE(laigd): for some reason we need to apply the reshape
    // unconditionally. The default shape has nbDims==-1 and it seems the
    // behavior is undefined in some cases.
    nvinfer1::Dims reshape_dims;
    reshape_dims.nbDims = 3;
    // 0 means copying from input; -1 means inferring from the rest.
    reshape_dims.d[0] = 0;
    reshape_dims.d[1] = original_dims.nbDims >= 2 ? 0 : 1;
    reshape_dims.d[2] = original_dims.nbDims >= 3 ? -1 : 1;
    shuffle_layer->setReshapeDimensions(reshape_dims);

    if (channel_index != 0) {
      shuffle_layer->setFirstTranspose(permutation);
    }
    tensor = out_tensor;
  }

  TRT_ShapedWeights weights = inputs.at(1).weights();
  nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
  if (weights.shape_.d[0] == 1) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  }

  TRT_ShapedWeights empty_weights(weights.TrtDType());
  nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
      *tensor->trt_tensor(), mode, weights.GetTrtWeights(),
      empty_weights.GetTrtWeights(), empty_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def, "scale");

  ITensorProxyPtr output_tensor = layer->getOutput(0);

  // Restore transpose & reshape.
  if (channel_index != 0 || original_dims.nbDims != 3) {
    nvinfer1::IShuffleLayer* shuffle_layer =
        params->converter->network()->addShuffle(*output_tensor->trt_tensor());
    TFTRT_RETURN_ERROR_IF_NULLPTR(shuffle_layer, node_def.name());
    params->converter->SetLayerName(shuffle_layer, node_def, "shuffle",
                                    /*op_instance=*/1);
    // NOTE: for same reason as mentioned above we need to apply the reshape
    // unconditionally.
    nvinfer1::Dims reshape_dims = original_dims;
    if (channel_index != 0) {
      // NOTE: according to NVIDIA dimension types are deprecated, so we don't
      // need to copy them back.
      reshape_dims.d[channel_index] = original_dims.d[0];
      reshape_dims.d[0] = original_dims.d[channel_index];
    }
    shuffle_layer->setReshapeDimensions(reshape_dims);

    if (channel_index != 0) {
      shuffle_layer->setSecondTranspose(permutation);
    }
    ITensorProxyPtr shuffle_tensor = shuffle_layer->getOutput(0);
    params->converter->MarkQuantizationRangesAsInferrable(&output_tensor,
                                                          &shuffle_tensor);
    output_tensor = shuffle_tensor;
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertBiasAdd(OpConverterParams* params) {
  if (params->precision_mode == TrtPrecisionMode::INT8 &&
      !params->use_calibration) {
    // NOTE(laigd): based on some observation, it seems TensorRT cannot fuse
    // IConvolutionLayer and IElementwiseLayer and will require range
    // information for the output of Conv2D. Using IScaleLayer will fix the
    // problem.
    return ConvertBiasAddInt8WithoutCalibration(params);
  }
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  if (inputs.size() != 2) {
    return errors::InvalidArgument(
        "BiasAdd expects exactly 2 inputs, but received ", inputs.size());
  }

  if (inputs[0].is_weights() && inputs[1].is_weights()) {
    return errors::InvalidArgument(
        "All inputs are weights, but Grappler is expected to fold them.");
  }

  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  TFAttrs attrs(node_def);
  const string& data_format = attrs.get<string>("data_format");

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
      TF_RETURN_IF_ERROR(ContainerToTrtDims(bias_shape_vec, &bias_shape));
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
  TF_RETURN_IF_ERROR(PrepareTensorForShape(params->converter, inputs.at(0),
                                           input_shape, params->validation_only,
                                           &input_tensor, node_def,
                                           /*op_instance=*/0));

  // Finally, reshape bias. Since the bias is usually a constant, this will
  // normally happen at conversion-time.
  ITensorProxyPtr bias_tensor{nullptr};
  TF_RETURN_IF_ERROR(PrepareTensorForShape(params->converter, inputs.at(1),
                                           bias_shape, params->validation_only,
                                           &bias_tensor, node_def,
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

void GetTensorDimsWithProtoShape(const Tensor& tensor, nvinfer1::Dims* dims) {
  if (tensor.dims() > 0) {
    *dims = GetTrtDimsForTensor(tensor);
  } else {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    dims->nbDims = 0;  // Use scalar weights to implement scalar constants.
#else
    dims->nbDims = 1;  // Use 1D weights to implement scalar constants.
#endif
    // No dimension provided. Flatten it.
    dims->d[0] = tensor.NumElements();
    for (int i = 1; i < nvinfer1::Dims::MAX_DIMS; ++i) {
      dims->d[i] = 0;
    }
  }
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

  nvinfer1::Dims weight_dims;
  GetTensorDimsWithProtoShape(tensor, &weight_dims);
  *weights = weight_store->GetTempWeights(trt_dtype, weight_dims);

  // Copy the tensor directly if the tensor does not require cast to the
  // supported type.
  if (converted_dtype == dtype) {
    char* dst = static_cast<char*>(weights->GetValues());
    memcpy(dst, tensor.tensor_data().data(), tensor.TotalBytes());
    return Status::OK();
  }

  Status status = Status::OK();
  // Copy tensor elements after casting them to the converted DataType.
  int32* dst = static_cast<int32*>(weights->GetValues());
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
        "Constant node is expected to have empty input list: ",
        node_def.name());
  }

  // Create shaped weights as output
  const auto& tensor_proto = node_def.attr().at("value").tensor();
  Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return errors::Internal("Cannot parse weight tensor proto: ",
                            node_def.name());
  }

  TFAttrs attrs(node_def);
  const DataType dtype = attrs.get<DataType>("dtype");
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
  params->outputs->push_back(params->inputs.at(0));
  return Status::OK();
}

const std::unordered_map<string, nvinfer1::ElementWiseOperation>*
BinaryOperationMap() {
  static auto* const m =
      new std::unordered_map<string, nvinfer1::ElementWiseOperation> {
    {"Add", nvinfer1::ElementWiseOperation::kSUM},
        {"AddV2", nvinfer1::ElementWiseOperation::kSUM},
        {"Mul", nvinfer1::ElementWiseOperation::kPROD},
        {"Sub", nvinfer1::ElementWiseOperation::kSUB},
        {"Div", nvinfer1::ElementWiseOperation::kDIV},
#if IS_TRT_VERSION_GE(6, 0, 1, 0)
        // Use TensorRT native FloorDiv.
        {"FloorDiv", nvinfer1::ElementWiseOperation::kFLOOR_DIV},
#elif IS_TRT_VERSION_GE(5, 1, 0, 0)
        // Emulate FloorDiv by doing Div then Floor.
        {"FloorDiv", nvinfer1::ElementWiseOperation::kDIV},
#endif
        {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},
        {"Minimum", nvinfer1::ElementWiseOperation::kMIN},
        {"Maximum", nvinfer1::ElementWiseOperation::kMAX},
        {"Pow", nvinfer1::ElementWiseOperation::kPOW},
  };
  return m;
}

Status ConvertBinary(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return errors::InvalidArgument(node_def.op(), " got ", inputs.size(),
                                   " inputs but expected 2, at ",
                                   node_def.name());
  }
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT32};
#else
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF};
#endif
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));

  // Constant folding should have been done by TensorFlow
  if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
    return errors::Unimplemented(
        "Constant folding is falled back to TensorFlow, binary op received "
        "both input as constant at: ",
        node_def.name());
  }
  const TRT_TensorOrWeights& operand_l = inputs.at(0);
  const TRT_TensorOrWeights& operand_r = inputs.at(1);

  auto op_pair = BinaryOperationMap()->find(node_def.op());
  if (op_pair == BinaryOperationMap()->end()) {
    return errors::Unimplemented("Binary op ", node_def.op(),
                                 " not supported at: ", node_def.name());
  }

  nvinfer1::Dims broadcasted_dims_l, broadcasted_dims_r;
  TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
      operand_l, operand_r, /*check_feasibility=*/true,
      params->use_implicit_batch, &broadcasted_dims_l, &broadcasted_dims_r));
  ITensorProxyPtr tensor_l = nullptr;
  ITensorProxyPtr tensor_r = nullptr;
  // This will also convert constants to tensors, and set quantization ranges.
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, operand_l, broadcasted_dims_l, params->validation_only,
      &tensor_l, node_def, /*op_instance=*/0));
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, operand_r, broadcasted_dims_r, params->validation_only,
      &tensor_r, node_def, /*op_instance=*/1));
  if (params->validation_only) return Status::OK();

  // Add ElementWise layer.
  nvinfer1::ILayer* layer = params->converter->network()->addElementWise(
      *tensor_l->trt_tensor(), *tensor_r->trt_tensor(), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr trt_tensor = layer->getOutput(0);

#if IS_TRT_VERSION_GE(5, 1, 0, 0) and !IS_TRT_VERSION_GE(6, 0, 1, 0)
  if (node_def.op() == "FloorDiv") {
    layer = params->converter->network()->addUnary(
        *trt_tensor->trt_tensor(), nvinfer1::UnaryOperation::kFLOOR);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    params->converter->SetLayerName(layer, node_def, "floor");
    trt_tensor = layer->getOutput(0);
  }
#endif
  params->outputs->push_back(TRT_TensorOrWeights(trt_tensor));
  return Status::OK();
}

Status ConvertRsqrt(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

  // TODO(tmorris): params->converter is null during validation. Allow
  // precision_mode and use_calibration to be accessed during validation and
  // include this check in validation.
  // We will need a quantization range for intermediate tensor if not using
  // calibration.
  //
  //   x -> [Sqrt] -> sqrt(x) -> [Recip] -> 1/sqrt(x)
  //                     ^
  //               need range here
  if (params->converter->precision_mode() == TrtPrecisionMode::INT8 &&
      !params->converter->use_calibration()) {
    return errors::Unimplemented(
        "Intermediate quantization range cannot be determined without"
        " calibration for Rsqrt, consider replacing with "
        "Sqrt -> FakeQuant -> Reciprocal ops, at ",
        node_def.name());
  }
  // Start conversion.
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  // Sqrt
  nvinfer1::IUnaryLayer* sqrt_layer = params->converter->network()->addUnary(
      *tensor->trt_tensor(), nvinfer1::UnaryOperation::kSQRT);
  TFTRT_RETURN_ERROR_IF_NULLPTR(sqrt_layer, node_def.name());
  params->converter->SetLayerName(sqrt_layer, node_def, "sqrt");
  // Recip
  nvinfer1::IUnaryLayer* recip_layer = params->converter->network()->addUnary(
      *sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
  TFTRT_RETURN_ERROR_IF_NULLPTR(recip_layer, node_def.name());
  params->converter->SetLayerName(recip_layer, node_def, "recip");
  params->outputs->push_back(TRT_TensorOrWeights(recip_layer->getOutput(0)));
  return Status::OK();
}

const std::unordered_map<string, nvinfer1::UnaryOperation>*
UnaryOperationMap() {
  static auto* const m =
      new std::unordered_map<string, nvinfer1::UnaryOperation>({
        {"Neg", nvinfer1::UnaryOperation::kNEG},
            {"Exp", nvinfer1::UnaryOperation::kEXP},
            {"Log", nvinfer1::UnaryOperation::kLOG},
            {"Sqrt", nvinfer1::UnaryOperation::kSQRT},
            {"Abs", nvinfer1::UnaryOperation::kABS},
            {"Reciprocal", nvinfer1::UnaryOperation::kRECIP},
#if IS_TRT_VERSION_GE(5, 1, 0, 0)
            {"Sin", nvinfer1::UnaryOperation::kSIN},
            {"Cos", nvinfer1::UnaryOperation::kCOS},
            {"Tan", nvinfer1::UnaryOperation::kTAN},
            {"Sinh", nvinfer1::UnaryOperation::kSINH},
            {"Cosh", nvinfer1::UnaryOperation::kCOSH},
            {"Asin", nvinfer1::UnaryOperation::kASIN},
            {"Acos", nvinfer1::UnaryOperation::kACOS},
            {"Atan", nvinfer1::UnaryOperation::kATAN},
            {"Asinh", nvinfer1::UnaryOperation::kASINH},
            {"Acosh", nvinfer1::UnaryOperation::kACOSH},
            {"Atanh", nvinfer1::UnaryOperation::kATANH},
            {"Ceil", nvinfer1::UnaryOperation::kCEIL},
            {"Floor", nvinfer1::UnaryOperation::kFLOOR},
#endif
#if IS_TRT_VERSION_GE(7, 0, 0, 0)
            {"Erf", nvinfer1::UnaryOperation::kERF},
#endif
      });
  return m;
}

Status ConvertUnary(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  auto op_pair = UnaryOperationMap()->find(node_def.op());
  if (op_pair == UnaryOperationMap()->end()) {
    return errors::Unimplemented("Unary op: ", node_def.op(),
                                 " not supported at: ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  // Start conversion.
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  nvinfer1::IUnaryLayer* layer = params->converter->network()->addUnary(
      *tensor->trt_tensor(), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr output_tensor = layer->getOutput(0);

  // Set quantization ranges.
  if (node_def.op() == "Sin" || node_def.op() == "Cos") {
    params->converter->ProvideQuantizationRange(&output_tensor, -1.0f, 1.0f);
  } else if (node_def.op() == "Asin" || node_def.op() == "Atan") {
    params->converter->ProvideQuantizationRange(&output_tensor, -M_PI_2,
                                                M_PI_2);
  } else if (node_def.op() == "Acos") {
    params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, M_PI);
  } else if (node_def.op() == "Neg" || node_def.op() == "Abs") {
    // Neg and Abs will have same range as input since TRT uses symmetric
    // quantization.
    // TODO(tmorris): Should we infer ranges for Ceil and Floor as well?
    params->converter->MarkQuantizationRangesAsInferrable(&tensor,
                                                          &output_tensor);
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertSquare(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
#if IS_TRT_VERSION_GE(6, 0, 1, 0)
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
#else
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
#endif
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
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  auto tf_axes_list = inputs.at(1).weights().GetSpan<int>();

  TFAttrs attrs(node_def);
  // Only expect to handle INT32 as attributes for now
  if (attrs.get<DataType>("Tidx") != DataType::DT_INT32) {
    return errors::Unimplemented("Tidx supports only DT_INT32");
  }

  int axes = 0;
  if (tf_axes_list.size() == 0) {
    return errors::InvalidArgument(
        "TRT cannot support reduce on all (batch) dimensions, at",
        node_def.name());
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
    return errors::Unimplemented("Op not supported ", node_def.op(), ", at ",
                                 node_def.name());
  }
  if (params->validation_only) return Status::OK();

  const auto keep_dims = attrs.get<bool>("keep_dims");
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

  TFAttrs attrs(node_def);
  const int num_inputs = attrs.get<int64>("N");
  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "Number of inputs for Pack is inconsistent with N attribute, at ",
        node_def.name());
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

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT32};
#else
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF};
#endif
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
  const nvinfer1::Dims dims = inputs.at(idx).GetTrtDims();
  // Convert axis from the TensorFlow format to TensorRT format.
  const int64 tf_axis = attrs.get<int64>("axis");
  int trt_axis;
  TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims + 1, node_def.name(),
                                 params->use_implicit_batch, &trt_axis));

  // Compute expanded dimensions and then reshape input tensors.
  std::vector<int> tensor_dims(dims.d, dims.d + dims.nbDims);
  tensor_dims.insert(tensor_dims.begin() + trt_axis, 1);
  nvinfer1::Dims expanded_dims;
  TF_RETURN_IF_ERROR(ContainerToTrtDims(tensor_dims, &expanded_dims));
  std::vector<ITensorProxyPtr> expanded_tensors;
  int input_index = 0;
  for (const TRT_TensorOrWeights& input : inputs) {
    ITensorProxyPtr expanded_tensor = nullptr;
    if (input.is_tensor() && !params->use_implicit_batch &&
        !HasStaticShape(dims)) {
      if (!params->validation_only) {
        TF_RETURN_IF_ERROR(params->converter->DynamicExpandDims(
            input.tensor(), dims, trt_axis, params, &expanded_tensor,
            input_index));
      }
    } else {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params->converter, input, expanded_dims, params->validation_only,
          &expanded_tensor, node_def, input_index));
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
    return errors::InvalidArgument("Convertpad requires at least 4D input, at ",
                                   node_def.name());
  }
  TRT_ShapedWeights pads = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  // Padding type here is done through TF type
  //   so I can leverage their EnumToDataType for my cast
  auto padding_type = attrs.get<DataType>("Tpaddings");
  // TODO(jie): handle data type conversion for TRT?

  if (pads.shape_.d[0] != nb_dims || pads.shape_.d[1] != 2) {
    return errors::InvalidArgument("Paddings at ", node_def.name(),
                                   " must be a weight with shape [n, 2], "
                                   "where n is the rank of input tensor");
  }

  // Only expect to handle INT32 as attributes for now
  if (padding_type != DataType::DT_INT32) {
    return errors::Unimplemented("Tpaddings supports only DT_INT32");
  }
  auto pad_data = static_cast<int*>(pads.GetValues());

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
  params->converter->MarkQuantizationRangesAsInferrable(&tensor,
                                                        &output_tensor);

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
    return errors::InvalidArgument(
        "Dimension ", tf_axis, " must have statically defined dimensions, at ",
        node_def.name());
  }

  // Dimension must equal num_splits for Unstack (when squeeze_after is true)
  if (squeeze_after && dims.d[trt_axis] != num_splits) {
    return errors::InvalidArgument(
        "Dimension ", tf_axis, " has size ", dims.d[trt_axis],
        " which is not equal to num of ", num_splits, ", at ", node_def.name());
  }
  // Dimension must be evenly divisible by num_splits.
  if (dims.d[trt_axis] % num_splits != 0) {
    return errors::InvalidArgument(
        "Dimension ", tf_axis, " of size ", dims.d[trt_axis],
        " is not evenly divisible by ", num_splits, ", at ", node_def.name());
  }

  // Create parameters for StridedSliceHelper.
  // Slice will begin on zero for all dims, except the one being split which
  // will change.
  std::vector<int> begin(dims.nbDims, 0);
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
  }
  // Create final shape for Unpack/Unstack, where split axis is squeezed.
  nvinfer1::Dims final_shape_for_unpack;
  nvinfer1::Dims* final_shape_for_unpack_ptr = nullptr;

  // We can't use final_shape_for_unpack_ptr when input dimensions are not
  // fully defined.
  const bool is_dynamic_shape = !HasStaticShape(dims);
  if (squeeze_after && !is_dynamic_shape) {
    std::vector<int> size_after_squeeze(size);
    const int tf_axis = trt_axis + (params->use_implicit_batch ? 1 : 0);
    size_after_squeeze.erase(size_after_squeeze.begin() + tf_axis);
    TF_RETURN_IF_ERROR(ContainerToTrtDims(size_after_squeeze,
                                          &final_shape_for_unpack,
                                          /*ignore_first_dim=*/
                                          params->use_implicit_batch));
    final_shape_for_unpack_ptr = &final_shape_for_unpack;
  }

  // Slice the input. ConvertStridedSliceHelper will push the outputs onto
  // params->outputs.
  for (int i = 0; i < num_splits; ++i) {
    const int tf_axis = trt_axis + (params->use_implicit_batch ? 1 : 0);
    begin[tf_axis] = i * split_size_on_axis;
    TF_RETURN_IF_ERROR(ConvertStridedSliceHelper(
        params, input, begin, size, stride, final_shape_for_unpack_ptr,
        /*op_instance=*/i));
  }
  if (params->validation_only) return Status::OK();

  // Squeeze for dynamic shapes
  if (squeeze_after && is_dynamic_shape) {
    for (int i = 0; i < params->outputs->size(); i++) {
      ITensorProxyPtr output_tensor = nullptr;
      std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
      input_dims[trt_axis] = 0;
      TF_RETURN_IF_ERROR(params->converter->SqueezeTensor(
          params->outputs->at(i).tensor(), &input_dims, params,
          &output_tensor));
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
    DataType::DT_FLOAT, DataType::DT_HALF,
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
        DataType::DT_INT32,
#endif
  }));
  int tf_axis = inputs.at(0).weights().GetSpan<int>()[0];
  TFAttrs attrs(node_def);
  const int num_split = attrs.get<int64>("num_split");

  return ConvertSplitHelper(params, inputs.at(1), tf_axis, num_split, false);
}

Status ConvertUnpack(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"value", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, {
    DataType::DT_FLOAT, DataType::DT_HALF,
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
        DataType::DT_INT32,
#endif
  }));
  // Input must be rank 1 or higher, since we can't unpack on axis 0.
  if (inputs.at(0).GetTrtDims().nbDims == 0) {
    return errors::Unimplemented(
        "Input \"value\" for Unpack must be rank 2 or greater, at ",
        node_def.name());
  }
  TFAttrs attrs(node_def);
  const int tf_axis = attrs.get<int64>("axis");
  const int num = attrs.get<int64>("num");

  return ConvertSplitHelper(params, inputs.at(0), tf_axis, num, true);
}

// Supports cast fp16=>fp32 through IIdentityLayer.
Status ConvertCast(OpConverterParams* params) {
  const NodeDef& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  auto unsupport_cast_error = [&]() {
    return errors::Unimplemented("Cast op: ", node_def.op(),
                                 " not supported at: ", node_def.name());
  };

  DataType input_type;
  TF_RETURN_IF_ERROR(GetInputTfType(*params, &input_type, 0));
  if (input_type != DataType::DT_HALF) {
    return unsupport_cast_error();
  }

  DataType output_type;
  TF_RETURN_IF_ERROR(GetNodeDefTfType(params->node_def, &output_type,
                                      kCastOutputTypeAttrName));

  if (output_type != DataType::DT_FLOAT) {
    return unsupport_cast_error();
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
  TFAttrs attrs(node_def);
  // Get number of tensor inputs.
  const int num_inputs = attrs.get<int64>("N");
  if (num_inputs != static_cast<int>(inputs.size()) - 1) {
    return errors::InvalidArgument(
        "Number of inputs for ConcatV2 is inconsistent with N attribute, at ",
        node_def.name());
  }
  // Validate inputs. Values must be tensors for now, although it would be
  // possible to accept weights in explicit batch mode. See CheckInputsWeights
  // for details. TODO(tfeher): Allow weight input in explicit batch mode.
  std::vector<std::pair<string, TrtInputArg>> inputs_kinds;
  TrtInputArg expected_input = TrtInputArg::kTensor;
  inputs_kinds.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs_kinds.push_back({StrCat("values_", i), expected_input});
  }
  inputs_kinds.push_back({"axis", TrtInputArg::kWeight});
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, inputs_kinds));

#if IS_TRT_VERSION_GE(7, 0, 0, 0)
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF,
                                   DataType::DT_INT32};
#else
  std::set<DataType> allowed_types{DataType::DT_FLOAT, DataType::DT_HALF};
#endif
  TF_RETURN_IF_ERROR(AllowDataTypes(*params, allowed_types));
  const auto axis = inputs.at(num_inputs).weights().GetSpan<int>();
  if (axis.size() != 1) {
    return errors::InvalidArgument("Axis for ConcatV2 must be a scalar, at ",
                                   node_def.name());
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
    input_tensors.push_back(inputs.at(i).tensor());
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
  TFAttrs attrs(node_def);
  float epsilon = attrs.get<float>("epsilon");
  auto data_format = attrs.get<string>("data_format");
  if (data_format != "NCHW") {
    return errors::Unimplemented(
        node_def.op(), " only supports data_format=NCHW, at ", node_def.name());
  }
  bool is_training = attrs.get<bool>("is_training");
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
                                 " only supports is_training=false, at ",
                                 node_def.name());
  }
  ITensorProxyPtr tensor = inputs.at(0).tensor();
  if (!params->use_implicit_batch && tensor->getDimensions().d[1] == -1) {
    // This check is to make sure that channel dimension is known during
    // conversion.
    //
    // We check this only in explicit batch mode and reject an op with unknown
    // channel dimension during segmentation. In implicit batch mode we have
    // known shapes during conversion even though the shapes may not be known
    // during segmentation (see the actual argument for input_shapes when
    // ConvertGraphDefToEngine is called from TRTEngineOp::BuildEngine).
    return errors::InvalidArgument("Channel dimension must be static, at ",
                                   node_def.name());
  }
  //  Check parameter types
  auto parameter_type = inputs.at(1).weights().TrtDType();
  if ((parameter_type != nvinfer1::DataType::kFLOAT) &&
      (parameter_type != nvinfer1::DataType::kHALF)) {
    return errors::Unimplemented(
        "Only float32 or float16 weight data type is supported, for node ",
        node_def.name(), " got ", DebugString(parameter_type));
  }
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).weights().TrtDType() != parameter_type) {
      return errors::Unimplemented(
          "Inconsistent parameter type for batchnorm is not supported, at: " +
          node_def.name());
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
      return errors::InvalidArgument(
          "Inconsistent batchnorm parameter count, at: " + node_def.name());
    }
  }
  if (params->validation_only) return Status::OK();

  //  We could technically have two weights with different shape.
  //  that requires two addScale op, arguably less performant
  TRT_ShapedWeights combined_scale_weights =
      params->weight_store->GetTempWeights(*ptr_shape_weights);
  TRT_ShapedWeights combined_offset_weights =
      params->weight_store->GetTempWeights(*ptr_shape_weights);

  const Eigen::half* cast_vals_array[4];
  const float* vals_array[4];
  for (int j = 0; j < 4; j++) {
    cast_vals_array[j] =
        static_cast<Eigen::half const*>(inputs.at(j + 1).weights().GetValues());
    vals_array[j] =
        static_cast<float const*>(inputs.at(j + 1).weights().GetValues());
  }
  Eigen::half* cast_combined_scale_vals =
      static_cast<Eigen::half*>(combined_scale_weights.GetValues());
  Eigen::half* cast_combined_offset_vals =
      static_cast<Eigen::half*>(combined_offset_weights.GetValues());
  float* combined_scale_vals =
      static_cast<float*>(combined_scale_weights.GetValues());
  float* combined_offset_vals =
      static_cast<float*>(combined_offset_weights.GetValues());

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

  nvinfer1::ScaleMode mode = nweight == 1 ? nvinfer1::ScaleMode::kUNIFORM
                                          : nvinfer1::ScaleMode::kCHANNEL;
  nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
      *tensor->trt_tensor(), mode, combined_offset_weights.GetTrtWeights(),
      combined_scale_weights.GetTrtWeights(),
      dummy_power_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  ITensorProxyPtr output_tensor = layer->getOutput(0);
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
                                   {"indices", TrtInputArg::kTensor},
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
    return errors::InvalidArgument("Axis for GatherV2 must be a scalar, at ",
                                   node_def.name());
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
  if (params->use_implicit_batch && params_input.is_tensor() &&
      indices_input.batch_size() != 1) {
    return errors::Unimplemented(
        "Indices must have a batch size of 1 when params is a tensor.");
  }
  // Both input are tensors, and the TF gather result will have rank:
  // (params.nbDims + 1) + (indices.nbDims + 1) - 1,
  // where "+ 1" adds the batch dim. If params is a weight, the TRT rank matches
  // the TF rank so we don't have to add + 1.
  const int params_tf_rank =
      params_input.GetTrtDims().nbDims +
      (params->use_implicit_batch && params_input.is_tensor() ? 1 : 0);
  const int indices_tf_rank =
      indices_input.GetTrtDims().nbDims + (params->use_implicit_batch ? 1 : 0);
  const int tf_gather_output_rank = params_tf_rank + indices_tf_rank - 1;
  if (tf_gather_output_rank >
      nvinfer1::Dims::MAX_DIMS + (params->use_implicit_batch ? 1 : 0)) {
    return errors::InvalidArgument(
        "Result of gather has dimension greater than ",
        nvinfer1::Dims::MAX_DIMS + 1);
  }
  if (params->validation_only) return Status::OK();

  // Convert params to tensor is it is a weight.
  ITensorProxyPtr params_tensor = nullptr;
  if (params_input.is_weights()) {
    params_tensor = params->converter->CreateConstantLayer(
        params_input.weights(), params_input.GetTrtDims());
  } else {
    params_tensor = params_input.tensor();
  }

  // Note on how IGatherLayer works: if both the data and indices tensors have
  // a batch size dimension of size N, it performs:
  // for batchid in xrange(N):
  //   output[batchid, a0, ..., an, i, ..., j, b0, ..., bn] = (
  //       data[batchid, a0, ..., an, indices[batchid, i, ..., j] b0, ..., bn])
  nvinfer1::IGatherLayer* layer = params->converter->network()->addGather(
      *params_tensor->trt_tensor(), *indices_input.tensor()->trt_tensor(),
      trt_axis);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  ITensorProxyPtr output_tensor = layer->getOutput(0);
  nvinfer1::Dims trt_gather_output_dims = output_tensor->getDimensions();
  // Note for the "- 2": one is for the output batch dim encapsulated by TF-TRT,
  // and the other is for the output dimension that is squeezed by IGatherLayer
  // because of the implicit batch dim in the indices (see the above note).
  const int expected_trt_output_rank =
      tf_gather_output_rank - (params_input.is_tensor() ? 2 : 1);
  if (params->use_implicit_batch &&
      trt_gather_output_dims.nbDims != expected_trt_output_rank) {
    return errors::Internal(
        "Get unexpected output dimensions of IGatherLayer. Expect nbDims: ",
        expected_trt_output_rank,
        ", actual nbDims: ", trt_gather_output_dims.nbDims);
  }
  // Reshape the output so after adding the implicit batch dim it'll match the
  // output shape of TF GatherV2.
  if (params->use_implicit_batch && params_input.is_tensor()) {
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
  nvinfer1::Dims reshape_dim{input_dim.nbDims + 2, {}};
  // The empty braces initialize the elements of reshap_dim.d to 0. A value 0 in
  // reshape_dim.d[i] will preserve the i-th dimension value from the shape of
  // input_a.
  reshape_dim.d[input_dim.nbDims] = 1;
  reshape_dim.d[input_dim.nbDims + 1] = 1;
  const NodeDef& node_def = params->node_def;
  TF_RETURN_IF_ERROR(PrepareTensorForShape(
      params->converter, input_a, reshape_dim,
      /*validation_only=*/false, &tensor_a, node_def, /*op_instance=*/0,
      /*origin_node_name=*/"FULLY_CONNECTED"));

  VLOG(2) << "New shape of A " << DebugString(tensor_a->getDimensions());

  TRT_ShapedWeights weights_b = input_b.weights();
  TRT_ShapedWeights weights_2D(weights_b);
  if (weights_b.shape_.nbDims > 2) {
    // Combine first nbDims-1 dims into a single dim, e.g. for a 4D tensor we
    // transform [N, H, W, C] -> [N*H*W, C].
    int k = weights_b.shape_.d[weights_b.shape_.nbDims - 1];
    nvinfer1::Dims dims{2, {static_cast<int>(weights_b.count() / k), k}};
    TF_RETURN_IF_ERROR(weights_2D.SetShape(dims));
  }

  // FC layer will transpose weights, so we need to pre-transpose.
  TRT_ShapedWeights weights(weights_2D.TrtDType());
  if (!transpose_b) {
    weights = params->weight_store->GetTempWeights(weights_2D);
    ReorderCKtoKC(weights_2D, &weights);
  } else {
    weights = weights_2D;
  }
  TRT_ShapedWeights biases(weights.TrtDType());
  int k = weights.shape_.d[weights.shape_.nbDims - 1];
  const int noutput = weights.count() / k;
  VLOG(2) << "Using fully connected layer with k=" << k
          << ", n_output=" << noutput
          << " weights shape: " << DebugString(weights.shape_) << " to convert "
          << node_def.op();
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
  if (inputs.size() != 2) {
    return errors::InvalidArgument(node_def.op(), " got ", inputs.size(),
                                   " inputs but expected 2, at ",
                                   node_def.name());
  }
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  TFAttrs attrs(node_def);
  bool transpose_a = attrs.get<bool>("transpose_a");
  bool transpose_b = attrs.get<bool>("transpose_b");

  return ConvertMatMulHelper(params, inputs.at(0), inputs.at(1), transpose_a,
                             transpose_b);
}

Status ConvertBatchMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return errors::InvalidArgument(node_def.op(), " got ", inputs.size(),
                                   " inputs but expected 2, at ",
                                   node_def.name());
  }
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params, {{"x", TrtInputArg::kBoth}, {"y", TrtInputArg::kBoth}}));
  // TODO(tfeher): Consider adding INT8 type because FC layer can support it.
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
    return errors::InvalidArgument(
        "All inputs are weights, but Grappler is expected to fold them.");
  }

  TFAttrs attrs(node_def);
  const bool transpose_a = attrs.get<bool>("adj_x");
  const bool transpose_b = attrs.get<bool>("adj_y");

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

// Finds the indices of elements in [begin, end) in array
// [array_begin, array_end), and appends the indices to permute. This is used to
// construct the permutation sequence for the operand with input labels
// [array_begin, array_end) to the desired permuted labels [begin, end).
template <typename Iterator>
Status FindIndices(Iterator begin, Iterator end, Iterator array_begin,
                   Iterator array_end, std::vector<int>* permute) {
  const int n = array_end - array_begin;
  if (n < end - begin) {
    return errors::Internal("Incorrect array size");
  }
  for (auto i = begin; i < end; i++) {
    int idx = std::find(array_begin, array_end, *i) - array_begin;
    if (idx >= n) {
      return errors::Internal("Label not found");
    }
    permute->push_back(idx);
  }
  return Status::OK();
}

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
// Layout of the einsum dimensions: Batch, Free and Contraction indices.
// Example: abcd,adef -> abde. The first tensor has layout BFC, the second BCF.
enum class EinsumLayout { BFC, BCF, MIX };

// Describes an operand: input shape, number of batch, free and contract
// dimensions, and the permutation that is needed to bring it to a matmul
// compatible form.
struct EinsumDescriptor {
  EinsumDescriptor() : b(0), f(0), c(0) {}

  // Deduces the number of batch, free, contract dimensions from the input
  // labels, decides what layout to use, and determines permutation indices for
  // that layout.
  Status InitDescriptor(const TRT_TensorOrWeights& operand, Labels input_labels,
                        std::vector<EinsumHelper::DimensionType>& label_types,
                        EinsumLayout preferred_layout,
                        EinsumDescriptor* other = nullptr) {
    if (preferred_layout == EinsumLayout::MIX)
      return errors::Internal("Preferred einsum layout cannot be MIX");
    const EinsumHelper::DimensionType kBatch =
        EinsumHelper::DimensionType::kBatch;
    const EinsumHelper::DimensionType kFree =
        EinsumHelper::DimensionType::kFree;
    const EinsumHelper::DimensionType kContract =
        EinsumHelper::DimensionType::kContract;

    // Map label indices to label types.
    std::vector<EinsumHelper::DimensionType> types;  // Input label types.
    std::transform(input_labels.begin(), input_labels.end(),
                   std::back_inserter(types),
                   [&label_types, kBatch](int i) { return label_types.at(i); });

    using label_t_iterator = std::vector<EinsumHelper::DimensionType>::iterator;
    auto count_labels = [](label_t_iterator begin, label_t_iterator end,
                           EinsumHelper::DimensionType val) {
      return std::count_if(begin, end, [val](EinsumHelper::DimensionType t) {
        return t == val;
      });
    };

    b = count_labels(types.begin(), types.end(), kBatch);
    f = count_labels(types.begin(), types.end(), kFree);
    c = count_labels(types.begin(), types.end(), kContract);

    if (c == 0 || f == 0) {
      VLOG(2) << "Einsum equation needs to have at least one free and one "
                 "contract dimension";
      return errors::Unimplemented("No conversion for einsum equation.");
    }

    // Checks whether input_labels[offset:offset+m] matches labels from other.
    auto order_matches = [other, &input_labels, kBatch, kFree, kContract](
                             int offset, int m,
                             EinsumHelper::DimensionType dim_type) {
      if (!other) return true;
      int offset_other = 0;
      if (dim_type == kFree)
        offset = other->offset_f;
      else if (dim_type == kContract)
        offset = other->offset_c;
      return std::equal(input_labels.begin() + offset,
                        input_labels.begin() + offset + m,
                        other->permuted_labels.begin() + offset_other);
    };

    // Check if the current layout is BFC or BCF. In that case we could avoid
    // transpose.
    layout = EinsumLayout::MIX;
    if (count_labels(types.begin(), types.begin() + b, kBatch) == b &&
        order_matches(0, b, kBatch)) {
      // Batch dims are the leading dims. They have the same order as other.
      if (count_labels(types.begin() + b, types.begin() + b + f, kFree) == f) {
        // All the free dims are placed consecutively after the batch dims.
        // Their order is arbitrary. The final transpose will ensure that the
        // output has correct order. We still have to check that the contract
        // indices have correct order.
        if (order_matches(b + f, c, kContract)) {
          layout = EinsumLayout::BFC;
        }
      } else if (count_labels(types.begin() + b, types.begin() + b + c,
                              kContract) == c) {
        // All the contract dims are placed consecutively after the batch
        // dims. Check whether the contract dims have the same order as the
        // contract dims in other.
        if (order_matches(b, c, kContract)) {
          layout = EinsumLayout::BCF;
        }
      }
    }

    if (layout == EinsumLayout::MIX) {
      // Input label types are mixed. Calculate a permutation that maps them
      // to the preferred layout (BCF or BFC).
      layout = preferred_layout;
      if (!other) {
        AppendMatchingIndicesToPermute(types, kBatch);
      } else {
        TF_RETURN_IF_ERROR(
            FindIndices(other->permuted_labels.begin(),
                        other->permuted_labels.begin() + other->b,
                        input_labels.begin(), input_labels.end(), &permute));
      }
      if (layout == EinsumLayout::BFC) {
        AppendMatchingIndicesToPermute(types, kFree);
        if (!other) {
          AppendMatchingIndicesToPermute(types, kContract);
        } else {
          TF_RETURN_IF_ERROR(FindIndices(
              other->permuted_labels.begin() + other->offset_c,
              other->permuted_labels.begin() + other->offset_c + other->c,
              input_labels.begin(), input_labels.end(), &permute));
        }
      } else {
        if (!other) {
          AppendMatchingIndicesToPermute(types, kContract);
        } else {
          TF_RETURN_IF_ERROR(FindIndices(
              other->permuted_labels.begin() + other->offset_c,
              other->permuted_labels.begin() + other->offset_c + other->c,
              input_labels.begin(), input_labels.end(), &permute));
        }
        AppendMatchingIndicesToPermute(types, kFree);
      }
    }

    if (layout == EinsumLayout::BFC) {
      offset_f = b;
      offset_c = f + b;
    } else {
      offset_f = b + c;
      offset_c = b;
    }

    dims = operand.GetTrtDims();
    for (int i = 0; i < b; i++) {
      // Set unknown batch dims to zero. These dims will be used in reshape op,
      // where zero is a special value for retaining the original dim size.
      if (dims.d[i] == -1) dims.d[i] = 0;
    }
    permuted_labels = input_labels;
    if (!permute.empty()) {
      // Apply the permutation on the dimension array.
      nvinfer1::Dims orig_dims = dims;
      for (int i = 0; i < permute.size(); i++) {
        dims.d[i] = orig_dims.d[permute[i]];
        permuted_labels[i] = input_labels[permute[i]];
      }
    }
    size_tensors.resize(dims.nbDims, nullptr);

    VLOG(2) << "Set up descriptor with "
            << (layout == EinsumLayout::BFC ? "BFC" : "BCF")
            << " layout, b=" << b << ", f=" << f << ", c=" << c;
    return Status::OK();
  }

  // Appends indices where types maches value.
  void AppendMatchingIndicesToPermute(
      const std::vector<EinsumHelper::DimensionType>& types,
      EinsumHelper::DimensionType val) {
    for (int i = 0; i < types.size(); i++) {
      if (types[i] == val) {
        permute.push_back(i);
      }
    }
  }

  // Returns whether the free and contract dimension have static shape.
  bool HasStaticShape() {
    return !std::any_of(dims.d + b, dims.d + dims.nbDims,
                        [](int k) { return k == -1; });
  }

  nvinfer1::Permutation GetPermutation() {
    nvinfer1::Permutation p;
    std::copy(permute.begin(), permute.end(), p.order);
    return p;
  }

  Status SetDynamicSize(OpConverterParams* params,
                        const TRT_TensorOrWeights& operand) {
    if (operand.GetTrtDims().nbDims != dims.nbDims)
      return errors::Internal("Operand dims must agree with descirptor dims");

    if (operand.is_weights()) {
      for (int i = 0; i < operand.GetTrtDims().nbDims; i++) {
        // dims.d stores the permuted dims.
        TF_RETURN_IF_ERROR(
            CreateScalarConstant(params, dims.d[i], &size_tensors[i]));
      }
      return Status::OK();
    }
    auto* shape_layer =
        params->converter->network()->addShape(*operand.tensor()->trt_tensor());
    TFTRT_RETURN_ERROR_IF_NULLPTR(shape_layer, params->node_def.name());
    ITensorProxyPtr shape = shape_layer->getOutput(0);
    for (int i = 0; i < operand.GetTrtDims().nbDims; i++) {
      int idx = permute.empty() ? i : permute.at(i);
      auto* layer = params->converter->network()->addSlice(
          *shape->trt_tensor(), {1, {idx}}, {1, {1}}, {1, {1}});
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
      size_tensors[i] = layer->getOutput(0);
      TFTRT_RETURN_ERROR_IF_NULLPTR(size_tensors[i], "error, slice is nullptr");
    }
    return Status::OK();
  }

  EinsumLayout layout;
  int b;  // number of batch dims
  int f;  // number of free dims
  int c;  // number of conraction dims
  int offset_f;
  int offset_c;
  nvinfer1::Dims dims;
  std::vector<int> permute;
  std::vector<ITensorProxyPtr> size_tensors;
  Labels permuted_labels;
};

Status GetDimsProd(nvinfer1::Dims dims, int offset, int n, int32_t* out) {
  size_t prod = std::accumulate(dims.d + offset, dims.d + offset + n, size_t(1),
                                std::multiplies<size_t>());
  if (prod > std::numeric_limits<int32_t>::max()) {
    return errors::Internal("Matrix too large");
  } else {
    *out = prod;
  }
  return Status::OK();
}

Status GetDimsProdDynamic(OpConverterParams* params,
                          std::vector<ITensorProxyPtr>::const_iterator begin,
                          std::vector<ITensorProxyPtr>::const_iterator end,
                          ITensorProxyPtr* out) {
  *out = *begin;
  begin++;
  while (begin != end) {
    nvinfer1::IElementWiseLayer* layer =
        params->converter->network()->addElementWise(
            *(*out)->trt_tensor(), *(*begin)->trt_tensor(),
            nvinfer1::ElementWiseOperation::kPROD);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
    *out = layer->getOutput(0);
    begin++;
  }
  return Status::OK();
}

Status ConcatenateShape(OpConverterParams* params,
                        const std::vector<ITensorProxyPtr> size_tensors,
                        ITensorProxyPtr* new_shape) {
  std::vector<nvinfer1::ITensor*> trt_size_tensors;
  for (const auto& t : size_tensors) {
    trt_size_tensors.push_back(t->trt_tensor());
  }
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          static_cast<nvinfer1::ITensor* const*>(trt_size_tensors.data()),
          size_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
  layer->setAxis(0);
  *new_shape = layer->getOutput(0);
  return Status::OK();
}

// Reshapes operand so that the free dimensions are combined into a single dim,
// and the contract dimensions are combined into another single dim.
Status GetEinsumNewDynamicShape(OpConverterParams* params,
                                const EinsumDescriptor& desc,
                                ITensorProxyPtr* new_shape) {
  std::vector<ITensorProxyPtr> size(desc.size_tensors.begin(),
                                    desc.size_tensors.begin() + desc.b + 2);

  int idx_f = desc.layout == EinsumLayout::BFC ? desc.b : desc.b + 1;
  int idx_c = desc.layout == EinsumLayout::BFC ? desc.b + 1 : desc.b;

  TF_RETURN_IF_ERROR(GetDimsProdDynamic(
      params, desc.size_tensors.begin() + desc.offset_f,
      desc.size_tensors.begin() + desc.offset_f + desc.f, &size[idx_f]));

  TF_RETURN_IF_ERROR(GetDimsProdDynamic(
      params, desc.size_tensors.begin() + desc.offset_c,
      desc.size_tensors.begin() + desc.offset_c + desc.c, &size[idx_c]));

  TF_RETURN_IF_ERROR(ConcatenateShape(params, size, new_shape));
  return Status::OK();
}

// Reshapes operand so that the free dimensions are combined into a single dim,
// and the contract dimensions are combined into another single dim.
Status GetEinsumNewStaticShape(const EinsumDescriptor& desc,
                               nvinfer1::Dims* new_dims) {
  new_dims->nbDims = desc.b + 2;
  // Copy batch dims.
  std::copy(desc.dims.d, desc.dims.d + desc.b, new_dims->d);
  // Combine free dims and contract dims.
  int idx_f = desc.layout == EinsumLayout::BFC ? desc.b : desc.b + 1;
  int idx_c = desc.layout == EinsumLayout::BFC ? desc.b + 1 : desc.b;
  TF_RETURN_IF_ERROR(
      GetDimsProd(desc.dims, desc.offset_f, desc.f, new_dims->d + idx_f));
  TF_RETURN_IF_ERROR(
      GetDimsProd(desc.dims, desc.offset_c, desc.c, new_dims->d + idx_c));
  return Status::OK();
}

// Adds shuffle layer (if needed) to bring einsum operand to a matmul compatible
// format.
Status ShuffleEinsumTensor(OpConverterParams* params,
                           std::unique_ptr<TRT_TensorOrWeights>* operand,
                           EinsumDescriptor* desc, int op_instance) {
  if (params->validation_only) return Status::OK();
  TF_RETURN_IF_ERROR(desc->SetDynamicSize(params, **operand));
  bool need_reshape = (desc->f != 1 || desc->c != 1);
  bool need_transpose = !desc->permute.empty();
  if ((*operand)->is_weights()) {
    nvinfer1::Dims new_dims;
    TF_RETURN_IF_ERROR(GetEinsumNewStaticShape(*desc, &new_dims));
    if (!need_transpose) {
      TRT_ShapedWeights weights((*operand)->weights());
      TF_RETURN_IF_ERROR(weights.SetShape(new_dims));
      operand->reset(new TRT_TensorOrWeights(weights));
      return Status::OK();
    }
    // TODO(tfeher): Instead of creating a tensor that will be transposed,
    // transpose the weight itself. Keeping it weight could enable FC layer.
    ITensorProxyPtr tensor = params->converter->CreateConstantLayer(
        (*operand)->weights(), (*operand)->GetTrtDims());
    operand->reset(new TRT_TensorOrWeights(tensor));
  }

  if (!need_transpose && !need_reshape) return Status::OK();
  ITensorProxyPtr operand_tensor = (*operand)->tensor();
  TFTRT_RETURN_ERROR_IF_NULLPTR(operand_tensor, "Null tensor at Einsum");
  nvinfer1::IShuffleLayer* layer =
      params->converter->network()->addShuffle(*operand_tensor->trt_tensor());

  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
  params->converter->SetLayerName(layer, params->node_def, "shuffle",
                                  /*op_instance=*/op_instance);
  // Set new shape.
  if (need_reshape) {
    if (desc->HasStaticShape()) {
      nvinfer1::Dims new_dims;
      TF_RETURN_IF_ERROR(GetEinsumNewStaticShape(*desc, &new_dims));
      layer->setReshapeDimensions(new_dims);
    } else {
      ITensorProxyPtr new_shape;
      TF_RETURN_IF_ERROR(GetEinsumNewDynamicShape(params, *desc, &new_shape));
      layer->setInput(1, *new_shape->trt_tensor());
    }
  }

  if (need_transpose) {
    layer->setFirstTranspose(desc->GetPermutation());
  }
  operand->reset(new TRT_TensorOrWeights(layer->getOutput(0)));
  return Status::OK();
}

// Combines output dims/labels by copying batch and free dims/labels from input
// A, and concatenating free values from input B.
template <typename InputIterator, typename OutputIterator>
void AssembleOutput(InputIterator begin_a, InputIterator begin_b,
                    const EinsumDescriptor& desc_a,
                    const EinsumDescriptor& desc_b, OutputIterator out) {
  std::copy(begin_a, begin_a + desc_a.b, out);
  begin_a += desc_a.offset_f;
  std::copy(begin_a, begin_a + desc_a.f, out + desc_a.b);
  begin_b += desc_b.offset_f;
  std::copy(begin_b, begin_b + desc_b.f, out + desc_a.b + desc_a.f);
}

// Restores free dimensions and sets final index order. Consider C = A * B,
// batched MatMul op, where A.shape = [B, x, k] and B.shape = [B, k, y]. Then
// C.shape = [B, x, y]. Here B can denote multiple batch indices while x, y, k
// are single indices. The original inputs to Einsum can have multiple free
// indices. These were combined into a singe free dimension x and y, for example
// x = f_a1 * f_a2 * f_a3, y = f_b1 * f_b2. This routine creates a shuffle layer
// to expand x into and y the original free dims, e.g. C is reshaped to
// [B, f_a1, f_a2, f_a3, f_b1, f_b2]. Finally, a permutation is applied to
// transform the shape to the shape of the original Einsum output.
Status ShuffleEinsumOutput(OpConverterParams* params, EinsumDescriptor desc_a,
                           EinsumDescriptor desc_b,
                           const std::vector<int>& permutation,
                           ITensorProxyPtr* output) {
  if (permutation.empty() && (desc_a.f == 1 && desc_b.f == 1))
    return Status::OK();

  nvinfer1::IShuffleLayer* layer =
      params->converter->network()->addShuffle(*(*output)->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
  params->converter->SetLayerName(layer, params->node_def, "shuffle",
                                  /*op_instance=*/2);

  int output_rank = desc_a.b + desc_a.f + desc_b.f;
  if (desc_a.f != 1 || desc_b.f != 1) {
    if (desc_a.HasStaticShape() && desc_b.HasStaticShape()) {
      nvinfer1::Dims dims_out = {output_rank, {}};
      AssembleOutput(desc_a.dims.d, desc_b.dims.d, desc_a, desc_b, dims_out.d);
      layer->setReshapeDimensions(dims_out);
    } else {
      std::vector<ITensorProxyPtr> size_tensors(output_rank);
      AssembleOutput(desc_a.size_tensors.begin(), desc_b.size_tensors.begin(),
                     desc_a, desc_b, size_tensors.begin());
      ITensorProxyPtr new_shape;
      TF_RETURN_IF_ERROR(ConcatenateShape(params, size_tensors, &new_shape));
      layer->setInput(1, *new_shape->trt_tensor());
    }
  }

  if (!permutation.empty()) {
    nvinfer1::Permutation p;
    std::copy(permutation.begin(), permutation.end(), p.order);
    layer->setSecondTranspose(p);
  }
  *output = layer->getOutput(0);
  return Status::OK();
}

// Prepares EinsumDescriptors after parsing the equation and determines the
// final transpose.
Status ParseEquation(OpConverterParams* params,
                     std::unique_ptr<TRT_TensorOrWeights>* input_a,
                     std::unique_ptr<TRT_TensorOrWeights>* input_b,
                     EinsumDescriptor* descriptor_a,
                     EinsumDescriptor* descriptor_b,
                     std::vector<int>* final_transpose) {
  TFAttrs attrs(params->node_def);
  std::string equation = attrs.get<string>("equation");
  VLOG(2) << "Einsum equation " << equation;

  OperandLabels input_labels;
  Labels output_labels;
  std::vector<EinsumHelper::DimensionType> label_types;
  OperandLabelCounts input_label_counts;
  LabelCounts output_label_counts;
  absl::InlinedVector<bool, 2> input_has_ellipsis;
  bool output_has_ellipsis;
  TF_RETURN_IF_ERROR(EinsumHelper::ParseEquation(
      equation, &input_labels, &output_labels, &label_types,
      &input_label_counts, &output_label_counts, &input_has_ellipsis,
      &output_has_ellipsis));

  VLOG(2) << "Output has ellipsis: " << output_has_ellipsis;

  if (input_has_ellipsis[0] || input_has_ellipsis[1] || output_has_ellipsis) {
    // TODO(tfeher): Handle ellipsis like EinsumHelper::ProcessDimensions.
    // Note: ProcessDimensions would introduce kBroadcasting labels, which we
    // need to replace with kBatch before we call InitDescriptor.
    VLOG(2) << "Ellipsis not yet supported";
    return errors::Unimplemented("No conversion for einsum equation.");
  }
  if (absl::c_any_of(label_types, [](auto l) {
        return l == EinsumHelper::DimensionType::kReduce ||
               l == EinsumHelper::DimensionType::kBroadcasting;
      })) {
    VLOG(2) << "Einsum reductions not implemented";
    return errors::Unimplemented("No conversion for einsum equation.");
  }

  auto no_duplicated_labels = [](const LabelCounts& label_counts) {
    return absl::c_any_of(label_counts, [](int i) { return i > 1; });
  };
  if (no_duplicated_labels(input_label_counts[0]) ||
      no_duplicated_labels(input_label_counts[1]) ||
      no_duplicated_labels(output_label_counts)) {
    VLOG(2) << "Einsum invalid label count";
    return errors::Unimplemented("No conversion for einsum equation.");
  }

  if ((*input_a)->is_weights() && (*input_b)->is_tensor()) {
    // We prefer to use FC layer, needs A as tensor and B as weight.
    std::swap(*input_a, *input_b);
    std::swap(input_labels[0], input_labels[1]);
    std::swap(input_label_counts[0], input_label_counts[1]);
  }

  TF_RETURN_IF_ERROR(descriptor_a->InitDescriptor(
      **input_a, input_labels[0], label_types, EinsumLayout::BFC));
  TF_RETURN_IF_ERROR(
      descriptor_b->InitDescriptor(**input_b, input_labels[1], label_types,
                                   EinsumLayout::BCF, descriptor_a));
  // TODO(tfeher): Update the permutation in the descriptors to avoid final
  // transpose (if possible). Consider swapping the input if it eliminates
  // final transpose.

  // Get final transpose.
  Labels matmul_output_labels(descriptor_a->b + descriptor_a->f +
                              descriptor_b->f);
  AssembleOutput(descriptor_a->permuted_labels.begin(),
                 descriptor_b->permuted_labels.begin(), *descriptor_a,
                 *descriptor_b, matmul_output_labels.begin());
  TF_RETURN_IF_ERROR(FindIndices(output_labels.begin(), output_labels.end(),
                                 matmul_output_labels.begin(),
                                 matmul_output_labels.end(), final_transpose));
  // Clear identity transpose.
  bool identity_transpose = true;
  for (int i = 0; i < final_transpose->size() && identity_transpose; i++) {
    identity_transpose &= final_transpose->at(i) == i;
  }
  if (identity_transpose) {
    final_transpose->clear();
  }
  return Status::OK();
}

Status ConvertEinsum(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (params->use_implicit_batch) {
    return errors::Unimplemented(
        "Einsum converter requires dynamic shape mode");
  }

  if (inputs.size() != 2) {
    VLOG(2) << "Einsum converter supports two operands at " << node_def.name()
            << " got " << inputs.size();
    return errors::Unimplemented("No conversion for einsum equation.");
  }
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  auto input_a = std::make_unique<TRT_TensorOrWeights>(inputs.at(0));
  auto input_b = std::make_unique<TRT_TensorOrWeights>(inputs.at(1));
  EinsumDescriptor descriptor_a;
  EinsumDescriptor descriptor_b;
  std::vector<int> final_transpose;
  TF_RETURN_IF_ERROR(ParseEquation(params, &input_a, &input_b, &descriptor_a,
                                   &descriptor_b, &final_transpose));

  TF_RETURN_IF_ERROR(ShuffleEinsumTensor(params, &input_a, &descriptor_a,
                                         /*op_instance=*/0));
  TF_RETURN_IF_ERROR(ShuffleEinsumTensor(params, &input_b, &descriptor_b,
                                         /*op_instance=*/1));
  if (params->validation_only) return Status::OK();

  StatusOr<ITensorProxyPtr> result = ConvertMatMulImpl(
      params, *input_a, *input_b, descriptor_a.layout == EinsumLayout::BCF,
      descriptor_b.layout == EinsumLayout::BFC);
  TF_RETURN_IF_ERROR(result.status());
  ITensorProxyPtr output = result.ValueOrDie();

  TF_RETURN_IF_ERROR(ShuffleEinsumOutput(params, descriptor_a, descriptor_b,
                                         final_transpose, &output));
  params->outputs->push_back(TRT_TensorOrWeights(output));
  return Status::OK();
}
#endif  // IS_TRT_VERSION_GE(7, 1, 3, 0)

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
        "TensorRT Softmax cannot apply on batch dimension, at",
        node_def.name());
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::ISoftMaxLayer* layer =
      params->converter->network()->addSoftMax(*tensor->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);
  // Tensorflow SoftMax assumes applying softmax on the last dimension.
  layer->setAxes(1 << (num_trt_dims - 1));

  ITensorProxyPtr output_tensor = layer->getOutput(0);
  // Quantization range for SoftMax is always (0, 1)
  params->converter->ProvideQuantizationRange(&output_tensor, 0.0f, 1.0f);
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
  // INT64 outputs are not supported by TRT.
  TFAttrs attrs(node_def);
  DataType output_dtype = attrs.get<DataType>("output_type");
  if (output_dtype != DataType::DT_INT32) {
    return errors::Unimplemented("Output type ", DataTypeString(output_dtype),
                                 " is not supported, at ", node_def.name());
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
      output_indices_tensor, &input_dims, params, &output_tensor));
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
  TFAttrs attrs(node_def);
  const bool sorted = attrs.get<bool>("sorted");
  if (!sorted) {
    // TensorRT only supports sorted output. Although TensorFlow API
    // doesn't specify the order of output elements in case sorted=false,
    // but it's safer to not convert because the output of TensorRT might
    // be different with TensorFlow which can cause confusion.
    return errors::InvalidArgument("Only sorted=True is supported, at",
                                   node_def.name());
  }

  ITensorProxyPtr tensor = inputs.at(0).tensor();
  const int num_dims = tensor->getDimensions().nbDims;
  if (num_dims == 0) {
    return errors::InvalidArgument(
        "TensorRT TopK cannot apply on batch dimension, at", node_def.name());
  }

  TRT_ShapedWeights k_w = inputs.at(1).weights();
  if (k_w.count() != 1) {
    return errors::InvalidArgument("k value of TopK should be a scalar, at",
                                   node_def.name());
  }
  // Note that ITopKLayer always have sorted outputs, so we don't need to handle
  // the 'sorted' attribute of the node.
  if (params->validation_only) return Status::OK();

  const nvinfer1::TopKOperation op = nvinfer1::TopKOperation::kMAX;
  const int k = *(static_cast<int*>(k_w.GetValues()));
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
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
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
#else
  return errors::Internal("Dynamic input requires TRT6");
#endif
}

Status ConvertDepthSpaceShuffle(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  TFAttrs attrs(node_def);
  const int block_size = attrs.get<int64>("block_size");
  if (block_size < 2) {
    return errors::InvalidArgument("Block size must be 2 or greater, at ",
                                   node_def.name());
  }
  const string data_format = attrs.get<string>("data_format");
  if (data_format != "NCHW" && data_format != "NHWC") {
    return errors::Unimplemented("Data format ", data_format,
                                 " is not supported, at ", node_def.name());
  }
  int idx_offset = params->use_implicit_batch ? 0 : 1;
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  const int required_rank = 3 + idx_offset;
  if (dims.nbDims != required_rank) {
    return errors::InvalidArgument("The input to ", node_def.op(),
                                   " must be rank 4, at ", node_def.name());
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
          "Number of channels must be divisible by block_size*block_size, at ",
          node_def.name());
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
          "Width and height must be divisible by block_size, at ",
          node_def.name());
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

  ITensorProxyPtr input_tensor = inputs.at(0).tensor();
  ITensorProxyPtr first_shuffle_tensor = first_shuffle->getOutput(0);
  ITensorProxyPtr second_shuffle_tensor = second_shuffle->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(&input_tensor,
                                                        &first_shuffle_tensor);
  params->converter->MarkQuantizationRangesAsInferrable(&first_shuffle_tensor,
                                                        &second_shuffle_tensor);
  params->outputs->push_back(TRT_TensorOrWeights(second_shuffle_tensor));
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

#if IS_TRT_VERSION_GE(7, 1, 3, 0)

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
        "TensorRT BatchedNMS Plugin input boxes must be 4-D including batch ",
        node_def.name());
  }
  const int class_idx = 1 + offset;
  const int num_classes = scores_dims.d[class_idx];
  const int num_boxes = boxes_dims.d[0 + offset];
  bool box_check =
      boxes_dims.d[class_idx] == 1 || boxes_dims.d[class_idx] == num_classes;
  if (!box_check) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin third dimension of boxes must be either 1 "
        "or num_classes ",
        node_def.name());
  }

  if (output_size_per_class.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class must be scalar ",
        node_def.name());
  }
  int max_size_per_class =
      *(static_cast<int*>(output_size_per_class.GetValues()));
  if (max_size_per_class <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class should be > 0",
        node_def.name());
  }
  if (total_size.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size must be scalar ",
        node_def.name());
  }
  int max_total_size = *(static_cast<int*>(total_size.GetValues()));
  if (max_total_size <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size should be > 0",
        node_def.name());
  }
  if (iou_threshold.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be scalar ",
        node_def.name());
  }
  float iou_thresh = *(static_cast<float*>(iou_threshold.GetValues()));
  if (iou_thresh < 0.0 || iou_thresh > 1.0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be in [0, 1]",
        node_def.name());
  }
  if (score_threshold.count() != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin score_threshold must be scalar ",
        node_def.name());
  }

  // TRT op is_normalized=False treats input corrdinates as pixels and
  // calculates width/height as (max - min + 1).
  //
  // TF op CombinedNonMaxSuppression doesn't care about the normalization and
  // calculates width/height  as (max-min).
  //
  // We set is_normalized = true to be consistent with TF IOU calculaton.
  const bool is_normalized = true;

  TFAttrs attrs(node_def);
  bool share_location = (boxes_dims.d[class_idx] == 1);
  const bool pad_per_class = attrs.get<bool>("pad_per_class");
  const bool clip_boxes = attrs.get<bool>("clip_boxes");
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
  float score_thresh = *(static_cast<float*>(score_threshold.GetValues()));
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

  // TRT6 fixes (removes) the extra last dimension in CombinedNMS outputs
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

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
Status ConvertResize(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"size", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  // Get input tensor. Transpose it from NHWC to NCHW.
  ITensorProxyPtr inputs_tensor = inputs.at(0).tensor();

  TFTRT_RETURN_ERROR_IF_NULLPTR(inputs_tensor, params->node_def.name());

  // Get output size. It must constain two values i.e. [H_out, W_out]
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.count() != 2) {
    return errors::Unimplemented("Resize to shape=[] is not supported, at ",
                                 node_def.name());
  }
  const int* weights_ptr = static_cast<int*>(weights.GetValues());

  // Verify and consume node attributes.
  TFAttrs attrs(node_def);
  bool align_corners = attrs.get<bool>("align_corners");
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
    return errors::Unimplemented(node_def.op(), " is not yet implemented at ",
                                 node_def.name());
  }

  // Validate inputs_tensor.
  // TODO: Allow dynamic shape for input-1 when shape input tensors are handled.
  const auto inputs_dims = inputs_tensor->getDimensions();
  if (!params->use_implicit_batch && !HasStaticShape(inputs_dims)) {
    return errors::Unimplemented(
        "TensorRT IResizeLayer requires input with static shape");
  }

  // return after validation if only validation is requested.
  if (params->validation_only) return Status::OK();

  // Transpose tensor from NHWC to NCHW format.
  TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
      inputs_tensor, {0, 3, 1, 2}, &inputs_tensor, node_def, "to_NCHW"));

  // Calculate output dimensions.
  // Given input dimensions [N, C, H, W] and output size [H_out, W_out],
  // output dimensions equals [N, C, H_out, W_out]
  nvinfer1::Dims output_dimensions;
  output_dimensions.nbDims = inputs_tensor->getDimensions().nbDims;
  for (int i = 0; i < output_dimensions.nbDims; ++i) {
    output_dimensions.d[i] = inputs_tensor->getDimensions().d[i];
  }
  output_dimensions.d[output_dimensions.nbDims - 2] = weights_ptr[0];
  output_dimensions.d[output_dimensions.nbDims - 1] = weights_ptr[1];

  // Add resize layer.
  nvinfer1::IResizeLayer* layer =
      params->converter->network()->addResize(*inputs_tensor->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->SetLayerName(layer, node_def);

  // Set layer parameters.
  layer->setResizeMode(resize_mode);
  layer->setOutputDimensions(output_dimensions);
  layer->setAlignCorners(align_corners);

  // Get output tensor. Transpose it from NCHW to NHWC.
  ITensorProxyPtr output = layer->getOutput(0);

  TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
      output, {0, 2, 3, 1}, &output, node_def, "to_NHWC"));
  params->outputs->push_back(TRT_TensorOrWeights(output));
  // Success
  return Status::OK();
}  // ConvertResize
#endif  // IS_TRT_VERSION_GE(6, 0, 0, 0)

Status ConvertAddN(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  TFAttrs attrs(node_def);
  const int num_inputs = attrs.get<int64>("N");
  if (num_inputs < 2) {
    return errors::InvalidArgument("AddN requires at least two inputs, at ",
                                   node_def.name());
  }
  if (inputs.size() != num_inputs) {
    return errors::InvalidArgument("Got ", inputs.size(),
                                   " inputs but expected ", num_inputs, ", at ",
                                   node_def.name());
  }
  for (const auto& input : inputs) {
    if (!input.is_tensor() && input.weights().shape_.d[0] != 1) {
      return errors::InvalidArgument(
          "Weights input to AddN is required to have batch dimension 1.");
    }
  }
  if (params->validation_only) return Status::OK();

  // AddN doesn't support broadcast.
  std::vector<ITensorProxyPtr> tensor_inputs;
  for (const auto& input : inputs) {
    if (input.is_tensor()) {
      tensor_inputs.push_back(input.tensor());
    } else {
      auto dims = input.weights().shape_;
      TF_RETURN_IF_ERROR(RemoveBatchDimension(&dims));
      tensor_inputs.push_back(
          params->converter->CreateConstantLayer(input.weights(), dims));
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

static void RegisterValidatableOpConverters(
    std::unordered_map<string, OpConverter>* registration) {
  (*registration)["BiasAdd"] = ConvertBiasAdd;
#if IS_TRT_VERSION_GE(5, 1, 2, 0)
  (*registration)["ClipByValue"] = ConvertClipByValue;
#endif
#if IS_TRT_VERSION_GE(7, 1, 3, 0)
  (*registration)["CombinedNonMaxSuppression"] = ConvertCombinedNMS;
#endif
  (*registration)["AddN"] = ConvertAddN;
  (*registration)["Cast"] = ConvertCast;
  (*registration)["ConcatV2"] = ConvertConcat;
  (*registration)["Const"] = ConvertConst;
  (*registration)["Conv2D"] = ConvertConv2D;
  (*registration)["Conv2DBackpropInput"] = ConvertConv2DBackpropInput;
  (*registration)["DepthToSpace"] = ConvertDepthSpaceShuffle;
  (*registration)["DepthwiseConv2dNative"] = ConvertConv2DDepthwise;
#if IS_TRT_VERSION_GE(7, 1, 3, 0)
  (*registration)["Einsum"] = ConvertEinsum;
#endif
  (*registration)["ExpandDims"] = ConvertExpandDims;
  (*registration)["FusedConv2DBiasActivation"] =
      ConvertFusedConv2DBiasActivation;
  (*registration)["GatherV2"] = ConvertGather;
  (*registration)["LeakyRelu"] = ConvertLeakyRelu;
  (*registration)["MatMul"] = ConvertMatMul;
  (*registration)["Pack"] = ConvertPack;
  (*registration)["Pad"] = ConvertPad;
  (*registration)["Relu6"] = ConvertRelu6;
  (*registration)["Reshape"] = ConvertReshape;
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  (*registration)["Conv3D"] = ConvertConv3D;
  (*registration)["Conv3DBackpropInputV2"] = ConvertConv3DBackpropInputV2;
  for (auto resize_mode : {"ResizeBilinear", "ResizeNearestNeighbor"}) {
    (*registration)[resize_mode] = ConvertResize;
  }
  for (auto pool_op_type : {"AvgPool3D", "MaxPool3D"}) {
    (*registration)[pool_op_type] = ConvertPool3D;
  }
#endif
  (*registration)["Shape"] = ConvertShape;
  (*registration)["Rsqrt"] = ConvertRsqrt;
  (*registration)["Slice"] = ConvertSlice;
  (*registration)["Softmax"] = ConvertSoftmax;
  (*registration)["SpaceToDepth"] = ConvertDepthSpaceShuffle;
  (*registration)["Split"] = ConvertSplit;
  (*registration)["Square"] = ConvertSquare;
  (*registration)["SquaredDifference"] = ConvertSquaredDifference;
  (*registration)["Squeeze"] = ConvertSqueeze;
  (*registration)["StridedSlice"] = ConvertStridedSlice;
  (*registration)["TopKV2"] = ConvertTopK;
  (*registration)["Transpose"] = ConvertTranspose;
  (*registration)["Unpack"] = ConvertUnpack;
  (*registration)["_CopyFromHostToGpu"] = ConvertIdentity;
  for (auto quantization_op_type :
       {"QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3",
        "FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxArgs"}) {
    (*registration)[quantization_op_type] = ConvertQuantize;
  }
  for (const auto& binary_op_pair : *BinaryOperationMap()) {
    (*registration)[binary_op_pair.first] = ConvertBinary;
  }
  for (const auto& activation_op_pair : *ActivationTypeMap()) {
    (*registration)[activation_op_pair.first] = ConvertActivation;
  }
  for (auto pool_op_type : {"AvgPool", "MaxPool"}) {
    (*registration)[pool_op_type] = ConvertPool;
  }
  for (auto normalization_op_type :
       {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"}) {
    (*registration)[normalization_op_type] = ConvertFusedBatchNorm;
  }
  for (const auto& unary_op_pair : *UnaryOperationMap()) {
    (*registration)[unary_op_pair.first] = ConvertUnary;
  }
  for (auto reduce_op_type : {"Sum", "Prod", "Max", "Min", "Mean"}) {
    (*registration)[reduce_op_type] = ConvertReduce;
  }
  for (auto arg_minmax_type : {"ArgMin", "ArgMax"}) {
    (*registration)[arg_minmax_type] = ConvertArgMinMax;
  }
  // The following are no-ops during inference and will not be mapped to any TRT
  // layer.
  for (auto identity_op_type : {"Identity", "Snapshot", "StopGradient"}) {
    (*registration)[identity_op_type] = ConvertIdentity;
  }
  for (auto batch_matmul_type : {"BatchMatMul", "BatchMatMulV2"}) {
    (*registration)[batch_matmul_type] = ConvertBatchMatMul;
  }
}

void TrtNodeValidator::RegisterOpValidators() {
  RegisterValidatableOpConverters(&op_validators_);
}

void Converter::RegisterOpConverters() {
  RegisterValidatableOpConverters(&op_registry_);
}

Status ConvertGraphDefToEngine(
    const GraphDef& gdef, TrtPrecisionMode precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<PartialTensorShape>& input_shapes,
    nvinfer1::ILogger* trt_logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, bool use_calibration,
    const bool use_implicit_batch, bool* convert_successfully,
    TrtShapeOptimizationProfile* profiles, absl::string_view engine_name) {
  engine->reset();
  if (convert_successfully) *convert_successfully = false;

  // Creating converter, TensorRT builder and network
  auto statusor = Converter::Create(precision_mode, use_calibration, trt_logger,
                                    use_implicit_batch, engine_name);
  TF_RETURN_IF_ERROR(statusor.status());
  auto converter = std::move(statusor.ValueOrDie());

  VLOG(1) << "Starting to convert TensorFlow ops to TensorRT layers";
  std::vector<Converter::EngineOutputInfo> output_tensors;
  int num_layers = converter->network()->getNbLayers();
  absl::flat_hash_set<const char*> layer_names;
  // Graph nodes are already topologically sorted during construction
  for (const auto& node_def : gdef.node()) {
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
        return Status(status.code(), error_message);
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
      TFAttrs attrs(node_def);
      DataType tf_dtype = attrs.get<DataType>("T");
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
        std::string error_message =
            absl::StrCat("Converting node ", node_name, ", op=", node_def.op(),
                         layer->getName() ? "create a layer with name collision"
                                          : "create a layer without a name");
        LOG_WARNING_WITH_PREFIX << error_message;
        return errors::Internal(error_message);
      }
    }
    num_layers = new_num_layers;
  }
  TF_RETURN_IF_ERROR(converter->RenameAndMarkOutputTensors(output_tensors));
  if (convert_successfully) *convert_successfully = true;

  // Apply user provided quantization ranges to tensors
  converter->MaybeApplyQuantizationRanges();

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
    std::vector<EngineConnection>* connections, GraphDef* segment_def,
    string* scope_name) {
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

  std::set<string> subgraph_node_names;
  for (const Node* node : subgraph_nodes) {
    subgraph_node_names.insert(node->name());
  }

  std::unordered_map<int, int> old_to_new_id_map;
  // Copy internal nodes to new graphdef
  string local_scope = subgraph_nodes.front()->name();
  for (const Node* node : subgraph_nodes) {
    local_scope = GetCommonNameScope(local_scope, node->name());
    old_to_new_id_map[node->id()] = segment_def->node_size();
    auto snode = segment_def->add_node();
    *snode = node->def();
    if (snode->op() == "Shape") {
      const std::string copy_op_name = snode->name();
      std::string shape_op_name = copy_op_name + "_cpu_result";

      // Add a node to copy the Shape OP output to GPU. Use the Shape OP node
      // name for this new node so that users switch to use the result of this
      // new node without having to change the name of the value they use.
      NodeDef* copy_op = segment_def->add_node();
      copy_op->set_name(copy_op_name);
      copy_op->set_op("_CopyFromHostToGpu");
      *copy_op->add_input() = shape_op_name + ":0";
      tensorflow::DataType type = snode->attr().at("out_type").type();
      AddNodeAttr("T", type, copy_op);
      AddNodeAttr("out_type", type, copy_op);

      // Rename the Shape OP node and add the new name to the set of node names
      // for the engine.
      snode->set_name(shape_op_name);
      subgraph_node_names.insert(shape_op_name);
      VLOG(2) << "Add copy node " << copy_op->DebugString();
    }
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
  *scope_name = local_scope;
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
