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

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/op_types.h"
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
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/strided_slice_op.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
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

bool IsEngineInput(absl::string_view name) {
  return absl::StartsWith(name, IONamePrefixes::kInputPHName);
}
bool IsEngineOutput(absl::string_view name) {
  return absl::StartsWith(name, IONamePrefixes::kOutputPHName);
}

using absl::StrAppend;
using absl::StrCat;

inline Status TfDataTypeToTrt(DataType tf_dtype,
                              nvinfer1::DataType* trt_dtype) {
  switch (tf_dtype) {
    case DataType::DT_FLOAT:
      *trt_dtype = nvinfer1::DataType::kFLOAT;
      break;
    case DataType::DT_HALF:
      *trt_dtype = nvinfer1::DataType::kHALF;
      break;
    case DataType::DT_INT32:
      *trt_dtype = nvinfer1::DataType::kINT32;
      break;
    default:
      return errors::InvalidArgument("Unsupported data type ",
                                     DataTypeString(tf_dtype));
  }
  return Status::OK();
}

inline Status TrtDataTypeToTf(nvinfer1::DataType trt_dtype,
                              DataType* tf_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *tf_dtype = DataType::DT_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *tf_dtype = DataType::DT_HALF;
      break;
    case nvinfer1::DataType::kINT32:
      *tf_dtype = DataType::DT_INT32;
      break;
    default:
      return errors::InvalidArgument("Unsupported data type ",
                                     DebugString(trt_dtype));
  }
  return Status::OK();
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
  TF_CHECK_OK(TfDataTypeToTrt(this->at(key)->type(), &trt_dtype));
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

template <typename Container>
Status TensorShapeArrayToTrtDims(const Container& shape, nvinfer1::Dims* out,
                                 bool ignore_first_dim = false) {
  PartialTensorShape tensor_shape;
  TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(shape, &tensor_shape));
  *out = TensorShapeToTrtDims(tensor_shape, ignore_first_dim);
  return Status::OK();
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

Status ValidateTensorProperties(const string& producer_node_type,
                                const DataType dtype,
                                const PartialTensorShape& shape,
                                const bool use_implicit_batch,
                                bool validation_only,
                                nvinfer1::DataType* trt_dtype,
                                nvinfer1::Dims* trt_dims, int* batch_size) {
  // Convert data type.
  TF_RETURN_IF_ERROR(TfDataTypeToTrt(dtype, trt_dtype));

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
  *trt_dims = TensorShapeToTrtDims(shape,
                                   /*ignore_first_dim=*/use_implicit_batch);
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
  // Following are validations at runtime.

  for (int d = first_trt_dim; d < shape.dims(); ++d) {
    if (shape.dim_size(d) < 0) {
      return errors::InvalidArgument(
          "Input tensor with shape ", shape.DebugString(),
          " has an unknown non-batch dimension at dim ", d);
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

nvinfer1::ITensor* Converter::CreateConstantLayer(
    const TRT_ShapedWeights& weights, const nvinfer1::Dims& dims) {
  nvinfer1::Weights trt_weights = weights.GetTrtWeights();
  nvinfer1::IConstantLayer* layer = network()->addConstant(dims, trt_weights);
  if (!layer) return nullptr;
  nvinfer1::ITensor* trt_tensor = layer->getOutput(0);
#if !IS_TRT_VERSION_GE(5, 1, 3, 0)
  // TODO(laigd): there is a bug in TensorRT 5.0 library that, if we don't set
  // the data type below, it will always be kFLOAT regardless what the data type
  // of the weights is. Once NVIDIA fixes this bug, we should remove the data
  // type setting logic below and test should still pass.
  trt_tensor->setType(trt_weights.type);
#endif
  return trt_tensor;
}

Status CreateBroadcastableScalarConstant(OpConverterParams* params, float value,
                                         const nvinfer1::Dims& dims,
                                         nvinfer1::ITensor** tensor,
                                         const char* dtype_attr_name = "T") {
  nvinfer1::DataType trt_dtype =
      nvinfer1::DataType::kFLOAT;  // Default to FP32.
  TFAttrs attrs(params->node_def);
  if (attrs.count(dtype_attr_name)) {
    DataType dtype = attrs.get<DataType>(dtype_attr_name);
    TF_RETURN_IF_ERROR(TfDataTypeToTrt(dtype, &trt_dtype));
  }

  // In order to be broadcastable, the number of dims has to match.
  nvinfer1::Dims broadcastable_dims(dims);
  for (int i = 0; i < broadcastable_dims.nbDims; i++) {
    broadcastable_dims.d[i] = 1;
  }
  TRT_ShapedWeights weights =
      params->weight_store->GetTempWeights(trt_dtype, broadcastable_dims);
  void* raw_ptr = weights.GetValues();
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      static_cast<float*>(raw_ptr)[0] = value;
      break;
    case nvinfer1::DataType::kHALF:
      static_cast<Eigen::half*>(raw_ptr)[0] = Eigen::half(value);
      break;
    default:
      return errors::InvalidArgument("Unsupported data type ",
                                     DebugString(trt_dtype));
  }
  *tensor = params->converter->CreateConstantLayer(weights, broadcastable_dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, params->node_def.name());
  params->converter->ProvideQuantizationRange(*tensor, value, value);
  return Status::OK();
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

// Returns total number of elements in a TensorRT weights dimensions.
// Returning 0 means either some dim is 0 or the number of dims is 0 (TensorRT
// doesn't allow scalar weights).
// Note that for TF scalar constant, we always convert to dims [1].
int64_t TrtWeightDimsNumElements(const nvinfer1::Dims& dims) {
  if (dims.nbDims == 0) return 0;
  return Prod(dims);
}

// Returns total number of elements in an ITensor dimension.
// Returns 1 if the number of dims is 0 (the total number is fully determined by
// the batch size).
// Returns -1 if any dimension is known.
int64_t TrtTensorDimsNumElements(const nvinfer1::Dims& dims) {
  if (!HasStaticShape(dims)) return -1;
  return Prod(dims);
}

bool DimsHaveSameSize(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs,
                      bool is_tensor) {
  if (is_tensor) {
    return TrtTensorDimsNumElements(lhs) == TrtTensorDimsNumElements(rhs);
  }
  return TrtWeightDimsNumElements(lhs) == TrtWeightDimsNumElements(rhs);
}

// Returns whether both dimensions are fully specified and the total number of
// elements equals.
bool AreDimsStaticWithSameSize(const nvinfer1::Dims& lhs,
                               const nvinfer1::Dims& rhs, bool is_tensor) {
  if (!HasStaticShape(lhs) || !HasStaticShape(rhs)) return false;
  return DimsHaveSameSize(lhs, rhs, is_tensor);
}

bool AreDimsStaticWithDifferentSize(const nvinfer1::Dims& lhs,
                                    const nvinfer1::Dims& rhs, bool is_tensor) {
  if (!HasStaticShape(lhs) || !HasStaticShape(rhs)) return false;
  return !DimsHaveSameSize(lhs, rhs, is_tensor);
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
}

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type,
                                     nvinfer1::Dims dims, Tensor tensor)
    : shape_(dims), type_(type), tensor_(tensor) {}

TRT_ShapedWeights::TRT_ShapedWeights(const TRT_ShapedWeights& rhs)
    : shape_(rhs.shape_), type_(rhs.type_), tensor_(rhs.tensor_) {}

int64_t TRT_ShapedWeights::count() const {
  return TrtWeightDimsNumElements(shape_);
}

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
  return nvinfer1::Weights{type_, GetValues(), count()};
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

// A fake ITensor implementation used to check whether the TF-TRT converter can
// handle specific node. We only need shape and type information, and the
// converter won't (and shouldn't) use this to build the TRT network.
class TRT_TensorOrWeights::SimpleITensor : public nvinfer1::ITensor {
 public:
  SimpleITensor(nvinfer1::DataType trt_dtype, const nvinfer1::Dims& trt_dims)
      : trt_dtype_(trt_dtype), trt_dims_(trt_dims) {}

  void setName(const char* name) override {}

  const char* getName() const override { return ""; }

  void setDimensions(nvinfer1::Dims dimensions) override {
    trt_dims_ = dimensions;
  }

  nvinfer1::Dims getDimensions() const override { return trt_dims_; }

  void setType(nvinfer1::DataType trt_dtype) override {
    trt_dtype_ = trt_dtype;
  }

  nvinfer1::DataType getType() const override { return trt_dtype_; }

  bool isNetworkInput() const override { return false; }

  bool isNetworkOutput() const override { return false; }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) override {}

  bool getBroadcastAcrossBatch() const override { return false; }

  nvinfer1::TensorLocation getLocation() const override {
    // This is arbitrary, since we don't use it.
    return nvinfer1::TensorLocation::kDEVICE;
  }

  void setLocation(nvinfer1::TensorLocation location) override {}

#if IS_TRT_VERSION_GE(5, 0, 0, 0)
  bool setDynamicRange(float min, float max) override { return true; }

  float getDynamicRange() const override { return 0; }
#endif

#if IS_TRT_VERSION_GE(5, 1, 0, 0)
  bool dynamicRangeIsSet() const override { return true; }

  void resetDynamicRange() override {}

  float getDynamicRangeMin() const override { return 0.f; }

  float getDynamicRangeMax() const override { return 0.f; }
#endif

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  void setAllowedFormats(nvinfer1::TensorFormats formats) override {}

  nvinfer1::TensorFormats getAllowedFormats() const override { return 1; }

  bool isShapeTensor() const override { return false; }

  bool isExecutionTensor() const override { return true; }
#endif

 private:
  nvinfer1::DataType trt_dtype_;
  nvinfer1::Dims trt_dims_;
};

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::ITensor* tensor,
                                         int batch_size)
    : tensor_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                                         const nvinfer1::Dims& trt_dims,
                                         int batch_size)
    : simple_itensor_(new SimpleITensor(trt_dtype, trt_dims)),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
    : weights_(weights), initialized_(true), is_tensor_(false) {}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
    : tensor_(rhs.tensor_),
      simple_itensor_(rhs.simple_itensor_),
      batch_size_(rhs.batch_size_),
      weights_(rhs.weights_),
      initialized_(rhs.initialized_),
      is_tensor_(rhs.is_tensor_) {}

void TRT_TensorOrWeights::operator=(const TRT_TensorOrWeights& rhs) {
  tensor_ = rhs.tensor_;
  simple_itensor_ = rhs.simple_itensor_;
  batch_size_ = rhs.batch_size_;
  weights_ = rhs.weights_;
  initialized_ = rhs.initialized_;
  is_tensor_ = rhs.is_tensor_;
}

nvinfer1::ITensor* TRT_TensorOrWeights::tensor() const {
  CHECK(is_tensor());
  return tensor_ == nullptr ? simple_itensor_.get() : tensor_;
}

nvinfer1::Dims TRT_TensorOrWeights::GetTrtDims() const {
  if (is_tensor()) {
    return tensor()->getDimensions();
  } else {
    return weights().shape_;
  }
}

string TRT_TensorOrWeights::DebugString() const {
  string output = "TRT_TensorOrWeights(type=";
  if (is_tensor()) {
    StrAppend(&output, "tensor=", tensorflow::tensorrt::DebugString(*tensor()),
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
void Reorder4(const nvinfer1::DimsNCHW& shape, const T* idata,
              const nvinfer1::DimsNCHW& istrides, T* odata,
              const nvinfer1::DimsNCHW& ostrides) {
  for (int n = 0; n < shape.n(); ++n) {
    for (int c = 0; c < shape.c(); ++c) {
      for (int h = 0; h < shape.h(); ++h) {
        for (int w = 0; w < shape.w(); ++w) {
          odata[n * ostrides.n() + c * ostrides.c() + h * ostrides.h() +
                w * ostrides.w()] = idata[n * istrides.n() + c * istrides.c() +
                                          h * istrides.h() + w * istrides.w()];
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
  const nvinfer1::DimsNCHW istrides = {1, k, s * k * c, c * k};
  const nvinfer1::DimsNCHW ostrides = {c * r * s, r * s, s, 1};
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
  TF_CHECK_OK(TrtDataTypeToTf(trt_dtype, &tf_dtype));
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

static void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
  static mutex plugin_mutex(LINKER_INITIALIZED);
  static bool plugin_initialized = false;
  mutex_lock lock(plugin_mutex);
  if (plugin_initialized) return;

  LOG(INFO) << "Linked TensorRT version: " << GetLinkedTensorRTVersion();
  LOG(INFO) << "Loaded TensorRT version: " << GetLoadedTensorRTVersion();

  plugin_initialized = initLibNvInferPlugins(trt_logger, "");
  if (!plugin_initialized) {
    LOG(ERROR) << "Failed to initialize TensorRT plugins, and conversion may "
                  "fail later.";
  }

  int num_trt_plugins = 0;
  nvinfer1::IPluginCreator* const* trt_plugin_creator_list =
      getPluginRegistry()->getPluginCreatorList(&num_trt_plugins);
  if (!trt_plugin_creator_list) {
    LOG(WARNING) << "Can not find any TensorRT plugins in registry.";
  } else {
    VLOG(1) << "Found the following " << num_trt_plugins
            << " TensorRT plugins in registry:";
    for (int i = 0; i < num_trt_plugins; ++i) {
      if (!trt_plugin_creator_list[i]) {
        LOG(WARNING) << "TensorRT plugin at index " << i
                     << " is not accessible (null pointer returned by "
                        "getPluginCreatorList for this plugin)";
      } else {
        VLOG(1) << "  " << trt_plugin_creator_list[i]->getPluginName();
      }
    }
  }
}

// static
StatusOr<std::unique_ptr<Converter>> Converter::Create(
    TrtPrecisionMode precision_mode, bool use_calibration,
    nvinfer1::ILogger* trt_logger, const bool use_implicit_batch) {
  std::unique_ptr<Converter> converter = absl::WrapUnique(new Converter(
      precision_mode, use_calibration, trt_logger, use_implicit_batch));
  TF_RETURN_IF_ERROR(converter->Init(trt_logger));
  return converter;
}

Converter::Converter(TrtPrecisionMode precision_mode, bool use_calibration,
                     nvinfer1::ILogger* trt_logger,
                     const bool use_implicit_batch)
    : precision_mode_(precision_mode),
      use_calibration_(use_calibration),
      use_implicit_batch_(use_implicit_batch) {
  InitializeTrtPlugins(trt_logger);
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
  Status status = MaybeUpdateBatchSize(batch_size);
  if (!status.ok()) {
    return Status(status.code(), StrCat("Batch size doesn't match for tensor ",
                                        name, ": ", status.error_message()));
  }
  nvinfer1::ITensor* tensor = network()->addInput(name.c_str(), dtype, dims);
  if (tensor == nullptr) {
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
  for (const auto& output : output_tensors) {
    TRT_TensorOrWeights tensor_or_weights;
    TF_RETURN_IF_ERROR(
        GetTensorOrWeights(output.source_tensor_name, &tensor_or_weights));
    if (!tensor_or_weights.is_tensor()) {
      return errors::InvalidArgument("Output ", output.source_tensor_name,
                                     " is weights not tensor");
    }
    nvinfer1::ITensor* tensor = tensor_or_weights.tensor();
    if (tensor == nullptr) {
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
      nvinfer1::IShuffleLayer* layer = network()->addShuffle(*tensor);
      TFTRT_RETURN_ERROR_IF_NULLPTR(
          layer, StrCat("Output Copy for ", tensor->getName()));
      MarkQuantizationRangesAsInferrable(tensor, layer->getOutput(0));
      tensor = layer->getOutput(0);
    }
    tensor->setName(output.dest_node_name.c_str());
    network()->markOutput(*tensor);
    // Set type after marking as output. TRT only supports setType for engine
    // outputs and inputs (type is inferred otherwise).
    tensor->setType(output.trt_dtype);
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

Status Converter::BuildCudaEngine(
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, int max_batch_size,
    size_t max_workspace_size_bytes, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator) {
  VLOG(1) << "Configuring TensorRT builder";
  trt_builder_->setMaxBatchSize(max_batch_size);
  trt_builder_->setGpuAllocator(allocator);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Create a network configuration and use it to build a TRT engine.
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
      trt_builder_->createBuilderConfig());
  builder_config->setMaxWorkspaceSize(max_workspace_size_bytes);
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

  VLOG(1) << "Building TensorRT engine";
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

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  string precision_mode_str;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(precision_mode_, &precision_mode_str));
  string trt_network_name = StrCat(
      "TF:", TF_VERSION_STRING, ", ", "TRT:", GetLoadedTensorRTVersion(), "-",
      "Precision:", precision_mode_str, ", ", "Calibration:", use_calibration_,
      ", ", "Max-Batch-Size:", max_batch_size, ", ",
      "Max-Workspace-Size:", max_workspace_size_bytes);
  VLOG(1) << "Setting TensorRT network name to " << trt_network_name;
  network()->setName(trt_network_name.c_str());
#endif  // #if IS_TRT_VERSION_GE(6, 0, 0, 0)

  VLOG(1) << "Building TensorRT engine";
  engine->reset(trt_builder_->buildCudaEngine(*network()));
#endif
  if (engine->get() == nullptr) {
    return errors::Internal("Failed to build TensorRT engine");
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

Status Converter::TransposeTensor(nvinfer1::ITensor* input_tensor,
                                  const std::vector<int>& order_with_batch_dim,
                                  nvinfer1::ITensor** output_tensor) {
  const auto dims = input_tensor->getDimensions();

  if (order_with_batch_dim.size() - 1 != size_t(dims.nbDims)) {
    return errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
  }
  if (order_with_batch_dim[0] != 0) {
    return errors::Unimplemented(
        "Transpose at batch dimension is not supported.");
  }

  nvinfer1::IShuffleLayer* layer = this->network()->addShuffle(*input_tensor);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Transpose");
  MarkQuantizationRangesAsInferrable(input_tensor, layer->getOutput(0));

  nvinfer1::Permutation permutation;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    permutation.order[i] = order_with_batch_dim[i + 1] - 1;
  }
  VLOG(1) << "TransposeTensor permutation: "
          << DebugString(permutation, dims.nbDims);
  layer->setFirstTranspose(permutation);

  nvinfer1::Dims reshape_dims;
  reshape_dims.nbDims = dims.nbDims;
  for (int32_t i = 0; i < reshape_dims.nbDims; ++i) {
    reshape_dims.d[i] = 0;
    // TODO(aaroey): why not transposing the types as well?
    reshape_dims.type[i] = dims.type[i];
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
      *out_min = Eigen::half_impl::half_to_float(*result.first);
      *out_max = Eigen::half_impl::half_to_float(*result.second);
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

Status Converter::PrepareTensorForShape(const TRT_TensorOrWeights& input,
                                        const nvinfer1::Dims& dims,
                                        const bool validation_only,
                                        nvinfer1::ITensor** tensor) {
  const nvinfer1::Dims input_dims = input.GetTrtDims();
  // If one of input_dims and dims doesn't have static shape, it means some of
  // the dims are unknown or need to be inferred. And we don't do further checks
  // but rely on the caller to not make mistakes.
  // Otherwise we do simple check to make sure the total sizes are the same.
  // If an input is a weight, it is going to become a tensor via
  // CreateConstantLayer. So we can treat it as a tensor for
  // AreDimsStaticWithDifferentSize(). This really only matters for 0-D tensors.
  if (AreDimsStaticWithDifferentSize(input_dims, dims, /*is_tensor=*/true)) {
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

  if (input.is_tensor()) {
    if (DimsEqual(input_dims, dims)) {
      *tensor = input.tensor();
    } else {
      nvinfer1::IShuffleLayer* layer =
          this->network()->addShuffle(*input.tensor());
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Reshape");
      layer->setReshapeDimensions(dims);
      MarkQuantizationRangesAsInferrable(input.tensor(), layer->getOutput(0));
      *tensor = layer->getOutput(0);
    }
  } else {
    *tensor = CreateConstantLayer(input.weights(), dims);
    TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, "TF-TRT Internal Reshape");
    if (precision_mode() == TrtPrecisionMode::INT8 && !use_calibration()) {
      // If we are in int8 mode and not calibrating, we need to explicitly set a
      // quantization range for the output tensor of the IConstantLayer. Here we
      // set the range to [min(weights), max(weights)].
      float min_range = 0.0f;
      float max_range = 0.0f;
      TF_RETURN_IF_ERROR(
          GetWeightRange(input.weights(), &min_range, &max_range));
      // Avoid setting range to 0 because TRT will throw an error. If the
      // weights are zero then the range doesn't matter: using 127.0f should
      // ensure the quantized weight will be exactly zero.
      if (min_range == 0.0f && max_range == 0.0f) {
        min_range = -127.0f;
        max_range = 127.0f;
      }
      ProvideQuantizationRange(*tensor, min_range, max_range);
    }
  }
  return Status::OK();
}

void Converter::MarkQuantizationRangesAsInferrable(nvinfer1::ITensor* input,
                                                   nvinfer1::ITensor* output) {
  quantization_infer_.push_back({input, output});
  quantization_infer_.push_back({output, input});
}

void Converter::ProvideQuantizationRange(nvinfer1::ITensor* tensor,
                                         float min_range, float max_range) {
  float symmetric_range = std::max(std::abs(min_range), std::abs(max_range));
  quantization_ranges_[tensor] = symmetric_range;
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
  std::map<nvinfer1::ITensor*, nvinfer1::ILayer*> tensor_layer;
  std::set<nvinfer1::ITensor*> all_tensors;
  for (int i = 0; i < this->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = this->network()->getLayer(i);
    layer_consumers[layer] = {};
    for (int j = 0; j < layer->getNbInputs(); j++) {
      all_tensors.insert(layer->getInput(j));
    }
    for (int j = 0; j < layer->getNbOutputs(); j++) {
      tensor_layer[layer->getOutput(j)] = layer;
      all_tensors.insert(layer->getOutput(j));
    }
  }
  for (int i = 0; i < this->network()->getNbLayers(); i++) {
    nvinfer1::ILayer* layer = this->network()->getLayer(i);
    layer_consumers[layer] = {};
    for (int j = 0; j < layer->getNbInputs(); j++) {
      nvinfer1::ITensor* input_tensor = layer->getInput(j);
      auto input_layer = tensor_layer.find(input_tensor);
      if (input_layer != tensor_layer.end()) {
        auto consumed_layer = layer_consumers.find(input_layer->second);
        if (consumed_layer != layer_consumers.end()) {
          consumed_layer->second.push_back(layer);
        }
      }
      all_tensors.insert(input_tensor);
    }
  }
  // Identify fused tensors.
  // Conv+BiasAdd+Add+Activation(Clip or Relu), Conv+BiasAdd+Add,
  // Conv+BiasAdd+Activation(Clip or Relu), Conv+BiasAdd,
  // Conv+Activation(Clip or Relu) are fused.
  std::set<nvinfer1::ITensor*> fused_tensors;
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
            fused_tensors.insert(fused_layer->getOutput(i));
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
      LOG(WARNING) << "Quantization range was not found for "
                   << tensor->getName() << ". "
                   << "Setting invalid quantization range.";
      // Set the range to something unusable so the engine will fail if it
      // tries to actually use the tensor's range.
      tensor->setDynamicRange(0, 0);
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
// as true are constant weights. true means that the input must be a weight,
// while false means the input must be a tensor. In the future, false will mean
// the input can be a tensor or weight.
Status CheckInputsWeights(
    const OpConverterParams& params,
    const std::vector<std::pair<string, bool>>& inputs_is_weight) {
  const auto& inputs = params.inputs;
  const auto& node_def = params.node_def;
  if (inputs.size() != inputs_is_weight.size()) {
    return errors::InvalidArgument(
        node_def.op(), " got ", inputs.size(), " inputs but expected ",
        inputs_is_weight.size(), ", at ", node_def.name());
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs_is_weight[i].second && inputs.at(i).is_tensor()) {
      return errors::Unimplemented("The input \"", inputs_is_weight[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a constant, at ", node_def.name());
    }
    // TODO(tmorris): Remove this check and provide a method to automatically
    // retrieve an input as a tensor, converting via CreateConstantLayer if it
    // was originally a weight. We will want a caching mechanism to prevent many
    // duplicate constants from being created.
    if (!inputs_is_weight[i].second && inputs.at(i).is_weights()) {
      return errors::Unimplemented("The input \"", inputs_is_weight[i].first,
                                   "\" for ", node_def.op(),
                                   " must be a tensor, at ", node_def.name());
    }
  }
  return Status::OK();
}

Status AllowDataTypes(const OpConverterParams& params,
                      const std::set<DataType>& allowed_dtypes,
                      const char* dtype_attr_name = "T") {
  const auto& node_def = params.node_def;
  TFAttrs attrs(node_def);
  if (!attrs.count(dtype_attr_name)) {
    return errors::InvalidArgument("Attribute with name ", dtype_attr_name,
                                   " not found.");
  }
  const auto op_dtype = attrs.get<DataType>(dtype_attr_name);
  if (!allowed_dtypes.count(op_dtype)) {
    // Build string list of allowed types.
    std::ostringstream ss;
    for (auto it = allowed_dtypes.begin(); it != allowed_dtypes.end(); ++it) {
      if (it != allowed_dtypes.begin()) ss << ", ";
      ss << DataTypeString(*it);
    }
    return errors::Unimplemented("Data type ", DataTypeString(op_dtype),
                                 " is not supported for ", node_def.op(),
                                 ", must be one of [", ss.str(), "], at ",
                                 node_def.name());
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
                           nvinfer1::ITensor* tensor,
                           std::vector<std::pair<int, int>>* padding,
                           nvinfer1::ITensor** padded_tensor) {
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
        *tensor, nvinfer1::DimsHW((*padding)[0].first, (*padding)[1].first),
        nvinfer1::DimsHW((*padding)[0].second, (*padding)[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, params->node_def.name());
    params->converter->MarkQuantizationRangesAsInferrable(
        tensor, pad_layer->getOutput(0));
    *padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
  }
  *padded_tensor = tensor;
  return Status::OK();
}

Status ConvertConv2DHelper(OpConverterParams* params, int group,
                           bool is_conv2d_backprop_input) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TRT_TensorOrWeights backprop_output_size;
  nvinfer1::ITensor* tensor = nullptr;
  if (is_conv2d_backprop_input) {
    // In the case when Conv2dBackpropInput is used for conv2d_transpose, these
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
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  if (params->validation_only) return Status::OK();

  // Transpose to NCHW (NCHW is required for IConvLayer).
  const bool need_transpose = (data_format == "NHWC");
  if (need_transpose) {
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 3, 1, 2}, &tensor));
  }
  // Dimensions of transposed tensor.
  const auto tensor_dim = tensor->getDimensions();

  // group == 0 signifies that this is a depthwise convolution, so set
  // num_groups to size of input's channel dim. For a non-depthwise conv,
  // num_groups will be 1.
  const int num_groups = (group == 0) ? tensor_dim.d[0] : group;

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
    // We use h_index and w_index instead of 1 and 2 because we havent
    // transposed backprop_output_size along with the input.
    auto output_size_weights =
        static_cast<int*>(backprop_output_size.weights().GetValues());
    input_dims = {output_size_weights[h_index], output_size_weights[w_index]};
  } else {
    // Use 1 and 2 because tensor_dim has the dimensions of the transposed
    // input.
    input_dims = {static_cast<int>(tensor_dim.d[1]),
                  static_cast<int>(tensor_dim.d[2])};
  }
  nvinfer1::ITensor* padded_tensor = nullptr;
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
            *tensor, noutput, kernel_size, weights.GetTrtWeights(),
            biases.GetTrtWeights());
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
    layer->setName(node_def.name().c_str());
    layer->setNbGroups(num_groups);
    conv_layer = layer;
  } else {
    nvinfer1::IConvolutionLayer* layer =
        params->converter->network()->addConvolution(
            *tensor, noutput, kernel_size, weights.GetTrtWeights(),
            biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStride(stride);
#if IS_TRT_VERSION_GE(5, 1, 3, 0)
    if (attrs.get<string>("padding") == "SAME") {
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
#else
    layer->setPadding(nvinfer1::DimsHW{padding[0].first, padding[1].first});
#endif
    layer->setName(node_def.name().c_str());
    layer->setNbGroups(num_groups);
    layer->setDilation(dilation);
    conv_layer = layer;
  }
  nvinfer1::ITensor* output_tensor = conv_layer->getOutput(0);
  // Add an extra padding for Deconv because TRT doesn't accept the
  // argument output_shape and thus the TRT output shape could be wrong
  // in case of strides>1.
  if (is_conv2d_backprop_input) {
    auto tf_output_shape =
        static_cast<int*>(backprop_output_size.weights().GetValues());
    nvinfer1::Dims trt_output_shape = output_tensor->getDimensions();
    // What determines the padding size is the difference between the given
    // input_sizes (tf_output_shape) and TRT computed size.
    const int height_diff = tf_output_shape[h_index] - trt_output_shape.d[1];
    const int width_diff = tf_output_shape[w_index] - trt_output_shape.d[2];
    if ((height_diff < 0) || (width_diff < 0)) {
      return errors::InvalidArgument(
          "input_sizes argument of Conv2DBackprop (i.e. output_shape argument "
          "of conv2d_transpose) ",
          "is too small for the given out_backprop argument of Conv2DBackprop "
          "(i.e. input argument of conv2d_transpose). Expect: ",
          "(", tf_output_shape[h_index], ", ", tf_output_shape[w_index],
          ") >= ", "(", trt_output_shape.d[1], ", ", trt_output_shape.d[2],
          ") for op ", node_def.name());
    }
    // Only add a padding layer if padding sizes are larger than 0
    if ((height_diff > 0) || (width_diff > 0)) {
      nvinfer1::DimsHW pre_padding(0, 0);
      nvinfer1::DimsHW post_padding(height_diff, width_diff);
      nvinfer1::IPaddingLayer* padding_layer =
          params->converter->network()->addPadding(*output_tensor, pre_padding,
                                                   post_padding);
      output_tensor = padding_layer->getOutput(0);
    }
  }
  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
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
  nvinfer1::ITensor* input_tensor = inputs.at(0).tensor();
  if (perm.size() - 1 != size_t(input_tensor->getDimensions().nbDims)) {
    return errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
  }
  if (perm[0] != 0) {
    return errors::Unimplemented(
        "Transpose at batch dimension is not supported.");
  }

  if (params->validation_only) return Status::OK();

  // Start conversion.
  nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(
      params->converter->TransposeTensor(input_tensor, perm, &output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertReshape(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"tensor", false}, {"shape", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  const TRT_TensorOrWeights& input_tensor = inputs.at(0);
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.count() == 0) {
    return errors::Unimplemented("Reshape to shape=[] is not supported, at ",
                                 node_def.name());
  }

  const int* weights_ptr = static_cast<int*>(weights.GetValues());

  // Check that it doesn't change the batch dimension. This check is
  // conservative, for example, when the first dim of the shape is -1 and input
  // tensor shape is not fixed, it is still possible that the reshape doesn't
  // change the batch dim, but as long as there is a possibility that it could
  // change the batch dim, it reject the conversion. The parameters are:
  //
  // * reshape_batch_dim: the value of the first dim of the input shape constant
  // * reshape_dims: all other dims of the input shape constant
  // * input_batch_dim: the value of the first dim of the input tensor to
  //   reshape
  // * input_dims: all other dims of the input tensor to reshape
  //
  // The validation logic is:
  //
  // if input_batch_dim is fixed:
  //   if reshape_batch_dim == input_batch_dim:
  //     ok
  //   elif reshape_batch_dim == -1 (meaning reshape_dims are fixed) and
  //        input_dims are fixed and
  //        prod(input_dims) == prod(reshape_dims)
  //     ok
  //   else:
  //     not ok
  // elif input_dims are fixed:
  //   if reshape_dims are fixed and
  //      prod(input_dims) == prod(reshape_dims):
  //     ok
  //   else:
  //     not ok
  // else:
  //   not ok
  //
  // Note that the following is ok no matter whether reshape_batch_dim is fixed
  // or not:
  //
  // ```
  // input_batch_dim is not fixed &&
  //     reshape_dims are fixed &&
  //     prod(input_dims) == prod(reshape_dims),
  // ```
  //
  // because the non-batch dims of the new and old shapes match, and TF runtime
  // should make sure the batch dim is not changed.

  const int input_batch_dim = input_tensor.batch_size();
  const int reshape_batch_dim = weights_ptr[0];
  const nvinfer1::Dims input_dims = input_tensor.GetTrtDims();

  nvinfer1::Dims reshape_dims;
  reshape_dims.nbDims = weights.count() - 1;
  for (int i = 1; i < weights.count(); i++) {
    reshape_dims.d[i - 1] = weights_ptr[i];
  }

  // Check that it doesn't change the batch dimension according to the logic
  // mentioned above.
  bool reshape_may_change_batch_dim = false;
  if (input_batch_dim > 0) {        // Batch size is fixed.
    if (reshape_batch_dim == -1) {  // Other dims of the shape must be fixed.
      if (!AreDimsStaticWithSameSize(input_dims, reshape_dims,
                                     /*is_tensor=*/true)) {
        reshape_may_change_batch_dim = true;
      }
    } else if (reshape_batch_dim != input_batch_dim) {
      reshape_may_change_batch_dim = true;
    } else {
      // This means (input_batch_dim>0 && input_batch_dim==reshape_batch_dim),
      // and TF runtime should make sure non-batch dims are matched.
    }
  } else if (!AreDimsStaticWithSameSize(input_dims, reshape_dims,
                                        /*is_tensor=*/true)) {
    reshape_may_change_batch_dim = true;
  }
  VLOG(1) << "input_batch_dim=" << input_batch_dim
          << ", input_dims=" << DebugString(input_dims)
          << "\nreshape_batch_dim=" << reshape_batch_dim
          << ", reshape_dims=" << DebugString(reshape_dims);
  if (reshape_may_change_batch_dim) {
    const string msg = StrCat(
        "Reshape on batch dimension is not supported, at ", node_def.name(),
        ". input_batch_dim=", input_batch_dim, ", ", DebugString(input_dims),
        "; reshape_batch_dim=", reshape_batch_dim, ", ",
        DebugString(reshape_dims));
    return errors::Unimplemented(msg);
  }

  // Start conversion.
  nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, reshape_dims, params->validation_only, &output_tensor));
  if (params->validation_only) return Status::OK();

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
                                 /*use_implicit_batch=*/true, &trt_axis));
  if (params->validation_only) return Status::OK();

  // ExpandDims: Insert new dim of size 1.
  input_dims.insert(input_dims.begin() + trt_axis, 1);
  // Reshape tensor.
  nvinfer1::Dims new_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims));
  nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, new_dims, /*validation_only=*/false, &output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
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
  // Mark axes to remove by setting them to 0.
  TFAttrs attrs(node_def);
  auto squeeze_dims = attrs.get<std::vector<int64>>("squeeze_dims");
  if (squeeze_dims.empty()) {
    return errors::Unimplemented(
        "Squeeze is only implemented for explicit dims, at ", node_def.name());
  }
  for (int tf_axis : squeeze_dims) {
    // Make sure axis is valid.
    int trt_axis;
    TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims, node_def.name(),
                                   /*use_implicit_batch=*/true, &trt_axis));
    // Make sure target dimension is size 1.
    if (input_dims[trt_axis] != 1) {
      return errors::InvalidArgument(
          "Dimension ", tf_axis, " with size ", input_dims[trt_axis],
          " cannot be squeezed because it must be size 1, at ",
          node_def.name());
    }
    // Mark dim for removal by setting to 0.
    input_dims[trt_axis] = 0;
  }
  if (params->validation_only) return Status::OK();

  // Remove all dims which are equal to 0.
  input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 0),
                   input_dims.end());
  // Reshape tensor.
  nvinfer1::Dims new_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims));
  nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, new_dims, /*validation_only=*/false, &output_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

template <typename Container>
Status ConvertStridedSliceHelper(OpConverterParams* params,
                                 const TRT_TensorOrWeights& input,
                                 Container begin, Container size,
                                 const Container& stride,
                                 const nvinfer1::Dims* final_shape = nullptr) {
  const auto& node_def = params->node_def;
  // Get input dims.
  nvinfer1::Dims dims = input.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  // Temporarily add batch dimension so that indexes line up properly.
  input_dims.insert(input_dims.begin(), -1);
  // Check bounds.
  for (int i = 1; i < input_dims.size(); i++) {
    if (begin[i] < 0 || begin[i] > input_dims[i]) {
      return errors::InvalidArgument("\"begin\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
    const int end = begin[i] + size[i];
    if (end < 0 || end > input_dims[i]) {
      return errors::InvalidArgument("\"begin\" + \"size\" for dimension ",
                                     std::to_string(i), " in ", node_def.op(),
                                     " is out of range, at ", node_def.name());
    }
    if (size[i] <= 0) {
      return errors::InvalidArgument("\"size\" cannot be negative or zero for ",
                                     node_def.op(), ", at ", node_def.name());
    }
  }
// TRT 5.1 adds ISliceLayer. For older versions, we attempt to use the
// padding layer with negative padding.
#if IS_TRT_VERSION_GE(5, 1, 3, 1)
  nvinfer1::Dims begin_dims, size_dims, stride_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(begin, &begin_dims,
                                               /*ignore_first_dim=*/true));
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(size, &size_dims,
                                               /*ignore_first_dim=*/true));
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(stride, &stride_dims,
                                               /*ignore_first_dim=*/true));
  if (params->validation_only) return Status::OK();

  nvinfer1::ISliceLayer* layer = params->converter->network()->addSlice(
      *input.tensor(), begin_dims, size_dims, stride_dims);
  nvinfer1::ITensor* tensor = layer->getOutput(0);
  // Reshape for shrink_axis.
  if (final_shape) {
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(tensor), *final_shape, /*validation_only=*/false,
        &tensor));
  }
  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return Status::OK();
#else
  // Use IPaddingLayer.
  // Strides must be 1 in this case.
  for (int x : stride) {
    if (x != 1) {
      return errors::Unimplemented(
          "Strides other than 1 are not supported with this version of TRT, "
          "at ",
          node_def.name());
    }
  }
  // Rank must be 2, 3 or 4.
  if (input_dims.size() > 4) {
    return errors::Unimplemented(node_def.op(),
                                 " for tensors with rank > 4 is not supported "
                                 "in this version of TRT, at ",
                                 node_def.name());
  }
  // Reshape if necessary to 4-D, since IPaddingLayer requires a 4-D input.
  const bool need_reshape = (input_dims.size() != 4);
  int reshape_dims_added = 0;
  nvinfer1::Dims reshape_dims;
  if (need_reshape) {
    // Add new dims after batch dim until tensor is 4D.
    while (input_dims.size() < 4) {
      input_dims.insert(input_dims.begin() + 1, 1);
      begin.insert(begin.begin() + 1, 0);
      size.insert(size.begin() + 1, 1);
      reshape_dims_added++;
    }
    TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &reshape_dims,
                                                 /*ignore_first_dim=*/true));
  }
  // Find dimensions which need to be sliced.
  std::vector<int> pad_dims;
  for (int i = 1; i < input_dims.size(); i++) {
    if ((begin[i] != 0) || (begin[i] + size[i] != input_dims[i])) {
      pad_dims.push_back(i);
    }
  }
  if (pad_dims.empty()) {
    // No dimensions are changed, so this is a no-op. We could just return the
    // input without creating a new layer. TRT will crash if an empty engine
    // with no layers is attempted to be created, so we add a no-op shuffle to
    // prevent our unit tests from breaking.
    // TODO(tmorris): Allow empty engines in the unit tests and return the input
    // as output here.
    if (params->validation_only) return Status::OK();
    nvinfer1::IShuffleLayer* layer =
        params->converter->network()->addShuffle(*input.tensor());
    params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
    return Status::OK();
  } else if (pad_dims.size() == 1) {
    // Only one dim is modified but we have to have 2, mark a second dim which
    // will have padding of 0. The dim we add is chosen to avoid an unnecessary
    // transpose.
    if (pad_dims[0] != 2) {
      pad_dims.push_back(2);
    } else {
      pad_dims.push_back(3);
    }
  } else if (pad_dims.size() > 2) {
    return errors::Unimplemented(
        node_def.op(),
        " can only modify up to 2 dimensions in this version of TRT, at ",
        node_def.name());
  }
  std::sort(pad_dims.begin(), pad_dims.end());
  // Convert to pre/post padding values. Since TRT does not have a StridedSlice
  // or Slice layer prior to 5.1, we instead create an IPaddingLayer with
  // negative padding.
  nvinfer1::DimsHW pre_padding, post_padding;
  for (int i = 0; i < pad_dims.size(); i++) {
    const int axis = pad_dims[i];
    pre_padding.d[i] = -begin[axis];
    post_padding.d[i] = (begin[axis] + size[axis]) - input_dims[axis];
  }

  // IPaddingLayer will always apply the padding to dims 2,3 (input format is
  // NCHW).
  const bool need_transpose = !(pad_dims[0] == 2 && pad_dims[1] == 3);
  std::vector<int> transpose_order(input_dims.size());
  std::vector<int> inv_transpose_order(input_dims.size());
  if (need_transpose) {
    if (pad_dims[0] == 1 && pad_dims[1] == 3) {
      transpose_order = {0, 2, 1, 3};
      inv_transpose_order = {0, 2, 1, 3};
    } else if (pad_dims[0] == 1 && pad_dims[1] == 2) {
      transpose_order = {0, 3, 1, 2};
      inv_transpose_order = {0, 2, 3, 1};
    }
  }
  if (params->validation_only) return Status::OK();

  // Start conversion.
  nvinfer1::ITensor* tensor = input.tensor();
  if (need_reshape) {
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        input, reshape_dims, /*validation_only=*/false, &tensor));
  }
  if (need_transpose) {
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, transpose_order, &tensor));
  }
  // Add padding layer
  nvinfer1::IPaddingLayer* layer = params->converter->network()->addPadding(
      *tensor, pre_padding, post_padding);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->MarkQuantizationRangesAsInferrable(tensor,
                                                        layer->getOutput(0));
  tensor = layer->getOutput(0);
  // Restore transpose
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, inv_transpose_order, &tensor));
  }
  // Reshape for shrink_axis.
  if (final_shape) {
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(tensor), *final_shape, /*validation_only=*/false,
        &tensor));
  } else if (need_reshape) {
    // Restore reshape.
    // Calculate output dimensions
    for (int i = 0; i < pad_dims.size(); i++) {
      const int axis = pad_dims[i];
      input_dims[axis] = size[axis];
    }
    // Remove added 1 dimensions
    for (int i = 0; i < reshape_dims_added; i++) {
      int value = input_dims[1];
      if (value != 1) {
        return errors::Internal("StridedSlice error when reshaping, at ",
                                node_def.name());
      }
      input_dims.erase(input_dims.begin() + 1);
    }

    nvinfer1::Dims new_dims;
    TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims,
                                                 /*ignore_first_dim=*/true));
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(tensor), new_dims, /*validation_only=*/false,
        &tensor));
  }

  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return Status::OK();
#endif
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
  input_dims.insert(input_dims.begin(), inputs.at(0).batch_size());
  if (!AllLengthsEqual({input_dims, begin, size})) {
    return errors::InvalidArgument(
        "Length of begin and size arguments must equal rank of input for "
        "Slice, at ",
        node_def.name());
  }
  // Check that batch dimension is unmodified.
  const bool begin_is_modified = begin[0] != 0;
  // If size[0]s is not -1, we can only know if the batch dimension is
  // unmodified when the batch size is defined. When the batch size is
  // undefined, we don't convert to be safe.
  const bool batch_size_is_defined = input_dims[0] > 0;
  const bool size_is_modified =
      size[0] != -1 && (!batch_size_is_defined ||
                        (batch_size_is_defined && size[0] != input_dims[0]));
  if (begin_is_modified || size_is_modified) {
    return errors::Unimplemented(
        "TensorRT does not allow modifications to the batch dimension, at ",
        node_def.name());
  }
  // Size of -1 signifies to take all remaining elements.
  for (int i = 1; i < input_dims.size(); i++) {
    if (size[i] == -1) {
      size[i] = input_dims[i] - begin[i];
    }
  }
  // Stride is 1 for all dims.
  std::vector<int> stride(begin.size(), 1);
  return ConvertStridedSliceHelper(params, inputs.at(0), begin, size, stride);
}

Status ConvertStridedSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(
      *params,
      {{"input", false}, {"begin", true}, {"end", true}, {"strides", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  TFAttrs attrs(node_def);
  // new_axis_mask is not supported.
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
  input_dims.insert(input_dims.begin(),
                    std::max(-1, inputs.at(0).batch_size()));

  const TRT_ShapedWeights& begin_weights = inputs.at(1).weights();
  const TRT_ShapedWeights& end_weights = inputs.at(2).weights();
  const TRT_ShapedWeights& stride_weights = inputs.at(3).weights();
  if (!AllLengthsEqual({begin_weights.ToVector<int>(),
                        end_weights.ToVector<int>(),
                        stride_weights.ToVector<int>()})) {
    return errors::InvalidArgument(
        "Length of begin, end, and stride must be equal, at ", node_def.name());
  }

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

  // Negative or zero strides currently not supported.
  for (int stride : strides) {
    if (stride <= 0) {
      return errors::Unimplemented(
          "Negative or zero stride values are not supported for StridedSlice, "
          "at ",
          node_def.name());
    }
  }

  // If batch dimension is covered by the ellipsis mask, it means it's left
  // untouched. Otherwise we check whether it modifies the batch dimension here.
  if (!(ellipsis_mask & 1) ||
      begin_weights.shape_.nbDims >= input_dims.size()) {
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
  if (shrink_axis_mask & 1) {
    return errors::Unimplemented(
        "TensorRT does not allow modifications to the batch dimension, at ",
        node_def.name());
  }
  // TRT Slice layer uses (begin, size) instead of (begin, end)
  absl::InlinedVector<int64, 4> size(input_dims.size());
  for (int i = 0; i < input_dims.size(); i++) {
    // Divide by stride (round up)
    size[i] = (end[i] - begin[i] + strides[i] - 1) / strides[i];
  }

  // shrink_axis_mask requires a reshape after the slice.
  nvinfer1::Dims final_shape_dims;
  nvinfer1::Dims* final_shape_dims_ptr = nullptr;
  if (shrink_axis_mask) {
    final_shape_dims =
        TensorShapeToTrtDims(final_shape, /*ignore_first_dim=*/true);
    final_shape_dims_ptr = &final_shape_dims;
  }
  return ConvertStridedSliceHelper(params, inputs.at(0), begin, size, strides,
                                   final_shape_dims_ptr);
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
  nvinfer1::ITensor* tensor = nullptr;
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

  // Finished validation checks
  if (params->validation_only) return Status::OK();

  // Transpose to NCDHW (NCDHW is required for IConvLayer).
  const bool need_transpose = is_ndhwc;
  if (need_transpose) {
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 4, 1, 2, 3}, &tensor));
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
            *tensor, noutput, kernel_size_drs, weights.GetTrtWeights(),
            biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);  // change to nd set stride

    // TensorRT 5.1.3 added support for padding modes.
    if (attrs.get<string>("padding") == "SAME") {
      VLOG(2) << "Using SAME padding";
      // SAME_UPPER means that post padding is preferred.
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    layer->setName(node_def.name().c_str());
    layer->setNbGroups(num_groups);
    conv_layer = layer;
  } else {
    nvinfer1::IConvolutionLayer* layer =
        params->converter->network()->addConvolutionNd(
            *tensor, noutput, kernel_size_drs, weights.GetTrtWeights(),
            biases.GetTrtWeights());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    layer->setStrideNd(stride_dhw);

    if (attrs.get<string>("padding") == "SAME") {
      VLOG(2) << "Using SAME padding";
      layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    layer->setName(node_def.name().c_str());
    layer->setNbGroups(num_groups);
    layer->setDilationNd(dilation_dhw);
    conv_layer = layer;
  }
  nvinfer1::ITensor* output_tensor = conv_layer->getOutput(0);

  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 4, 1}, &output_tensor));
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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  if (data_format == "NDHWC") {
    // NDHWC => NCDHW
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 4, 1, 2, 3}, &tensor));
  }

  const nvinfer1::Dims3 stride(tf_stride[d_index], tf_stride[h_index],
                               tf_stride[w_index]);
  const nvinfer1::Dims3 ksize(tf_kernel[d_index], tf_kernel[h_index],
                              tf_kernel[w_index]);

  nvinfer1::IPoolingLayer* layer =
      params->converter->network()->addPoolingNd(*tensor, type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  params->converter->MarkQuantizationRangesAsInferrable(tensor,
                                                        layer->getOutput(0));

  layer->setStrideNd(stride);
  // VALID padding is the default TRT behavior.
  if (padding_type == "SAME") {
    // SAME_UPPER means that post padding is preferred.
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  layer->setName(node_def.name().c_str());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (data_format == "NDHWC") {
    // NCDHW => NDHWC
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 4, 1}, &output_tensor));
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
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
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
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 3, 1, 2}, &tensor));
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
  nvinfer1::ITensor* padded_tensor = nullptr;
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
        *tensor, weights.shape_.d[0], kernel_size, weights.GetTrtWeights(),
        biases.GetTrtWeights());
  } else {
    // For conv, TF weights are RSCK, and TRT expects KCRS.
    DCHECK_EQ(filter_format, "HWIO");
    TRT_ShapedWeights weights_kcrs =
        params->weight_store->GetTempWeights(weights);
    ReorderRSCKToKCRS(weights, &weights_kcrs, 1);
    conv_layer = params->converter->network()->addConvolution(
        *tensor, weights.shape_.d[3], kernel_size, weights_kcrs.GetTrtWeights(),
        biases.GetTrtWeights());
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
  conv_layer->setName(node_def.name().c_str());
  conv_layer->setNbGroups(1);
  conv_layer->setDilation(dilation);
  nvinfer1::ITensor* output_tensor = conv_layer->getOutput(0);

  // Add activation if there is one.
  if (op_pair != ActivationTypeMap()->end()) {
    nvinfer1::IActivationLayer* activation_layer =
        params->converter->network()->addActivation(*output_tensor,
                                                    op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(activation_layer, node_def.name());
    output_tensor = activation_layer->getOutput(0);
  }
  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertPool(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"input", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  int h_index = 2;
  int w_index = 3;
  const auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    h_index = 1;
    w_index = 2;
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 3, 1, 2}, &tensor));
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
        *tensor, nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, node_def.name());
    params->converter->MarkQuantizationRangesAsInferrable(
        tensor, pad_layer->getOutput(0));
    padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
  }
#endif

  nvinfer1::IPoolingLayer* layer =
      params->converter->network()->addPooling(*tensor, type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  // TODO(tmorris): Average pooling may not be entirely safe to infer
  // quantization range through (at least forwards - backwards should be fine).
  // Max pooling is okay.
  params->converter->MarkQuantizationRangesAsInferrable(tensor,
                                                        layer->getOutput(0));

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
  layer->setName(node_def.name().c_str());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (data_format == "NHWC") {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, &output_tensor));
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
          *inputs.at(0).tensor(), nvinfer1::ActivationType::kLEAKY_RELU);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  // Create const for alpha.
  nvinfer1::ITensor* const_alpha_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, alpha, tensor->getDimensions(), &const_alpha_tensor));
  // alpha * x
  nvinfer1::IElementWiseLayer* mul_layer =
      params->converter->network()->addElementWise(
          *tensor, *const_alpha_tensor, nvinfer1::ElementWiseOperation::kPROD);
  TFTRT_RETURN_ERROR_IF_NULLPTR(mul_layer, node_def.name());
  // max(x, alpha * x)
  nvinfer1::IElementWiseLayer* max_layer =
      params->converter->network()->addElementWise(
          *tensor, *mul_layer->getOutput(0),
          nvinfer1::ElementWiseOperation::kMAX);
  TFTRT_RETURN_ERROR_IF_NULLPTR(max_layer, node_def.name());
  nvinfer1::ITensor* output_tensor = max_layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(
      output_tensor, mul_layer->getOutput(0));

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
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
    clip_value_min = Eigen::half_impl::half_to_float(
        inputs.at(1).weights().GetSpan<Eigen::half>()[0]);
    clip_value_max = Eigen::half_impl::half_to_float(
        inputs.at(2).weights().GetSpan<Eigen::half>()[0]);
  }

  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *inputs.at(0).tensor(), nvinfer1::ActivationType::kCLIP);
  layer->setAlpha(clip_value_min);
  layer->setBeta(clip_value_max);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->converter->ProvideQuantizationRange(output_tensor, clip_value_min,
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
      params->converter->network()->addActivation(*inputs.at(0).tensor(),
                                                  op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // Set quantization range for output when known.
  if (node_def.op() == "Sigmoid") {
    params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 1.0f);
  } else if (node_def.op() == "Tanh") {
    params->converter->ProvideQuantizationRange(output_tensor, -1.0f, 1.0f);
  } else if (node_def.op() == "Softsign") {
    params->converter->ProvideQuantizationRange(output_tensor, -1.0f, 1.0f);
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
  params->converter->ProvideQuantizationRange(inputs.at(0).tensor(), min_range,
                                              max_range);
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
          *inputs.at(0).tensor(), nvinfer1::ActivationType::kCLIP);
  layer->setAlpha(0.0f);
  layer->setBeta(6.0f);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 6.0f);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
#else
  // Convert using min(Relu(x), 6) before TRT 5.1
  // Input Tensor
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  // Relu operation i.e. Relu(x) = max(0, x)
  nvinfer1::IActivationLayer* relu_layer =
      params->converter->network()->addActivation(
          *tensor, nvinfer1::ActivationType::kRELU);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu_layer, node_def.name());

  // Large range of relu is problematic during quantization in INT8 precision
  // mode. Setting dynamic range of relu = [0.f, 6.0f] helps with quantization.
  // TRT only uses dynamic ranges in INT8 precision mode,
  // and this does not affect the FP32 path.
  params->converter->ProvideQuantizationRange(relu_layer->getOutput(0), 0.0f,
                                              6.0f);

  // Create a constant layer to store the floating point weight i.e. 6.0f
  nvinfer1::ITensor* const6_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, 6.0f, relu_layer->getOutput(0)->getDimensions(), &const6_tensor));

  // ElementWise Min Operation
  // Min op is a nop for INT8 execution path, as the input tensor
  // to this layer will only have values in range [0.f, 6.0f].
  nvinfer1::IElementWiseLayer* relu6_layer =
      params->converter->network()->addElementWise(
          *relu_layer->getOutput(0), *const6_tensor,
          nvinfer1::ElementWiseOperation::kMIN);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu6_layer, node_def.name());
  nvinfer1::ITensor* output_tensor = relu6_layer->getOutput(0);
  params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 6.0f);

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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
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
        params->converter->network()->addShuffle(*tensor);
    TFTRT_RETURN_ERROR_IF_NULLPTR(shuffle_layer, node_def.name());
    params->converter->MarkQuantizationRangesAsInferrable(
        tensor, shuffle_layer->getOutput(0));

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
    tensor = shuffle_layer->getOutput(0);
  }

  TRT_ShapedWeights weights = inputs.at(1).weights();
  nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
  if (weights.shape_.d[0] == 1) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  }

  TRT_ShapedWeights empty_weights(weights.TrtDType());
  nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
      *tensor, mode, weights.GetTrtWeights(), empty_weights.GetTrtWeights(),
      empty_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // Restore transpose & reshape.
  if (channel_index != 0 || original_dims.nbDims != 3) {
    nvinfer1::IShuffleLayer* shuffle_layer =
        params->converter->network()->addShuffle(*output_tensor);
    TFTRT_RETURN_ERROR_IF_NULLPTR(shuffle_layer, node_def.name());
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
    params->converter->MarkQuantizationRangesAsInferrable(
        output_tensor, shuffle_layer->getOutput(0));
    output_tensor = shuffle_layer->getOutput(0);
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
  // If the input is NCHW, then we need to unsqueeze the bias such that its last
  // dimensions are 1s (and the first dimension is C).
  if (data_format == "NCHW") {
    bias_shape.nbDims = input_shape.nbDims;
    std::fill(bias_shape.d + 1, bias_shape.d + bias_shape.nbDims, 1);
  } else {
    // Next, broadcast the bias across the input.
    TF_RETURN_IF_ERROR(GetTrtBroadcastShape(inputs.at(0), inputs.at(1),
                                            /*check_feasibility=*/true,
                                            params->use_implicit_batch,
                                            &input_shape, &bias_shape));
  }

  // Convert input to a TRT tensor
  nvinfer1::ITensor* input_tensor{nullptr};
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(0), input_shape, params->validation_only, &input_tensor));

  // Finally, reshape bias. Since the bias is usually a constant, this will
  // normally happen at conversion-time.
  nvinfer1::ITensor* bias_tensor{nullptr};
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(1), bias_shape, params->validation_only, &bias_tensor));
  VLOG(2) << "Bias shape adjusted to " << DebugString(bias_shape);

  if (params->validation_only) return Status::OK();

  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *input_tensor, *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

void GetTensorDimsWithProtoShape(const Tensor& tensor, nvinfer1::Dims* dims) {
  if (tensor.dims() > 0) {
    *dims = GetTrtDimsForTensor(tensor);
  } else {
    dims->nbDims = 1;
    // No dimension provided. Flatten it.
    dims->d[0] = tensor.NumElements();
    dims->type[0] = nvinfer1::DimensionType::kSPATIAL;
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
  TF_RETURN_IF_ERROR(TfDataTypeToTrt(converted_dtype, &trt_dtype));

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
#if IS_TRT_VERSION_GE(5, 1, 0, 0)
        // This op applies Floor after Div.
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
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

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
  nvinfer1::ITensor* tensor_l = nullptr;
  nvinfer1::ITensor* tensor_r = nullptr;
  // This will also convert constants to tensors, and set quantization ranges.
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      operand_l, broadcasted_dims_l, params->validation_only, &tensor_l));
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      operand_r, broadcasted_dims_r, params->validation_only, &tensor_r));
  if (params->validation_only) return Status::OK();

  // Add ElementWise layer.
  nvinfer1::ILayer* layer = params->converter->network()->addElementWise(
      *tensor_l, *tensor_r, op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* trt_tensor = layer->getOutput(0);

#if IS_TRT_VERSION_GE(5, 1, 0, 0)
  if (node_def.op() == "FloorDiv") {
    layer = params->converter->network()->addUnary(
        *trt_tensor, nvinfer1::UnaryOperation::kFLOOR);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  // Sqrt
  nvinfer1::IUnaryLayer* sqrt_layer = params->converter->network()->addUnary(
      *tensor, nvinfer1::UnaryOperation::kSQRT);
  TFTRT_RETURN_ERROR_IF_NULLPTR(sqrt_layer, node_def.name());
  // Recip
  nvinfer1::IUnaryLayer* recip_layer = params->converter->network()->addUnary(
      *sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
  TFTRT_RETURN_ERROR_IF_NULLPTR(recip_layer, node_def.name());
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
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  nvinfer1::IUnaryLayer* layer =
      params->converter->network()->addUnary(*tensor, op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // Set quantization ranges.
  if (node_def.op() == "Sin" || node_def.op() == "Cos") {
    params->converter->ProvideQuantizationRange(output_tensor, -1.0f, 1.0f);
  } else if (node_def.op() == "Asin" || node_def.op() == "Atan") {
    params->converter->ProvideQuantizationRange(output_tensor, -M_PI_2, M_PI_2);
  } else if (node_def.op() == "Acos") {
    params->converter->ProvideQuantizationRange(output_tensor, 0.0f, M_PI);
  } else if (node_def.op() == "Neg" || node_def.op() == "Abs") {
    // Neg and Abs will have same range as input since TRT uses symmetric
    // quantization.
    // TODO(tmorris): Should we infer ranges for Ceil and Floor as well?
    params->converter->MarkQuantizationRangesAsInferrable(tensor,
                                                          output_tensor);
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertSquare(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (params->validation_only) return Status::OK();

  // Constant 2 with same rank as input
  nvinfer1::ITensor* const2_tensor = nullptr;
  TF_RETURN_IF_ERROR(CreateBroadcastableScalarConstant(
      params, 2.0f, inputs.at(0).GetTrtDims(), &const2_tensor));

  // ElementWise Pow Operation
  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *inputs.at(0).tensor(), *const2_tensor,
          nvinfer1::ElementWiseOperation::kPOW);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
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
                    node_def.name(), /*use_implicit_batch=*/true, &trt_axis));
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
      *tensor, reduce_operation, axes, keep_dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

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

  // Validate inputs. Values must be tensors for now.
  std::vector<std::pair<string, bool>> inputs_is_weight;
  for (int i = 0; i < num_inputs; ++i) {
    inputs_is_weight.push_back({StrCat("values_", i), false});
  }
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, inputs_is_weight));

  // TODO(hinsu): Enable INT32 with TensorRT version 5.1.3 after testing.
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  if (num_inputs > 1) {
    // Verify that inputs are compatible for concatenation after the expansion.
    TF_RETURN_IF_ERROR(
        VerifyShapesMatch(inputs, /*masked_dim=*/-1, node_def.name()));
  }

  // Convert axis from the TensorFlow format to TensorRT format.
  const nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  const int64 tf_axis = attrs.get<int64>("axis");
  int trt_axis;
  TF_RETURN_IF_ERROR(ConvertAxis(tf_axis, dims.nbDims + 1, node_def.name(),
                                 /*use_implicit_batch=*/true, &trt_axis));

  // Compute expanded dimensions and then reshape input tensors.
  std::vector<int> tensor_dims(dims.d, dims.d + dims.nbDims);
  tensor_dims.insert(tensor_dims.begin() + trt_axis, 1);
  nvinfer1::Dims expanded_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(tensor_dims, &expanded_dims));
  std::vector<nvinfer1::ITensor*> expanded_tensors;
  for (const TRT_TensorOrWeights& tensor : inputs) {
    nvinfer1::ITensor* expanded_tensor = nullptr;
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        tensor, expanded_dims, params->validation_only, &expanded_tensor));
    if (!params->validation_only) {
      expanded_tensors.push_back(expanded_tensor);
    }
  }
  if (params->validation_only) return Status::OK();

  // If there is only one tensor in the input, return the expanded tensor.
  if (num_inputs == 1) {
    params->outputs->push_back(TRT_TensorOrWeights(expanded_tensors[0]));
    return Status::OK();
  }

  // Otherwise, concatenate expanded tensors.
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          const_cast<nvinfer1::ITensor**>(expanded_tensors.data()),
          expanded_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));

  // Implement tensor binaryOp weight [channel wise] for now;
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  const auto dims = tensor->getDimensions();
  // Restore implicit batch dimension
  const int nb_dims = dims.nbDims + 1;

  TRT_ShapedWeights pads = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  // Padding type here is done through TF type
  //   so I can leverage their EnumToDataType for my cast
  auto padding_type = attrs.get<DataType>("Tpaddings");
  // TODO(jie): handle data type conversion for TRT?

  if (pads.shape_.d[0] != nb_dims || pads.shape_.d[1] != 2) {
    return errors::InvalidArgument(
        "Pad only supports explicit padding on 4 dimensional tensor, at ",
        node_def.name());
  }

  // Only expect to handle INT32 as attributes for now
  if (padding_type != DataType::DT_INT32) {
    return errors::Unimplemented("Tpaddings supports only DT_INT32");
  }
  auto pad_data = static_cast<int*>(pads.GetValues());

  std::vector<int32_t> pad_index;
  for (int i = 0; i < nb_dims; i++) {
    if (pad_data[2 * i] != 0 || pad_data[2 * i + 1] != 0) {
      pad_index.push_back(i);
    }
  }

  // No padding at all, we should exit
  if (pad_index.empty()) {
    params->outputs->push_back(inputs.at(0));
    return Status::OK();
  }

  // Only supports padding on less than 2 axis GIE-2579
  if (pad_index.size() > 2) {
    return errors::InvalidArgument(
        "Padding layer does not support padding on > 2");
  }

  // Padding on batch dimension is not supported
  if (pad_index[0] == 0) {
    return errors::InvalidArgument(
        "Padding layer does not support padding on batch dimension");
  }

  // Not doing the legit thing here. ignoring padding on dim 1 and 3;
  // TODO(jie): implement pad as uff parser
  if (pad_index.size() == 2 && pad_index[0] == 0 && pad_index[1] == 3) {
    return errors::Unimplemented(
        "Padding layer does not support padding on dimension 1 and 3 yet");
  }
  if (params->validation_only) return Status::OK();

  bool legit_pad = true;
  nvinfer1::DimsHW pre_padding(0, 0);
  nvinfer1::DimsHW post_padding(0, 0);

  std::vector<int32_t> permuted_pad_index(pad_index);
  if (pad_index[0] == 1) {
    legit_pad = false;
    TF_RETURN_IF_ERROR(
        params->converter->TransposeTensor(tensor, {0, 3, 2, 1}, &tensor));
    permuted_pad_index[0] = 3;
  }

  for (size_t i = 0; i < pad_index.size(); i++) {
    int index = pad_index[i];
    if (permuted_pad_index[i] == 2) {
      pre_padding.h() = pad_data[index * 2];
      post_padding.h() = pad_data[index * 2 + 1];
    } else if (permuted_pad_index[i] == 3) {
      pre_padding.w() = pad_data[index * 2];
      post_padding.w() = pad_data[index * 2 + 1];
    }
  }

  nvinfer1::IPaddingLayer* layer = params->converter->network()->addPadding(
      *tensor, pre_padding, post_padding);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(tensor, output_tensor);

  if (!legit_pad) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 3, 2, 1}, &output_tensor));
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
                                 /*use_implicit_batch=*/true, &trt_axis));
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
        " is not evenly divisble by ", num_splits, ", at ", node_def.name());
  }

  // Create parameters for StridedSliceHelper.
  // Slice will begin on zero for all dims, except the one being split which
  // will change.
  std::vector<int> begin(dims.nbDims, 0);
  // Determine size of split. Slice will get the full length of all dims, except
  // the one being split.
  std::vector<int> size(dims.d, dims.d + dims.nbDims);
  const int split_size_on_axis = dims.d[trt_axis] / num_splits;
  size[trt_axis] = split_size_on_axis;
  // Stride will always be 1
  std::vector<int> stride(dims.nbDims, 1);
  // Add dummy batch dimension
  begin.insert(begin.begin(), 0);
  size.insert(size.begin(), 1);
  stride.insert(stride.begin(), 1);
  // Create final shape for Unpack/Unstack, where split axis is squeezed.
  nvinfer1::Dims final_shape_for_unpack;
  nvinfer1::Dims* final_shape_for_unpack_ptr = nullptr;
  if (squeeze_after) {
    std::vector<int> size_after_squeeze(size);
    size_after_squeeze.erase(size_after_squeeze.begin() + trt_axis + 1);
    TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(
        size_after_squeeze, &final_shape_for_unpack, /*ignore_frst_dim=*/true));
    final_shape_for_unpack_ptr = &final_shape_for_unpack;
  }

  // Slice the input. ConvertStridedSliceHelper will push the outputs onto
  // params->outputs.
  for (int i = 0; i < num_splits; ++i) {
    begin[trt_axis + 1] = i * split_size_on_axis;
    TF_RETURN_IF_ERROR(ConvertStridedSliceHelper(
        params, input, begin, size, stride, final_shape_for_unpack_ptr));
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
  // Validate inputs. Values must be tensors for now.
  std::vector<std::pair<string, bool>> inputs_is_weight;
  for (int i = 0; i < num_inputs; ++i) {
    inputs_is_weight.push_back({StrCat("values_", i), false});
  }
  inputs_is_weight.push_back({"axis", true});
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, inputs_is_weight));
  // TODO(tmorris): There is a bug with Concat and INT32 in TRT - it is supposed
  // to be supported.
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  const auto axis = inputs.at(num_inputs).weights().GetSpan<int>();
  if (axis.size() != 1) {
    return errors::InvalidArgument("Axis for ConcatV2 must be a scalar, at ",
                                   node_def.name());
  }
  int trt_axis = 0;
  const auto dim = inputs.at(0).GetTrtDims();
  TF_RETURN_IF_ERROR(ConvertAxis(axis[0], dim.nbDims, node_def.name(),
                                 /*use_implicit_batch=*/true, &trt_axis));
  // Check that dimensions match on non-concatenate axis.
  TF_RETURN_IF_ERROR(VerifyShapesMatch(
      absl::Span<const TRT_TensorOrWeights>(inputs).first(num_inputs), trt_axis,
      node_def.name()));
  if (params->validation_only) return Status::OK();

  // Gather inputs as tensors
  std::vector<nvinfer1::ITensor const*> input_tensors;
  for (int i = 0; i < num_inputs; i++) {
    input_tensors.push_back(inputs.at(i).tensor());
  }
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          const_cast<nvinfer1::ITensor* const*>(input_tensors.data()),
          input_tensors.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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
    LOG(WARNING) << node_def.op() << " only supports is_training=false. If you "
                 << "are using Keras, please call "
                 << "keras.backend.set_learning_phase(0) before constructing "
                 << "your model. At " << node_def.name();
    return errors::Unimplemented(node_def.op(),
                                 " only supports is_training=false, at ",
                                 node_def.name());
  }
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();

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
          batchnorm_data[j] =
              Eigen::half_impl::half_to_float(cast_vals_array[j][i]);
        }
      } else {
        if (parameter_type == nvinfer1::DataType::kFLOAT) {
          batchnorm_data[j] = vals_array[j][0];
        } else if (parameter_type == nvinfer1::DataType::kHALF) {
          batchnorm_data[j] =
              Eigen::half_impl::half_to_float(cast_vals_array[j][0]);
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
      *tensor, mode, combined_offset_weights.GetTrtWeights(),
      combined_scale_weights.GetTrtWeights(),
      dummy_power_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertGather(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // TODO(tmorris): Use CheckInputsWeights by changing bool to enum with an
  // option for an input to be either tensor or weight.
  if (inputs.size() != 3) {
    return errors::InvalidArgument("GatherV2 got ", inputs.size(),
                                   " inputs but expected 3, at ",
                                   node_def.name());
  }
  const auto& params_input = inputs.at(0);
  const auto& indices_input = inputs.at(1);
  const auto& axis_input = inputs.at(2);
  if (!axis_input.is_weights()) {
    return errors::Unimplemented(
        "The input \"axis\" for GatherV2 must be a constant, at ",
        node_def.name());
  }
  if (!indices_input.is_tensor()) {
    return errors::Unimplemented(
        "The input \"indices\" for GatherV2 must be a tensor, at ",
        node_def.name());
  }

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
  TF_RETURN_IF_ERROR(ConvertAxis(axis[0], params_input.GetTrtDims().nbDims,
                                 node_def.name(), params_input.is_tensor(),
                                 &trt_axis));
  if (params_input.is_weights() && trt_axis != 0) {
    return errors::Unimplemented(
        "The input axis must be zero when params is a weight.");
  }
  if (params_input.is_tensor() && indices_input.batch_size() != 1) {
    return errors::Unimplemented(
        "Indices must have a batch size of 1 when params is a tensor.");
  }
  // Both input are tensors, and the TF gather result will have rank:
  // (params.nbDims + 1) + (indices.nbDims + 1) - 1,
  // where "+ 1" adds the batch dim. If params is a weight, the TRT rank matches
  // the TF rank so we don't have to add + 1.
  const int params_tf_rank =
      params_input.GetTrtDims().nbDims + (params_input.is_tensor() ? 1 : 0);
  const int indices_tf_rank = indices_input.GetTrtDims().nbDims + 1;
  const int tf_gather_output_rank = params_tf_rank + indices_tf_rank - 1;
  if (tf_gather_output_rank > nvinfer1::Dims::MAX_DIMS + 1) {
    return errors::InvalidArgument(
        "Result of gather has dimension greater than ",
        nvinfer1::Dims::MAX_DIMS + 1);
  }
  if (params->validation_only) return Status::OK();

  // Convert params to tensor is it is a weight.
  nvinfer1::ITensor* params_tensor = nullptr;
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
      *params_tensor, *indices_input.tensor(), trt_axis);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  nvinfer1::Dims trt_gather_output_dims = output_tensor->getDimensions();
  // Note for the "- 2": one is for the output batch dim encapsulated by TF-TRT,
  // and the other is for the output dimension that is squeezed by IGatherLayer
  // because of the implicit batch dim in the indices (see the above note).
  const int expected_trt_output_rank =
      tf_gather_output_rank - (params_input.is_tensor() ? 2 : 1);
  if (trt_gather_output_dims.nbDims != expected_trt_output_rank) {
    return errors::Internal(
        "Get unexpected output dimensions of IGatherLayer. Expect nbDims: ",
        expected_trt_output_rank,
        ", actual nbDims: ", trt_gather_output_dims.nbDims);
  }
  // Reshape the output so after adding the implicit batch dim it'll match the
  // output shape of TF GatherV2.
  if (params_input.is_tensor()) {
    for (int i = trt_gather_output_dims.nbDims; i > trt_axis; --i) {
      trt_gather_output_dims.d[i] = trt_gather_output_dims.d[i - 1];
    }
    trt_gather_output_dims.d[trt_axis] = 1;
    ++trt_gather_output_dims.nbDims;

    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(output_tensor), trt_gather_output_dims,
        /*validation_only=*/false, &output_tensor));
  }

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertFullyConnectedHelper(OpConverterParams* params,
                                   nvinfer1::ITensor* tensor_a,
                                   TRT_ShapedWeights weights_b,
                                   bool transpose_b, const string& node_name) {
  // Reshape input to 3D - this will be a no-op unless using int8 precision.
  auto input_dim = tensor_a->getDimensions();
  while (input_dim.nbDims < 3) {
    input_dim.d[input_dim.nbDims++] = 1;
  }
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      TRT_TensorOrWeights(tensor_a), input_dim, /*validation_only=*/false,
      &tensor_a));

  // FC layer will transpose weights, so we need to pre-transpose.
  TRT_ShapedWeights weights(weights_b.TrtDType());
  if (!transpose_b) {
    weights = params->weight_store->GetTempWeights(weights_b);
    ReorderCKtoKC(weights_b, &weights);
  } else {
    weights = weights_b;
  }
  TRT_ShapedWeights biases(weights.TrtDType());
  const int noutput = weights.shape_.d[0];
  nvinfer1::IFullyConnectedLayer* layer =
      params->converter->network()->addFullyConnected(
          *tensor_a, noutput, weights.GetTrtWeights(), biases.GetTrtWeights());

  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_name);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // Reshape output to 1D - this will be a no-op unless using int8 precision.
  auto output_dim = output_tensor->getDimensions();
  output_dim.nbDims = 1;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      TRT_TensorOrWeights(output_tensor), output_dim, /*validation_only=*/false,
      &output_tensor));

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertMatMulHelper(OpConverterParams* params,
                           TRT_TensorOrWeights input_a,
                           TRT_TensorOrWeights input_b, bool transpose_a,
                           bool transpose_b, string node_name) {
  // TODO: ReorderCKtoKC is currently not general enough to transpose weights
  // that are not 2D.
  if ((transpose_a && input_a.is_weights() &&
       input_a.GetTrtDims().nbDims != 2) ||
      (transpose_b && input_b.is_weights() &&
       input_b.GetTrtDims().nbDims != 2)) {
    return errors::InvalidArgument(
        "Cannot currently transpose constant input if it is not 2 dimensional");
  }

  // If A is a tensor, we can only transpose if it is at least 3D in TF,
  // or TRT will not do the correct transposition.
  if (transpose_a && input_a.is_tensor() && input_a.GetTrtDims().nbDims < 2) {
    return errors::InvalidArgument(
        "Cannot transpose first input if it is a tensor with fewer than 2 "
        "non-batch dimensions.");
  }

  // If B is a tensor, then it must be at least 3D in TF,
  // or TRT won't be able to handle the multiply correctly.
  if (input_b.is_tensor() && input_b.GetTrtDims().nbDims < 2) {
    return errors::InvalidArgument(
        "Second input must either be a constant, or contain at least 2 "
        "non-batch dimensions.");
  }
  if (params->validation_only) return Status::OK();

  // If an FC layer can be used and would be faster, use that instead.
  const bool can_use_fc =
      !transpose_a && input_a.is_tensor() && input_b.is_weights();
  const bool should_use_fc = can_use_fc && input_a.GetTrtDims().nbDims >= 3 &&
                             input_b.GetTrtDims().nbDims == 2;
  // If int8 is specified, FC must be used unless it is not compatible, as MM
  // does not support int8 at this time.
  if (should_use_fc || (can_use_fc && params->converter->precision_mode() ==
                                          TrtPrecisionMode::INT8)) {
    return ConvertFullyConnectedHelper(
        params, input_a.tensor(), input_b.weights(), transpose_b, node_name);
  }

  const auto get_matrix_op = [](nvinfer1::ITensor* in,
                                bool transpose) -> nvinfer1::MatrixOperation {
    return (in->getDimensions().nbDims < 2)
               ? nvinfer1::MatrixOperation::kVECTOR
               : (transpose) ? nvinfer1::MatrixOperation::kTRANSPOSE
                             : nvinfer1::MatrixOperation::kNONE;
  };

  // If the MatMul operand is a constant, applies transposes at conversion-time
  // as necessary. If the operand is a tensor, does nothing. If required
  // transposes were applied, sets transpose to false.
  const auto prepare_matmul_operand =
      [&params](TRT_TensorOrWeights operand,
                bool* transpose) -> nvinfer1::ITensor* {
    if (operand.is_tensor()) {
      return operand.tensor();
    } else {
      TRT_ShapedWeights weights(operand.weights().TrtDType());
      if (*transpose) {
        weights = params->weight_store->GetTempWeights(operand.weights());
        ReorderCKtoKC(operand.weights(), &weights);
        // Weights have been transposed, can set transpose to false
        *transpose = false;
      } else {
        weights = operand.weights();
      }
      return params->converter->CreateConstantLayer(weights, weights.shape_);
    }
  };

  nvinfer1::ITensor* tensor_a = prepare_matmul_operand(input_a, &transpose_a);
  nvinfer1::ITensor* tensor_b = prepare_matmul_operand(input_b, &transpose_b);

  nvinfer1::IMatrixMultiplyLayer* layer =
      params->converter->network()->addMatrixMultiply(
          *tensor_a, get_matrix_op(tensor_a, transpose_a), *tensor_b,
          get_matrix_op(tensor_b, transpose_b));

  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_name);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
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
                             transpose_b, node_def.name());
}

Status ConvertBatchMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return errors::InvalidArgument(node_def.op(), " got ", inputs.size(),
                                   " inputs but expected 2, at ",
                                   node_def.name());
  }
  // TODO(tmorris): Enable once false is updated to mean either tensor or weight
  // TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"x", false}, {"y",
  // false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
    return errors::InvalidArgument(
        "All inputs are weights, but Grappler is expected to fold them.");
  }
  if (inputs.at(0).is_tensor() && inputs.at(1).is_tensor() &&
      inputs.at(0).GetTrtDims().nbDims != inputs.at(1).GetTrtDims().nbDims) {
    return errors::Unimplemented(
        "Inputs must have the same rank if they are both tensors.");
  }

  TFAttrs attrs(node_def);
  const bool transpose_a = attrs.get<bool>("adj_x");
  const bool transpose_b = attrs.get<bool>("adj_y");

  // There is no way to batch constants in TRT. Example:
  // Tensor with TF Dims: 12 5 3 -> TRT Dims: 5 3
  // Weight with TF Dims: 12 3 6 -> TRT Dims: 12 3 6
  // It is not possible to treat the weight input as a batched [3, 6] tensor.
  const auto check_weight_is_not_batched =
      [](const TRT_TensorOrWeights& input_l,
         const TRT_TensorOrWeights& input_r) {
        // If input_l is a weight, then input_r must be a tensor because
        // otherwise the op would be handled by Grappler.
        if (input_l.is_weights() &&
            input_l.GetTrtDims().nbDims > input_r.GetTrtDims().nbDims &&
            input_l.GetTrtDims().d[0] != 1) {
          return errors::Unimplemented(
              "TensorRT does not support batched constants.");
        }
        return Status::OK();
      };
  TF_RETURN_IF_ERROR(check_weight_is_not_batched(inputs.at(0), inputs.at(1)));
  TF_RETURN_IF_ERROR(check_weight_is_not_batched(inputs.at(1), inputs.at(0)));

  // Broadcast inputs. We don't check feasibility since the dimensions in a
  // MatMul don't need to match. For example, consider a valid set of inputs
  // which would produce an output of shape [N, T, K]:
  // input 0: [N, T, C]
  // input 1: [1, C, K]
  // Since C != K and T != C, check feasiblity would fail.
  nvinfer1::Dims broadcasted_dims_l, broadcasted_dims_r;
  TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
      inputs.at(0), inputs.at(1), /*check_feasibility=*/false,
      params->use_implicit_batch, &broadcasted_dims_l, &broadcasted_dims_r));
  nvinfer1::ITensor* tensor_l = nullptr;
  nvinfer1::ITensor* tensor_r = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(0), broadcasted_dims_l, params->validation_only, &tensor_l));
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(1), broadcasted_dims_r, params->validation_only, &tensor_r));
  if (params->validation_only) return Status::OK();

  return ConvertMatMulHelper(params, TRT_TensorOrWeights(tensor_l),
                             TRT_TensorOrWeights(tensor_r), transpose_a,
                             transpose_b, node_def.name());
}

Status ConvertSoftmax(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(CheckInputsWeights(*params, {{"logits", false}}));
  TF_RETURN_IF_ERROR(
      AllowDataTypes(*params, {DataType::DT_FLOAT, DataType::DT_HALF}));
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  const int num_trt_dims = tensor->getDimensions().nbDims;
  if (num_trt_dims == 0) {
    return errors::InvalidArgument(
        "TensorRT Softmax cannot apply on batch dimension, at",
        node_def.name());
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::ISoftMaxLayer* layer =
      params->converter->network()->addSoftMax(*tensor);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  // Tensorflow SoftMax assumes applying softmax on the last dimension.
  layer->setAxes(1 << (num_trt_dims - 1));

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // Quantization range for SoftMax is always (0, 1)
  params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 1.0f);
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
                                 /*use_implicit_batch=*/true, &trt_axis));
  nvinfer1::TopKOperation topk_op;
  if (node_def.op() == "ArgMin") {
    topk_op = nvinfer1::TopKOperation::kMIN;
  } else if (node_def.op() == "ArgMax") {
    topk_op = nvinfer1::TopKOperation::kMAX;
  } else {
    return errors::InvalidArgument("Unsupported ArgMin/Max operation");
  }
  if (params->validation_only) return Status::OK();

  // Use TopK with k = 1. Only indices output is needed (output 1).
  const uint32_t reduce_axes = 1 << trt_axis;
  nvinfer1::ITopKLayer* layer = params->converter->network()->addTopK(
      *inputs.at(0).tensor(), topk_op, 1, reduce_axes);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_indices_tensor = layer->getOutput(1);

  // Squeeze on axis.
  std::vector<int> size(dims.d, dims.d + dims.nbDims);
  size.erase(size.begin() + trt_axis);
  nvinfer1::Dims new_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(size, &new_dims));
  nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      TRT_TensorOrWeights(output_indices_tensor), new_dims,
      /*validation_only=*/false, &output_tensor));

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

  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
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
  nvinfer1::ITopKLayer* layer =
      params->converter->network()->addTopK(*tensor, op, k, reduce_axes);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  nvinfer1::ITensor* output_value_tensor = layer->getOutput(0);
  nvinfer1::ITensor* output_indices_tensor = layer->getOutput(1);
  params->outputs->push_back(TRT_TensorOrWeights(output_value_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_indices_tensor));
  return Status::OK();
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
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  if (dims.nbDims != 3) {
    return errors::InvalidArgument("The input to ", node_def.op(),
                                   " must be rank 4, at ", node_def.name());
  }
  const int num_channels = data_format == "NCHW" ? dims.d[0] : dims.d[2];
  const int h = data_format == "NCHW" ? dims.d[1] : dims.d[0];
  const int w = data_format == "NCHW" ? dims.d[2] : dims.d[1];
  // Get shuffle parameters.
  nvinfer1::Dims first_shuffle_shape;
  nvinfer1::Permutation transpose_perm;
  nvinfer1::Dims second_shuffle_shape;
  if (node_def.op() == "DepthToSpace") {
    if (num_channels % (block_size * block_size) != 0) {
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
        nvinfer1::DimsCHW(num_channels / (block_size * block_size),
                          h * block_size, w * block_size);
  } else if (node_def.op() == "SpaceToDepth") {
    if (h % block_size != 0 || w % block_size != 0) {
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
    second_shuffle_shape = nvinfer1::DimsCHW(
        num_channels * block_size * block_size, h / block_size, w / block_size);
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::IShuffleLayer* first_shuffle =
      params->converter->network()->addShuffle(*inputs.at(0).tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(first_shuffle, node_def.name());
  if (data_format == "NHWC") {
    first_shuffle->setFirstTranspose({2, 0, 1});
  }
  first_shuffle->setReshapeDimensions(first_shuffle_shape);
  first_shuffle->setSecondTranspose(transpose_perm);

  nvinfer1::IShuffleLayer* second_shuffle =
      params->converter->network()->addShuffle(*first_shuffle->getOutput(0));
  TFTRT_RETURN_ERROR_IF_NULLPTR(second_shuffle, node_def.name());
  second_shuffle->setReshapeDimensions(second_shuffle_shape);
  if (data_format == "NHWC") {
    second_shuffle->setSecondTranspose({1, 2, 0});
  }

  params->converter->MarkQuantizationRangesAsInferrable(
      inputs.at(0).tensor(), first_shuffle->getOutput(0));
  params->converter->MarkQuantizationRangesAsInferrable(
      first_shuffle->getOutput(0), second_shuffle->getOutput(0));
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
  nvinfer1::ITensor* tensor_l = nullptr;
  nvinfer1::ITensor* tensor_r = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(0), broadcasted_dims_l, params->validation_only, &tensor_l));
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(1), broadcasted_dims_r, params->validation_only, &tensor_r));
  if (params->validation_only) return Status::OK();

  // Subtract x - y.
  nvinfer1::IElementWiseLayer* sub =
      params->converter->network()->addElementWise(
          *tensor_l, *tensor_r, nvinfer1::ElementWiseOperation::kSUB);
  TFTRT_RETURN_ERROR_IF_NULLPTR(sub, node_def.name());
  // Multiply (x - y) * (x - y).
  nvinfer1::IElementWiseLayer* mul =
      params->converter->network()->addElementWise(
          *sub->getOutput(0), *sub->getOutput(0),
          nvinfer1::ElementWiseOperation::kPROD);
  TFTRT_RETURN_ERROR_IF_NULLPTR(mul, node_def.name());

  params->outputs->push_back(TRT_TensorOrWeights(mul->getOutput(0)));
  return Status::OK();
}

#if IS_TRT_VERSION_GE(5, 1, 0, 0)
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

  nvinfer1::ITensor* boxes_tensor = inputs.at(0).tensor();
  nvinfer1::ITensor* scores_tensor = inputs.at(1).tensor();
  TRT_ShapedWeights output_size_per_class = inputs.at(2).weights();
  TRT_ShapedWeights total_size = inputs.at(3).weights();
  TRT_ShapedWeights iou_threshold = inputs.at(4).weights();
  TRT_ShapedWeights score_threshold = inputs.at(5).weights();

  // Validate tensors and weights (also set some of the needed plugin fields)
  const auto boxes_dims = boxes_tensor->getDimensions();
  const auto scores_dims = scores_tensor->getDimensions();
  if (boxes_dims.nbDims != 3) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin input boxes must be 3-D excluding batch ",
        node_def.name());
  }
  const int num_classes = scores_dims.d[1];
  bool box_check = boxes_dims.d[1] == 1 || boxes_dims.d[1] == num_classes;
  if (!box_check) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin third dimension of boxes must be either 1 "
        "or num_classes ",
        node_def.name());
  }
  if (output_size_per_class.shape_.nbDims != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class must be 0-D ",
        node_def.name());
  }
  int max_size_per_class =
      *(static_cast<int*>(output_size_per_class.GetValues()));
  if (max_size_per_class <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_output_size_per_class should be > 0",
        node_def.name());
  }
  if (total_size.shape_.nbDims != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size must be 0-D ",
        node_def.name());
  }
  int max_total_size = *(static_cast<int*>(total_size.GetValues()));
  if (max_total_size <= 0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin max_total_size should be > 0",
        node_def.name());
  }
  if (iou_threshold.shape_.nbDims != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be 0-D ",
        node_def.name());
  }
  float iou_thresh = *(static_cast<float*>(iou_threshold.GetValues()));
  if (iou_thresh < 0.0 || iou_thresh > 1.0) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin iou_threshold must be in [0, 1]",
        node_def.name());
  }
  if (score_threshold.shape_.nbDims != 1) {
    return errors::InvalidArgument(
        "TensorRT BatchedNMS Plugin score_threshold must be 0-D ",
        node_def.name());
  }

  if (params->validation_only) return Status::OK();

  // TF op CombinedNonMaxSuppression doesn't have the option of
  // not normalizing coordinates.
  const bool is_normalized = true;
  // Set plugin fields and the field collection
  TFAttrs attrs(node_def);
  bool share_location = (boxes_dims.d[1] == 1);
  const bool pad_per_class = attrs.get<bool>("pad_per_class");
  int top_k;
  if (pad_per_class) {
    top_k = std::min(max_size_per_class * num_classes, max_total_size);
  } else {
    top_k = max_total_size;
  }
  const int keep_top_k = top_k;
  float score_thresh = *(static_cast<float*>(score_threshold.GetValues()));
  const int background_id = -1;
  nvinfer1::PluginField fields[8] = {
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
  };
  nvinfer1::PluginFieldCollection fc{8, fields};

  // Get plugin creator
  auto creator =
      getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1", "");
  TFTRT_RETURN_ERROR_IF_NULLPTR(creator, node_def.name());

  // Create plugin
  TrtUniquePtrType<nvinfer1::IPluginV2> plugin(
      creator->createPlugin(node_def.name().c_str(), &fc));
  TFTRT_RETURN_ERROR_IF_NULLPTR(plugin, node_def.name());

  // Set plugin inputs
  std::vector<nvinfer1::ITensor*> plugin_inputs;
  plugin_inputs.push_back(boxes_tensor);
  plugin_inputs.push_back(scores_tensor);

  // Add plugin to network
  nvinfer1::IPluginV2Layer* layer = params->converter->network()->addPluginV2(
      &plugin_inputs[0], static_cast<int>(plugin_inputs.size()), *plugin);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  // Set plugin outputs
  nvinfer1::ITensor* output_nmsed_boxes = layer->getOutput(1);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // TRT6 fixes (removes) the extra last dimension in CombinedNMS outputs
  nvinfer1::ITensor* output_num_detections = layer->getOutput(0);
  nvinfer1::ITensor* output_nmsed_scores = layer->getOutput(2);
  nvinfer1::ITensor* output_nmsed_classes = layer->getOutput(3);
#else
  nvinfer1::ITensor* output_num_detections = nullptr;
  nvinfer1::ITensor* output_nmsed_scores = nullptr;
  nvinfer1::ITensor* output_nmsed_classes = nullptr;

  auto shrink_last_dim = [params](nvinfer1::ITensor* in_tensor,
                                  nvinfer1::ITensor** out_tensor) {
    nvinfer1::Dims dims = in_tensor->getDimensions();
    if (dims.d[dims.nbDims - 1] != 1) {
      return errors::Internal("Expect last dims to be 1, for tensor ",
                              DebugString(*in_tensor));
    }
    --dims.nbDims;
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(in_tensor), dims,
        /*validation_only=*/false, out_tensor));
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(
      shrink_last_dim(layer->getOutput(2), &output_nmsed_scores));
  TF_RETURN_IF_ERROR(
      shrink_last_dim(layer->getOutput(3), &output_nmsed_classes));
  TF_RETURN_IF_ERROR(
      shrink_last_dim(layer->getOutput(0), &output_num_detections));
#endif  // IS_TRT_VERSION_GE(6, 0, 0, 0)

  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_boxes));
  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_scores));
  params->outputs->push_back(TRT_TensorOrWeights(output_nmsed_classes));
  params->outputs->push_back(TRT_TensorOrWeights(output_num_detections));

  return Status::OK();
}
#endif  // IS_TRT_VERSION_GE(5, 1, 0, 0)

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
Status ConvertResize(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"input", false}, {"size", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));

  // Get input tensor. Transpose it from NHWC to NCHW.
  nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  TFTRT_RETURN_ERROR_IF_NULLPTR(tensor, params->node_def.name());

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
    resize_mode = nvinfer1::ResizeMode::kLINEAR;
  } else if (node_def.op() == "ResizeNearestNeighbor") {
    resize_mode = nvinfer1::ResizeMode::kNEAREST;
  } else {
    return errors::Unimplemented(node_def.op(), " is not yet implemented at ",
                                 node_def.name());
  }

  // return after validation if only validation is requested.
  if (params->validation_only) return Status::OK();

  // Transpose tensor from NHWC to NCHW format.
  TF_RETURN_IF_ERROR(
      params->converter->TransposeTensor(tensor, {0, 3, 1, 2}, &tensor));

  // Calculate output dimensions.
  // Given input dimensions [N, C, H, W] and output size [H_out, W_out],
  // output dimensions equals [N, C, H_out, W_out]
  nvinfer1::Dims output_dimensions;
  output_dimensions.nbDims = tensor->getDimensions().nbDims;
  for (int i = 0; i < output_dimensions.nbDims; ++i) {
    output_dimensions.d[i] = tensor->getDimensions().d[i];
  }
  output_dimensions.d[output_dimensions.nbDims - 2] = weights_ptr[0];
  output_dimensions.d[output_dimensions.nbDims - 1] = weights_ptr[1];

  // Add resize layer.
  nvinfer1::IResizeLayer* layer =
      params->converter->network()->addResize(*tensor);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  // Set layer parameters.
  layer->setResizeMode(resize_mode);
  layer->setOutputDimensions(output_dimensions);
  layer->setAlignCorners(align_corners);

  // Get output tensor. Transpose it from NCHW to NHWC.
  nvinfer1::ITensor* output = layer->getOutput(0);

  TF_RETURN_IF_ERROR(
      params->converter->TransposeTensor(output, {0, 2, 3, 1}, &output));
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
  std::vector<nvinfer1::ITensor*> tensor_inputs;
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
  nvinfer1::ITensor* lhs = tensor_inputs[0];
  for (int i = 1; i < num_inputs; ++i) {
    nvinfer1::ITensor* rhs = tensor_inputs[i];
    nvinfer1::ILayer* layer = params->converter->network()->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kSUM);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
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
#if IS_TRT_VERSION_GE(5, 1, 0, 0)
  (*registration)["CombinedNonMaxSuppression"] = ConvertCombinedNMS;
#endif
  (*registration)["AddN"] = ConvertAddN;
  (*registration)["ConcatV2"] = ConvertConcat;
  (*registration)["Const"] = ConvertConst;
  (*registration)["Conv2D"] = ConvertConv2D;
  (*registration)["Conv2DBackpropInput"] = ConvertConv2DBackpropInput;
  (*registration)["DepthToSpace"] = ConvertDepthSpaceShuffle;
  (*registration)["DepthwiseConv2dNative"] = ConvertConv2DDepthwise;
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
  for (auto unary_op_pair : *UnaryOperationMap()) {
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
    const bool use_implicit_batch, bool* convert_successfully) {
  engine->reset();
  if (convert_successfully) *convert_successfully = false;

  // Creating converter, TensorRT builder and network
  auto statusor = Converter::Create(precision_mode, use_calibration, trt_logger,
                                    use_implicit_batch);
  TF_RETURN_IF_ERROR(statusor.status());
  auto converter = std::move(statusor.ValueOrDie());

  VLOG(1) << "Starting to convert TensorFlow ops to TensorRT layers";
  std::vector<Converter::EngineOutputInfo> output_tensors;
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
        LOG(WARNING) << error_message;
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
      TF_RETURN_IF_ERROR(TfDataTypeToTrt(tf_dtype, &trt_dtype));
      if (output_tensors.size() <= slot_number) {
        output_tensors.resize(slot_number + 1);
      }
      output_tensors.at(slot_number) = {node_def.input(0), node_name,
                                        trt_dtype};
    } else {
      TF_RETURN_IF_ERROR(converter->ConvertNode(node_def));
    }
  }
  TF_RETURN_IF_ERROR(converter->RenameAndMarkOutputTensors(output_tensors));
  if (convert_successfully) *convert_successfully = true;

  // Apply user provided quantization ranges to tensors
  converter->MaybeApplyQuantizationRanges();

  // Build the engine.
  TF_RETURN_IF_ERROR(converter->BuildCudaEngine(
      engine, max_batch_size, max_workspace_size_bytes, allocator, calibrator));

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

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
