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
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resource_manager.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"        // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

// Check if the types are equal. Cast to int first so that failure log message
// would work!
#define TFTRT_CHECK_EQ_TYPE(val1, val2) CHECK_EQ((int)val1, (int)val2)

#define TFTRT_INTERNAL_ERROR_AT_NODE(node)                                \
  do {                                                                    \
    return tensorflow::errors::Internal(                                  \
        "TFTRT::", __FUNCTION__, " failed to add TRT layer, at: ", node); \
  } while (0)

#define TFTRT_RETURN_ERROR_IF_FALSE(status, node) \
  do {                                            \
    if (status == false) {                        \
      TFTRT_INTERNAL_ERROR_AT_NODE(node);         \
    }                                             \
  } while (0)

#define TFTRT_RETURN_ERROR_IF_NULLPTR(ptr, node) \
  do {                                           \
    if (ptr == nullptr) {                        \
      TFTRT_INTERNAL_ERROR_AT_NODE(node);        \
    }                                            \
  } while (0)

namespace tensorflow {
namespace tensorrt {
// TODO(aaroey): put these constants into some class.
const char* const kInputPHName = "TensorRTInputPH_";
const char* const kOutputPHName = "TensorRTOutputPH_";

namespace convert {
using absl::StrAppend;
using absl::StrCat;
using ::tensorflow::str_util::Split;

inline tensorflow::Status ConvertDType(tensorflow::DataType tf_dtype,
                                       nvinfer1::DataType* trt_dtype) {
  switch (tf_dtype) {
    case tensorflow::DataType::DT_FLOAT:
      *trt_dtype = nvinfer1::DataType::kFLOAT;
      break;
    // TODO(aaroey): this should be DT_QINT8 which is not a well supported type.
    case tensorflow::DataType::DT_INT8:
      *trt_dtype = nvinfer1::DataType::kINT8;
      break;
    case tensorflow::DataType::DT_HALF:
      *trt_dtype = nvinfer1::DataType::kHALF;
      break;
    case tensorflow::DataType::DT_INT32:
      *trt_dtype = nvinfer1::DataType::kINT32;
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type ", tensorflow::DataTypeString(tf_dtype));
  }
  return tensorflow::Status::OK();
}

template <typename TensorShapeType>
inline nvinfer1::Dims TensorShapeToTrtDims(const TensorShapeType& shape,
                                           bool ignore_first_dim) {
  nvinfer1::Dims trt_dims;
  const int offset = (ignore_first_dim ? 1 : 0);
  for (int i = offset; i < shape.dims(); i++) {
    trt_dims.d[i - offset] = shape.dim_size(i);
  }
  trt_dims.nbDims = shape.dims() - offset;
  return trt_dims;
}

Status TensorShapeArrayToTrtDims(const std::vector<int>& shape,
                                 nvinfer1::Dims* out,
                                 bool ignore_first_dim = false) {
  PartialTensorShape tensor_shape;
  TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(shape, &tensor_shape));
  *out = TensorShapeToTrtDims(tensor_shape, ignore_first_dim);
  return tensorflow::Status::OK();
}

void GetOutputProperties(const grappler::GraphProperties& graph_properties,
                         const Node* node, const int out_port,
                         PartialTensorShape* shape,
                         tensorflow::DataType* dtype) {
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
                        PartialTensorShape* shape,
                        tensorflow::DataType* dtype) {
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
                                const tensorflow::DataType dtype,
                                const PartialTensorShape& shape,
                                bool validation_only,
                                nvinfer1::DataType* trt_dtype,
                                nvinfer1::Dims* trt_dims, int* batch_size) {
  // Convert data type.
  TF_RETURN_IF_ERROR(ConvertDType(dtype, trt_dtype));

  // Convert shape.
  if (shape.dims() < 0) {
    return errors::InvalidArgument("Input tensor rank is unknown.");
  }
  if (shape.dims() > nvinfer1::Dims::MAX_DIMS + 1) {  // +1 for batch dim
    return errors::OutOfRange("Input tensor rank is greater than ",
                              nvinfer1::Dims::MAX_DIMS + 1);
  }
  if (producer_node_type != "Const" && shape.dims() < 2) {
    return errors::InvalidArgument(
        "Input tensor with rank<2 is not supported since the first dimension "
        "is treated as batch dimension by TRT");
  }
  *trt_dims = TensorShapeToTrtDims(shape, /*ignore_first_dim=*/true);
  *batch_size = shape.dim_size(0);

  if (validation_only) return Status::OK();
  // Following are validations at runtime.

  for (int d = 1; d < shape.dims(); ++d) {
    if (shape.dim_size(d) < 0) {
      return errors::InvalidArgument(
          "Input tensor with shape ", shape.DebugString(),
          " has an unknown non-batch dimension at dim ", d);
    }
  }
  return Status::OK();
}

string DebugString(const nvinfer1::DimensionType type) {
  switch (type) {
    case nvinfer1::DimensionType::kSPATIAL:
      return "kSPATIAL";
    case nvinfer1::DimensionType::kCHANNEL:
      return "kCHANNEL";
    case nvinfer1::DimensionType::kINDEX:
      return "kINDEX";
    case nvinfer1::DimensionType::kSEQUENCE:
      return "kSEQUENCE";
    default:
      return StrCat(static_cast<int>(type), "=unknown");
  }
}

string DebugString(const nvinfer1::DataType trt_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "kFLOAT";
    case nvinfer1::DataType::kHALF:
      return "kHALF";
    case nvinfer1::DataType::kINT8:
      return "kINT8";
    case nvinfer1::DataType::kINT32:
      return "kINT32";
    default:
      return "Invalid TRT data type";
  }
}

string DebugString(const nvinfer1::Dims& dims) {
  string out = StrCat("nvinfer1::Dims(nbDims=", dims.nbDims, ", d=");
  for (int i = 0; i < dims.nbDims; ++i) {
    StrAppend(&out, dims.d[i], "[", DebugString(dims.type[i]), "],");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const nvinfer1::Permutation& permutation, int len) {
  string out = "nvinfer1::Permutation(";
  for (int i = 0; i < len; ++i) {
    StrAppend(&out, permutation.order[i], ",");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const nvinfer1::ITensor& tensor) {
  return StrCat("nvinfer1::ITensor(@", reinterpret_cast<uintptr_t>(&tensor),
                ", name=", tensor.getName(),
                ", dtype=", DebugString(tensor.getType()),
                ", dims=", DebugString(tensor.getDimensions()), ")");
}

Status Converter::GetTrtBroadcastShape(
    const TRT_TensorOrWeights& operand_l, const TRT_TensorOrWeights& operand_r,
    nvinfer1::Dims* operand_l_new_dims,
    nvinfer1::Dims* operand_r_new_dims) const {
  // ***************************************************************************
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
  auto compute_output_dims =
      [max_nb_dims](const TRT_TensorOrWeights& input, int broadcast_num_dims,
                    int* output_dims_array, nvinfer1::Dims* output_dims) {
        const nvinfer1::Dims input_dims = input.GetTrtDims();
        std::fill(output_dims_array, output_dims_array + max_nb_dims, 1);
        std::copy(input_dims.d, input_dims.d + input_dims.nbDims,
                  output_dims_array + broadcast_num_dims - input_dims.nbDims);
        if (input.is_tensor()) {
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
        // Copy to output dimensions (stripping the batch dimension).
        output_dims->nbDims = broadcast_num_dims - 1;
        std::copy(output_dims_array + 1, output_dims_array + broadcast_num_dims,
                  output_dims->d);
        return Status::OK();
      };

  // Compute the output dimensions.
  const int broadcast_num_dims =
      std::max(operand_l.GetTrtDims().nbDims + (operand_l.is_tensor() ? 1 : 0),
               operand_r.GetTrtDims().nbDims + (operand_r.is_tensor() ? 1 : 0));
  int output_l[max_nb_dims], output_r[max_nb_dims];
  TF_RETURN_IF_ERROR(compute_output_dims(operand_l, broadcast_num_dims,
                                         output_l, operand_l_new_dims));
  TF_RETURN_IF_ERROR(compute_output_dims(operand_r, broadcast_num_dims,
                                         output_r, operand_r_new_dims));

  // Compare broadcast feasibility
  for (int i = 0; i < broadcast_num_dims; ++i) {
    if ((output_l[i] != output_r[i]) && (output_l[i] != 1) &&
        (output_r[i] != 1)) {
      return errors::InvalidArgument(
          "Infeasible broadcast scheme (", "batch_dim: ", output_l[0], ", ",
          DebugString(*operand_l_new_dims), " vs ", "batch_dim: ", output_r[0],
          ", ", DebugString(*operand_r_new_dims), ")");
    }
  }
  return Status::OK();
}

nvinfer1::ITensor* Converter::CreateConstantLayer(
    const TRT_ShapedWeights& weights, const nvinfer1::Dims& dims) {
  nvinfer1::Weights trt_weights = weights.GetTrtWeights();
  nvinfer1::IConstantLayer* layer = network()->addConstant(dims, trt_weights);
  if (!layer) return nullptr;
  const nvinfer1::DataType trt_dtype = trt_weights.type;
  nvinfer1::ITensor* trt_tensor = layer->getOutput(0);
  // TODO(laigd): there is a bug in TensorRT 5.0 library that, if we don't set
  // the data type below, it will always be kFLOAT regardless what the data type
  // of the weights is. Once NVIDIA fixes this bug, we should remove the data
  // type setting logic below and test should still pass.
  trt_tensor->setType(trt_dtype);
  return trt_tensor;
}

tensorflow::Status Converter::CreateBroadcastableScalarConstant(
    OpConverterParams* params, float value, const nvinfer1::Dims& dims,
    const nvinfer1::ITensor** tensor) {
  // In order to be broadcastable, the number of dims has to match.
  nvinfer1::Dims broadcastable_dims(dims);
  for (int i = 0; i < broadcastable_dims.nbDims; i++) {
    broadcastable_dims.d[i] = 1;
  }
  TRT_ShapedWeights weights = params->weight_store->GetTempWeights(
      tensorflow::DataType::DT_FLOAT, broadcastable_dims);
  auto weights_ptr =
      static_cast<float*>(const_cast<void*>(weights.GetValues()));
  weights_ptr[0] = value;
  *tensor = params->converter->CreateConstantLayer(weights, broadcastable_dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, params->node_def.name());
  params->converter->ProvideQuantizationRange(
      const_cast<nvinfer1::ITensor*>(*tensor), value, value);
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

inline nvinfer1::Dims GetTrtDimsForTensor(const tensorflow::Tensor& tensor) {
  nvinfer1::Dims dims;
  dims.nbDims = tensor.dims();
  for (int i = 0; i < dims.nbDims; i++) {
    dims.d[i] = tensor.dim_size(i);
  }
  return dims;
}

inline bool HasStaticShape(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 0) return false;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < 0) return false;
  }
  return true;
}

// Returns total number of elements in dims. Returning 0 means either some dim
// is 0 or the number of dims is 0.
// Note that for TF scalar constant, we always convert to dims [1].
int64_t TrtDimsNumElements(const nvinfer1::Dims& dims) {
  if (dims.nbDims == 0) return 0;
  int64_t count = 1;
  for (int d = 0; d < dims.nbDims; ++d) {
    count *= dims.d[d];
  }
  return count;
}

static std::vector<std::pair<int, int>> CreateSamePadding(
    const nvinfer1::DimsHW& stride, const nvinfer1::DimsHW& kernel,
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

TRT_ShapedWeights::TRT_ShapedWeights(DataType type) : type_(type) {
  shape_.nbDims = 0;
}

TRT_ShapedWeights::TRT_ShapedWeights(DataType type, nvinfer1::Dims dims,
                                     Tensor tensor)
    : shape_(dims), type_(type), tensor_(tensor) {}

TRT_ShapedWeights::TRT_ShapedWeights(const TRT_ShapedWeights& rhs)
    : shape_(rhs.shape_), type_(rhs.type_), tensor_(rhs.tensor_) {}

int64_t TRT_ShapedWeights::count() const { return TrtDimsNumElements(shape_); }

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
  nvinfer1::DataType trt_type(nvinfer1::DataType::kFLOAT);
  TF_CHECK_OK(ConvertDType(type_, &trt_type));
  return nvinfer1::Weights{trt_type, GetValues(), count()};
}

size_t TRT_ShapedWeights::size_bytes() const {
  return this->count() * tensorflow::DataTypeSize(this->type_);
}

string TRT_ShapedWeights::DebugString() const {
  return StrCat("TRT_ShapedWeights(shape=", convert::DebugString(shape_),
                ", type=", DataTypeString(type_),
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

#if NV_TENSORRT_MAJOR >= 5
  bool setDynamicRange(float min, float max) override { return true; }

  float getDynamicRange() const override { return 0; }
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

nvinfer1::ITensor* TRT_TensorOrWeights::tensor() {
  CHECK(is_tensor());
  return tensor_ == nullptr ? simple_itensor_.get() : tensor_;
}

const nvinfer1::ITensor* TRT_TensorOrWeights::tensor() const {
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
    StrAppend(&output, "tensor=", convert::DebugString(*tensor()),
              ", batch_size=", batch_size_);
  } else {
    StrAppend(&output, "weights=", weights_.DebugString());
  }
  StrAppend(&output, ")");
  return output;
}

class TFAttrs {
 public:
  explicit TFAttrs(const tensorflow::NodeDef& tf_node) {
    for (const auto& attr : tf_node.attr()) {
      attrs_.insert({attr.first, &attr.second});
    }
  }

  bool count(const string& key) const { return attrs_.count(key); }

  tensorflow::AttrValue const* at(const string& key) const {
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

  std::vector<string> GetAllAttrKeys() const {
    std::vector<string> attr_list;
    for (const auto& attr_item : attrs_) {
      attr_list.emplace_back(attr_item.first);
    }
    return attr_list;
  }

 private:
  typedef std::map<string, tensorflow::AttrValue const*> AttrMap;
  AttrMap attrs_;
};

template <>
string TFAttrs::get<string>(const string& key) const {
  return this->at(key)->s();
}

template <>
std::vector<int> TFAttrs::get<std::vector<int>>(const string& key) const {
  auto attr = this->at(key)->list().i();
  return std::vector<int>(attr.begin(), attr.end());
}

template <>
std::vector<float> TFAttrs::get<std::vector<float>>(const string& key) const {
  auto attr = this->at(key)->list().f();
  return std::vector<float>(attr.begin(), attr.end());
}

template <>
nvinfer1::DataType TFAttrs::get<nvinfer1::DataType>(const string& key) const {
  nvinfer1::DataType trt_dtype(nvinfer1::DataType::kFLOAT);
  TF_CHECK_OK(ConvertDType(this->at(key)->type(), &trt_dtype));
  return trt_dtype;
}

template <>
tensorflow::DataType TFAttrs::get<tensorflow::DataType>(
    const string& key) const {
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
int TFAttrs::get<int>(const string& key) const {
  return this->at(key)->i();
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
  switch (iweights.type_) {
    case tensorflow::DataType::DT_FLOAT: {
      Reorder2({k, c}, static_cast<float const*>(iweights.GetValues()),
               istrides,
               // TODO(aaroey): get rid of all the const_cast like this.
               static_cast<float*>(const_cast<void*>(oweights->GetValues())),
               ostrides);
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      Reorder2(
          {k, c}, static_cast<Eigen::half const*>(iweights.GetValues()),
          istrides,
          static_cast<Eigen::half*>(const_cast<void*>(oweights->GetValues())),
          ostrides);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type in reorder expected fp32 or fp16 but got "
                 << DataTypeString(iweights.type_);
  }
}

void ReorderRSCKToKCRS(const TRT_ShapedWeights& iweights,
                       TRT_ShapedWeights* oweights, const int num_groups) {
  CHECK_EQ(iweights.type_, oweights->type_);
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
  switch (iweights.type_) {
    case tensorflow::DataType::DT_FLOAT: {
      Reorder4({k, c, r, s}, static_cast<float const*>(iweights.GetValues()),
               istrides,
               static_cast<float*>(const_cast<void*>(oweights->GetValues())),
               ostrides);
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      Reorder4(
          {k, c, r, s}, static_cast<Eigen::half const*>(iweights.GetValues()),
          istrides,
          static_cast<Eigen::half*>(const_cast<void*>(oweights->GetValues())),
          ostrides);
      break;
    }

    default:
      LOG(FATAL) << "Unsupported type, expected fp32 or fp16 but got "
                 << DataTypeString(iweights.type_);
  }
}

TRT_ShapedWeights TrtWeightStore::GetTempWeights(tensorflow::DataType type,
                                                 const nvinfer1::Dims& dims) {
  TensorShape shape;
  // TODO(laigd): make it return a status.
  TF_CHECK_OK(TensorShapeUtils::MakeShape(dims.d, dims.nbDims, &shape));
  // TODO(jie): check weights size_bytes. 0 means type error
  Tensor tensor(type, shape);
  TRT_ShapedWeights weights(type, dims, tensor);
  store_.emplace_back(std::move(tensor));
  return weights;
}

TrtNodeValidator::TrtNodeValidator() { RegisterOpValidators(); }

Status TrtNodeValidator::ConvertToTensorOrWeights(
    const NodeDef& node_def, int output_port,
    const grappler::GraphProperties& graph_properties,
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
  if (!graph_properties.HasOutputProperties(node_def.name())) {
    return errors::InvalidArgument("Shape and data type are unknown");
  }

  // Validate and convert shape and dtype.
  const auto& output_params =
      graph_properties.GetOutputProperties(node_def.name());
  const auto& tensor_properties = output_params.at(output_port);
  const DataType dtype = tensor_properties.dtype();
  const PartialTensorShape shape = tensor_properties.shape();
  nvinfer1::DataType trt_dtype;
  nvinfer1::Dims trt_dims;
  int batch_size = -1;
  TF_RETURN_IF_ERROR(ValidateTensorProperties(
      node_def.op(), dtype, shape, /*validation_only_=*/true, &trt_dtype,
      &trt_dims, &batch_size));

  // Adds a fake ITensor. This is fine since op converter operates in
  // validation-only mode and it won't (and shouldn't) use the tensor to do
  // any TRT network operations.
  *tensor_or_weights = TRT_TensorOrWeights(trt_dtype, trt_dims, batch_size);
  return Status::OK();
}

Status TrtNodeValidator::ValidateNode(
    const tensorflow::NodeDef& node_def,
    const std::vector<std::pair<const NodeDef*, int>>& input_node_and_ports,
    const grappler::GraphProperties& graph_properties) {
  // Convert input NodeDef and corresponding output ports to
  // TRT_TensorOrWeights.
  std::vector<TRT_TensorOrWeights> inputs;
  for (int i = 0; i < input_node_and_ports.size(); ++i) {
    const auto& pair = input_node_and_ports[i];
    TRT_TensorOrWeights tensor_or_weights;
    Status status = ConvertToTensorOrWeights(
        *pair.first, pair.second, graph_properties, &tensor_or_weights);
    if (!status.ok()) {
      return errors::Internal(
          "Failed to convert input with index ", i,
          " to a TRT_TensorOrWeights: ", status.error_message());
    }
    inputs.push_back(tensor_or_weights);
  }

  // Validate the node.
  const auto iter = op_validators_.find(node_def.op());
  if (iter == op_validators_.end()) {
    // If validator is not registered, it means no validation is needed.
    return Status::OK();
  }

  OpConverter validator = iter->second;
  OpConverterParams params(
      /*arg_converter=*/nullptr, node_def, inputs, /*arg_outputs=*/nullptr,
      /*arg_validation_only=*/true, &weight_store_);
  return validator(&params);
}

Status TrtNodeValidator::ConvertConstToWeights(
    const NodeDef& const_node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    TRT_TensorOrWeights* output) {
  std::vector<TRT_TensorOrWeights> outputs;
  OpConverterParams params(
      /*arg_converter=*/nullptr, const_node_def, inputs, &outputs,
      /*arg_validation_only=*/true, &weight_store_);
  Status status = op_validators_["Const"](&params);
  if (status.ok() && output) *output = outputs[0];
  return status;
}

Converter::Converter(nvinfer1::INetworkDefinition* trt_network,
                     int precision_mode, bool use_calibration)
    : trt_network_(trt_network),
      precision_mode_(precision_mode),
      use_calibration_(use_calibration) {
  this->RegisterOpConverters();
}

Status Converter::ConvertNode(const NodeDef& node_def) {
  std::vector<TRT_TensorOrWeights> inputs, outputs;
  TF_RETURN_IF_ERROR(this->GetInputs(node_def, &inputs));

  OpConverterParams params(this, node_def, inputs, &outputs,
                           /*arg_validation_only=*/false, &weight_store_);
  const string& op = node_def.op();
  if (PluginFactoryTensorRT::GetInstance()->IsPlugin(op)) {
    TF_RETURN_IF_ERROR(plugin_converter_(&params));
  } else {
    if (!op_registry_.count(op)) {
      return errors::Unimplemented("No converter registered for op: " + op);
    }
    OpConverter op_converter = op_registry_.at(op);
    TF_RETURN_IF_ERROR(op_converter(&params));
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    TRT_TensorOrWeights& output = outputs[i];
    string output_name = node_def.name();
    if (i != 0) output_name = StrCat(output_name, ":", i);
    // We need to check the name before setting it. If the input is one of the
    // engine input, setting the name here will overwrite engine input
    // bindings which will cause runtime error.
    // TODO(tmorris): Remove this work-around once we use TRT's IIdentityLayer
    // in ConvertIdentity.
    if (output.is_tensor()) {
      const char* tensor_name = output.tensor()->getName();
      if (!tensorflow::str_util::StartsWith(tensor_name, kInputPHName)) {
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
    // Check if this tensor has already been marked as an output.
    // ConvertIdentity can cause the same tensor to be repeated in
    // output_tensors, which can cause us to overwrite the name of the output
    // tensor binding. For example, if we rename OutputPH_0 to OutputPH_1 then
    // we won't be able to locate OutputPH_0 during runtime. To fix this,
    // duplicate the tensor using no-op shuffle.
    // TODO(tmorris): Remove this work-around once we use TRT's IIdentityLayer
    // in ConvertIdentity.
    if (tensorflow::str_util::StartsWith(tensor->getName(), kOutputPHName)) {
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
            << ", which feeds TF node " << output.dest_node_name;
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
  if (input.is_tensor()) input.set_batch_size(batch_size_);
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
                                  const nvinfer1::ITensor** output_tensor) {
  const auto dims = input_tensor->getDimensions();

  if (order_with_batch_dim.size() - 1 != size_t(dims.nbDims)) {
    return tensorflow::errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
  }
  if (order_with_batch_dim[0] != 0) {
    return tensorflow::errors::Unimplemented(
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
  return tensorflow::Status::OK();
}

Status Converter::GetWeightRange(const TRT_ShapedWeights& weights,
                                 float* out_min, float* out_max) const {
  switch (weights.type_) {
    case DataType::DT_FLOAT: {
      auto inp = static_cast<float const*>(weights.GetValues());
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = *result.first;
      *out_max = *result.second;
      break;
    }
    case DataType::DT_HALF: {
      auto inp = static_cast<Eigen::half const*>(weights.GetValues());
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = Eigen::half_impl::half_to_float(*result.first);
      *out_max = Eigen::half_impl::half_to_float(*result.second);
      break;
    }
    case DataType::DT_INT32: {
      auto inp = static_cast<int const*>(weights.GetValues());
      auto result = std::minmax_element(inp, inp + weights.count());
      *out_min = static_cast<float>(*result.first);
      *out_max = static_cast<float>(*result.second);
      break;
    }
    default:
      return errors::Unimplemented(
          "Data type not supported for GetWeightRange: ",
          DataTypeString(weights.type_));
  }
  return Status::OK();
}

Status Converter::PrepareTensorForShape(const TRT_TensorOrWeights& input,
                                        const nvinfer1::Dims& dims,
                                        const nvinfer1::ITensor** tensor) {
  // If -1 is not used for one of the dims, we can check if the shapes are
  // compatible.
  bool can_check_shapes = true;
  for (int i = 0; i < dims.nbDims; i++) {
    if (dims.d[i] == -1) {
      can_check_shapes = false;
      break;
    }
  }
  if (can_check_shapes &&
      TrtDimsNumElements(input.GetTrtDims()) != TrtDimsNumElements(dims)) {
    return errors::InvalidArgument("Reshape shapes are not compatible (",
                                   DebugString(input.GetTrtDims()), " vs ",
                                   DebugString(dims), ")");
  }

  if (input.is_tensor()) {
    if (DimsEqual(input.GetTrtDims(), dims)) {
      *tensor = input.tensor();
    } else {
      nvinfer1::IShuffleLayer* layer = this->network()->addShuffle(
          *const_cast<nvinfer1::ITensor*>(input.tensor()));
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, "TF-TRT Internal Reshape");
      layer->setReshapeDimensions(dims);
      MarkQuantizationRangesAsInferrable(
          const_cast<nvinfer1::ITensor*>(input.tensor()), layer->getOutput(0));
      *tensor = layer->getOutput(0);
    }
  } else {
    *tensor = CreateConstantLayer(input.weights(), dims);
    TFTRT_RETURN_ERROR_IF_NULLPTR(*tensor, "TF-TRT Internal Reshape");
    if (precision_mode() == INT8MODE && !use_calibration()) {
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
      ProvideQuantizationRange(const_cast<nvinfer1::ITensor*>(*tensor),
                               min_range, max_range);
    }
  }
  return tensorflow::Status::OK();
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

void Converter::MaybeApplyQuantizationRanges() {
  if (precision_mode() != INT8MODE) return;

  // Infer ranges across marked ops.
  PropagateQuantizationRanges();
  // Apply ranges.
#if NV_TENSORRT_MAJOR >= 5
  for (auto pair : quantization_ranges_) {
    nvinfer1::ITensor* tensor = pair.first;
    const float range = pair.second;
    VLOG(1) << "Setting range for: " << tensor->getName() << ": " << range;
    // TODO(laigd): if 'tensor' already has a range set which doesn't match
    // 'range', it should report error.
    tensor->setDynamicRange(-range, range);
  }
#endif

  // Warn user about tensors that are missing ranges. If TRT fuses some layers
  // then these tensors may not actually be required, which is why this is
  // just a warning. If we are still missing ranges even after fusion,
  // Builder::buildCudaEngine() will return nullptr and we will catch the
  // error at that point.
  if (!use_calibration()) {
    // Get all tensors from network
    std::set<nvinfer1::ITensor*> all_tensors;
    for (int i = 0; i < this->network()->getNbLayers(); i++) {
      nvinfer1::ILayer* layer = this->network()->getLayer(i);
      for (int j = 0; j < layer->getNbInputs(); j++) {
        all_tensors.insert(layer->getInput(j));
      }
      for (int j = 0; j < layer->getNbOutputs(); j++) {
        all_tensors.insert(layer->getOutput(j));
      }
    }
    // Find tensors with no ranges
    for (auto tensor : all_tensors) {
      if (!quantization_ranges_.count(tensor)) {
        // Note: there may be some warnings for "(Unnamed ITensor* N)". These
        // are tensors which are created internally by TF-TRT. The ranges for
        // these unnamed ITensors are always inferred from user provided ranges,
        // thus there will also be a warning for the range(s) the user missed.
        LOG(WARNING) << "Quantization range was not found for "
                     << tensor->getName() << ". "
                     << "This is okay if TensorRT does not need the range "
                     << "(e.g. due to node fusion).";
      }
    }
  }
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

Status Converter::GetInputs(const tensorflow::NodeDef& node_def,
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
      return tensorflow::errors::InvalidArgument(msg);
    }
  }
  return tensorflow::Status::OK();
}

TRT_ShapedWeights ConvertFP32ToFP16(TrtWeightStore* store,
                                    const TRT_ShapedWeights& weights_src) {
  auto dtype_new = tensorflow::DataType::DT_HALF;
  TRT_ShapedWeights weights =
      store->GetTempWeights(dtype_new, weights_src.shape_);
  const float* src = static_cast<const float*>(weights_src.GetValues());
  Eigen::half* dst = const_cast<Eigen::half*>(
      static_cast<Eigen::half const*>(weights.GetValues()));
  for (int64_t i = 0; i < weights_src.count(); i++) {
    dst[i] = Eigen::half_impl::float_to_half_rtne(src[i]);
  }
  return weights;
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
        return [](T t) -> T { return 1.0 / sqrt(t); };
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
        return Eigen::half(1.0 / sqrt(static_cast<float>(t)));
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

tensorflow::Status UnaryCompute(const TRT_ShapedWeights& iweights,
                                TRT_ShapedWeights* oweights,
                                LambdaFactory unary_op) {
  CHECK_EQ(iweights.type_, oweights->type_);
  switch (iweights.type_) {
    case tensorflow::DataType::DT_FLOAT: {
      auto inp = static_cast<float const*>(iweights.GetValues());
      auto oup = static_cast<float*>(const_cast<void*>(oweights->GetValues()));
      std::transform(inp, inp + iweights.count(), oup, unary_op.unary<float>());
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      auto inp = static_cast<Eigen::half const*>(iweights.GetValues());
      auto oup =
          static_cast<Eigen::half*>(const_cast<void*>(oweights->GetValues()));
      std::transform(inp, inp + iweights.count(), oup,
                     unary_op.unary<Eigen::half>());
      break;
    }
    default:
      return tensorflow::errors::Unimplemented(
          "Data type not supported: " +
          tensorflow::DataTypeString(iweights.type_));
  }
  return tensorflow::Status::OK();
}

// If swapped_inputs is false, 'tensor' is the left operand and 'weights' is the
// right operand. If swapped_inputs is true, those two are swapped.
//
// TODO(jie): broadcast is needed yet not implemented.
// Only implemented channel wise for the time being.
Status BinaryTensorOpWeight(OpConverterParams* params,
                            const nvinfer1::ITensor* tensor,
                            TRT_ShapedWeights weights, bool swapped_inputs) {
  static const std::unordered_set<string> supported_ops = {"Sub", "Add", "Mul",
                                                           "Div", "RealDiv"};
  const auto& node_def = params->node_def;
  if (!supported_ops.count(node_def.op())) {
    return errors::Unimplemented(node_def.op(), " is not supported, at ",
                                 node_def.name());
  }

  // Check type consistency.
  nvinfer1::DataType trt_dtype;
  TF_RETURN_IF_ERROR(ConvertDType(weights.type_, &trt_dtype));

  // Check scale mode.
  auto dims_w = weights.shape_;
  const auto dims_t = tensor->getDimensions();

  // TODO(jie): addScale checks for input tensor dimension
  if (dims_t.nbDims != 3) {
    return errors::InvalidArgument("addScale requires tensor with rank 3, at ",
                                   node_def.name());
  }

  // Default to element-wise
  auto scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;

  // TODO(jie): maybe use a permutation instead to support more cases;
  bool need_to_permute = false;

  if (weights.count() == 1) {
    scale_mode = nvinfer1::ScaleMode::kUNIFORM;
  } else {
    VLOG(2) << "weights dims: " << DebugString(dims_w)
            << "; tensor dims: " << DebugString(dims_t);
    // Make sure no broadcasting on batch dimension.
    if (dims_w.nbDims == dims_t.nbDims + 1) {
      if (dims_w.d[0] == 1) {
        for (int i = 1; i < dims_w.nbDims; i++) {
          dims_w.d[i - 1] = dims_w.d[i];
        }
        dims_w.nbDims--;
      } else {
        return errors::InvalidArgument("Binary op cannot operate on batch, at ",
                                       node_def.name());
      }
    }

    if (dims_w.nbDims == dims_t.nbDims && dims_w.d[0] == dims_t.d[0]) {
      scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;
      // Default is element-wise
      for (int i = 1; i < dims_w.nbDims; i++) {
        if (dims_w.d[i] != dims_t.d[i]) {
          // If dimension does not match, switch back to per-channel
          scale_mode = nvinfer1::ScaleMode::kCHANNEL;
          break;
        }
      }
      // If the mode is per-channel, since channel dimension is assumed to be
      // the third to last dimension, we need to make sure all other dimensions
      // have size 1.
      if (scale_mode == nvinfer1::ScaleMode::kCHANNEL) {
        for (int i = 1; i < dims_w.nbDims; i++) {
          if (dims_w.d[i] != 1)
            return errors::InvalidArgument(
                "Weight dims not compatible for channel-wise broadcast at ",
                node_def.name());
        }
      }
    } else if (dims_w.nbDims == 1 &&
               dims_w.d[0] == dims_t.d[dims_t.nbDims - 1]) {
      // Channel wise and broadcast required. We compare the last dimension of
      // the tensor shape because of tensorflow default broadcasting rules.
      need_to_permute = true;
      scale_mode = nvinfer1::ScaleMode::kCHANNEL;
    } else {
      return errors::InvalidArgument("Weight dims not compatible at ",
                                     node_def.name());
    }
  }
  // TODO(laigd): we should add validation_only support in TransposeTensor() and
  // PrepareTensorForShape().
  if (params->validation_only) return Status::OK();

  // Transpose last dimension.
  std::vector<int> permutation(dims_t.nbDims + 1);
  if (need_to_permute) {
    // We swap the last dimension into channel for trt, because of tensorflow
    // default broadcasting rules.
    for (int i = 0; i < static_cast<int>(permutation.size()); i++) {
      permutation[i] = i;
    }
    permutation[1] = dims_t.nbDims;
    permutation[dims_t.nbDims] = 1;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(tensor), permutation, &tensor));
  }

  if (params->converter->precision_mode() == FP16MODE) {
    weights = ConvertFP32ToFP16(params->weight_store, weights);
  }

  // Prepare weights
  TRT_ShapedWeights shift_weights(weights.type_);
  TRT_ShapedWeights scale_weights(weights.type_);
  TRT_ShapedWeights power_weights(weights.type_);

  if (node_def.op() == "Sub") {
    if (swapped_inputs) {
      shift_weights = weights;
      nvinfer1::IUnaryLayer* layer = params->converter->network()->addUnary(
          *const_cast<nvinfer1::ITensor*>(tensor),
          nvinfer1::UnaryOperation::kNEG);
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
      // Since quantization ranges are symmetric, the same range as the input
      // will work for the negation of the input.
      params->converter->MarkQuantizationRangesAsInferrable(
          const_cast<nvinfer1::ITensor*>(tensor), layer->getOutput(0));
      tensor = layer->getOutput(0);
    } else {
      TRT_ShapedWeights neg_weights =
          params->weight_store->GetTempWeights(weights);
      LambdaFactory unary_op;
      unary_op.op = LambdaFactory::OP_CATEGORY::NEG;
      TF_RETURN_IF_ERROR(UnaryCompute(weights, &neg_weights, unary_op));
      shift_weights = neg_weights;
    }
  } else if (node_def.op() == "Div" || node_def.op() == "RealDiv") {
    if (swapped_inputs) {
      // We need to infer the quantization range for this intermediate tensor.
      //
      //   x -> [Recip] -> 1/x -> [Scale] -> s/x
      //                    ^
      //            need range for this
      //
      // We have the quantization scales for x and s/x - can we divide the scale
      // for s/x by s? Only if it is a scalar.
      //
      // Because of this issue, fall back to BinaryTensorOpTensor if we are
      // doing INT8 with no calibration. There is most likely no performance
      // penalty by falling back here.
      if (params->converter->precision_mode() == INT8MODE &&
          !params->converter->use_calibration()) {
        return errors::Unimplemented(
            "Intermediate quantization range cannot be determined without"
            " calibration. Falling back to BinaryTensorOpTensor for ",
            node_def.op(), ", at ", node_def.name());
      }
      scale_weights = weights;
      nvinfer1::IUnaryLayer* layer = params->converter->network()->addUnary(
          *const_cast<nvinfer1::ITensor*>(tensor),
          nvinfer1::UnaryOperation::kRECIP);
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
      tensor = layer->getOutput(0);
    } else {
      TRT_ShapedWeights recip_weights =
          params->weight_store->GetTempWeights(weights);
      LambdaFactory unary_op;
      unary_op.op = LambdaFactory::OP_CATEGORY::RECIP;
      TF_RETURN_IF_ERROR(UnaryCompute(weights, &recip_weights, unary_op));
      scale_weights = recip_weights;
    }
  } else if (node_def.op() == "Mul") {
    scale_weights = weights;
  } else if (node_def.op() == "Add") {
    shift_weights = weights;
  } else {
    // This should not happen.
    return errors::Unimplemented("Binary op not supported at ", node_def.op());
  }

  nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
      *const_cast<nvinfer1::ITensor*>(tensor), scale_mode,
      shift_weights.GetTrtWeights(), scale_weights.GetTrtWeights(),
      power_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  const nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // Transpose back dimension
  if (need_to_permute) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), permutation,
        &output_tensor));
  }

  // Pass the output
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

enum class ConvolutionType { DEFAULT, DEPTHWISE_CONV };

tensorflow::Status ConvertConv2DHelper(OpConverterParams* params, int group) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return tensorflow::errors::InvalidArgument("Two inputs are expected for ",
                                               node_def.op(), ", at ",
                                               node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        node_def.op(), " is only implemented for tensors, not weights, at ",
        node_def.name());
  }
  if (inputs.at(1).is_tensor()) {
    return tensorflow::errors::Unimplemented("Kernel for ", node_def.op(),
                                             " must be constant weights, at ",
                                             node_def.name());
  }
  TRT_ShapedWeights weights_rsck = inputs.at(1).weights();
  if (weights_rsck.shape_.nbDims != 4) {
    return tensorflow::errors::InvalidArgument(
        "Conv2D expects kernel of dimension 4, at " + node_def.name());
  }
  TFAttrs attrs(node_def);
  auto data_format = attrs.get<string>("data_format");
  int c_index = (data_format == "NHWC") ? 3 : 1;
  int h_index = (data_format == "NHWC") ? 1 : 2;
  int w_index = (data_format == "NHWC") ? 2 : 3;
  auto tf_dilations = attrs.get<std::vector<int>>("dilations");
  if (tf_dilations.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "Convolution dilations field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_dilations[0] != 1 || tf_dilations[c_index] != 1) {
    return tensorflow::errors::Unimplemented(
        "Dilation rate must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const nvinfer1::DimsHW dilation(tf_dilations[h_index], tf_dilations[w_index]);

  const auto tf_stride = attrs.get<std::vector<int>>("strides");
  if (tf_stride.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "Convolution strides field must specify 4 dimensions, at ",
        node_def.name());
  }
  if (tf_stride[0] != 1 || tf_stride[c_index] != 1) {
    return tensorflow::errors::Unimplemented(
        "Stride must be 1 for batch and channel dimensions, at ",
        node_def.name());
  }
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);
  if (params->validation_only) return tensorflow::Status::OK();

  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  // Transpose to NCHW (NCHW is required for IConvLayer).
  const bool need_transpose = (data_format == "NHWC");
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(tensor), {0, 3, 1, 2}, &tensor));
  }
  // Dimensions of transposed tensor.
  const auto tensor_dim = tensor->getDimensions();

  // For depthwise convolution, group will be 0 so set num_groups to size of
  // input's channel dim. For a non-depthwise conv, num_groups will be 1.
  const int num_groups = (group == 0) ? tensor_dim.d[0] : group;

  if (params->converter->precision_mode() == FP16MODE) {
    weights_rsck =
        ConvertFP32ToFP16(params->weight_store, inputs.at(1).weights());
  }
  TRT_ShapedWeights weights =
      params->weight_store->GetTempWeights(weights_rsck);
  ReorderRSCKToKCRS(weights_rsck, &weights, num_groups);
  TRT_ShapedWeights biases(weights.type_);
  const int noutput = weights.shape_.d[0] * num_groups;
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = weights.shape_.d[2];
  kernel_size.w() = weights.shape_.d[3];

  // Add padding.
  std::vector<std::pair<int, int>> padding;
  if (attrs.get<string>("padding") == "SAME") {
    nvinfer1::DimsHW effective_kernel_size = kernel_size;
    effective_kernel_size.h() += (kernel_size.h() - 1) * (dilation.h() - 1);
    effective_kernel_size.w() += (kernel_size.w() - 1) * (dilation.w() - 1);
    padding = CreateSamePadding(
        stride, effective_kernel_size,
        {static_cast<int>(tensor_dim.d[1]), static_cast<int>(tensor_dim.d[2])});
  } else {
    padding = {{0, 0}, {0, 0}};
  }
  if (padding[0].first != padding[0].second ||
      padding[1].first != padding[1].second) {
    // Handle asymmetric padding.
    auto pad_layer = params->converter->network()->addPadding(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, node_def.name());
    params->converter->MarkQuantizationRangesAsInferrable(
        const_cast<nvinfer1::ITensor*>(tensor), pad_layer->getOutput(0));
    padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
  }

  // Add convolution.
  nvinfer1::IConvolutionLayer* layer =
      params->converter->network()->addConvolution(
          *const_cast<nvinfer1::ITensor*>(tensor), noutput, kernel_size,
          weights.GetTrtWeights(), biases.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  layer->setStride(stride);
  layer->setPadding({padding[0].first, padding[1].first});
  layer->setName(node_def.name().c_str());
  layer->setNbGroups(num_groups);
  layer->setDilation(dilation);
  const nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), {0, 2, 3, 1},
        &output_tensor));
  }
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConv2DHelper(OpConverterParams* params,
                                       ConvolutionType type) {
  switch (type) {
    case ConvolutionType::DEFAULT:
      return ConvertConv2DHelper(params, 1);
    case ConvolutionType::DEPTHWISE_CONV:
      return ConvertConv2DHelper(params, 0);
  }
  return tensorflow::errors::Unimplemented("Unsupported convolution type, at ",
                                           params->node_def.name());
}

Status BinaryTensorOpTensor(OpConverterParams* params,
                            const TRT_TensorOrWeights& operand_l,
                            const TRT_TensorOrWeights& operand_r) {
  const auto& node_def = params->node_def;
  static const std::unordered_map<string, nvinfer1::ElementWiseOperation> ops{
      {"Add", nvinfer1::ElementWiseOperation::kSUM},
      {"Mul", nvinfer1::ElementWiseOperation::kPROD},
      {"Sub", nvinfer1::ElementWiseOperation::kSUB},
      {"Div", nvinfer1::ElementWiseOperation::kDIV},
      {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},
      {"Minimum", nvinfer1::ElementWiseOperation::kMIN},
      {"Maximum", nvinfer1::ElementWiseOperation::kMAX},
  };
  auto op_pair = ops.find(node_def.op());
  if (op_pair == ops.end()) {
    return errors::Unimplemented("Binary op ", node_def.op(),
                                 " not supported at: ", node_def.name());
  }

  nvinfer1::Dims broadcasted_dims_l, broadcasted_dims_r;
  Status status = params->converter->GetTrtBroadcastShape(
      operand_l, operand_r, &broadcasted_dims_l, &broadcasted_dims_r);
  if (!status.ok()) {
    return errors::InvalidArgument(
        "Unsupported binary op broadcast scheme for op ", node_def.name(), ": ",
        status.error_message());
  }
  TFAttrs attrs(node_def);
  nvinfer1::DataType dtype = attrs.get<nvinfer1::DataType>("T");
  if (dtype == nvinfer1::DataType::kINT32) {
    return errors::Unimplemented("Binary op ", node_def.op(),
                                 " does not support INT32, at ",
                                 node_def.name());
  }
  if (params->validation_only) return Status::OK();

  const nvinfer1::ITensor* tensor_l = nullptr;
  const nvinfer1::ITensor* tensor_r = nullptr;
  status = params->converter->PrepareTensorForShape(
      operand_l, broadcasted_dims_l, &tensor_l);
  if (status.ok()) {
    status = params->converter->PrepareTensorForShape(
        operand_r, broadcasted_dims_r, &tensor_r);
  }
  if (!status.ok()) {
    return errors::Internal("Failed to convert binary op ", node_def.name(),
                            ": ", status.error_message());
  }

  // Check type consistency.
  TFTRT_CHECK_EQ_TYPE(tensor_l->getType(), dtype)
      << DebugString(tensor_l->getType()) << " vs " << DebugString(dtype);
  TFTRT_CHECK_EQ_TYPE(tensor_r->getType(), dtype)
      << DebugString(tensor_r->getType()) << " vs " << DebugString(dtype);

  // Add ElementWise layer.
  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *const_cast<nvinfer1::ITensor*>(tensor_l),
          *const_cast<nvinfer1::ITensor*>(tensor_r), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // Pass the output
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPlugin(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // prepare input
  std::vector<nvinfer1::ITensor*> all_inputs;
  all_inputs.reserve(inputs.size());
  for (auto input : inputs) {
    all_inputs.emplace_back(const_cast<nvinfer1::ITensor*>(input.tensor()));
  }

  // plugin is owned by PluginFactory
  // TODO(jie): destroy plugins later (resource management)
  PluginTensorRT* plugin =
      PluginFactoryTensorRT::GetInstance()->CreatePlugin(node_def.op());

  // passing attributes
  // TODO(jie): support more general attribute
  TFAttrs attrs(node_def);
  auto attr_key_vector = attrs.GetAllAttrKeys();
  for (auto attr_key : attr_key_vector) {
    // TODO(jie): support only list of float for toy example here.
    auto data = attrs.get<std::vector<float>>(attr_key);
    size_t size_data = data.size() * sizeof(float);
    if (!plugin->SetAttribute(attr_key, static_cast<void*>(data.data()),
                              size_data)) {
      return tensorflow::errors::InvalidArgument("plugin SetAttribute failed");
    }
  }

  nvinfer1::IPluginLayer* layer = params->converter->network()->addPlugin(
      &all_inputs[0], static_cast<int>(inputs.size()), *plugin);

  for (int i = 0; i < layer->getNbOutputs(); i++) {
    nvinfer1::ITensor* output_tensor = layer->getOutput(i);
    params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertTranspose(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at ", params->node_def.name());
  }

  // Get the permutation from weights.
  TRT_ShapedWeights weights = inputs.at(1).weights();
  const int* weights_ptr =
      static_cast<int*>(const_cast<void*>(weights.GetValues()));
  std::vector<int> perm(weights_ptr, weights_ptr + weights.count());

  // Verify the permutation.
  nvinfer1::ITensor* input_tensor =
      const_cast<nvinfer1::ITensor*>(inputs.at(0).tensor());
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
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(
      params->converter->TransposeTensor(input_tensor, perm, &output_tensor));
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertReshape(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2 || !inputs.at(1).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "Input expects weights for shape, at ", node_def.name());
  }

  TRT_TensorOrWeights input_tensor = inputs.at(0);
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.count() == 0) {
    return tensorflow::errors::Unimplemented(
        "Reshape to shape=[] is not supported, at ", node_def.name());
  }

  const int* weights_ptr =
      static_cast<int*>(const_cast<void*>(weights.GetValues()));

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
      if (!HasStaticShape(input_dims) ||
          TrtDimsNumElements(reshape_dims) != TrtDimsNumElements(input_dims)) {
        reshape_may_change_batch_dim = true;
      }
    } else if (reshape_batch_dim != input_batch_dim) {
      reshape_may_change_batch_dim = true;
    }
  } else if (HasStaticShape(input_dims)) {
    if (!HasStaticShape(reshape_dims) ||
        TrtDimsNumElements(reshape_dims) != TrtDimsNumElements(input_dims)) {
      reshape_may_change_batch_dim = true;
    }
  } else {
    reshape_may_change_batch_dim = true;
  }
  VLOG(1) << "input_batch_dim=" << input_batch_dim
          << ", input_dims=" << DebugString(input_dims)
          << "\nreshape_batch_dim=" << reshape_batch_dim
          << ", reshape_dims=" << DebugString(reshape_dims);
  if (reshape_may_change_batch_dim) {
    const string msg = StrCat(
        "Reshape on batch dimension is not supported, at ", node_def.name());
    return errors::Unimplemented(msg);
  }
  if (params->validation_only) return Status::OK();

  // Start conversion.
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, reshape_dims, &output_tensor));
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertExpandDims(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return tensorflow::errors::InvalidArgument(
        "Two inputs expected for ExpandDims, at ", node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        "ExpandDims expects tensor for input, at ", node_def.name());
  }
  if (!inputs.at(1).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "ExpandDims expects weights for axis, at ", node_def.name());
  }
  // Get input shape as vector.
  TRT_TensorOrWeights input_tensor = inputs.at(0);
  const nvinfer1::Dims dims = input_tensor.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  // Add batch dim back.
  input_dims.insert(input_dims.begin(), -1);
  const int input_rank = input_dims.size();
  // Get axis to expand on.
  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (weights.count() != 1) {
    return tensorflow::errors::InvalidArgument(
        "ExpandDims axis must be a scalar, at ", node_def.name());
  }
  const int* weights_ptr =
      static_cast<int*>(const_cast<void*>(weights.GetValues()));
  int axis = weights_ptr[0];
  // Make sure axis is valid.
  if ((axis < (-input_rank - 1)) || (axis > input_rank)) {
    return tensorflow::errors::InvalidArgument(
        "Axis for ExpandDims is invalid, must be in the range "
        "[-rank(input) - 1, rank(input)], at ",
        node_def.name());
  }
  // Convert negative axis to corresponding positive axis.
  if (axis < 0) axis += input_rank + 1;
  if (axis == 0) {
    return tensorflow::errors::Unimplemented(
        "Modifying batch dimension is not supported for ExpandDims, at ",
        node_def.name());
  }
  if (params->validation_only) return Status::OK();

  // ExpandDims: Insert new dim of size 1.
  input_dims.insert(input_dims.begin() + axis, 1);
  // Reshape tensor.
  nvinfer1::Dims new_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims,
                                               /*ignore_first_dim=*/true));
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, new_dims, &output_tensor));
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSqueeze(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        "One input expected for Squeeze, at ", node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        "Squeeze expects tensor for input, at ", node_def.name());
  }
  // Get input shape.
  TRT_TensorOrWeights input_tensor = inputs.at(0);
  const nvinfer1::Dims dims = input_tensor.GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  // Add batch dim back.
  input_dims.insert(input_dims.begin(), -1);
  const int input_rank = input_dims.size();
  // Mark axes to remove by setting them to 0.
  TFAttrs attrs(node_def);
  auto squeeze_dims = attrs.get<std::vector<int>>("squeeze_dims");
  if (squeeze_dims.size() == 0) {
    return tensorflow::errors::Unimplemented(
        "Squeeze is only implemented for explicit dims, at ", node_def.name());
  }
  for (int axis : squeeze_dims) {
    // Make sure axis is valid.
    if ((axis < -input_rank) || (axis >= input_rank)) {
      return tensorflow::errors::InvalidArgument(
          "Axis for Squeeze is invalid, must be in the range "
          "[-rank(input), rank(input)), at ",
          node_def.name());
    }
    // Convert negative axis to corresponding positive axis.
    if (axis < 0) axis += input_rank;
    // Don't squeeze batch dim.
    if (axis == 0) {
      return tensorflow::errors::Unimplemented(
          "Cannot squeeze batch dimension, at ", node_def.name());
    }
    // Make sure target dimension is size 1.
    if (input_dims[axis] != 1) {
      return tensorflow::errors::InvalidArgument(
          "Cannot squeeze a dimension which isn't size 1, at ",
          node_def.name());
    }
    // Mark dim for removal by setting to 0.
    input_dims[axis] = 0;
  }
  if (params->validation_only) return Status::OK();

  // Remove all dims which are equal to 0.
  input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 0),
                   input_dims.end());
  // Reshape tensor.
  nvinfer1::Dims new_dims;
  TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims,
                                               /*ignore_first_dim=*/true));
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      input_tensor, new_dims, &output_tensor));
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

// Gets the bounds (start or end) from the weights of a StridedSlice op.
tensorflow::Status GetStridedSliceBound(const std::vector<int>& input_dims,
                                        const TRT_ShapedWeights& bound_weights,
                                        int mask, bool begin, string node_name,
                                        std::vector<int>* output_bound) {
  const string bound_name = (begin) ? "begin" : "end";
  const int* weights_ptr = static_cast<int*>(bound_weights.GetValues());
  *output_bound =
      std::vector<int>(weights_ptr, weights_ptr + bound_weights.count());
  if (output_bound->size() != input_dims.size()) {
    return tensorflow::errors::InvalidArgument(
        "StridedSlice \"", bound_name, "\" specified ",
        std::to_string(output_bound->size()), " dimensions, but input rank is ",
        std::to_string(input_dims.size()), ", at ", node_name);
  }
  for (int i = 0; i < output_bound->size(); i++) {
    if ((1 << i) & mask) {
      // Apply mask.
      (*output_bound)[i] = (begin) ? 0 : input_dims[i];
      // Masked bound will always result in a valid, non-negative bound, so we
      // don't need the following checks. For the common case of using masks on
      // a undefined batch dim (-1), we specifically don't want to do the
      // following checks because they will erroneously detect an out of range
      // bound or try to correct the negative value.
      continue;
    }
    // Make sure bound is valid.
    if (((*output_bound)[i] < -input_dims[i]) ||
        ((*output_bound)[i] > input_dims[i])) {
      return tensorflow::errors::InvalidArgument(
          bound_name, " value of ", std::to_string((*output_bound)[i]),
          " for StridedSlice is invalid, must be in the range "
          "[-dim_size(i), dim_size(i)], at ",
          node_name);
    }
    // Convert negative values to their positive equivalent.
    if ((*output_bound)[i] < 0) {
      (*output_bound)[i] += input_dims[i];
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertStridedSlice(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "StridedSlice expects 4 inputs, at ", node_def.name());
  }
  if (!inputs.at(1).is_weights() || !inputs.at(2).is_weights() ||
      !inputs.at(3).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "StridedSlice expects weights for begin, end, and strides, at ",
        node_def.name());
  }
  if (!inputs.at(0).is_tensor()) {
    return tensorflow::errors::Unimplemented(
        "StridedSlice is only implemented for tensors, at ", node_def.name());
  }
  // Get input dims.
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  std::vector<int> input_dims(dims.d, dims.d + dims.nbDims);
  if (inputs.at(0).is_tensor()) {
    // Temporarily add batch dimension so that indexes line up properly.
    input_dims.insert(input_dims.begin(), inputs.at(0).batch_size());
  }
  if (input_dims.size() > 4) {
    return tensorflow::errors::Unimplemented(
        "StridedSlice is not implemented for tensors with rank > 4, at ",
        node_def.name());
  }
  TFAttrs attrs(node_def);
  // Get begin and end bounds per axis.
  std::vector<int> begin, end;
  TF_RETURN_IF_ERROR(GetStridedSliceBound(input_dims, inputs.at(1).weights(),
                                          attrs.get<int>("begin_mask"), true,
                                          node_def.name(), &begin));
  TF_RETURN_IF_ERROR(GetStridedSliceBound(input_dims, inputs.at(2).weights(),
                                          attrs.get<int>("end_mask"), false,
                                          node_def.name(), &end));
  // Get strides per axis (must all be 1).
  TRT_ShapedWeights stride_weights = inputs.at(3).weights();
  const int* stride_weights_ptr = static_cast<int*>(stride_weights.GetValues());
  std::vector<int> strides(stride_weights_ptr,
                           stride_weights_ptr + stride_weights.count());
  for (int x : strides) {
    if (x != 1) {
      return tensorflow::errors::Unimplemented(
          "StridedSlice is only implemented for stride of 1, at ",
          node_def.name());
    }
  }
  // Unsupported mask options.
  for (const string& attr :
       {"ellipsis_mask", "new_axis_mask", "shrink_axis_mask"}) {
    int attr_val = attrs.get<int>(attr);
    if (attr_val != 0) {
      return tensorflow::errors::Unimplemented(
          attr, " is not supported for StridedSlice, at ", node_def.name());
    }
  }

  nvinfer1::ITensor* tensor =
      const_cast<nvinfer1::ITensor*>(inputs.at(0).tensor());
  // Reshape if necessary to 4-D, since IPaddingLayer requires a 4-D input.
  const bool need_reshape = (input_dims.size() != 4);
  int reshape_dims_added = 0;
  nvinfer1::Dims reshape_dims;
  if (need_reshape) {
    // Add new dims after batch dim until tensor is 4D.
    while (input_dims.size() < 4) {
      input_dims.insert(input_dims.begin() + 1, 1);
      begin.insert(begin.begin() + 1, 0);
      end.insert(end.begin() + 1, 1);
      reshape_dims_added++;
    }
    TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &reshape_dims,
                                                 /*ignore_first_dim=*/true));
  }
  // Find dimensions which need to be sliced.
  std::vector<int> pad_dims;
  for (int i = 0; i < input_dims.size(); i++) {
    if ((begin[i] != 0) || (end[i] != input_dims[i])) {
      if (i == 0) {
        return tensorflow::errors::Unimplemented(
            "StridedSlice can't modify batch dim, at ", node_def.name());
      } else if ((end[i] - begin[i]) < 0) {
        return tensorflow::errors::InvalidArgument(
            "New size of sliced dimension is negative, at ", node_def.name());
      }
      pad_dims.push_back(i);
    }
  }
  if (pad_dims.size() == 0) {
    // No dimensions are changed. We could create a padding layer anyway with
    // values of 0.
    if (params->validation_only) return Status::OK();
    params->outputs->push_back(inputs.at(0));
    return tensorflow::Status::OK();
  } else if (pad_dims.size() == 1) {
    // Only one dim is modified but we have to have 2, mark a second dim which
    // will have padding of 0. The dim we add is chosen to avoid an unecessary
    // transpose.
    if (pad_dims[0] != 2) {
      pad_dims.push_back(2);
    } else {
      pad_dims.push_back(3);
    }
  } else if (pad_dims.size() > 2) {
    return tensorflow::errors::Unimplemented(
        "StridedSlice can only modify 2 dimensions, at ", node_def.name());
  }
  std::sort(pad_dims.begin(), pad_dims.end());
  // Convert to pre/post padding values. Since TRT does not have a StridedSlice
  // or Slice layer, we instead create an IPaddingLayer with negative padding.
  nvinfer1::DimsHW pre_padding, post_padding;
  for (int i = 0; i < pad_dims.size(); i++) {
    const int axis = pad_dims[i];
    pre_padding.d[i] = -begin[axis];
    post_padding.d[i] = end[axis] - input_dims[axis];
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
  if (need_reshape) {
    const nvinfer1::ITensor* output_tensor = nullptr;
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        inputs.at(0), reshape_dims, &output_tensor));
    tensor = const_cast<nvinfer1::ITensor*>(output_tensor);
  }
  if (need_transpose) {
    const nvinfer1::ITensor* output_tensor = nullptr;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, transpose_order, &output_tensor));
    tensor = const_cast<nvinfer1::ITensor*>(output_tensor);
  }

  // Add padding layer
  nvinfer1::IPaddingLayer* layer = params->converter->network()->addPadding(
      *const_cast<nvinfer1::ITensor*>(tensor), pre_padding, post_padding);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  params->converter->MarkQuantizationRangesAsInferrable(tensor,
                                                        layer->getOutput(0));
  tensor = layer->getOutput(0);

  // Restore transpose
  if (need_transpose) {
    const nvinfer1::ITensor* output_tensor = nullptr;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        tensor, inv_transpose_order, &output_tensor));
    tensor = const_cast<nvinfer1::ITensor*>(output_tensor);
  }
  // Restore reshape
  if (need_reshape) {
    // Calculate output dimensions
    for (int i = 0; i < pad_dims.size(); i++) {
      const int axis = pad_dims[i];
      input_dims[axis] = end[axis] - begin[axis];
    }
    // Remove added 1 dimensions
    for (int i = 0; i < reshape_dims_added; i++) {
      int value = input_dims[1];
      if (value != 1) {
        return tensorflow::errors::Internal(
            "StridedSlice error when reshaping, at ", node_def.name());
      }
      input_dims.erase(input_dims.begin() + 1);
    }

    nvinfer1::Dims new_dims;
    TF_RETURN_IF_ERROR(TensorShapeArrayToTrtDims(input_dims, &new_dims,
                                                 /*ignore_first_dim=*/true));
    const nvinfer1::ITensor* output_tensor = nullptr;
    TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
        TRT_TensorOrWeights(tensor), new_dims, &output_tensor));
    tensor = const_cast<nvinfer1::ITensor*>(output_tensor);
  }

  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConv2D(OpConverterParams* params) {
  return ConvertConv2DHelper(params, ConvolutionType::DEFAULT);
}

tensorflow::Status ConvertConv2DDepthwise(OpConverterParams* params) {
  return ConvertConv2DHelper(params, ConvolutionType::DEPTHWISE_CONV);
}

tensorflow::Status ConvertPool(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        node_def.op(), " is only implemented for tensors, not weights, at ",
        node_def.name());
  }
  nvinfer1::PoolingType type;
  if (node_def.op() == "MaxPool") {
    type = nvinfer1::PoolingType::kMAX;
  } else if (node_def.op() == "AvgPool") {
    type = nvinfer1::PoolingType::kAVERAGE;
  } else {
    return tensorflow::errors::Unimplemented(
        "Unsupported pooling type: ", node_def.op(), ", at ", node_def.name());
  }
  TFAttrs attrs(node_def);
  const string padding_type = attrs.get<string>("padding");
  if ((padding_type != "SAME") && (padding_type != "VALID")) {
    return tensorflow::errors::Unimplemented(
        "Unsupported padding type: ", padding_type, ", at ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  int h_index = 2;
  int w_index = 3;
  const auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    h_index = 1;
    w_index = 2;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(tensor), {0, 3, 1, 2}, &tensor));
  }

  const auto tf_stride = attrs.get<std::vector<int>>("strides");
  const nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);

  const auto tf_kernel = attrs.get<std::vector<int>>("ksize");
  const nvinfer1::DimsHW ksize(tf_kernel[h_index], tf_kernel[w_index]);

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

  if (padding[0].first != padding[0].second ||
      padding[1].first != padding[1].second) {
    VLOG(2) << "Padding!!!: " << padding[0].first << padding[0].second
            << padding[1].first << padding[1].second;
    auto pad_layer = params->converter->network()->addPadding(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    TFTRT_RETURN_ERROR_IF_NULLPTR(pad_layer, node_def.name());
    params->converter->MarkQuantizationRangesAsInferrable(
        const_cast<nvinfer1::ITensor*>(tensor), pad_layer->getOutput(0));
    padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
  }

  nvinfer1::IPoolingLayer* layer = params->converter->network()->addPooling(
      *const_cast<nvinfer1::ITensor*>(tensor), type, ksize);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  // TODO(tmorris): Average pooling may not be entirely safe to infer
  // quantization range through (at least forwards - backwards should be fine).
  // Max pooling is okay.
  params->converter->MarkQuantizationRangesAsInferrable(
      const_cast<nvinfer1::ITensor*>(tensor), layer->getOutput(0));

  layer->setStride(stride);
  layer->setPadding({padding[0].first, padding[1].first});
  layer->setName(node_def.name().c_str());
  const nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (data_format == "NHWC") {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), {0, 2, 3, 1},
        &output_tensor));
  }
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

// TODO(tmorris): Use ActivationType::kLEAKY_RELU in TRT 5.1+ once perf
// improves.
tensorflow::Status ConvertLeakyRelu(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        node_def.op(), " expects one input, at ", node_def.name());
  }
  if (!inputs.at(0).is_tensor()) {
    return tensorflow::errors::Unimplemented(
        node_def.op(), " is only implemented for tensors, at ",
        node_def.name());
  }
  TFAttrs attrs(node_def);
  const float alpha = attrs.get<float>("alpha");
  if (alpha < 0.0f || alpha > 1.0f) {
    return tensorflow::errors::Unimplemented(
        "Alpha value for LeakyRelu must be between 0 and 1, at ",
        node_def.name());
  }
  if (params->validation_only) return tensorflow::Status::OK();

  // Input Tensor
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  // Create const for alpha.
  const nvinfer1::ITensor* const_alpha_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->CreateBroadcastableScalarConstant(
      params, alpha, tensor->getDimensions(), &const_alpha_tensor));
  // alpha * x
  nvinfer1::IElementWiseLayer* mul_layer =
      params->converter->network()->addElementWise(
          *const_cast<nvinfer1::ITensor*>(tensor),
          *const_cast<nvinfer1::ITensor*>(const_alpha_tensor),
          nvinfer1::ElementWiseOperation::kPROD);
  TFTRT_RETURN_ERROR_IF_NULLPTR(mul_layer, node_def.name());
  // max(x, alpha * x)
  nvinfer1::IElementWiseLayer* max_layer =
      params->converter->network()->addElementWise(
          *const_cast<nvinfer1::ITensor*>(tensor),
          *const_cast<nvinfer1::ITensor*>(mul_layer->getOutput(0)),
          nvinfer1::ElementWiseOperation::kMAX);
  TFTRT_RETURN_ERROR_IF_NULLPTR(max_layer, node_def.name());
  nvinfer1::ITensor* output_tensor = max_layer->getOutput(0);
  params->converter->MarkQuantizationRangesAsInferrable(
      output_tensor, const_cast<nvinfer1::ITensor*>(mul_layer->getOutput(0)));

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

tensorflow::Status ConvertActivation(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        node_def.op(), " expects one input, at ", node_def.name());
  }
  if (!inputs.at(0).is_tensor()) {
    return tensorflow::errors::Unimplemented(
        node_def.op(), " is only implemented for tensors, at ",
        node_def.name());
  }
  static const std::unordered_map<string, nvinfer1::ActivationType> ops{
      {"Relu", nvinfer1::ActivationType::kRELU},
      {"Sigmoid", nvinfer1::ActivationType::kSIGMOID},
      {"Tanh", nvinfer1::ActivationType::kTANH},
  };
  auto op_pair = ops.find(node_def.op());
  if (op_pair == ops.end()) {
    return tensorflow::errors::Unimplemented("Activation op: ", node_def.op(),
                                             " not supported at: ",
                                             node_def.name());
  }
  if (params->validation_only) return tensorflow::Status::OK();

  // Start conversion.
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  nvinfer1::IActivationLayer* layer =
      params->converter->network()->addActivation(
          *const_cast<nvinfer1::ITensor*>(tensor), op_pair->second);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // Set quantization range for output of Sigmoid, Tanh.
  if (node_def.op() == "Sigmoid") {
    params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 1.0f);
  } else if (node_def.op() == "Tanh") {
    params->converter->ProvideQuantizationRange(output_tensor, -1.0f, 1.0f);
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

Status ConvertQuantize(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if ((inputs.size() == 0) ||
      (node_def.op() == "FakeQuantWithMinMaxArgs" && inputs.size() != 1) ||
      (node_def.op() == "FakeQuantWithMinMaxVars" && inputs.size() != 3) ||
      (node_def.op() == "QuantizeAndDequantizeV2" && inputs.size() != 3) ||
      (node_def.op() == "QuantizeAndDequantizeV3" && inputs.size() != 4)) {
    return errors::InvalidArgument("Invalid number of inputs for ",
                                   node_def.op(), ", at ", node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    // TensorRT will automatically quantize weights, so we will ignore ranges
    // for weights.
    params->outputs->push_back(inputs.at(0));
    return Status::OK();
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
    if (!inputs.at(1).is_weights() || !inputs.at(2).is_weights()) {
      return errors::InvalidArgument("Min and max inputs for ", node_def.op(),
                                     " must be weights not tensors, at ",
                                     node_def.name());
    }
    auto get_weights_value = [&inputs](int index) {
      auto raw_weights = static_cast<float*>(
          const_cast<void*>(inputs.at(index).weights().GetValues()));
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
  params->converter->ProvideQuantizationRange(
      const_cast<nvinfer1::ITensor*>(inputs.at(0).tensor()), min_range,
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

// TODO(tmorris): Use ActivationType::kCLIP in TRT 5.1+ once perf improves.
tensorflow::Status ConvertRelu6(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        "Invalid number of inputs for Relu6, at ", node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        "Relu6 is only implemented for tensors, not weights, at ",
        node_def.name());
  }
  if (params->validation_only) return Status::OK();
  // ***************************************************************************
  // TensorRT does not implement Relu6 natively. This function converts Relu6 op
  // to available TensorRT ops: Relu6(x) = min(Relu(x), 6)
  // ***************************************************************************

  // Input Tensor
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  // Relu operation i.e. Relu(x) = max(0, x)
  nvinfer1::IActivationLayer* relu_layer =
      params->converter->network()->addActivation(
          *const_cast<nvinfer1::ITensor*>(tensor),
          nvinfer1::ActivationType::kRELU);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu_layer, node_def.name());

  // Large range of relu is problematic during quantization in INT8 precision
  // mode. Setting dynamic range of relu = [0.f, 6.0f] helps with quantization.
  // TRT only uses dynamic ranges in INT8 precision mode,
  // and this does not affect the FP32 path.
  params->converter->ProvideQuantizationRange(relu_layer->getOutput(0), 0.0f,
                                              6.0f);

  // Create a constant layer to store the floating point weight i.e. 6.0f
  const nvinfer1::ITensor* const6_tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->CreateBroadcastableScalarConstant(
      params, 6.0f, relu_layer->getOutput(0)->getDimensions(), &const6_tensor));

  // ElementWise Min Operation
  // Min op is a nop for INT8 execution path, as the input tensor
  // to this layer will only have values in range [0.f, 6.0f].
  nvinfer1::IElementWiseLayer* relu6_layer =
      params->converter->network()->addElementWise(
          *const_cast<nvinfer1::ITensor*>(relu_layer->getOutput(0)),
          *const_cast<nvinfer1::ITensor*>(const6_tensor),
          nvinfer1::ElementWiseOperation::kMIN);
  TFTRT_RETURN_ERROR_IF_NULLPTR(relu6_layer, node_def.name());
  nvinfer1::ITensor* output_tensor = relu6_layer->getOutput(0);
  params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 6.0f);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

tensorflow::Status ConvertBiasAdd(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return errors::InvalidArgument("Input expects tensor and weights, at ",
                                   node_def.name());
  }
  TFAttrs attrs(node_def);
  tensorflow::DataType tf_dtype = attrs.get<tensorflow::DataType>("T");
  if (tf_dtype != DataType::DT_FLOAT && tf_dtype != DataType::DT_HALF) {
    return errors::Unimplemented("Data type is not supported, for node ",
                                 node_def.name(), " got ",
                                 DataTypeString(tf_dtype));
  }
  if (params->validation_only) return Status::OK();

  nvinfer1::ITensor* tensor =
      const_cast<nvinfer1::ITensor*>(inputs.at(0).tensor());
  const nvinfer1::Dims original_dims = tensor->getDimensions();
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
  if (params->converter->precision_mode() == FP16MODE) {
    weights = ConvertFP32ToFP16(params->weight_store, weights);
  }
  nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
  if (weights.shape_.d[0] == 1) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  }

  TRT_ShapedWeights empty_weights(weights.type_);
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

Status TfTensorToTrtWeights(const Tensor& tensor, TrtWeightStore* weight_store,
                            TRT_ShapedWeights* weights) {
  const DataType dtype = tensor.dtype();

  // We always convert the integer constants to INT32, since TRT INT8 is for
  // quantized inference.
  //
  // TODO(aaroey): FP16 will remain in half format and is not converted to
  // FP32, but the converter currently uses all float weights as FP32. Fix
  // this.
  const DataType converted_dtype =
      (dtype == DT_INT16 || dtype == DT_INT8 || dtype == DT_UINT8 ? DT_INT32
                                                                  : dtype);

  // Verify that the dtype is supported by TensorRT. Otherwise, return an error.
  nvinfer1::DataType trt_dtype;
  TF_RETURN_IF_ERROR(ConvertDType(converted_dtype, &trt_dtype));

  if (tensor.NumElements() == 0) {
    // Return empty weights having converted dtype.
    *weights = TRT_ShapedWeights(converted_dtype);
    return Status::OK();
  }

  nvinfer1::Dims weight_dims;
  GetTensorDimsWithProtoShape(tensor, &weight_dims);
  *weights = weight_store->GetTempWeights(converted_dtype, weight_dims);

  // Copy the tensor directly if the tensor does not require cast to the
  // supported type.
  if (converted_dtype == dtype) {
    char* dst = static_cast<char*>(const_cast<void*>(weights->GetValues()));
    memcpy(dst, tensor.tensor_data().data(), tensor.TotalBytes());
    return Status::OK();
  }

  // Copy tensor elements after casting them to the converted DataType.
  int32* dst = static_cast<int32*>(const_cast<void*>(weights->GetValues()));
  if (dtype == DT_INT16) {
    const int16* src = tensor.flat<int16>().data();
    std::copy(src, src + tensor.NumElements(), dst);
  } else if (dtype == DT_INT8) {
    const int8* src = tensor.flat<int8>().data();
    std::copy(src, src + tensor.NumElements(), dst);
  } else {
    // dtype can only be DT_UINT8 at this point.
    TFTRT_CHECK_EQ_TYPE(dtype, DT_UINT8);
    const uint8* src = tensor.flat<uint8>().data();
    std::copy(src, src + tensor.NumElements(), dst);
  }
  return Status::OK();
}

// Convert a Const NodeDef to TRT_ShapedWeights. This is a special converter, it
// always ignores the params->validation_only parameter but adds the converted
// weights to params->outputs. We did this since TrtNodeValidator needs the
// weights as input to other nodes, and use it to determine whether those nodes
// are supported by TRT.
tensorflow::Status ConvertConst(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (!inputs.empty()) {
    return errors::InvalidArgument(
        "Constant node is expected to have empty input list: ",
        node_def.name());
  }

  // Create shaped weights as output
  const auto& tensor_proto = node_def.attr().at("value").tensor();
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return tensorflow::errors::Internal("Cannot parse weight tensor proto: ",
                                        node_def.name());
  }

  TFAttrs attrs(node_def);
  const DataType dtype = attrs.get<tensorflow::DataType>("dtype");
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

tensorflow::Status ConvertIdentity(OpConverterParams* params) {
  // TODO(tmorris): TRT's Identity layer does not get optimized away as of TRT
  // 5.0, however once we know that it does it would be nice to use that
  // instead.
  params->outputs->push_back(params->inputs.at(0));
  return tensorflow::Status::OK();
}

Status ConvertBinary(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2) {
    return errors::InvalidArgument("Binary ops require two inputs, at ",
                                   node_def.name());
  }

  // Constant folding should have been done by TensorFlow
  if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
    return errors::Unimplemented(
        "Constant folding is falled back to TensorFlow, binary op received "
        "both input as constant at: ",
        node_def.name());
  }

  // TODO(tmorris): TRT plans to deprecate IScaleLayer and will replace it with
  // IElementwiseLayer. At that point, we can remove BinaryTensorOpWeight. For
  // now, the performance will be slightly better with IScaleLayer because it
  // can be fused in more situations. However, most of the benefits of
  // IScaleLayer are when the layer performs both a shift and a scale, which we
  // don't do except for convolutions.
  //
  // Try to convert into Scale layer first (for better performance).
  // Since scale layer supports restricted broadcast policy and op types, we
  // allow failure and try to handle it through Elementwise op
  // (BinaryTensorOpTensor).
  Status status = Status::OK();
  if (inputs.at(0).is_tensor() && inputs.at(1).is_weights()) {
    status = BinaryTensorOpWeight(params, inputs.at(0).tensor(),
                                  inputs.at(1).weights(), false);
  } else if (inputs.at(0).is_weights() && inputs.at(1).is_tensor()) {
    status = BinaryTensorOpWeight(params, inputs.at(1).tensor(),
                                  inputs.at(0).weights(), true);
  }
  // If both input are tensors, or one of them is weights but the conversion
  // above failed, try the conversion using BinaryTensorOpTensor.
  if ((inputs.at(0).is_tensor() && inputs.at(1).is_tensor()) || !status.ok()) {
    if (!status.ok()) VLOG(1) << status;
    status = BinaryTensorOpTensor(params, inputs.at(0), inputs.at(1));
  }
  return status;
}

tensorflow::Status ConvertUnary(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  static const std::unordered_map<string, nvinfer1::UnaryOperation> ops{
      {"Neg", nvinfer1::UnaryOperation::kNEG},
      {"Exp", nvinfer1::UnaryOperation::kEXP},
      {"Log", nvinfer1::UnaryOperation::kLOG},
      {"Sqrt", nvinfer1::UnaryOperation::kSQRT},
      {"Abs", nvinfer1::UnaryOperation::kABS},
      {"Reciprocal", nvinfer1::UnaryOperation::kRECIP},
  };

  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Unary ops require single tensor input, at ", node_def.name());
  }

  // TODO(jie): check type
  const nvinfer1::ITensor* tensor = nullptr;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(0), inputs.at(0).GetTrtDims(), &tensor));

  nvinfer1::IUnaryLayer* layer;
  if (node_def.op() == "Rsqrt") {
    // We will need a quantization range for intermediate tensor if not using
    // calibration.
    //
    //   x -> [Sqrt] -> sqrt(x) -> [Recip] -> 1/sqrt(x)
    //                     ^
    //               need range here
    if (params->converter->precision_mode() == INT8MODE &&
        !params->converter->use_calibration()) {
      return errors::Unimplemented(
          "Intermediate quantization range cannot be determined without"
          " calibration for Rsqrt, consider replacing with "
          "Sqrt -> FakeQuant -> Reciprocal ops, at ",
          node_def.name());
    }
    layer = params->converter->network()->addUnary(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::UnaryOperation::kSQRT);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    tensor = layer->getOutput(0);
    layer = params->converter->network()->addUnary(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::UnaryOperation::kRECIP);
  } else if (ops.count(node_def.op()) != 0) {
    layer = params->converter->network()->addUnary(
        *const_cast<nvinfer1::ITensor*>(tensor), ops.at(node_def.op()));
  } else {
    return tensorflow::errors::InvalidArgument(
        "Binary op: ", node_def.op(), " not supported, at ", node_def.name());
  }

  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSquare(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument("Square expects one input, at ",
                                               node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        "Square is only implemented for tensors, at ", node_def.name());
  }
  if (params->validation_only) return Status::OK();

  // Constant 2 with same rank as input
  nvinfer1::Dims dims = inputs.at(0).GetTrtDims();
  for (int i = 0; i < dims.nbDims; i++) {
    dims.d[i] = 1;
  }
  TRT_ShapedWeights weights = params->weight_store->GetTempWeights(
      tensorflow::DataType::DT_FLOAT, dims);
  auto weights_ptr =
      static_cast<float*>(const_cast<void*>(weights.GetValues()));
  weights_ptr[0] = 2.f;
  nvinfer1::ITensor* const2_tensor =
      params->converter->CreateConstantLayer(weights, dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(const2_tensor, node_def.name());

  // ElementWise Pow Operation
  nvinfer1::IElementWiseLayer* layer =
      params->converter->network()->addElementWise(
          *const_cast<nvinfer1::ITensor*>(inputs.at(0).tensor()),
          *const2_tensor, nvinfer1::ElementWiseOperation::kPOW);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertReduce(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at", node_def.name());
  }

  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  TRT_ShapedWeights index_list = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  auto index_type = attrs.get<tensorflow::DataType>("Tidx");

  // Only expect to handle INT32 as attributes for now
  if (index_type != tensorflow::DataType::DT_INT32) {
    return tensorflow::errors::Unimplemented("Tidx supports only DT_INT32");
  }

  int axes = 0;
  if (index_list.count() == 0) {
    return tensorflow::errors::InvalidArgument(
        "TRT cannot support reduce on all (batch) dimensions, at",
        node_def.name());
  } else {
    auto index_list_data =
        static_cast<int*>(const_cast<void*>(index_list.GetValues()));
    for (int i = 0; i < index_list.count(); i++) {
      int axis = index_list_data[i];
      if (axis < 0) axis += tensor->getDimensions().nbDims + 1;
      if (axis == 0) {
        return tensorflow::errors::InvalidArgument(
            "TRT cannot reduce at batch dimension, at", node_def.name());
      }
      axes |= (1 << (axis - 1));
    }
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
    return tensorflow::errors::Unimplemented("Op not supported ", node_def.op(),
                                             " , at ", node_def.name());
  }

  const auto keep_dims = attrs.get<bool>("keep_dims");
  nvinfer1::ILayer* layer = params->converter->network()->addReduce(
      *const_cast<nvinfer1::ITensor*>(tensor), reduce_operation, axes,
      keep_dims);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  params->outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPad(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // TODO(aaroey): make a routine for this check and reuse it.
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at", node_def.name());
  }

  // Implement tensor binaryOp weight [channel wise] for now;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  const auto dims = tensor->getDimensions();
  // Restore implicit batch dimension
  const int nb_dims = dims.nbDims + 1;

  TRT_ShapedWeights pads = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  // Padding type here is done through TF type
  //   so I can leverage their EnumToDataType for my cast
  auto padding_type = attrs.get<tensorflow::DataType>("Tpaddings");
  // TODO(jie): handle data type conversion for TRT?

  if (pads.shape_.d[0] != nb_dims || pads.shape_.d[1] != 2) {
    return tensorflow::errors::InvalidArgument(
        "Pad only supports explicit padding on 4 dimensional tensor, at ",
        node_def.name());
  }

  // Only expect to handle INT32 as attributes for now
  if (padding_type != tensorflow::DataType::DT_INT32) {
    return tensorflow::errors::Unimplemented(
        "Tpaddings supports only DT_INT32");
  }
  auto pad_data = static_cast<int*>(const_cast<void*>(pads.GetValues()));

  std::vector<int32_t> pad_index;
  for (int i = 0; i < nb_dims; i++) {
    if (pad_data[2 * i] != 0 || pad_data[2 * i + 1] != 0) {
      pad_index.push_back(i);
    }
  }

  // No padding at all, we should exit
  if (pad_index.size() == 0) {
    params->outputs->push_back(inputs.at(0));
    return tensorflow::Status::OK();
  }

  // Only supports padding on less than 2 axis GIE-2579
  if (pad_index.size() > 2) {
    return tensorflow::errors::InvalidArgument(
        "Padding layer does not support padding on > 2");
  }

  // Padding on batch dimension is not supported
  if (pad_index[0] == 0) {
    return tensorflow::errors::InvalidArgument(
        "Padding layer does not support padding on batch dimension");
  }

  // Not doing the legit thing here. ignoring padding on dim 1 and 3;
  // TODO(jie): implement pad as uff parser
  if (pad_index.size() == 2 && pad_index[0] == 0 && pad_index[1] == 3) {
    return tensorflow::errors::Unimplemented(
        "Padding layer does not support padding on dimension 1 and 3 yet");
  }
  if (params->validation_only) return Status::OK();

  bool legit_pad = true;
  nvinfer1::DimsHW pre_padding(0, 0);
  nvinfer1::DimsHW post_padding(0, 0);

  std::vector<int32_t> permuted_pad_index(pad_index);
  if (pad_index[0] == 1) {
    legit_pad = false;
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(tensor), {0, 3, 2, 1}, &tensor));
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
      *const_cast<nvinfer1::ITensor*>(tensor), pre_padding, post_padding);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  const nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (!legit_pad) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), {0, 3, 2, 1},
        &output_tensor));
  }

  params->outputs->push_back(
      TRT_TensorOrWeights(const_cast<nvinfer1::ITensor*>(output_tensor)));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConcat(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  // not including the last input (axis) here
  int input_size = static_cast<int>(inputs.size()) - 1;

  if (!inputs.at(0).is_tensor()) {
    return tensorflow::errors::InvalidArgument(
        "Concat in TRT support only Tensor input, at ", node_def.name());
  }

  // We are retrieving the axis
  TRT_ShapedWeights axis = inputs.at(input_size).weights();

  TFAttrs attrs(node_def);
  auto index_type = attrs.get<tensorflow::DataType>("Tidx");

  // TODO(jie): handle data type
  // Only expect to handle INT32 as index attributes for now
  if (index_type != tensorflow::DataType::DT_INT32)
    return tensorflow::errors::Unimplemented("Tidx supports only DT_INT32, at ",
                                             node_def.name());

  int index = *(static_cast<int*>(const_cast<void*>(axis.GetValues())));

  // TODO(jie): early termination with no-op (attr_size==1)

  auto dim = inputs.at(0).tensor()->getDimensions();
  // dimension check
  if (index > dim.nbDims + 1) {
    return tensorflow::errors::InvalidArgument(
        "Concatenate on axis out of dimension range, at ", node_def.name());
  }
  if (index == 0) {
    return tensorflow::errors::InvalidArgument(
        "Concatenate on batch dimension not supported, at ", node_def.name());
  }
  if (index < 0) {
    index = dim.nbDims + index + 1;
  }

  std::vector<nvinfer1::ITensor const*> inputs_vec;
  // Shap chack (all input tensor should have same shape)
  // starting from 0 since we are probably also doing transpose here;
  for (int i = 0; i < input_size; i++) {
    auto tensor_i = inputs.at(i).tensor();
    auto dim_i = tensor_i->getDimensions();
    if (dim_i.nbDims != dim.nbDims) {
      return tensorflow::errors::InvalidArgument(
          "Concatenate receives inputs with inconsistent dimensions, at ",
          node_def.name());
    }
    for (int j = 0; j < dim.nbDims; j++) {
      // check dimension consistency on non-concatenate axis
      if (j != index - 1 && dim_i.d[j] != dim.d[j]) {
        return tensorflow::errors::InvalidArgument(
            "Concatenate receives inputs with inconsistent shape, at",
            node_def.name());
      }
    }

    inputs_vec.push_back(tensor_i);
  }
  if (params->validation_only) return tensorflow::Status::OK();

  // nvinfer1::ITensor const* tensor = inputs.at(0).tensor();
  nvinfer1::IConcatenationLayer* layer =
      params->converter->network()->addConcatenation(
          const_cast<nvinfer1::ITensor* const*>(inputs_vec.data()),
          inputs_vec.size());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  layer->setAxis(index - 1);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFusedBatchNorm(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TFAttrs attrs(node_def);
  float epsilon = attrs.get<float>("epsilon");
  auto data_format = attrs.get<string>("data_format");
  if (data_format != "NCHW") {
    return tensorflow::errors::Unimplemented(
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
    return tensorflow::errors::Unimplemented(
        node_def.op(), " only supports is_training=false, at ",
        node_def.name());
  }
  if (inputs.at(0).is_weights()) {
    return tensorflow::errors::Unimplemented(
        node_def.op(),
        " is only implemented for tensor inputs, not weights, at ",
        node_def.name());
  }
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).is_tensor()) {
      return tensorflow::errors::Unimplemented(
          node_def.op(),
          " must have constant inputs for scale, offset, mean and variance, "
          "at ",
          node_def.name());
    }
  }
  nvinfer1::ITensor const* tensor = inputs.at(0).tensor();

  //  Check parameter types
  auto parameter_type = inputs.at(1).weights().type_;
  if ((parameter_type != tensorflow::DataType::DT_FLOAT) &&
      (parameter_type != tensorflow::DataType::DT_HALF)) {
    return tensorflow::errors::Unimplemented(
        "only float32 or float16 weight data type is supported, for node " +
        node_def.name() + " got " + tensorflow::DataTypeString(parameter_type));
  }
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).weights().type_ != parameter_type) {
      return tensorflow::errors::Unimplemented(
          "Inconsistent parameter type for batchnorm is not supported, at: " +
          node_def.name());
    }
  }

  TRT_ShapedWeights dummy_power_weights(parameter_type);
  size_t nweight = 0;
  for (int i = 1; i < 5; i++) {
    nweight = std::max(nweight, (size_t)inputs.at(i).weights().count());
  }
  TRT_ShapedWeights* ptr_shape_weights = nullptr;
  for (int i = 1; i < 5; i++) {
    if (inputs.at(i).weights().count() == nweight) {
      ptr_shape_weights =
          const_cast<TRT_ShapedWeights*>(&(inputs.at(i).weights()));
    } else if (inputs.at(i).weights().count() != 1) {
      return tensorflow::errors::InvalidArgument(
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
  Eigen::half* cast_combined_scale_vals = const_cast<Eigen::half*>(
      static_cast<Eigen::half const*>(combined_scale_weights.GetValues()));
  Eigen::half* cast_combined_offset_vals = const_cast<Eigen::half*>(
      static_cast<Eigen::half const*>(combined_offset_weights.GetValues()));
  float* combined_scale_vals = const_cast<float*>(
      static_cast<float const*>(combined_scale_weights.GetValues()));
  float* combined_offset_vals = const_cast<float*>(
      static_cast<float const*>(combined_offset_weights.GetValues()));

  for (size_t i = 0; i < nweight; ++i) {
    float batchnorm_data[4];
    for (int j = 0; j < 4; j++) {
      if (inputs.at(j + 1).weights().count() != 1) {
        if (parameter_type == tensorflow::DT_FLOAT) {
          batchnorm_data[j] = vals_array[j][i];
        } else if (parameter_type == tensorflow::DT_HALF) {
          batchnorm_data[j] =
              Eigen::half_impl::half_to_float(cast_vals_array[j][i]);
        }
      } else {
        if (parameter_type == tensorflow::DT_FLOAT) {
          batchnorm_data[j] = vals_array[j][0];
        } else if (parameter_type == tensorflow::DT_HALF) {
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
    if (parameter_type == tensorflow::DT_FLOAT) {
      combined_scale_vals[i] = combined_scale_val;
      combined_offset_vals[i] = combined_offset_val;
    } else if (parameter_type == tensorflow::DT_HALF) {
      cast_combined_scale_vals[i] = Eigen::half(combined_scale_val);
      cast_combined_offset_vals[i] = Eigen::half(combined_offset_val);
    }
  }

  nvinfer1::ScaleMode mode = nweight == 1 ? nvinfer1::ScaleMode::kUNIFORM
                                          : nvinfer1::ScaleMode::kCHANNEL;
  nvinfer1::IScaleLayer* layer = params->converter->network()->addScale(
      *const_cast<nvinfer1::ITensor*>(tensor), mode,
      combined_offset_weights.GetTrtWeights(),
      combined_scale_weights.GetTrtWeights(),
      dummy_power_weights.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertMatMulHelper(OpConverterParams* params,
                                       TRT_TensorOrWeights tensor_input,
                                       TRT_ShapedWeights weights_raw,
                                       bool transpose_weight,
                                       string node_name) {
  nvinfer1::ITensor* output_tensor;
  if (!tensor_input.is_tensor()) {
    return tensorflow::errors::InvalidArgument("Input 0 expects tensor");
  }
  const nvinfer1::ITensor* tensor = tensor_input.tensor();

  TRT_ShapedWeights weights(weights_raw.type_);
  if (transpose_weight) {
    weights = weights_raw;
  } else {
    weights = params->weight_store->GetTempWeights(weights_raw);
    ReorderCKtoKC(weights_raw, &weights);
  }
  TRT_ShapedWeights biases(weights.type_);

  int noutput = weights.shape_.d[0];

  auto input_dim = tensor->getDimensions();
  while (input_dim.nbDims != 3) {
    input_dim.d[input_dim.nbDims++] = 1;
  }
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      tensor_input, input_dim, &tensor));

  nvinfer1::IFullyConnectedLayer* layer =
      params->converter->network()->addFullyConnected(
          *const_cast<nvinfer1::ITensor*>(tensor), noutput,
          weights.GetTrtWeights(), biases.GetTrtWeights());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_name);
  output_tensor = layer->getOutput(0);

  const nvinfer1::ITensor* temp_tensor = nullptr;
  auto output_dim = output_tensor->getDimensions();
  output_dim.nbDims = 1;
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      TRT_TensorOrWeights(output_tensor), output_dim, &temp_tensor));
  output_tensor = const_cast<nvinfer1::ITensor*>(temp_tensor);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

// inputs are both two dimensional (tensorflow::ops::MatMul)
tensorflow::Status ConvertMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return errors::InvalidArgument("Input expects tensor and weights, at ",
                                   node_def.name());
  }

  TFAttrs attrs(node_def);
  tensorflow::DataType tf_dtype = attrs.get<tensorflow::DataType>("T");
  if (tf_dtype != DataType::DT_FLOAT && tf_dtype != DataType::DT_HALF) {
    return errors::Unimplemented("Data type is not supported, for node ",
                                 node_def.name(), " got ",
                                 DataTypeString(tf_dtype));
  }
  bool transpose_a = attrs.get<bool>("transpose_a");
  bool transpose_b = attrs.get<bool>("transpose_b");

  // FullyConnected:
  if (transpose_a) {
    return errors::InvalidArgument(
        "transpose_a is not supported for TensorRT FullyConnected (op: ",
        node_def.op(), "), at: ", node_def.name());
  }
  if (params->validation_only) return Status::OK();
  return ConvertMatMulHelper(params, inputs.at(0), inputs.at(1).weights(),
                             transpose_b, node_def.name());
}

tensorflow::Status ConvertBatchMatMul(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  TFAttrs attrs(node_def);

  tensorflow::DataType tf_dtype = attrs.get<tensorflow::DataType>("T");
  if (tf_dtype != tensorflow::DataType::DT_FLOAT &&
      tf_dtype != tensorflow::DataType::DT_HALF) {
    return tensorflow::errors::Unimplemented(
        "data type is not supported, for node " + node_def.name() + " got " +
        tensorflow::DataTypeString(tf_dtype));
  }

  bool transpose_a = attrs.get<bool>("adj_x");
  bool transpose_b = attrs.get<bool>("adj_y");

  auto dims = inputs.at(0).GetTrtDims();
  if (dims.nbDims == 1) {  // NC * CK is only supported through fully connected
    if (transpose_a == false && inputs.at(0).is_tensor() &&
        inputs.at(1).is_weights()) {
      return ConvertMatMulHelper(params, inputs.at(0), inputs.at(1).weights(),
                                 transpose_b, node_def.name());
    } else {
      return tensorflow::errors::InvalidArgument(
          "Invalid configuration for MatMul, at: " + node_def.name());
    }
  }

  const nvinfer1::ITensor* tensor_l;
  const nvinfer1::ITensor* tensor_r;
  auto dims_l = inputs.at(0).GetTrtDims();
  auto dims_r = inputs.at(1).GetTrtDims();
  if (inputs.at(0).is_weights()) {
    if (inputs.at(0).GetTrtDims().d[0] != 1) {
      return tensorflow::errors::InvalidArgument(
          "Input 0 as weight assumes broadcast across batch for MatMul, at: " +
          node_def.name());
    } else {
      for (int i = 0; i < dims_l.nbDims - 1; i++) {
        dims_l.d[i] = dims_l.d[i + 1];
      }
      dims_l.nbDims--;
    }
  }
  if (inputs.at(1).is_weights()) {
    if (inputs.at(1).GetTrtDims().d[0] != 1) {
      return tensorflow::errors::InvalidArgument(
          "Input 1 as weight assumes broadcast across batch for MatMul, at: " +
          node_def.name());
    } else {
      for (int i = 0; i < dims_r.nbDims - 1; i++) {
        dims_r.d[i] = dims_r.d[i + 1];
      }
      dims_r.nbDims--;
    }
  }
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(0), dims_l, &tensor_l));
  TF_RETURN_IF_ERROR(params->converter->PrepareTensorForShape(
      inputs.at(1), dims_r, &tensor_r));

  nvinfer1::IMatrixMultiplyLayer* layer =
      params->converter->network()->addMatrixMultiply(
          *const_cast<nvinfer1::ITensor*>(tensor_l), transpose_a,
          *const_cast<nvinfer1::ITensor*>(tensor_r), transpose_b);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSoftmax(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  int nbDims = tensor->getDimensions().nbDims;
  if (nbDims == 0) {
    return tensorflow::errors::InvalidArgument(
        "TensorRT Softmax cannot apply on batch dimension, at" +
        node_def.name());
  }
  nvinfer1::ISoftMaxLayer* layer = params->converter->network()->addSoftMax(
      *const_cast<nvinfer1::ITensor*>(tensor));
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
  // Tensorflow SoftMax assumes applying softmax on the last dimension.
  layer->setAxes(1 << (nbDims - 1));

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // Quantization range for SoftMax is always (0, 1)
  params->converter->ProvideQuantizationRange(output_tensor, 0.0f, 1.0f);
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertTopK(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights()) {
    return errors::InvalidArgument("Input expects tensor and weights, at ",
                                   params->node_def.name());
  }

  const auto& node_def = params->node_def;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
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
  const int k = *(static_cast<int*>(const_cast<void*>(k_w.GetValues())));
  const uint32_t reduce_axes = 1 << (num_dims - 1);
  nvinfer1::ITopKLayer* layer = params->converter->network()->addTopK(
      *const_cast<nvinfer1::ITensor*>(tensor), op, k, reduce_axes);
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

  nvinfer1::ITensor* output_value_tensor = layer->getOutput(0);
  nvinfer1::ITensor* output_indices_tensor = layer->getOutput(1);
  params->outputs->push_back(TRT_TensorOrWeights(output_value_tensor));
  params->outputs->push_back(TRT_TensorOrWeights(output_indices_tensor));
  return tensorflow::Status::OK();
}

static void RegisterValidatableOpConverters(
    std::unordered_map<string, OpConverter>* registration) {
  // TODO(laigd): support all op types.
  (*registration)["BiasAdd"] = ConvertBiasAdd;
  (*registration)["ConcatV2"] = ConvertConcat;
  (*registration)["Const"] = ConvertConst;
  (*registration)["Conv2D"] = ConvertConv2D;
  (*registration)["DepthwiseConv2dNative"] = ConvertConv2DDepthwise;
  (*registration)["ExpandDims"] = ConvertExpandDims;
  (*registration)["LeakyRelu"] = ConvertLeakyRelu;
  (*registration)["MatMul"] = ConvertMatMul;
  (*registration)["Pad"] = ConvertPad;
  (*registration)["Relu6"] = ConvertRelu6;
  (*registration)["Reshape"] = ConvertReshape;
  (*registration)["Square"] = ConvertSquare;
  (*registration)["Squeeze"] = ConvertSqueeze;
  (*registration)["StridedSlice"] = ConvertStridedSlice;
  (*registration)["Transpose"] = ConvertTranspose;
  (*registration)["TopKV2"] = ConvertTopK;

  for (auto quantization_op_type :
       {"QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3",
        "FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxArgs"}) {
    (*registration)[quantization_op_type] = ConvertQuantize;
  }
  for (auto binary_op_type :
       {"Add", "Mul", "Sub", "Div", "RealDiv", "Maximum", "Minimum"}) {
    (*registration)[binary_op_type] = ConvertBinary;
  }
  for (auto activation_op_type : {"Relu", "Sigmoid", "Tanh"}) {
    (*registration)[activation_op_type] = ConvertActivation;
  }
  for (auto pool_op_type : {"AvgPool", "MaxPool"}) {
    (*registration)[pool_op_type] = ConvertPool;
  }
  for (auto normalization_op_type : {"FusedBatchNorm", "FusedBatchNormV2"}) {
    (*registration)[normalization_op_type] = ConvertFusedBatchNorm;
  }
}

void TrtNodeValidator::RegisterOpValidators() {
  RegisterValidatableOpConverters(&op_validators_);
}

void Converter::RegisterOpConverters() {
  RegisterValidatableOpConverters(&op_registry_);
  // TODO(ben,jie): this is a temp hack.
  op_registry_["Identity"] = ConvertIdentity;  // Identity should be removed
  op_registry_["Snapshot"] = ConvertIdentity;  // Snapshot should be removed

  op_registry_["Rsqrt"] = ConvertUnary;
  op_registry_["Reciprocal"] = ConvertUnary;
  op_registry_["Exp"] = ConvertUnary;
  op_registry_["Log"] = ConvertUnary;
  op_registry_["Sqrt"] = ConvertUnary;
  op_registry_["Abs"] = ConvertUnary;
  op_registry_["Neg"] = ConvertUnary;

  op_registry_["Sum"] = ConvertReduce;
  op_registry_["Prod"] = ConvertReduce;
  op_registry_["Max"] = ConvertReduce;
  op_registry_["Min"] = ConvertReduce;
  op_registry_["Mean"] = ConvertReduce;
  op_registry_["Softmax"] = ConvertSoftmax;
  op_registry_["BatchMatMul"] = ConvertBatchMatMul;

  plugin_converter_ = ConvertPlugin;
}

tensorflow::Status ConvertGraphDefToEngine(
    const tensorflow::GraphDef& gdef, int precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<tensorflow::PartialTensorShape>& input_shapes,
    Logger* logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, bool use_calibration,
    bool* convert_successfully) {
  engine->reset();
  if (convert_successfully) *convert_successfully = false;

  // Create the builder.
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(*logger));
  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(max_workspace_size_bytes);
  builder->setGpuAllocator(allocator);
  if (precision_mode == FP16MODE) {
    builder->setHalf2Mode(true);
  } else if (precision_mode == INT8MODE) {
    builder->setInt8Mode(true);
    if (use_calibration) {
      builder->setInt8Calibrator(calibrator);
    } else {
      builder->setInt8Calibrator(nullptr);
    }
  }

  // Create the network.
  auto trt_network =
      TrtUniquePtrType<nvinfer1::INetworkDefinition>(builder->createNetwork());
  if (!trt_network) {
    return tensorflow::errors::Internal(
        "Failed to create TensorRT network object");
  }

  // Build the network
  VLOG(1) << "Starting engine conversion ";
  Converter converter(trt_network.get(), precision_mode, use_calibration);
  std::vector<Converter::EngineOutputInfo> output_tensors;
  // Graph nodes are already topologically sorted during construction
  for (const auto& node_def : gdef.node()) {
    string node_name = node_def.name();
    VLOG(2) << "Converting op name=" << node_name << ", op=" << node_def.op();
    if (tensorflow::str_util::StartsWith(node_name, kInputPHName) &&
        (node_def.op() == "Placeholder")) {
      int32 slot_number = -1;
      if (!tensorflow::strings::safe_strto32(  // non-absl ok
              node_name.c_str() + strlen(kInputPHName), &slot_number)) {
        return tensorflow::errors::InvalidArgument(
            "Failed to parse slot number from ", node_name);
      }
      nvinfer1::DataType trt_dtype;
      nvinfer1::Dims trt_dims;
      int batch_size = -1;
      auto shape = input_shapes.at(slot_number);
      auto status = ValidateTensorProperties(
          node_def.op(), node_def.attr().at("dtype").type(), shape,
          /*validation_only=*/false, &trt_dtype, &trt_dims, &batch_size);
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
      TF_RETURN_IF_ERROR(
          converter.AddInputTensor(node_name, trt_dtype, trt_dims, batch_size));
    } else if (tensorflow::str_util::StartsWith(node_name, kOutputPHName) &&
               (node_def.op() == "Identity")) {
      int32 slot_number = -1;
      if (!tensorflow::strings::safe_strto32(  // non-absl ok
              node_name.c_str() + strlen(kOutputPHName), &slot_number)) {
        return tensorflow::errors::InvalidArgument(
            "Failed to parse slot number from ", node_name);
      }
      // Get output type that TensorFlow expects
      TFAttrs attrs(node_def);
      tensorflow::DataType tf_dtype = attrs.get<tensorflow::DataType>("T");
      nvinfer1::DataType trt_dtype;
      TF_RETURN_IF_ERROR(ConvertDType(tf_dtype, &trt_dtype));
      if (output_tensors.size() <= slot_number) {
        output_tensors.resize(slot_number + 1);
      }
      output_tensors.at(slot_number) = {node_def.input(0), node_name,
                                        trt_dtype};
    } else {
      VLOG(2) << "Converting node: " << node_def.name() << " , "
              << node_def.op();
      TF_RETURN_IF_ERROR(converter.ConvertNode(node_def));
    }
  }
  TF_RETURN_IF_ERROR(converter.RenameAndMarkOutputTensors(output_tensors));
  if (convert_successfully) *convert_successfully = true;

  // Apply user provided quantization ranges to tensors
  converter.MaybeApplyQuantizationRanges();

  // Build the engine.
  VLOG(1) << "Starting engine creation";
  engine->reset(builder->buildCudaEngine(*converter.network()));
  if (engine->get() == nullptr) {
    return tensorflow::errors::Internal("Failed to build TensorRT engine");
  }
  VLOG(1) << "Finished conversion";
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSegmentToGraphDef(
    const tensorflow::Graph* graph,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::vector<const Node*>& subgraph_nodes,  // In topological order
    std::vector<EngineConnection>* connections,
    tensorflow::GraphDef* segment_def, string* common_scope) {
  std::set<string> marker_nodes;
  // Update connection shapes/data types and add corresponding input/output
  // nodes in the segment graphdef.
  for (size_t i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    if (connection.is_control_edge()) continue;
    auto outside_node = graph->FindNodeId(connection.outside_id);
    if (!outside_node) {
      // This should never happen, unless the original graph is problematic.
      return tensorflow::errors::NotFound(
          "Cannot find node with id ", connection.outside_id, " in the graph.");
    }
    // Updates the shape and data types of input/output connections.
    tensorflow::DataType dtype;
    tensorflow::PartialTensorShape partial_shape;
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
      const string node_name = StrCat(kInputPHName, connection.port_number);
      if (marker_nodes.count(node_name)) {
        VLOG(1) << "Reusing input " << node_name << " for the edge "
                << connection.outside_node_name << ":"
                << connection.outside_port << " -> "
                << connection.inside_node_name << ":" << connection.inside_port;
        continue;
      }
      marker_nodes.insert(node_name);
      auto seg_node = segment_def->add_node();
      tensorflow::NodeDefBuilder builder(node_name, "Placeholder");
      auto status = builder.Attr("shape", partial_shape)
                        .Attr("dtype", dtype)
                        .Finalize(seg_node);
      VLOG(1) << "Constructing input " << node_name << " for the edge "
              << connection.outside_node_name << ":" << connection.outside_port
              << " -> " << connection.inside_node_name << ":"
              << connection.inside_port;
    } else {
      const string node_name = StrCat(kOutputPHName, connection.port_number);
      if (marker_nodes.count(node_name)) {
        VLOG(1) << "Reusing output " << node_name << " for the edge "
                << connection.inside_node_name << ":" << connection.inside_port
                << " -> " << connection.outside_node_name << ":"
                << connection.outside_port;
        continue;
      }
      marker_nodes.insert(node_name);
      auto seg_node = segment_def->add_node();
      tensorflow::NodeDefBuilder builder(node_name, "Identity");
      auto status =
          builder
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
    snode->CopyFrom(node->def());
    VLOG(2) << "Copying " << snode->name() << " to subgraph";
  }
  // Update the inputs of the new input nodes to point to placeholder nodes.
  for (int i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    if (connection.is_control_edge() || !connection.is_input_edge) continue;
    auto snode =
        segment_def->mutable_node(old_to_new_id_map[connection.inside_id]);
    const string placeholder_name =
        StrCat(kInputPHName, connection.port_number);
    VLOG(1) << "Updating " << snode->name() << ":" << connection.inside_port
            << " from " << snode->input(connection.inside_port) << " to "
            << placeholder_name;
    snode->set_input(connection.inside_port, placeholder_name);
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
          !str_util::StartsWith(input.first, kInputPHName)) {
        if (input.second == Graph::kControlSlot) {
          VLOG(1) << "... removing control inputs " << input.first
                  << " from subgraph.";
          ++input_idx;
          continue;
        } else {
          return tensorflow::errors::InvalidArgument(
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
  *common_scope = local_scope;
  VLOG(1) << "Converted TensorRT candidate segment @scope '" << local_scope
          << "' to a GraphDef";
  return tensorflow::Status::OK();
}

bool OutputEdgeValidator::operator()(const tensorflow::Edge* out_edge) const {
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
