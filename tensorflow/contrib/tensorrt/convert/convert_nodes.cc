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

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
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
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

//  Check if the types are equal. Cast to int first so that failure log message
//  would work!
#define CHECK_EQ_TYPE(val1, val2) CHECK_EQ((int)val1, (int)val2)

namespace tensorflow {
namespace tensorrt {
namespace convert {
using ::tensorflow::str_util::Split;

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

namespace {

inline tensorflow::Status ConvertDType(tensorflow::DataType tf_dtype,
                                       nvinfer1::DataType* trt_dtype) {
  switch (tf_dtype) {
    case tensorflow::DataType::DT_FLOAT:
      *trt_dtype = nvinfer1::DataType::kFLOAT;
      break;
    case tensorflow::DataType::DT_INT8:
      *trt_dtype = nvinfer1::DataType::kINT8;
      break;
    case tensorflow::DataType::DT_HALF:
      *trt_dtype = nvinfer1::DataType::kHALF;
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type " + tensorflow::DataTypeString(tf_dtype));
  }
  return tensorflow::Status::OK();
}

inline nvinfer1::Dims GetTensorShape(const tensorflow::Tensor& tensor) {
  nvinfer1::Dims dims;
  dims.nbDims = tensor.dims();
  for (int i = 0; i < dims.nbDims; i++) {
    dims.d[i] = tensor.dim_size(i);
  }
  return dims;
}

inline int64_t GetShapeSize(nvinfer1::Dims shape) {
  // Returns total number of elements in shape
  int64_t count = 1;
  for (int d = 0; d < shape.nbDims; ++d) {
    count *= shape.d[d];
  }
  return count;
}

static std::vector<std::pair<int, int>> CreateSamePadding(
    const nvinfer1::DimsHW& stride, const nvinfer1::DimsHW& kernel,
    const std::vector<int64_t>& input_dims) {
  std::vector<std::pair<int, int>> padding(input_dims.size());
  CHECK_EQ((size_t)stride.nbDims, input_dims.size());  // TODO(jie): N+C? NC+?

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
  for (size_t i = 0; i < std::min(op_name_a.size(), op_name_b.size()); ++i) {
    if (op_name_a[i] != op_name_b[i]) {
      break;
    } else if (op_name_a[i] == '/') {
      last_scope_separator = i + 1;
    }
  }
  return op_name_a.substr(0, last_scope_separator);
}

class TRT_ShapedWeights {
 public:
  TRT_ShapedWeights(tensorflow::DataType type, const void* values,
                    nvinfer1::Dims shape)
      : shape_(shape), type_(type), values_(values), empty_weight_flag_(false) {
    // Note: this->shape.type[] is not used
  }

  explicit TRT_ShapedWeights(tensorflow::DataType type)
      : shape_(), type_(type), values_(nullptr), empty_weight_flag_(true) {}

  TRT_ShapedWeights(const TRT_ShapedWeights& rhs)
      : shape_(rhs.shape_),
        type_(rhs.type_),
        values_(rhs.values_),
        empty_weight_flag_(rhs.empty_weight_flag_) {}

  int64_t count() const {
    int64_t c = 1;
    for (int i = 0; i < shape_.nbDims; i++) c *= shape_.d[i];
    return c;
  }

  nvinfer1::Weights GetWeightsForTRT() const {
    nvinfer1::DataType trt_type(nvinfer1::DataType::kFLOAT);
    TF_CHECK_OK(ConvertDType(type_, &trt_type));
    if (empty_weight_flag_) return nvinfer1::Weights{trt_type, nullptr, 0};

    // Note: this->shape.type[] is not used
    return nvinfer1::Weights{trt_type, GetValues(), GetShapeSize(shape_)};
  }

  const void* GetValues() const { return values_; }

  void SetValues(const void* values) { values_ = values; }

  size_t size_bytes() const {
    int type_size = tensorflow::DataTypeSize(this->type_);
    return this->count() * type_size;
  }

  // Default converter
  operator nvinfer1::Weights() const { return GetWeightsForTRT(); }

  nvinfer1::Dims shape_;
  tensorflow::DataType type_;

 private:
  const void* values_;
  bool empty_weight_flag_;
};

class TRT_TensorOrWeights {
 public:
  explicit TRT_TensorOrWeights(nvinfer1::ITensor* tensor)
      : tensor_(tensor), weights_(DT_FLOAT), variant_(TRT_NODE_TENSOR) {}
  explicit TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
      : tensor_(nullptr), weights_(weights), variant_(TRT_NODE_WEIGHTS) {}
  TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
      : tensor_(rhs.tensor_), weights_(rhs.weights_), variant_(rhs.variant_) {}
  ~TRT_TensorOrWeights() {}

  bool is_tensor() const { return variant_ == TRT_NODE_TENSOR; }
  bool is_weights() const { return variant_ == TRT_NODE_WEIGHTS; }

  nvinfer1::ITensor* tensor() {
    CHECK_EQ(is_tensor(), true);
    return tensor_;
  }
  const nvinfer1::ITensor* tensor() const {
    CHECK_EQ(is_tensor(), true);
    return tensor_;
  }
  TRT_ShapedWeights& weights() {
    CHECK_EQ(is_weights(), true);
    return weights_;
  }
  const TRT_ShapedWeights& weights() const {
    CHECK_EQ(is_weights(), true);
    return weights_;
  }
  nvinfer1::Dims shape() const {
    if (is_tensor()) {
      return tensor()->getDimensions();
    } else {
      return weights().shape_;
    }
  }

 private:
  nvinfer1::ITensor* tensor_;
  TRT_ShapedWeights weights_;
  enum { TRT_NODE_TENSOR, TRT_NODE_WEIGHTS } variant_;
};

class TFAttrs {
 public:
  explicit TFAttrs(const tensorflow::NodeDef& tf_node) {
    for (const auto& attr : tf_node.attr()) {
      attrs_.insert({attr.first, &attr.second});
    }
  }
  bool count(string key) const { return attrs_.count(key); }
  tensorflow::AttrValue const* at(string key) const {
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

  std::vector<string> GetAllAttrKey() {
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
std::vector<string> TFAttrs::get<std::vector<string>>(const string& key) const {
  auto attr = this->at(key)->list().s();
  return std::vector<string>(attr.begin(), attr.end());
}
template <>
nvinfer1::Dims TFAttrs::get<nvinfer1::Dims>(const string& key) const {
  auto values = this->get<std::vector<int>>(key);
  nvinfer1::Dims dims;
  dims.nbDims = values.size();
  std::copy(values.begin(), values.end(), dims.d);
  // Note: No dimension type information is included
  return dims;
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

// TODO(jie): reorder4 & reorder2 should be merged?
template <typename T>
void Reorder4(nvinfer1::DimsNCHW shape, const T* idata,
              nvinfer1::DimsNCHW istrides, T* odata,
              nvinfer1::DimsNCHW ostrides) {
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
void Reorder2(nvinfer1::DimsHW shape, const T* idata, nvinfer1::DimsHW istrides,
              T* odata, nvinfer1::DimsHW ostrides) {
  for (int h = 0; h < shape.h(); ++h) {
    for (int w = 0; w < shape.w(); ++w) {
      odata[h * ostrides.h() + w * ostrides.w()] =
          idata[h * ostrides.h() + w * ostrides.w()];
    }
  }
}

// TODO(jie): fallback to tensorflow!!
void ReorderCKtoKC(const TRT_ShapedWeights& iweights,
                   TRT_ShapedWeights* oweights) {
  int c = iweights.shape_.d[0];
  int k = iweights.shape_.d[1];
  oweights->shape_.d[0] = k;
  oweights->shape_.d[1] = c;
  nvinfer1::DimsHW istrides = {1, k};
  nvinfer1::DimsHW ostrides = {c, 1};
  switch (iweights.type_) {
    case tensorflow::DataType::DT_FLOAT: {
      Reorder2({k, c}, static_cast<float const*>(iweights.GetValues()),
               istrides,
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
                       TRT_ShapedWeights* oweights, int num_groups) {
  CHECK_EQ(iweights.type_, oweights->type_);
  CHECK_EQ(iweights.size_bytes(), oweights->size_bytes());
  int r = iweights.shape_.d[0];
  int s = iweights.shape_.d[1];
  // TRT requires GKcRS, while TF depthwise has RSCK
  //   where c=1, C=G
  VLOG(2) << "num_groups: " << num_groups;
  int c = iweights.shape_.d[2] / num_groups;
  VLOG(2) << "c" << iweights.shape_.d[2] << " then " << c;
  int k = iweights.shape_.d[3] * num_groups;
  VLOG(2) << "k" << iweights.shape_.d[3] << " then " << k;
  oweights->shape_.d[0] = k / num_groups;
  oweights->shape_.d[1] = c * num_groups;
  oweights->shape_.d[2] = r;
  oweights->shape_.d[3] = s;
  nvinfer1::DimsNCHW istrides = {1, k, s * k * c, c * k};
  nvinfer1::DimsNCHW ostrides = {c * r * s, r * s, s, 1};
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

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj) {
  return std::shared_ptr<T>(obj, InferDeleter());
}

class Converter;

using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&,
                                     const std::vector<TRT_TensorOrWeights>&,
                                     std::vector<TRT_TensorOrWeights>*)>;

class Converter {
  std::unordered_map<string, TRT_TensorOrWeights> trt_tensors_;
  std::unordered_map<string, OpConverter> op_registry_;
  OpConverter plugin_converter_;
  nvinfer1::INetworkDefinition* trt_network_;
  std::list<std::vector<uint8_t>> temp_bufs_;
  tensorflow::tensorrt::TRTWeightStore* weight_store_;
  bool fp16_;
  void register_op_converters();
  tensorflow::Status get_inputs(const tensorflow::NodeDef& node_def,
                                std::vector<TRT_TensorOrWeights>* inputs) {
    for (auto const& input_name : node_def.input()) {
      /*************************************************************************
       * TODO(jie) handle case 1) here
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
      auto first = name.find_first_of(':');
      if (first != string::npos && first + 2 == name.size() &&
          name[first + 1] == '0')
        name.erase(first);

      VLOG(2) << "retrieve input: " << name;
      if (trt_tensors_.count(name)) {
        inputs->push_back(trt_tensors_.at(name));
      } else {
        string str("Node ");
        StrAppend(&str, node_def.name(), " should have an input named '", name,
                  "' but it is not available");
        LOG(WARNING) << "input: " << name << " not available for node at "
                     << node_def.name();
        return tensorflow::errors::InvalidArgument(str);
      }
    }
    return tensorflow::Status::OK();
  }

 public:
  explicit Converter(nvinfer1::INetworkDefinition* trt_network,
                     tensorflow::tensorrt::TRTWeightStore* ws, bool fp16)
      : trt_network_(trt_network), weight_store_(ws), fp16_(fp16) {
    this->register_op_converters();
  }
  tensorflow::tensorrt::TRTWeightStore* weight_store() { return weight_store_; }
  TRT_ShapedWeights get_temp_weights(tensorflow::DataType type,
                                     nvinfer1::Dims shape) {
    TRT_ShapedWeights weights(type, nullptr, shape);
    // TODO(jie): check weights size_bytes. 0 means type error
    weight_store_->store_.push_back(std::vector<uint8_t>(weights.size_bytes()));
    weights.SetValues(weight_store_->store_.back().data());
    return weights;
  }
  bool isFP16() { return fp16_; }
  TRT_ShapedWeights get_temp_weights_like(const TRT_ShapedWeights& weights) {
    return this->get_temp_weights(weights.type_, weights.shape_);
  }

  tensorflow::Status convert_node(const tensorflow::NodeDef& node_def) {
    std::vector<TRT_TensorOrWeights> inputs;
    TF_RETURN_IF_ERROR(this->get_inputs(node_def, &inputs));
    string op = node_def.op();
    std::vector<TRT_TensorOrWeights> outputs;
    if (PluginFactoryTensorRT::GetInstance()->IsPlugin(op)) {
      TF_RETURN_IF_ERROR(plugin_converter_(*this, node_def, inputs, &outputs));
    } else {
      if (!op_registry_.count(op)) {
        return tensorflow::errors::Unimplemented(
            "No converter registered for op: " + op);
      }
      OpConverter op_converter = op_registry_.at(op);
      TF_RETURN_IF_ERROR(op_converter(*this, node_def, inputs, &outputs));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      TRT_TensorOrWeights output = outputs.at(i);
      // TODO(jie): tf protobuf seems to be omitting the :0 suffix
      string output_name = node_def.name();
      if (i != 0) output_name = StrCat(output_name, ":", i);
      if (output.is_tensor()) {
        output.tensor()->setName(output_name.c_str());
      }
      VLOG(2) << "Write out tensor: " << output_name;
      if (!trt_tensors_.insert({output_name, output}).second) {
        return tensorflow::errors::AlreadyExists(
            "Output tensor already exists for op: " + op);
      }
    }
    return tensorflow::Status::OK();
  }

  nvinfer1::INetworkDefinition* network() { return trt_network_; }

  TRT_TensorOrWeights get_tensor(string name) {
    if (!trt_tensors_.count(name)) {
      return TRT_TensorOrWeights(nullptr);
    }
    return trt_tensors_.at(name);
  }

  bool insert_input_tensor(string name, nvinfer1::ITensor* tensor) {
    return trt_tensors_.insert({name, TRT_TensorOrWeights(tensor)}).second;
  }

  nvinfer1::ITensor* TransposeTensor(nvinfer1::ITensor* input_tensor,
                                     std::vector<int> order) {
    auto dims = input_tensor->getDimensions();

    // TODO(jie): change the return to status and properly exit
    if (order.size() - 1 != size_t(dims.nbDims))
      LOG(ERROR) << "Dimension does not match, fail gracefully";

    nvinfer1::IShuffleLayer* layer = this->network()->addShuffle(*input_tensor);
    nvinfer1::Permutation permutation;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
      permutation.order[i] = order[i + 1] - 1;
    }
    layer->setFirstTranspose(permutation);

    nvinfer1::Dims reshape_dims;
    reshape_dims.nbDims = dims.nbDims;
    for (int32_t i = 0; i < reshape_dims.nbDims; ++i) {
      reshape_dims.d[i] = 0;
      reshape_dims.type[i] = dims.type[i];
    }
    layer->setReshapeDimensions(reshape_dims);
    return layer->getOutput(0);
  }
};

TRT_ShapedWeights ConvertFP32ToFP16(Converter& ctx,
                                    const TRT_ShapedWeights& weights_src) {
  auto dtype_new = tensorflow::DataType::DT_HALF;
  TRT_ShapedWeights weights =
      ctx.get_temp_weights(dtype_new, weights_src.shape_);
  const float* src = static_cast<const float*>(weights_src.GetValues());
  Eigen::half* dst = const_cast<Eigen::half*>(
      static_cast<Eigen::half const*>(weights.GetValues()));
  for (int64_t i = 0; i < weights_src.count(); i++) {
    dst[i] = Eigen::half_impl::float_to_half_rtne(src[i]);
  }
  return weights;
}
// ****************************************************************************
// Constant folding functions
// TODO(jie): once optimizer kicks in, we should have done constant folding
// there.
//*****************************************************************************/
struct LambdaFactory {
  enum class OP_CATEGORY : int { RSQRT = 0, NEG, ADD, MUL, SUB };
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
      default:
        VLOG(2) << "Not supported op for unary: " << static_cast<int>(op);
        return nullptr;
    }
  }

  template <typename T>
  std::function<T(T, T)> binary() {
    switch (op) {
      case OP_CATEGORY::ADD:
        return [](T l, T r) -> T { return l + r; };
      case OP_CATEGORY::SUB:
        return [](T l, T r) -> T { return l - r; };
      case OP_CATEGORY::MUL:
        return [](T l, T r) -> T { return l * r; };
      default:
        LOG(WARNING) << "Not supported op for binary: " << static_cast<int>(op);
    }
    return [](T l, T r) -> T {
      LOG(FATAL) << "Unsupported op type ";
      return l;
    };
  }

  template <typename T>
  std::function<T(T)> broadcast_r(T val) {
    VLOG(2) << "LAMBDA VAL : " << val;
    switch (op) {
      case OP_CATEGORY::ADD:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return l + val;
        };
      // Return [val](T l)-> T {return l+val;};
      case OP_CATEGORY::SUB:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return l - val;
        };
      case OP_CATEGORY::MUL:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return l * val;
        };
      default:
        LOG(WARNING) << "Not supported op for binary: " << static_cast<int>(op);
    }
    return [val](T l) -> T {
      LOG(FATAL) << "Unsupported op type ";
      return l;
    };
  }

  template <typename T>
  std::function<T(T)> broadcast_l(T val) {
    VLOG(2) << "LAMBDA VAL : " << val;
    switch (op) {
      case OP_CATEGORY::ADD:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return val + l;
        };
      case OP_CATEGORY::SUB:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return val - l;
        };
      case OP_CATEGORY::MUL:
        return [val](T l) -> T {
          VLOG(2) << "LAMBDA VAL : " << val;
          return val * l;
        };
      default:
        LOG(ERROR) << "Not supported op for binary: " << static_cast<int>(op);
    }
    return [val](T l) -> T {
      LOG(FATAL) << "Unsupported op type ";
      return l;
    };
  }
};

template <>
std::function<Eigen::half(Eigen::half)> LambdaFactory::unary<Eigen::half>() {
  switch (op) {
    case OP_CATEGORY::RSQRT: {
      VLOG(2) << "RSQRT GETS DONE";
      return [](Eigen::half t) -> Eigen::half {
        return Eigen::half(1.0 / sqrt(static_cast<float>(t)));
      };
    }
    case OP_CATEGORY::NEG:
      return [](Eigen::half t) -> Eigen::half { return -t; };
    default:
      VLOG(2) << "Not supported op for unary: " << static_cast<int>(op);
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

tensorflow::Status BinaryCompute(const TRT_ShapedWeights& iweights_l,
                                 const TRT_ShapedWeights& iweights_r,
                                 TRT_ShapedWeights* oweights,
                                 LambdaFactory binary_op) {
  // Assume iweights_l.type == iweight_r.type
  CHECK_EQ(iweights_l.type_, oweights->type_);
  CHECK_EQ(iweights_r.type_, oweights->type_);
  VLOG(2) << "SANITY CHECK!";

  switch (iweights_l.type_) {
    case tensorflow::DataType::DT_FLOAT: {
      auto inp_l = static_cast<const float*>(iweights_l.GetValues());
      auto inp_r = static_cast<const float*>(iweights_r.GetValues());
      auto oup = static_cast<float*>(const_cast<void*>(oweights->GetValues()));

      if (iweights_l.count() != iweights_r.count()) {
        // We only supports broadcast of RankZero
        if (iweights_l.count() == 1) {
          VLOG(2) << "I bet it is not working!" << (*inp_l);
          std::transform(inp_r, inp_r + iweights_r.count(), oup,
                         binary_op.broadcast_l<float>(*inp_l));
        } else if (iweights_r.count() == 1) {
          VLOG(2) << "I bet it is not working!" << (*inp_r);
          std::transform(inp_l, inp_l + iweights_l.count(), oup,
                         binary_op.broadcast_r<float>(*inp_r));
        } else {
          return tensorflow::errors::Unimplemented(
              "Binary op with non-rankZero broadcast not supported");
        }
      } else {
        std::transform(inp_l, inp_l + iweights_l.count(), inp_r, oup,
                       binary_op.binary<float>());
      }
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      auto inp_l = static_cast<const Eigen::half*>(iweights_l.GetValues());
      auto inp_r = static_cast<const Eigen::half*>(iweights_r.GetValues());
      auto oup =
          static_cast<Eigen::half*>(const_cast<void*>(oweights->GetValues()));

      if (iweights_l.count() != iweights_r.count()) {
        // We only supports broadcast of RankZero
        if (iweights_l.count() == 1) {
          VLOG(2) << "I bet it is not working!" << (*inp_l);
          std::transform(inp_r, inp_r + iweights_r.count(), oup,
                         binary_op.broadcast_l<Eigen::half>(*inp_l));
        } else if (iweights_r.count() == 1) {
          VLOG(2) << "I bet it is not working!" << (*inp_r);
          std::transform(inp_l, inp_l + iweights_l.count(), oup,
                         binary_op.broadcast_r<Eigen::half>(*inp_r));
        } else {
          return tensorflow::errors::Unimplemented(
              "Binary op with non-rankZero broadcast not supported");
        }
      } else {
        std::transform(inp_l, inp_l + iweights_l.count(), inp_r, oup,
                       binary_op.binary<Eigen::half>());
      }
      break;
    }
    default:
      return tensorflow::errors::Unimplemented(
          "Data type not supported: " +
          tensorflow::DataTypeString(iweights_l.type_));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status ConstantFoldUnary(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  TRT_ShapedWeights weights_input = inputs.at(0).weights();

  // Allocate output weights
  TRT_ShapedWeights weights_output = ctx.get_temp_weights_like(weights_input);

  // FIXME assume type matches input weights
  // Get trt type & shape
  // Maybe this part has to be moved into the block of rsqrt later
  // Check type consistency
  CHECK_EQ(weights_input.type_,
           TFAttrs(node_def).get<tensorflow::DataType>("T"));

  LambdaFactory unary_op;
  if (node_def.op() == "Rsqrt") {
    // Compute rsqrt
    unary_op.op = LambdaFactory::OP_CATEGORY::RSQRT;
    auto ret = UnaryCompute(weights_input, &weights_output, unary_op);
    // Pass the output
    if (ret == tensorflow::Status::OK()) {
      outputs->push_back(TRT_TensorOrWeights(weights_output));
    }
    return ret;
  } else {
    return tensorflow::errors::Unimplemented("Binary op not supported: " +
                                             node_def.op());
  }
}

// TODO(jie,ben) broadcast is needed yet not implemented
// Let's get the simple stuff working first. Maybe we should fall back to TF
//   approach for constant folding
tensorflow::Status ConstantFoldBinary(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  TRT_ShapedWeights weights_input_l = inputs.at(0).weights();
  TRT_ShapedWeights weights_input_r = inputs.at(1).weights();

  // Check type consistency
  CHECK_EQ(weights_input_l.type_, weights_input_r.type_);

  if (weights_input_l.shape_.nbDims != weights_input_r.shape_.nbDims)
    return tensorflow::errors::Unimplemented(
        "Binary op implicit broadcast not supported: " + node_def.op());

  // TODO(jie): constant fold should really fall back to TF.
  int num_dims = weights_input_l.shape_.nbDims;
  nvinfer1::Dims output_shape;
  output_shape.nbDims = num_dims;
  VLOG(2) << "nb_dims: " << num_dims
          << ", the other: " << weights_input_r.shape_.nbDims;
  for (int i = 0; i < num_dims; i++) {
    if (weights_input_l.shape_.d[i] == weights_input_r.shape_.d[i]) {
      output_shape.d[i] = weights_input_l.shape_.d[i];
    } else if (weights_input_l.shape_.d[i] == 1 ||
               weights_input_r.shape_.d[i] == 1) {
      output_shape.d[i] =
          std::max(weights_input_l.shape_.d[i], weights_input_r.shape_.d[i]);
    } else {
      return tensorflow::errors::Unimplemented(
          "Binary op with incompatible shape at, " + node_def.op());
    }
    VLOG(2) << "left: " << weights_input_l.shape_.d[i]
            << "right: " << weights_input_r.shape_.d[i]
            << "output: " << output_shape.d[i];
  }

  // FIXME assume type matches input weights
  // Get trt type & shape
  TFAttrs attrs(node_def);
  // Maybe this part has to be moved into the block of rsqrt later
  tensorflow::DataType dtype = attrs.get<tensorflow::DataType>("T");

  // Allocate output weights
  TRT_ShapedWeights weights_output = ctx.get_temp_weights(dtype, output_shape);

  LambdaFactory binary_op;
  if (node_def.op() == "Sub") {
    binary_op.op = LambdaFactory::OP_CATEGORY::SUB;
  } else if (node_def.op() == "Mul") {
    binary_op.op = LambdaFactory::OP_CATEGORY::MUL;
  } else if (node_def.op() == "Add") {
    binary_op.op = LambdaFactory::OP_CATEGORY::ADD;
  } else {
    return tensorflow::errors::Unimplemented("Binary op not supported: " +
                                             node_def.op());
  }
  auto ret = BinaryCompute(weights_input_l, weights_input_r, &weights_output,
                           binary_op);

  // Pass the output
  if (ret == tensorflow::Status::OK()) {
    outputs->push_back(TRT_TensorOrWeights(weights_output));
  }

  return ret;
}

// TODO(jie): broadcast is needed yet not implemented.
// Only implemented channel wise for the time being
tensorflow::Status BinaryTensorOpWeight(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const nvinfer1::ITensor* tensor, TRT_ShapedWeights weights,
    std::vector<TRT_TensorOrWeights>* outputs) {
  // FIXME assume type matches input weights
  // Get trt type & shape
  // Maybe this part has to be moved into the block of rsqrt later

  // Check type consistency
  nvinfer1::DataType ttype;
  TF_RETURN_IF_ERROR(ConvertDType(weights.type_, &ttype));

  // Check scale mode
  auto dims_w = weights.shape_;
  auto dims_t = tensor->getDimensions();

  // default to element-wise
  auto scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;

  // TODO(jie): maybe use a permutation instead to support more cases;
  bool permutation_flag = false;

  if (weights.count() == 1) {
    VLOG(2) << "UNIFORM";
    scale_mode = nvinfer1::ScaleMode::kUNIFORM;
  } else {
    // no broadcasting on Batch dimension;
    VLOG(2) << "WEIGHTS DIM: " << dims_w.nbDims
            << " tensor DIM: " << dims_t.nbDims;
    if (dims_w.nbDims == dims_t.nbDims + 1) {
      if (dims_w.d[0] == 1) {
        for (int i = 1; i < dims_w.nbDims; i++) {
          dims_w.d[i - 1] = dims_w.d[i];
        }
        dims_w.nbDims--;
      } else {
        return tensorflow::errors::InvalidArgument(
            "Binary op cannot operate on batch, " + node_def.name());
      }
    }

    if (dims_w.nbDims == dims_t.nbDims && dims_w.d[0] == dims_t.d[0]) {
      scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;
      // default is element;
      for (int i = 1; i < dims_w.nbDims; i++) {
        if (dims_w.d[i] != dims_t.d[i]) {
          // if dimension does not match, switch back to channel;
          VLOG(2) << "channel";
          scale_mode = nvinfer1::ScaleMode::kCHANNEL;
          break;
        }
      }
      // if channel as candidate, validate it
      if (scale_mode == nvinfer1::ScaleMode::kCHANNEL) {
        for (int i = 1; i < dims_w.nbDims; i++) {
          if (dims_w.d[i] != 1)
            return tensorflow::errors::InvalidArgument(
                "Weight shape not compatible at, " + node_def.name());
        }
      } else {
        VLOG(2) << "elementwise";
      }
    } else if (dims_w.nbDims == 1 &&
               dims_w.d[0] == dims_t.d[dims_t.nbDims - 1]) {
      // channel wise and broadcast required;
      permutation_flag = true;
      scale_mode = nvinfer1::ScaleMode::kCHANNEL;
    } else {
      return tensorflow::errors::InvalidArgument(
          "Weight shape not compatible at, " + node_def.name());
    }
  }

  // transpose last dimension
  std::vector<int> permutation(dims_t.nbDims + 1);
  if (permutation_flag) {
    if (scale_mode == nvinfer1::ScaleMode::kCHANNEL && dims_t.nbDims > 1) {
      // we swap the last dimension into channel for trt.
      // because of tensorflow default broadcasting rules.
      for (int i = 0; i < static_cast<int>(permutation.size()); i++) {
        permutation[i] = i;
      }
      permutation[1] = dims_t.nbDims;
      permutation[dims_t.nbDims] = 1;
      tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                   permutation);
    } else {
      return tensorflow::errors::InvalidArgument(
          "Transpose cannot be applied, " + node_def.name());
    }
  }

  if (ctx.isFP16()) {
    weights = ConvertFP32ToFP16(ctx, weights);
  }

  // prepare weights
  TRT_ShapedWeights shift_weights(weights.type_);
  TRT_ShapedWeights scale_weights(weights.type_);
  TRT_ShapedWeights power_weights(weights.type_);

  // Maybe I should do a switch
  if (node_def.op() == "Sub") {
    TRT_ShapedWeights neg_weights = ctx.get_temp_weights_like(weights);
    LambdaFactory unary_op;
    unary_op.op = LambdaFactory::OP_CATEGORY::NEG;
    TF_RETURN_IF_ERROR(UnaryCompute(weights, &neg_weights, unary_op));
    shift_weights = neg_weights;
  } else if (node_def.op() == "Mul") {
    scale_weights = weights;
  } else if (node_def.op() == "Add") {
    shift_weights = weights;
  } else {
    return tensorflow::errors::Unimplemented("Binary op not supported: " +
                                             node_def.op());
  }

  nvinfer1::IScaleLayer* layer = ctx.network()->addScale(
      *const_cast<nvinfer1::ITensor*>(tensor), scale_mode, shift_weights,
      scale_weights, power_weights);

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  // transpose back dimension
  if (permutation_flag) {
    output_tensor = ctx.TransposeTensor(output_tensor, permutation);
  }

  // Pass the output
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

enum class ConvolutionType { DEFAULT, DEPTHWISE_CONV };

tensorflow::Status ConvertConv2DHelper(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs, int group) {
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  TFAttrs attrs(node_def);

  int h_index = 2;
  int w_index = 3;
  auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                 {0, 3, 1, 2});
    h_index = 1;
    w_index = 2;
    // TODO(jie): transpose it
  }

  // tensor after transpose (NCHW)
  auto tensor_dim = tensor->getDimensions();

  int num_groups = group;
  if (num_groups == 0)  // depthwise convolution
    num_groups = tensor_dim.d[0];
  VLOG(2) << "groups count: " << num_groups;

  TRT_ShapedWeights weights_rsck = inputs.at(1).weights();
  if (ctx.isFP16()) {
    weights_rsck = ConvertFP32ToFP16(ctx, inputs.at(1).weights());
  }

  TRT_ShapedWeights weights = ctx.get_temp_weights_like(weights_rsck);
  ReorderRSCKToKCRS(weights_rsck, &weights, num_groups);
  TRT_ShapedWeights biases(weights.type_);
  int noutput = weights.shape_.d[0] * num_groups;
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = weights.shape_.d[2];
  kernel_size.w() = weights.shape_.d[3];
  VLOG(2) << "kernel size: " << kernel_size.h() << ", " << kernel_size.w();

  // TODO(jie): stride. (NHWC/NCHW)
  auto tf_stride = attrs.get<std::vector<int>>("strides");
  VLOG(2) << "h_INDEX" << h_index << ", w_index " << w_index;
  VLOG(2) << "stride!!!: " << tf_stride[0] << tf_stride[1] << tf_stride[2]
          << tf_stride[3];
  nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);

  std::vector<std::pair<int, int>> padding;
  // TODO(jie): padding.
  if (attrs.get<string>("padding") == "SAME") {
    // This is NCHW tensor with no batch dimension.
    //  1 -> h
    //  2 -> w
    padding = CreateSamePadding(
        stride, kernel_size,
        {static_cast<int>(tensor_dim.d[1]), static_cast<int>(tensor_dim.d[2])});
  } else {
    padding = {{0, 0}, {0, 0}};
  }

  if (padding[0].first != padding[0].second ||
      padding[1].first != padding[1].second) {
    // TODO(jie): handle asymmetric padding
    VLOG(2) << "Padding!!!: " << padding[0].first << padding[0].second
            << padding[1].first << padding[1].second;

    auto dim_before = tensor->getDimensions();
    VLOG(2) << "TENSOR before: " << dim_before.d[0] << ", " << dim_before.d[1]
            << dim_before.d[2] << ", " << dim_before.d[3];
    auto pad_layer = ctx.network()->addPadding(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
    auto dim_after = tensor->getDimensions();
    VLOG(2) << "TENSOR after: " << dim_after.d[0] << ", " << dim_after.d[1]
            << dim_after.d[2] << ", " << dim_after.d[3];
  }

  nvinfer1::IConvolutionLayer* layer =
      ctx.network()->addConvolution(*const_cast<nvinfer1::ITensor*>(tensor),
                                    noutput, kernel_size, weights, biases);

  layer->setStride(stride);
  layer->setPadding({padding[0].first, padding[1].first});
  layer->setName(node_def.name().c_str());
  layer->setNbGroups(num_groups);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  auto dim_after = output_tensor->getDimensions();
  VLOG(2) << "TENSOR out: " << dim_after.d[0] << ", " << dim_after.d[1] << ", "
          << dim_after.d[2] << ", " << dim_after.d[3];

  if (data_format == "NHWC") {
    // TODO(jie): transpose it back!
    output_tensor = ctx.TransposeTensor(output_tensor, {0, 2, 3, 1});
  } else {
    VLOG(2) << "NCHW !!!!";
  }
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConv2DHelper(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs, ConvolutionType type) {
  switch (type) {
    case ConvolutionType::DEFAULT:
      return ConvertConv2DHelper(ctx, node_def, inputs, outputs, 1);
    case ConvolutionType::DEPTHWISE_CONV:
      return ConvertConv2DHelper(ctx, node_def, inputs, outputs, 0);
  }
  return tensorflow::errors::Unimplemented("unsupported convolution type at, " +
                                           node_def.name());
}

tensorflow::Status BinaryTensorOpTensor(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const nvinfer1::ITensor* tensor_l, const nvinfer1::ITensor* tensor_r,
    std::vector<TRT_TensorOrWeights>* outputs) {
  static const std::unordered_map<string, nvinfer1::ElementWiseOperation> ops{
      {"Add", nvinfer1::ElementWiseOperation::kSUM},
      {"Mul", nvinfer1::ElementWiseOperation::kPROD},
      {"Sub", nvinfer1::ElementWiseOperation::kSUB},
      {"Div", nvinfer1::ElementWiseOperation::kDIV},
  };

  // FIXME assume type matches input weights
  // get trt type & shape
  TFAttrs attrs(node_def);
  // maybe this part has to be moved into the block of rsqrt later
  nvinfer1::DataType dtype = attrs.get<nvinfer1::DataType>("T");

  // check type consistency
  CHECK_EQ_TYPE(tensor_l->getType(), dtype);
  CHECK_EQ_TYPE(tensor_r->getType(), dtype);
  auto op_pair = ops.find(node_def.op());
  if (op_pair == ops.end())
    return tensorflow::errors::Unimplemented(
        "binary op: " + node_def.op() +
        " not supported at: " + node_def.name());

  nvinfer1::IElementWiseLayer* layer = ctx.network()->addElementWise(
      *const_cast<nvinfer1::ITensor*>(tensor_l),
      *const_cast<nvinfer1::ITensor*>(tensor_r), op_pair->second);

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  // pass the output
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPlugin(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  // prepare input
  std::vector<nvinfer1::ITensor*> all_inputs;
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
  auto attr_key_vector = attrs.GetAllAttrKey();
  for (auto attr_key : attr_key_vector) {
    // TODO(jie): support only list of float for toy example here.
    auto data = attrs.get<std::vector<float>>(attr_key);
    size_t size_data = data.size() * sizeof(float);
    if (!plugin->SetAttribute(attr_key, static_cast<void*>(data.data()),
                              size_data)) {
      return tensorflow::errors::InvalidArgument("plugin SetAttribute failed");
    }
  }

  nvinfer1::IPluginLayer* layer = ctx.network()->addPlugin(
      &all_inputs[0], static_cast<int>(inputs.size()), *plugin);

  for (int i = 0; i < layer->getNbOutputs(); i++) {
    nvinfer1::ITensor* output_tensor = layer->getOutput(i);
    outputs->push_back(TRT_TensorOrWeights(output_tensor));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPlaceholder(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  VLOG(2) << "Placeholder should have been replace already";
  return tensorflow::errors::Unimplemented("cannot convert Placeholder op");
  // OK this make sense since we are supposed to replace it with input
  TFAttrs attrs(node_def);
  nvinfer1::DataType dtype = attrs.get<nvinfer1::DataType>("dtype");
  nvinfer1::Dims dims = attrs.get<nvinfer1::Dims>("shape");

  dims.nbDims--;
  for (int i = 0; i < dims.nbDims; i++) dims.d[i] = dims.d[i + 1];

  nvinfer1::ITensor* output =
      ctx.network()->addInput(node_def.name().c_str(), dtype, dims);
  if (!output) {
    return tensorflow::errors::InvalidArgument("Failed to create Input layer");
  }
  outputs->push_back(TRT_TensorOrWeights(output));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConv2D(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  return ConvertConv2DHelper(ctx, node_def, inputs, outputs,
                             ConvolutionType::DEFAULT);
}

tensorflow::Status ConvertConv2DDepthwise(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  return ConvertConv2DHelper(ctx, node_def, inputs, outputs,
                             ConvolutionType::DEPTHWISE_CONV);
}

tensorflow::Status ConvertPool(Converter& ctx,
                               const tensorflow::NodeDef& node_def,
                               const std::vector<TRT_TensorOrWeights>& inputs,
                               std::vector<TRT_TensorOrWeights>* outputs) {
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  TFAttrs attrs(node_def);

  int h_index = 2;
  int w_index = 3;
  auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    h_index = 1;
    w_index = 2;
    tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                 {0, 3, 1, 2});
  } else {
    VLOG(2) << "NCHW !!!!";
  }
  nvinfer1::PoolingType type;
  // TODO(jie): support other pooling type
  if (node_def.op() == "MaxPool")
    type = nvinfer1::PoolingType::kMAX;
  else if (node_def.op() == "AvgPool")
    type = nvinfer1::PoolingType::kAVERAGE;
  else
    return tensorflow::errors::Unimplemented("Only supports Max pool");

  // TODO(jie): NCHW
  auto tf_stride = attrs.get<std::vector<int>>("strides");
  nvinfer1::DimsHW stride(tf_stride[h_index], tf_stride[w_index]);

  auto tf_kernel = attrs.get<std::vector<int>>("ksize");
  nvinfer1::DimsHW ksize(tf_kernel[h_index], tf_kernel[w_index]);

  auto tensor_dim = tensor->getDimensions();
  std::vector<std::pair<int, int>> padding;
  // TODO(jie): padding.
  if (attrs.get<string>("padding") == "SAME") {
    // This is NCHW tensor with no batch dimension.
    //  1 -> h
    //  2 -> w
    padding = CreateSamePadding(
        stride, ksize,
        {static_cast<int>(tensor_dim.d[1]), static_cast<int>(tensor_dim.d[2])});
  } else if (attrs.get<string>("padding") == "VALID") {
    // No padding for valid padding here
    VLOG(2) << "No padding added for VALID padding in pool" << node_def.name();
    padding = {{0, 0}, {0, 0}};
  } else {
    return tensorflow::errors::Unimplemented(
        "Current MaxPool cannot support padding other than SAME");
  }

  if (padding[0].first != padding[0].second ||
      padding[1].first != padding[1].second) {
    // TODO(jie): handle asymmetric padding
    VLOG(2) << "Padding!!!: " << padding[0].first << padding[0].second
            << padding[1].first << padding[1].second;
    auto pad_layer = ctx.network()->addPadding(
        *const_cast<nvinfer1::ITensor*>(tensor),
        nvinfer1::DimsHW(padding[0].first, padding[1].first),
        nvinfer1::DimsHW(padding[0].second, padding[1].second));
    padding = {{0, 0}, {0, 0}};
    tensor = pad_layer->getOutput(0);
  }

  nvinfer1::IPoolingLayer* layer = ctx.network()->addPooling(
      *const_cast<nvinfer1::ITensor*>(tensor), type, ksize);

  layer->setStride(stride);
  layer->setPadding({padding[0].first, padding[1].first});
  layer->setName(node_def.name().c_str());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (data_format == "NHWC") {
    // TODO(jie): transpose it back!
    output_tensor = ctx.TransposeTensor(output_tensor, {0, 2, 3, 1});
  } else {
    VLOG(2) << "NCHW !!!!";
  }
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertActivation(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  nvinfer1::IActivationLayer* layer = ctx.network()->addActivation(
      *const_cast<nvinfer1::ITensor*>(tensor), nvinfer1::ActivationType::kRELU);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertScale(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TRT_TensorOrWeights>& inputs,
                                std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights())
    return tensorflow::errors::Unimplemented(
        "Only supports tensor op weight for now, at " + node_def.name());
  // Implement tensor binaryOp weight [channel wise] for now;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  TRT_ShapedWeights weights = inputs.at(1).weights();
  if (ctx.isFP16()) {
    weights = ConvertFP32ToFP16(ctx, inputs.at(1).weights());
  }

  TRT_ShapedWeights empty_weights(weights.type_);

  TFAttrs attrs(node_def);

  // Transpose NHWC
  auto data_format = attrs.get<string>("data_format");
  if (data_format == "NHWC") {
    tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                 {0, 3, 1, 2});
    // TODO(jie): transpose it
  } else {
    VLOG(2) << "NCHW !!!!";
  }

  auto dims = tensor->getDimensions();
  VLOG(2) << "tensor dimensions: " << dims.nbDims;
  for (int i = 0; i < dims.nbDims; i++) {
    VLOG(2) << "i: " << dims.d[i];
  }
  dims = weights.shape_;
  VLOG(2) << "tensor dimensions: " << dims.nbDims;
  for (int i = 0; i < dims.nbDims; i++) {
    VLOG(2) << "i: " << dims.d[i];
  }

  nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
  if (weights.shape_.d[0] == 1) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  }

  nvinfer1::IScaleLayer* layer =
      ctx.network()->addScale(*const_cast<nvinfer1::ITensor*>(tensor), mode,
                              weights, empty_weights, empty_weights);

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  if (data_format == "NHWC") {
    // TODO(jie): transpose it back!
    output_tensor = ctx.TransposeTensor(output_tensor, {0, 2, 3, 1});
  } else {
    VLOG(2) << "NCHW !!!!";
  }
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConst(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TRT_TensorOrWeights>& inputs,
                                std::vector<TRT_TensorOrWeights>* outputs) {
  const auto& weights_tensor = node_def.attr().at("value").tensor();

  // Get trt type & shape
  TFAttrs attrs(node_def);
  const tensorflow::DataType dtype = attrs.get<tensorflow::DataType>("dtype");

  // Create shaped weights as output
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(weights_tensor))
    return tensorflow::errors::Internal("Cannot parse weight tensor proto: " +
                                        node_def.name());

  TRT_ShapedWeights weights(dtype);
  if (!weights_tensor.float_val().empty()) {
    VLOG(2) << "SCALAR!!!" << node_def.name();
    nvinfer1::Dims scalar_shape;
    if (tensor.dims() > 0) {
      VLOG(2) << "dimensions: " << tensor.dims();
      VLOG(2) << "size: " << weights_tensor.float_val_size();
      scalar_shape = GetTensorShape(tensor);
      for (int i = 0; i < scalar_shape.nbDims; i++)
        VLOG(2) << scalar_shape.d[i];
      if (GetShapeSize(scalar_shape) != weights_tensor.float_val_size()) {
        if (weights_tensor.float_val_size() == 1 ||
            scalar_shape.d[0] == weights_tensor.float_val_size()) {
          scalar_shape.nbDims = 1;
          // no dimension provided. flatten it
          scalar_shape.d[0] = weights_tensor.float_val_size();
          scalar_shape.type[0] = nvinfer1::DimensionType::kSPATIAL;
        } else {
          LOG(WARNING) << "Broadcast on weights only supports kCHANNEL and"
                       << " kUNIFORM, at: " << node_def.name();
          string err_str("Broadcast method is not supported for '");
          StrAppend(&err_str, node_def.name(), "' of type ", node_def.op());
          return tensorflow::errors::InvalidArgument(err_str);
        }
      }
    } else {
      VLOG(2) << "Dimensions: " << tensor.dims();
      scalar_shape.nbDims = 1;
      // no dimension provided. flatten it
      scalar_shape.d[0] = weights_tensor.float_val_size();
      scalar_shape.type[0] = nvinfer1::DimensionType::kSPATIAL;
      for (int i = 1; i < nvinfer1::Dims::MAX_DIMS; i++) {
        scalar_shape.d[i] = 0;
        scalar_shape.type[i] = nvinfer1::DimensionType::kSPATIAL;
      }
    }
    size_t len_data = tensorflow::DataTypeSize(dtype);
    for (int i = 0; i < scalar_shape.nbDims; i++) len_data *= scalar_shape.d[i];
    ctx.weight_store()->store_.push_back(std::vector<uint8_t>(len_data));
    void* dst = static_cast<void*>(&(ctx.weight_store()->store_.back()[0]));
    std::vector<float> tensor_data(
        weights_tensor.float_val().begin(),
        weights_tensor.float_val()
            .end());  //  make a local copy first to flatten
    memcpy(dst, tensor_data.data(), len_data);  // store into weight store
    weights = TRT_ShapedWeights(dtype, dst, scalar_shape);
  } else if (!weights_tensor.int_val().empty()) {
    VLOG(2) << "int!!!" << node_def.name();
    nvinfer1::Dims scalar_shape;
    if (tensor.dims() > 0) {
      VLOG(2) << "dimensions: " << tensor.dims();
      scalar_shape = GetTensorShape(tensor);
      if (GetShapeSize(scalar_shape) != weights_tensor.int_val_size()) {
        if (weights_tensor.int_val_size() == 1 ||
            scalar_shape.d[0] == weights_tensor.int_val_size()) {
          scalar_shape.nbDims = 1;
          // no dimension provided. flatten it
          scalar_shape.d[0] = weights_tensor.int_val_size();
          scalar_shape.type[0] = nvinfer1::DimensionType::kSPATIAL;
        } else {
          LOG(WARNING) << "Broadcast on weights only supports kCHANNEL and"
                       << " kUNIFORM, at: " << node_def.name();
          string err_str("Broadcast method is not supported for '");
          StrAppend(&err_str, node_def.name(), "' of type ", node_def.op());
          return tensorflow::errors::InvalidArgument(err_str);
        }
      }
    } else {
      VLOG(2) << "dimensions: " << tensor.dims();
      scalar_shape.nbDims = 1;
      // no dimension provided. flatten it
      scalar_shape.d[0] = weights_tensor.int_val_size();
      scalar_shape.type[0] = nvinfer1::DimensionType::kSPATIAL;
      for (int i = 1; i < nvinfer1::Dims::MAX_DIMS; i++) {
        scalar_shape.d[i] = 0;
        scalar_shape.type[i] = nvinfer1::DimensionType::kSPATIAL;
      }
    }
    //  we should not have converted //if (ctx.isFP16()) {
    size_t len_data = tensorflow::DataTypeSize(dtype);
    for (int i = 0; i < scalar_shape.nbDims; i++) len_data *= scalar_shape.d[i];
    size_t len_tensor = weights_tensor.int_val_size() * sizeof(int32);
    len_data = std::max(len_data, len_tensor);
    ctx.weight_store()->store_.push_back(std::vector<uint8_t>(len_data));
    void* dst = static_cast<void*>(&(ctx.weight_store()->store_.back()[0]));
    std::vector<int32> tensor_data(
        weights_tensor.int_val().begin(),
        weights_tensor.int_val().end());  //  make a local copy first to flatten
                                          //  doesn't have to be contigous
    memcpy(dst, tensor_data.data(), len_tensor);  // store into weight store
    weights = TRT_ShapedWeights(dtype, dst, scalar_shape);
  } else if (!weights_tensor.tensor_content().empty()) {
    //  obsolete method.
    //  After optimization path, we do not see weights in this format.
    //  fp16 conversion technically should be needed here.
    VLOG(2) << "TENSOR!!!" << node_def.name();
    const auto& content = weights_tensor.tensor_content();

    weights = ctx.get_temp_weights(dtype, GetTensorShape(tensor));
    if (content.size() > 0) {
      const int dtype_size = tensorflow::DataTypeSize(dtype);
      CHECK_EQ(0, content.size() % dtype_size)
          << "Tensor content size (" << content.size()
          << ") is not a multiple of " << dtype_size;
      port::CopyToArray(
          content, static_cast<char*>(const_cast<void*>(weights.GetValues())));
    }
  } else {
    return tensorflow::errors::Unimplemented(
        "Not supported constant type, at " + node_def.name());
  }
  // Pass the output
  outputs->push_back(TRT_TensorOrWeights(weights));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertIdentity(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  outputs->push_back(inputs.at(0));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBinary(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 2)
    return tensorflow::errors::FailedPrecondition(
        "Binary ops require two tensor input, at " + node_def.name());

  if (inputs.at(0).is_weights() && inputs.at(1).is_weights())
    return ConstantFoldBinary(ctx, node_def, inputs, outputs);

  if (inputs.at(0).is_tensor() && inputs.at(1).is_weights())
    return BinaryTensorOpWeight(ctx, node_def, inputs.at(0).tensor(),
                                inputs.at(1).weights(), outputs);

  if (inputs.at(0).is_weights() && inputs.at(1).is_tensor())
    return BinaryTensorOpWeight(ctx, node_def, inputs.at(1).tensor(),
                                inputs.at(0).weights(), outputs);

  if (inputs.at(0).is_tensor() && inputs.at(1).is_tensor())
    return BinaryTensorOpTensor(ctx, node_def, inputs.at(0).tensor(),
                                inputs.at(1).tensor(), outputs);

  return tensorflow::errors::Unknown("Binary op input error, at " +
                                     node_def.name());
}

tensorflow::Status ConvertUnary(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TRT_TensorOrWeights>& inputs,
                                std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 1)
    return tensorflow::errors::FailedPrecondition(
        "Unary ops require single tensor input, at " + node_def.name());

  if (inputs.at(0).is_weights())
    return ConstantFoldUnary(ctx, node_def, inputs, outputs);
  else if (inputs.at(0).is_tensor())
    return tensorflow::errors::Unimplemented(
        "Unary op for tensor not supported, at " + node_def.name());

  return tensorflow::errors::Unknown("Binary op input error, at " +
                                     node_def.name());
}

tensorflow::Status ConvertReduce(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights())
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at" + node_def.name());

  // Implement tensor binaryOp weight [channel wise] for now;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  auto dims = tensor->getDimensions();
  // Restore implicit batch dimension
  int nb_dims = dims.nbDims + 1;

  TRT_ShapedWeights index_list = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  // TODO(jie): handle data type.
  // Index type here is done through TF type, so I can leverage their
  // EnumToDataType for my cast
  auto index_type = attrs.get<tensorflow::DataType>("Tidx");

  // Only expect to handle INT32 as attributes for now
  if (index_type != tensorflow::DataType::DT_INT32)
    return tensorflow::errors::Unimplemented("Tidx supports only DT_INT32");
  auto index_list_data =
      static_cast<int*>(const_cast<void*>(index_list.GetValues()));

  // Hack warning: have to fall back to pool layer since reduce is not in public
  // TRT yet.
  if (nb_dims != 4)
    return tensorflow::errors::InvalidArgument(
        "TRT only support reduce on 4 dimensional tensors, at" +
        node_def.name());
  if (index_list.count() > 2)
    return tensorflow::errors::InvalidArgument(
        "TRT cannot support reduce on more than 2 dimensions, at" +
        node_def.name());

  std::set<int> idx_set;
  // We cannot operate on Channel. permutation flag used to transpose tensor
  int permuted_index = -1;
  for (int i = 0; i < index_list.count(); i++) {
    if (index_list_data[i] == 0)
      return tensorflow::errors::InvalidArgument("TRT cannot reduce at 0, at" +
                                                 node_def.name());
    if (index_list_data[i] == 1) permuted_index = 1;

    idx_set.emplace(index_list_data[i]);
  }

  std::vector<int> permutation_order(nb_dims);
  nvinfer1::DimsHW pool_kernel;
  if (permuted_index == 1) {
    for (int i = 2; i < nb_dims; i++) {
      if (idx_set.count(i) == 0) {
        permuted_index = i;
        break;
      }
    }
    for (int i = 0; i < nb_dims; i++) permutation_order[i] = i;

    permutation_order[permuted_index] = 1;
    permutation_order[1] = permuted_index;

    // Apply permutation before extracting dimension for pool_kernel
    tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                 permutation_order);
  }

  // Apply permutation before extracting dimension for pool_kernel
  pool_kernel.d[0] = (idx_set.count(2) || permuted_index == 2) ? dims.d[1] : 1;
  pool_kernel.d[1] = (idx_set.count(3) || permuted_index == 3) ? dims.d[2] : 1;

  nvinfer1::ITensor* output_tensor;

  if (node_def.op() == "Mean") {
    nvinfer1::IPoolingLayer* layer =
        ctx.network()->addPooling(*const_cast<nvinfer1::ITensor*>(tensor),
                                  nvinfer1::PoolingType::kAVERAGE, pool_kernel);
    output_tensor = layer->getOutput(0);
  } else {
    return tensorflow::errors::Unimplemented(
        "Op not supported " + node_def.op() + " , at " + node_def.name());
  }
  if (permuted_index != -1) {
    // Apply permutation before extracting dimension for pool_kernel
    output_tensor = ctx.TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), permutation_order);
  }
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPad(Converter& ctx,
                              const tensorflow::NodeDef& node_def,
                              const std::vector<TRT_TensorOrWeights>& inputs,
                              std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights())
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at" + node_def.name());

  // Implement tensor binaryOp weight [channel wise] for now;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  auto dims = tensor->getDimensions();
  // Restore implicit batch dimension
  int nb_dims = dims.nbDims + 1;

  TRT_ShapedWeights pads = inputs.at(1).weights();

  TFAttrs attrs(node_def);
  // Padding type here is done through TF type
  //   so I can leverage their EnumToDataType for my cast
  auto padding_type = attrs.get<tensorflow::DataType>("Tpaddings");
  // TODO(jie): handle data type conversion for TRT?

  if (pads.shape_.d[0] != nb_dims || pads.shape_.d[1] != 2)
    return tensorflow::errors::InvalidArgument(
        "Pad only supports explicit padding on 4 dimensional tensor, at " +
        node_def.name());

  // Only expect to handle INT32 as attributes for now
  if (padding_type != tensorflow::DataType::DT_INT32)
    return tensorflow::errors::Unimplemented(
        "Tpaddings supports only DT_INT32");
  auto pad_data = static_cast<int*>(const_cast<void*>(pads.GetValues()));

  std::vector<int32_t> pad_index;
  for (int i = 0; i < nb_dims; i++) {
    if (pad_data[2 * i] != 0 || pad_data[2 * i + 1] != 0)
      pad_index.push_back(i);
  }

  // No padding at all, we should exit
  if (pad_index.size() == 0) {
    outputs->push_back(inputs.at(0));
    return tensorflow::Status::OK();
  }

  // Only supports padding on less than 2 axis GIE-2579
  if (pad_index.size() > 2)
    return tensorflow::errors::InvalidArgument(
        "Padding layer does not support padding on > 2");

  // Padding on batch dimension is not supported
  if (pad_index[0] == 0)
    return tensorflow::errors::InvalidArgument(
        "Padding layer does not support padding on batch dimension");

  // Not doing the legit thing here. ignoring padding on dim 1 and 3;
  // TODO(jie): implement pad as uff parser
  if (pad_index.size() == 2 && pad_index[0] == 0 && pad_index[1] == 3)
    return tensorflow::errors::Unimplemented(
        "Padding layer does not support padding on dimension 1 and 3 yet");

  bool legit_pad = true;
  nvinfer1::DimsHW pre_padding(0, 0);
  nvinfer1::DimsHW post_padding(0, 0);

  std::vector<int32_t> permuted_pad_index(pad_index);
  if (pad_index[0] == 1) {
    legit_pad = false;
    tensor = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor),
                                 {0, 3, 2, 1});
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

  nvinfer1::IPaddingLayer* layer = ctx.network()->addPadding(
      *const_cast<nvinfer1::ITensor*>(tensor), pre_padding, post_padding);
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (!legit_pad)
    output_tensor = ctx.TransposeTensor(
        const_cast<nvinfer1::ITensor*>(output_tensor), {0, 3, 2, 1});

  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConcat(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  // not including the last input (axis) here
  int input_size = static_cast<int>(inputs.size()) - 1;

  if (!inputs.at(0).is_tensor())
    return tensorflow::errors::InvalidArgument(
        "Concat in TRT support only Tensor input, at " + node_def.name());

  // We are retrieving the axis
  TRT_ShapedWeights axis = inputs.at(input_size).weights();

  TFAttrs attrs(node_def);
  auto index_type = attrs.get<tensorflow::DataType>("Tidx");

  // TODO(jie): handle data type
  // Only expect to handle INT32 as index attributes for now
  if (index_type != tensorflow::DataType::DT_INT32)
    return tensorflow::errors::Unimplemented(
        "Tidx supports only DT_INT32, at " + node_def.name());

  int index = *(static_cast<int*>(const_cast<void*>(axis.GetValues())));

  // TODO(jie): early termination with no-op (attr_size==1)

  auto dim = inputs.at(0).tensor()->getDimensions();
  // dimension check
  if (index > dim.nbDims + 1)
    return tensorflow::errors::InvalidArgument(
        "Concatenate on axis out of dimension range, at " + node_def.name());

  if (index == 0)
    return tensorflow::errors::InvalidArgument(
        "Concatenate on batch dimension not supported, at " + node_def.name());

  // incase we need permutation;
  std::vector<int> permutation_order(dim.nbDims + 1);

  for (int i = 0; i < dim.nbDims + 1; i++) permutation_order[i] = i;

  if (index != 1) {
    permutation_order[1] = index - 1;
    permutation_order[index - 1] = 1;
  }

  std::vector<nvinfer1::ITensor const*> inputs_vec;
  // Shap chack (all input tensor should have same shape)
  // starting from 0 since we are probably also doing transpose here;
  for (int i = 0; i < input_size; i++) {
    auto tensor_i = inputs.at(i).tensor();
    auto dim_i = tensor_i->getDimensions();
    if (dim_i.nbDims != dim.nbDims)
      return tensorflow::errors::InvalidArgument(
          "Concatenate receives inputs with inconsistent dimensions, at " +
          node_def.name());

    for (int j = 0; j < dim.nbDims; j++) {
      // check dimension consistency on non-concatenate axis
      if (j != index - 1 && dim_i.d[j] != dim.d[j])
        return tensorflow::errors::InvalidArgument(
            "Concatenate receives inputs with inconsistent shape, at" +
            node_def.name());
    }

    // TRT does concatenation only on channel!
    if (index != 1)
      tensor_i = ctx.TransposeTensor(const_cast<nvinfer1::ITensor*>(tensor_i),
                                     permutation_order);

    inputs_vec.push_back(tensor_i);
  }

  // nvinfer1::ITensor const* tensor = inputs.at(0).tensor();
  nvinfer1::IConcatenationLayer* layer = ctx.network()->addConcatenation(
      const_cast<nvinfer1::ITensor* const*>(inputs_vec.data()),
      inputs_vec.size());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);

  if (index != 1) {
    output_tensor = ctx.TransposeTensor(output_tensor, permutation_order);
  }
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFusedBatchNorm(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  TFAttrs attrs(node_def);
  float epsilon = attrs.get<float>("epsilon");
  auto data_format = attrs.get<string>("data_format");
  if (data_format != "NCHW") {
    return tensorflow::errors::Unimplemented(
        "only data_format=NCHW is supported, at " + node_def.name());
  }
  bool is_training = attrs.get<bool>("is_training");
  if (is_training) {
    return tensorflow::errors::Unimplemented(
        "only is_training=false is supported, at " + node_def.name());
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
          "Inconsistent parameter type for batchnormis not supported, at: " +
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
  //  We could technically have two weights with different shape.
  //  that requires two addScale op, arguably less performant
  TRT_ShapedWeights combined_scale_weights =
      ctx.get_temp_weights_like(*ptr_shape_weights);
  TRT_ShapedWeights combined_offset_weights =
      ctx.get_temp_weights_like(*ptr_shape_weights);

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
  nvinfer1::IScaleLayer* layer =
      ctx.network()->addScale(*const_cast<nvinfer1::ITensor*>(tensor), mode,
                              combined_offset_weights.GetWeightsForTRT(),
                              combined_scale_weights.GetWeightsForTRT(),
                              dummy_power_weights.GetWeightsForTRT());
  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertMatMul(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TRT_TensorOrWeights>& inputs,
                                 std::vector<TRT_TensorOrWeights>* outputs) {
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();

  // TODO(jie): transpose!
  TFAttrs attrs(node_def);

  TRT_ShapedWeights weights_ck = inputs.at(1).weights();
  TRT_ShapedWeights weights = ctx.get_temp_weights_like(weights_ck);
  ReorderCKtoKC(weights_ck, &weights);
  TRT_ShapedWeights biases(weights.type_);

  int noutput = weights.shape_.d[0];

  nvinfer1::IFullyConnectedLayer* layer = ctx.network()->addFullyConnected(
      *const_cast<nvinfer1::ITensor*>(tensor), noutput, weights, biases);

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertReshape(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TRT_TensorOrWeights>& inputs,
    std::vector<TRT_TensorOrWeights>* outputs) {
  if (inputs.size() != 2 || !inputs.at(0).is_tensor() ||
      !inputs.at(1).is_weights())
    return tensorflow::errors::InvalidArgument(
        "Input expects tensor and weights, at" + node_def.name());

  // implement tensor binaryOp weight [channel wise] for now;
  const nvinfer1::ITensor* tensor = inputs.at(0).tensor();
  auto dims = tensor->getDimensions();
  // restore implicit batch dimension

  TRT_ShapedWeights shape = inputs.at(1).weights();

  TFAttrs attrs(node_def);

  auto padding_type = attrs.get<tensorflow::DataType>("Tshape");

  if (shape.shape_.nbDims != 1)
    return tensorflow::errors::InvalidArgument(
        "reshape new shape is not 1 dimensional, at " + node_def.name());

  // Only expect to handle INT32 as attributes for now
  if (padding_type != tensorflow::DataType::DT_INT32)
    return tensorflow::errors::Unimplemented(
        "reshape new shape supports only DT_INT32, at " + node_def.name());

  auto shape_data = static_cast<int*>(const_cast<void*>(shape.GetValues()));

  if (shape_data[0] != -1)
    return tensorflow::errors::InvalidArgument(
        "reshape new shape first dimension is not -1, at " + node_def.name());

  auto shape_num_dims = shape.shape_.d[0];
  VLOG(2) << "shape dimensions: " << shape_num_dims;
  int volume_w = 1;
  for (int i = 1; i < shape.shape_.d[0]; i++) volume_w *= shape_data[i];

  int volume_t = 1;
  for (int i = 0; i < dims.nbDims; i++) volume_t *= dims.d[i];

  VLOG(2) << "volume: " << volume_t << " volume weights: " << volume_w;
  if (volume_w != volume_t)
    return tensorflow::errors::InvalidArgument(
        "volume does not agree between tensor and new shape, at " +
        node_def.name());

  nvinfer1::IShuffleLayer* layer =
      ctx.network()->addShuffle(*const_cast<nvinfer1::ITensor*>(tensor));

  nvinfer1::Dims reshape_dims;
  VLOG(2) << "new dimension: " << shape_num_dims - 1;
  reshape_dims.nbDims = shape_num_dims - 1;
  for (int32_t i = 0; i < reshape_dims.nbDims; ++i) {
    reshape_dims.d[i] = shape_data[i + 1];
  }
  layer->setReshapeDimensions(reshape_dims);
  VLOG(2) << "new dimension: " << shape_num_dims - 1;

  nvinfer1::ITensor* output_tensor = layer->getOutput(0);
  auto dims_output = output_tensor->getDimensions();
  VLOG(2) << "output tensor dimension:" << dims_output.nbDims;
  outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return tensorflow::Status::OK();
}

void Converter::register_op_converters() {
  // vgg_16 slim implementation
  op_registry_["Placeholder"] = ConvertPlaceholder;
  op_registry_["Conv2D"] = ConvertConv2D;
  op_registry_["DepthwiseConv2dNative"] = ConvertConv2DDepthwise;
  op_registry_["Relu"] = ConvertActivation;
  op_registry_["MaxPool"] = ConvertPool;
  op_registry_["AvgPool"] = ConvertPool;
  // This could be really handled as ConvertBinary
  op_registry_["BiasAdd"] = ConvertScale;
  op_registry_["Const"] = ConvertConst;
  // TODO(ben,jie): this is a temp hack.
  op_registry_["Identity"] = ConvertIdentity;  // Identity should be removed
  op_registry_["Snapshot"] = ConvertIdentity;  // Snapshot should be removed

  // resnet_50_v1 slim implementation
  op_registry_["Add"] = ConvertBinary;
  op_registry_["Mul"] = ConvertBinary;
  op_registry_["Sub"] = ConvertBinary;
  op_registry_["Rsqrt"] = ConvertUnary;
  op_registry_["Mean"] = ConvertReduce;
  op_registry_["Pad"] = ConvertPad;
  // TODO(ben,jie): Add more ops

  op_registry_["ConcatV2"] = ConvertConcat;
  op_registry_["MatMul"] = ConvertMatMul;
  op_registry_["Reshape"] = ConvertReshape;
  op_registry_["FusedBatchNorm"] = ConvertFusedBatchNorm;
  op_registry_["FusedBatchNormV2"] = ConvertFusedBatchNorm;

  plugin_converter_ = ConvertPlugin;
}

}  // namespace

tensorflow::Status ConvertSubgraphToEngine(
    const tensorflow::GraphDef& gdef, nvinfer1::IBuilder* builder,
    const std::vector<tensorflow::PartialTensorShape>& input_shapes,
    nvinfer1::ICudaEngine** engine, int precision_mode) {
  auto trt_network = infer_object(builder->createNetwork());
  if (!trt_network) {
    return tensorflow::errors::Internal(
        "Failed to create TensorRT network object");
  }
  auto ws = std::unique_ptr<tensorflow::tensorrt::TRTWeightStore>(
      new TRTWeightStore());
  // Build the network
  VLOG(1) << "Starting engine conversion ";
  Converter converter(trt_network.get(), ws.get(), precision_mode == FP16MODE);
  std::vector<std::pair<string, string>> output_tensors;
  for (const auto& node_def : gdef.node()) {
    string node_name = node_def.name();
    VLOG(1) << "Converting op name=" << node_name << ", op=" << node_def.op();
    if (tensorflow::str_util::StartsWith(node_name, kInputPHName) &&
        (node_def.op() == "Placeholder")) {
      nvinfer1::DimsCHW input_dim_pseudo_chw;
      for (int i = 0; i < 8; i++) input_dim_pseudo_chw.d[i] = 0;
      nvinfer1::DataType dtype(nvinfer1::DataType::kFLOAT);
      auto type_status =
          ConvertDType(node_def.attr().at("dtype").type(), &dtype);
      if (type_status != tensorflow::Status::OK()) {
        LOG(WARNING) << "Type conversion failed for " << node_name;
        return type_status;
      }
      int32 slot_number = -1;
      if (!tensorflow::strings::safe_strto32(node_name.c_str() + 8,
                                             &slot_number)) {
        LOG(ERROR) << "Failed to parse slot number from " << node_name
                   << " +8= " << node_name.c_str() + 8;
      }
      auto shape = input_shapes.at(slot_number);
      if (shape.dims() > 8) {
        LOG(ERROR) << "Tensor rank is greater than 8 for " << node_name
                   << " at input slot " << slot_number;
        return tensorflow::errors::OutOfRange(
            "Input tensor rank is greater than 8");
      }
      if (VLOG_IS_ON(1)) {
        string dim_str("dims=");
        StrAppend(&dim_str, "[ ", shape.dim_size(0));
        for (int i = 1; i < shape.dims(); i++) {
          StrAppend(&dim_str, ", ", shape.dim_size(i));
        }
        StrAppend(&dim_str, " ]");
        VLOG(1) << dim_str;
      }
      for (int i = 1; i < shape.dims(); i++) {
        input_dim_pseudo_chw.d[i - 1] = shape.dim_size(i);
      }

      input_dim_pseudo_chw.nbDims = shape.dims() - 1;
      nvinfer1::ITensor* input_tensor = converter.network()->addInput(
          node_name.c_str(), dtype, input_dim_pseudo_chw);
      if (!input_tensor) {
        return tensorflow::errors::InvalidArgument(
            StrCat("Failed to create Input layer tensor ", node_name,
                   " rank=", shape.dims() - 1));
      }
      VLOG(1) << "Input tensor name :" << node_name;
      if (!converter.insert_input_tensor(node_name, input_tensor)) {
        return tensorflow::errors::AlreadyExists(
            "Output tensor already exists for op: " + node_name);
      }
    } else if (tensorflow::str_util::StartsWith(node_name, kOutputPHName) &&
               (node_def.op() == "Identity")) {
      tensorflow::int32 slot_number = -1;
      if (!tensorflow::strings::safe_strto32(node_name.c_str() + 9,
                                             &slot_number)) {
        LOG(ERROR) << "Failed to parse slot number from " << node_name
                   << " +9=" << node_name.c_str() + 9;
      }
      if (output_tensors.size() <= slot_number) {
        output_tensors.resize(slot_number + 1);
      }
      output_tensors.at(slot_number) = {node_def.input(0), node_name};
    } else {
      VLOG(2) << "Converting node: " << node_def.name() << " , "
              << node_def.op();
      TF_RETURN_IF_ERROR(converter.convert_node(node_def));
    }
  }
  for (const auto& output : output_tensors) {
    auto tensor_or_weights = converter.get_tensor(output.first);
    if (!tensor_or_weights.is_tensor()) {
      return tensorflow::errors::InvalidArgument(
          "Output node '" + output.first + "' is weights not tensor");
    }
    nvinfer1::ITensor* tensor = tensor_or_weights.tensor();
    tensor->setName(output.second.c_str());
    if (!tensor) {
      return tensorflow::errors::NotFound("Output tensor not found: " +
                                          output.first);
    }
    VLOG(1) << "Marking output tensor " << output.first << ", as output tensor "
            << output.second;

    converter.network()->markOutput(*tensor);
  }
  VLOG(1) << "Starting engine creation";
  *engine = builder->buildCudaEngine(*converter.network());
  VLOG(1) << "Finished conversion";
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSegmentToGraphDef(
    const tensorflow::Graph* graph,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::vector<int>& subgraph_node_ids,
    std::vector<EngineConnections>* connections,
    tensorflow::GraphDef* segment_def, string* common_scope) {
  std::set<string> marker_nodes;
  for (size_t i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    auto outside_node = graph->FindNodeId(connection.outside_id);
    if (outside_node) {
      tensorflow::DataType input_type = tensorflow::DT_FLOAT;
      tensorflow::PartialTensorShape partial_shape;
      if (connection.is_input_edge) {
        if (graph_properties.HasOutputProperties(
                connection.outside_node_name)) {
          auto output_params = graph_properties.GetOutputProperties(
              connection.outside_node_name);
          auto out_shape = output_params.at(connection.outside_port);
          input_type = out_shape.dtype();
          std::vector<tensorflow::int64> dims;
          partial_shape = out_shape.shape();
          connection.outside_shape = partial_shape;
        } else {
          VLOG(0) << "Unknown output shape" << outside_node->name();
          input_type = graph->FindNodeId(connection.outside_id)
                           ->output_type(connection.outside_port);
        }
        connection.connection_type = input_type;

      } else {  // output edge
        if (graph_properties.HasInputProperties(connection.outside_node_name)) {
          auto input_params =
              graph_properties.GetInputProperties(connection.outside_node_name);
          auto in_shape = input_params.at(connection.outside_port);
          input_type = in_shape.dtype();
          partial_shape = in_shape.shape();
          connection.inside_shape = partial_shape;
        } else {
          input_type = graph->FindNodeId(connection.inside_id)
                           ->output_type(connection.outside_port);
        }
        connection.connection_type = input_type;
      }

      tensorflow::NodeDef dummy_placeholder;
      string node_name;
      if (connection.is_input_edge) {
        StrAppend(&node_name, kInputPHName, connection.port_number);
        if (marker_nodes.count(node_name)) {
          VLOG(1) << "Reusing input " << node_name << " for the edge "
                  << connection.outside_node_name << ":"
                  << connection.outside_port << " -> "
                  << connection.inside_node_name << ":"
                  << connection.inside_port;
          continue;
        }
        marker_nodes.insert(node_name);
        auto seg_node = segment_def->add_node();
        tensorflow::NodeDefBuilder dph_builder(node_name, "Placeholder");
        auto status = dph_builder.Attr("shape", partial_shape)
                          .Attr("dtype", input_type)
                          .Finalize(seg_node);
        VLOG(1) << "Constructing input " << node_name << " for the edge "
                << connection.outside_node_name << ":"
                << connection.outside_port << " -> "
                << connection.inside_node_name << ":" << connection.inside_port;
      } else {
        StrAppend(&node_name, kOutputPHName, connection.port_number);
        if (marker_nodes.count(node_name)) {
          VLOG(1) << "Reusing output " << node_name << " for the edge "
                  << connection.inside_node_name << ":"
                  << connection.inside_port << " -> "
                  << connection.outside_node_name << ":"
                  << connection.outside_port;
          continue;
        }
        marker_nodes.insert(node_name);
        auto seg_node = segment_def->add_node();
        tensorflow::NodeDefBuilder dph_builder(node_name, "Identity");
        auto status =
            dph_builder.Input(connection.inside_node_name, 0, input_type)
                .Finalize(seg_node);
        VLOG(1) << "Constructing output " << node_name << " for the edge "
                << connection.inside_node_name << ":" << connection.inside_port
                << " -> " << connection.outside_node_name << ":"
                << connection.outside_port;
      }
    }
  }
  std::unordered_map<int, int> newIdMap;
  // Copy nodes to new graphdef
  string local_scope = graph->FindNodeId(*subgraph_node_ids.begin())->name();
  for (const auto node_id : subgraph_node_ids) {
    const auto node = graph->FindNodeId(node_id);
    local_scope = GetCommonNameScope(local_scope, node->name());
    if (node) {
      newIdMap[node_id] = segment_def->node_size();
      auto snode = segment_def->add_node();
      snode->CopyFrom(node->def());
      VLOG(1) << "Copying " << snode->name() << " to subgraph";
    }
  }
  // update the inputs of the new nodes to point to dummy inputs
  for (int i = 0; i < connections->size(); ++i) {
    auto& connection = connections->at(i);
    if (!connection.is_input_edge) continue;
    auto snode = segment_def->mutable_node(newIdMap[connection.inside_id]);
    string placeholder_name(kInputPHName);
    StrAppend(&placeholder_name, connection.port_number);
    VLOG(1) << "Updating " << snode->name() << ":" << connection.inside_port
            << " from " << snode->input(connection.inside_port) << " to "
            << placeholder_name;
    snode->set_input(connection.inside_port, placeholder_name);
  }
  *common_scope = local_scope;
  VLOG(0) << "Segment @scope '" << local_scope << "', converted to graph";
  return tensorflow::Status::OK();
}
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
