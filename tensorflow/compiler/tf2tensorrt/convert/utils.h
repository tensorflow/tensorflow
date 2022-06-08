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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

#define TFTRT_ERROR(func, ...)                                              \
  do {                                                                      \
    return func("TFTRT::", __FUNCTION__, ":", __LINE__, ": ", __VA_ARGS__); \
  } while (0)

#define TFTRT_CHECK_SHAPE_TENSOR(tensor)                                 \
  if (!IsTrtShapeTensorCompatible(tensor)) {                             \
    TFTRT_ERROR(errors::InvalidArgument, "Tensor of type ",              \
                DebugString(tensor.dtype()), " having shape ",           \
                tensor.shape().DebugString(), " is not TRT compatible"); \
  }

namespace tensorflow {
namespace tensorrt {

static constexpr char kCastOutputTypeAttrName[] = "DstT";

#if !IS_TRT_VERSION_GE(8, 2, 0, 0)
template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
    if (t) t->destroy();
  }
};
template <typename T>
using TrtUniquePtrType = std::unique_ptr<T, TrtDestroyer<T>>;
#else
template <typename T>
using TrtUniquePtrType = std::unique_ptr<T>;
#endif

// Define a hash function for vector<TensorShape> because it is used as the key
// for the engine cache.
struct VectorTensorShapeHasher {
  std::size_t operator()(const std::vector<TensorShape>& key) const {
    return std::hash<std::string>()(TensorShapeUtils::ShapeListString(key));
  }
};

using absl::StrAppend;
using absl::StrCat;

// This utility template converts an arithmetic type to a string. This function
// is necessary to allow the following function to behave recursively:
// `string DebugString(const std::vector<CType>&)`.
template <typename CType, typename = typename std::enable_if<
                              std::is_arithmetic<CType>::value, CType>::type>
string DebugString(const CType& el) {
  string el_str = std::to_string(el);
  // Prettify std::to_string which can sometimes returns 1.50000 instead of 1.5.
  // In short it removes trailing 0s in a string-formatted number.
  el_str.erase(el_str.find_last_not_of('0') + 1, std::string::npos);
  return el_str;
}
// This utility template converts nested vectors to a string for debug purposes.
template <typename CType>
string DebugString(const std::vector<CType>& vector) {
  string tmp_s = "";
  for (const auto el : vector) {
    StrAppend(&tmp_s, StrCat(DebugString(el), ", "));
  }
  return StrCat("{", tmp_s.substr(0, tmp_s.length() - 2), "}");
}
string DebugString(const nvinfer1::Dims& dims);
string DebugString(const nvinfer1::DataType trt_dtype);
string DebugString(const DataType tf_type);
string DebugString(const nvinfer1::Permutation& permutation, int len);
string DebugString(const ITensorProxyPtr& tensor);
string DebugString(const nvinfer1::ITensor& tensor);
string DebugString(const std::vector<nvinfer1::Dims>& dimvec);
string DebugString(const std::vector<TensorShape>& shapes);
string DebugString(const std::vector<PartialTensorShape>& shapes);

template <size_t N>
string DebugString(const absl::InlinedVector<int64, N>& data) {
  return absl::StrCat("[", absl::StrJoin(data, ","), "]");
}

inline bool HasStaticShape(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 0) return false;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < 0) return false;
  }
  return true;
}

template <typename T>
bool HasStaticShape(const T& dims) {
  return !absl::c_any_of(dims, [](int i) { return i < 0; });
}

// Returns whether a shape is compatible with a TRT shape tensor.
template <typename TensorShapeType>
inline bool IsTrtShapeTensorCompatible(const TensorShapeType& shape) {
  return (
      shape.dims() == 0 ||
      (shape.dims() == 1 && shape.num_elements() <= nvinfer1::Dims::MAX_DIMS));
}

// Returns whether a TF tensor could be interpreted as a TRT shape tensor.
inline bool IsTrtShapeTensorCompatible(const Tensor& tensor) {
  return tensor.dtype() == DT_INT32 &&
         IsTrtShapeTensorCompatible(tensor.shape());
}

// Adapts various representations of shape (TF Shape, TRT Dims, plain
// containers) and provides methods for properties (length, volume) and
// conversion between types. Note that unlike TF's TensorShape, the underlying
// storage will only contain active dimensions. In the case of scalar shapes,
// `NumDims` is allowed to return 0 or 1, but the `storage_` vector will contain
// 1 element in both cases. In the non-scalar case, `NumDims() ==
// storage_.size()`.
class DimsAdapter {
 public:
  using StorageType = absl::InlinedVector<int64_t, 4>;

 private:
  template <typename T>
  using EnableIfNotTensorShapeType =
      std::enable_if_t<!std::is_base_of<TensorShapeBase<T>, T>::value>;

  template <typename T>
  using EnableIfInt = std::enable_if_t<std::is_arithmetic<T>::value &&
                                       std::is_integral<T>::value>;

 public:
  //----- Constructors ------

  // Constructs from an absl::Span.
  template <typename T>
  explicit DimsAdapter(absl::Span<T> shape)
      : num_dims_(static_cast<int32_t>(shape.size())) {
    absl::c_copy(shape, std::back_inserter(storage_));
  }

  // Constructs from an absl::Span.
  template <typename T>
  explicit DimsAdapter(const std::vector<T>& shape)
      : num_dims_(static_cast<int32_t>(shape.size())) {
    absl::c_copy(shape, std::back_inserter(storage_));
  }

  // Constructs from a TRT dims object.
  DimsAdapter(const nvinfer1::Dims& dims) : num_dims_(dims.nbDims) {
    absl::c_copy(absl::MakeSpan(dims.d, dims.d + std::max(dims.nbDims, 0)),
                 std::back_inserter(storage_));
  }

  // Constructs explicitly specifing num_dims and storage data.
  DimsAdapter(int32_t num_dims, StorageType data)
      : num_dims_(num_dims), storage_(std::forward<StorageType>(data)) {}

  // Constructs from a TensorShape or PartialTensorShape.
  template <typename T>
  static StatusOr<DimsAdapter> Create(const TensorShapeBase<T>& shape,
                                      bool ignore_first_dim = false) {
    if (shape.dims() > nvinfer1::Dims::MAX_DIMS)
      return errors::InvalidArgument("dims of TensorShape exceed MAX_DIMS");
    if (ignore_first_dim && shape.dims() <= 0)
      return errors::InvalidArgument(
          "removing first dim requires explicit batch dimension");
    if (shape.dims() == -1) {
      return DimsAdapter(-1, StorageType{});
    }
    if (shape.dims() == 0) {
      return DimsAdapter(0, StorageType{1});
    }
    auto offt = (ignore_first_dim ? 1 : 0);
    return DimsAdapter(
        absl::MakeSpan(shape.dim_sizes().begin() + offt, shape.dims() - offt));
  }

  // Constructs from a container.
  template <typename InputSequence,
            typename = EnableIfNotTensorShapeType<InputSequence>>
  static StatusOr<DimsAdapter> Create(const InputSequence& shape,
                                      bool ignore_first_dim = false) {
    if (ignore_first_dim && shape.size() <= 0) {
      return errors::InvalidArgument(
          "removing first dim requires explicit batch dimension");
    }
    return DimsAdapter(
        absl::MakeSpan(shape).subspan(ignore_first_dim ? 1 : 0, shape.size()));
  }

  //----- Conversion Utilities ------

  //  Converts to an nvinfers::Dims and assign the result to the object passed
  //  in via the result pointer.
  void TrtDims(nvinfer1::Dims* result) const {
    result->nbDims = num_dims_;
    absl::c_copy(storage_, static_cast<int32_t*>(result->d));
  }

  // Converts to an nvinfer1::Dims and return by value.
  nvinfer1::Dims AsTrtDims() const {
    nvinfer1::Dims result;
    TrtDims(&result);
    return result;
  }

  // Converts to a TensorShape and assigns the result to the object passed in
  // via the shape pointer.
  Status TensorShape(TensorShape* shape,
                     std::optional<int> batch_size = std::nullopt) const {
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        static_cast<const int64_t*>(storage_.data()), storage_.size(), shape));
    if (batch_size) shape->InsertDim(0, *batch_size);
    return Status::OK();
  }

  // Converts to a PartialTensorShape and assigns the result to the object
  // passed in via the shape pointer.
  Status PartialTensorShape(
      PartialTensorShape* shape,
      std::optional<int> batch_size = std::nullopt) const {
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        static_cast<const int64_t*>(storage_.data()), storage_.size(), shape));
    if (batch_size) shape->InsertDim(0, *batch_size);
    return Status::OK();
  }

  // Copies the dimension values to the vector passed in via the shape pointer.
  template <typename T, typename = EnableIfInt<T>>
  Status Vector(std::vector<T>* shape) const {
    shape->clear();
    absl::c_copy(storage_, std::back_inserter(*shape));
    return Status::OK();
  }

  //----- Property Accessors ------

  // Returns true if the shape has no dynamic dimensions.
  bool IsStatic() const {
    return !absl::c_any_of(storage_, [](auto i) { return i < 0; });
  }

  // Returns product of all dimensions.
  int64_t Volume() const {
    return absl::c_accumulate(storage_, static_cast<int64_t>(1),
                              std::multiplies<>());
  }

  int32_t NumDims() const { return num_dims_; }

  // Returns true if the shape should be interpreted as a scalar. This follows
  // TensorRT conversions: a scalar shape can have NumDims()==1 or NumDims()==0,
  // but the underlying storage_ container has a single dimension of size 1.
  bool IsScalar() const {
    return (num_dims_ == 0 || num_dims_ == 1) && storage_.size() == 1 &&
           storage_[0] == 1;
  }

  // Returns true if the dimension storage is empty. This indicates an empty
  // shape in both the scalar and non-scalar case.
  bool IsEmpty() const { return storage_.empty(); }

  string DebugString() const {
    auto vol = absl::c_accumulate(storage_, static_cast<int64_t>(1),
                                  std::multiplies<>());
    return absl::StrCat("DimsAdapter(num_dims=", num_dims_, ",shape=[",
                        absl::StrJoin(storage_, ","), "],", "vol=", vol, ")");
  }

  // Returns beginning iterator for the underlying storage.
  StorageType::const_iterator begin() const { return storage_.begin(); }

  // Returns ending iterator for the underlying storage.
  StorageType::const_iterator end() const { return storage_.end(); }

  // Returns the size of the dimension at `idx`.
  StorageType::value_type dim(size_t idx) const { return storage_[idx]; }

  // Returns a references to the dimension at `idx`.
  StorageType::value_type& dim(size_t idx) { return storage_[idx]; }

  //----- Non-Const Operators ------

  DimsAdapter& Append(int32_t dim) {
    StatusOr<bool> is_scalar = IsScalar();
    if (!is_scalar.ok()) return *this;
    num_dims_ = *is_scalar ? 2 : num_dims_ + 1;
    storage_.push_back(dim);
    return *this;
  }

  DimsAdapter& Prepend(std::optional<int32_t> dim) {
    if (dim) {
      num_dims_ = IsScalar() ? 2 : num_dims_ + 1;
      storage_.insert(storage_.begin(), *dim);
    }
    return *this;
  }

  Status RemoveBatchDimension() {
    if (storage_.empty())
      return errors::InvalidArgument(
          "attempted to remove batch dim from scalar");
    num_dims_ -= 1;
    storage_.erase(storage_.begin());
    return Status::OK();
  }

  //----- Comparison Operators ------

  bool operator==(const DimsAdapter& rhs) const {
    if (rhs.num_dims_ != num_dims_) return false;
    for (int i = 0; i < num_dims_; i++) {
      if (rhs.storage_[i] != storage_[i]) return false;
    }
    return true;
  }

  bool operator!=(const DimsAdapter& rhs) const { return !(*this == rhs); }

 private:
  int32_t num_dims_{0};
  StorageType storage_{};
};

Status GetNetworkInputShapes(const nvinfer1::INetworkDefinition* network,
                             std::vector<PartialTensorShape>* input_shapes);

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type);
Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type);

// Returns true if an engine built for cached_shapes can also run actual_shapes.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes);

// Returns the number of inputs for the engine, which also correspends to the
// number of input tensors for the network. This can differ from the number of
// input bindings, because the number of total input bindings equals the number
// of profiles times the number of engine inputs.
int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine);

// Returns the string representation for the assigned device or the requested
// device of the given node.
absl::string_view GetDeviceName(const Node* node);

// Returns the ParsedName representation for the assigned device or the
// requested device string of the given node. If the device string is invalid,
// returns std::nullopt.
std::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node);

// If the given two device assignments as compatible, returns the merge of the
// two assignments. Otherwise, returns std::nullopt.
std::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, const DeviceNameUtils::ParsedName& b);
// Similar to the above, except that the second device assignment is represented
// by a string_view.
std::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b);

bool isExperimentalFeatureActivated(string feature_name);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
