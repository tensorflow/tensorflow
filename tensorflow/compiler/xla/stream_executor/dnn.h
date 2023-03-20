/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Neural Net operation support for StreamExecutor instances.
//
// This is an abstract interface for a platform to optionally support common
// neural net operations; it accommodates implementations such as the cudnn
// library operations.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DNN_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DNN_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/compiler/xla/stream_executor/data_type.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.pb.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.pb.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace Eigen {
struct half;
}  // namespace Eigen

namespace stream_executor {

class HostBuffer;
class Stream;
class ScratchAllocator;

namespace dnn {

// Specifies an index to use when accessing specific spatial dimensions.
enum class DimIndex : int {
  X = 0,
  Y = 1,
  Z = 2,
};

// Return a reordered dims.
std::vector<int64_t> ReorderDims(const std::vector<int64_t>& input,
                                 const DataLayout& from, const DataLayout& to);

// Helper functions to make methods more readable.
inline int64_t GetDim(absl::Span<const int64_t> data, DimIndex dim) {
  return data.rbegin()[static_cast<int64_t>(dim)];
}

inline void SetDim(absl::Span<int64_t> data, DimIndex dim, int64_t value) {
  data.rbegin()[static_cast<int64_t>(dim)] = value;
}

inline void SetDim(std::vector<int64_t>* data, DimIndex dim, int64_t value) {
  return SetDim(absl::MakeSpan(*data), dim, value);
}

// int64_t is not the same type as tensorflow::protobuf_int64 in open-source.
// This wrapper function gives an int64_t array slice view of a repeated int64
// protobuf field.
//
// T should be a protobuf RepeatedField.
template <typename T>
inline absl::Span<const int64_t> AsInt64Slice(const T& repeated_field) {
  using data_ty =
      typename std::remove_reference<decltype(*repeated_field.data())>::type;
  static_assert(std::is_integral<data_ty>::value &&
                    std::is_signed<data_ty>::value && sizeof(data_ty) == 8,
                "repeated_field.data() must return a pointer to a signed "
                "64-bit integer type.");
  return absl::Span<const int64_t>(
      reinterpret_cast<const int64_t*>(repeated_field.data()),
      repeated_field.size());
}
template <typename T>
inline absl::Span<int64_t> AsInt64Slice(T* repeated_field) {
  using data_ty =
      typename std::remove_reference<decltype(*repeated_field->data())>::type;
  static_assert(std::is_integral<data_ty>::value &&
                    std::is_signed<data_ty>::value && sizeof(data_ty) == 8,
                "repeated_field->data() must return a pointer to a signed "
                "64-bit integer type.");
  return absl::Span<int64_t>(
      reinterpret_cast<int64_t*>(repeated_field->mutable_data()),
      repeated_field->size());
}

// Returns a string representation of the given data layout.
std::string DataLayoutString(DataLayout layout);

// Specifies a quantization for activations in a given BatchDescriptor.
enum class QuantizedActivationMode {
  k8Bit = 1,
  k16Bit = 2,
  k32Bit = 4,
};

// Specifies the types of a RNN model.
enum class RnnMode {
  kRnnRelu = 0,
  kRnnTanh = 1,
  kRnnLstm = 2,
  kRnnGru = 3,
};

// Specifies the input model and whether there is a linear transformation
// between the input state and the first layer hidden state.
enum class RnnInputMode {
  kRnnLinearSkip = 0,
  kRnnSkipInput = 1,
};

// Specifies the number of directions used in a RNN model. When bidirection
// is used, the input states and output sequence contain data for both
// directions.
enum class RnnDirectionMode {
  kRnnUnidirectional = 0,
  kRnnBidirectional = 1,
};

// Relevant to DepthToSpace and SpaceToDepth. This is the write layout when
// performing depth to space and the read layout when performing space to depth.
// It's specified with most-major dimension first and most-minor dimension last.
// In DepthToSpace, the D*M^2 values are read in and then, for DepthHeightWidth,
// written out to the output patch, by varying first width, then height, then
// depth. In C array format, it looks like [depth][height][width]. See
// DepthToSpace comment for more information.
enum class DepthToSpaceLayout { DepthHeightWidth };

class TensorDescriptor {
 public:
  TensorDescriptor() {}
  tsl::StatusOr<std::vector<int64_t>> GetPhysicalDimensionsMajorToMinor() const;
  std::vector<int64_t> GetPhysicalStridesMajorToMinor() const;
  std::vector<int64_t> GetLogicalStrides() const;

  static TensorDescriptor For(DataType type,
                              absl::Span<const int64_t> dimensions,
                              absl::Span<const int64_t> minor_to_major);
  int ndims() const;
  std::vector<int64_t> dimensions() const { return dimensions_; }
  std::vector<int64_t> minor_to_major() const { return minor_to_major_; }
  DataType type() const { return d_type_; }
  const std::string ToString() const;

 protected:
  TensorDescriptor(DataType type, std::vector<int64_t> dimensions,
                   std::vector<int64_t> minor_to_major)
      : d_type_(type),
        dimensions_(dimensions),
        minor_to_major_(minor_to_major) {}

 private:
  DataType d_type_;
  std::vector<int64_t> dimensions_;
  std::vector<int64_t> minor_to_major_;
};

class MatmulTensorDescriptor {
 public:
  MatmulTensorDescriptor() {}
  tsl::StatusOr<std::vector<int64_t>> GetNonContractingDims() const;
  std::vector<int64_t> GetCudnnCompatibleDimensions(
      bool is_lhs
      /*if not lhs, then rhs*/) const;
  std::vector<int64_t> GetCudnnCompatibleStrides(
      bool is_lhs
      /*if not lhs, then rhs*/) const;
  tsl::StatusOr<std::vector<int64_t>> MakeCudnnCompatible(
      const std::vector<int64_t>&, bool is_lhs) const;

  static MatmulTensorDescriptor For(DataType type,
                                    absl::Span<const int64_t> dimensions,
                                    absl::Span<const int64_t> minor_to_major,
                                    absl::Span<const int64_t> batch_dims,
                                    absl::Span<const int64_t> contracting_dims);
  std::vector<int64_t> dimensions() const { return tensor_.dimensions(); }
  std::vector<int64_t> minor_to_major() const {
    return tensor_.minor_to_major();
  }
  DataType type() const { return tensor_.type(); }

  const std::string ToString() const;

 protected:
  MatmulTensorDescriptor(TensorDescriptor tensor,
                         std::vector<int64_t> batch_dims,
                         std::vector<int64_t> contracting_dims)
      : tensor_(tensor),
        batch_dimension_numbers_(batch_dims),
        contracting_dim_(contracting_dims) {}

 private:
  TensorDescriptor tensor_;
  std::vector<int64_t> batch_dimension_numbers_;
  std::vector<int64_t> contracting_dim_;
};

// Specifies the descriptor for a RNN model.
//
// An example use case:
//   * The user first creates a model through createRnnDescriptor.
//   * The user queries the size of the underlying opaque parameter buffer.
//   * The user creates and initializes a parameter buffer of the proper size.
//   * The user runs forward and backward operations using this RNN descriptor.
//   * Once a while, user queries maintainable weights and bias regions from
//       the underlying parameter buffer. They are more likely to be forward
//       compatible and should used in saving and restoring a model.
//   * The user releases the RNN descriptor when the model is no longer in use.
class RnnDescriptor {
 public:
  struct ParamsRegion {
    int64_t offset;
    int64_t size;
  };
  typedef std::vector<ParamsRegion> ParamsRegions;
  virtual ~RnnDescriptor() {}
  virtual int64_t ParamsSizeInBytes() const { return -1; }
  virtual ParamsRegions ParamsWeightRegions() const { return ParamsRegions(); }
  virtual ParamsRegions ParamsBiasRegions() const { return ParamsRegions(); }
};

// Specifies the sequence in a RNN model.
//
// The user is responsible for releasing this descriptor when it is no longer
// in use. The destructor releases the underlying descriptors.
class RnnSequenceTensorDescriptor {
 public:
  virtual ~RnnSequenceTensorDescriptor() {}
};

// Specifies either the input and hidden state in a RNN model.
//
// The user is responsible for releasing this descriptor when it is no longer
// in use. The destructor releases the underlying descriptors.
class RnnStateTensorDescriptor {
 public:
  virtual ~RnnStateTensorDescriptor() {}
};

// Returns a string representation of the given quantization mode.
std::string QuantizedActivationModeString(QuantizedActivationMode mode);

// Describes the dimensions that a layer consumes/produces.
//
// This is a matrix (height, width), its "depth" (feature_map_count),
// how many of these matrices are present (count),
// and the maximum and minimum values expected in the matrix (value_max,
// value_min).
// If input is quantized, all values greater
// than value_max will be clipped to value_max and all values less than
// value_min will be clipped to value_min.
// When quantized output is dequantized no value will be greater than
// value_max or less than value_min.
//
// Uses the named argument construction form:
//
//  auto input_batch_dimensions =
//      BatchDescriptor().set_count(42).set_feature_map_count(7)...
//
// Details:
//
// For a convolutional layer, a single inference takes a 3-dimensional matrix
// of input and produces a 3-dimensional matrix of output. We call the three
// dimensions height, width and feature_map_count, where for an image, the
// height and width correspond to the Y and X pixel indices, respectively, and
// the feature_map_count corresponds to the RGB dimension of the input data.
// Then the count indicates how many 3D matrices are being presented to be
// processed at once; this corresponds to the neural network concept of
// minibatch size.
//
// For a fully connected layer, it's better to put the nodes of the layer in
// the feature_map_count, and leave the height and weight as degenerate (== 1).
// Count indicates how many input vectors (degenerate 3D matrices) are to be
// processed.
//
// If unspecified, value_max and value_min default to 0.0.
// If value_max == value_min the Stream will attempt to derive valid values -
// for example the output of Relu6 activation will always be in the range
// [0.0, 6.0].
//
// If unspecified, layout defaults to kYXDepthBatch.
class BatchDescriptor {
 public:
  // Creates a "blank" batch descriptor, which should be initialized via the
  // named argument helpers.
  BatchDescriptor();
  explicit BatchDescriptor(int ndims);

  // Clones values from 'other' for initialization.
  void CloneFrom(const BatchDescriptor& other);

  std::string ToString() const;
  std::string ToShortString() const;

  // Pre-condition:
  //   value_max_ == 0
  //   value_min_ == 0
  //   quantized_activation_mode_ == QuantizedActivationMode::k8Bit
  TensorDescriptorProto ToProto(DataType data_type) const;

  // Accessors.
  int64_t count() const { return tensor_.dimensions(0); }
  int64_t feature_map_count() const { return tensor_.dimensions(1); }
  int64_t height() const { return GetDim(spatial_size(), DimIndex::Y); }
  int64_t width() const { return GetDim(spatial_size(), DimIndex::X); }
  int64_t spatial_dim(DimIndex dim) const {
    return GetDim(spatial_size(), dim);
  }
  int ndims() const { return spatial_size().size(); }
  float value_max() const { return value_max_; }
  float value_min() const { return value_min_; }
  DataLayout layout() const { return tensor_.data_layout(); }
  QuantizedActivationMode quantized_activation_mode() const {
    return quantized_activation_mode_;
  }
  // Full dimensions of the underlying data, ordered according to a specific
  // layout.
  std::vector<int64_t> full_dims(const DataLayout& layout) const;

  // Full strides of the underlying data, ordered according to a specific
  // layout.
  std::vector<int64_t> full_strides(const DataLayout& layout) const;

  // Vectorized dimensions where users can specify the dimension that the number
  // of dimensions is reported rather than the full number of elements.
  std::vector<int64_t> vectorized_dims(const DataLayout& layout,
                                       int vector_size, int vector_dim) const;

  // Vectorized strides correspond to the vectorized_dims.
  std::vector<int64_t> vectorized_strides(const DataLayout& layout,
                                          int vector_size,
                                          int vector_dim) const;

  // Named-argument helpers for avoiding user error during construction.
  BatchDescriptor& set_count(int64_t value) {
    tensor_.set_dimensions(0, value);
    return *this;
  }
  BatchDescriptor& set_feature_map_count(int64_t value) {
    tensor_.set_dimensions(1, value);
    return *this;
  }
  BatchDescriptor& set_height(int64_t value) {
    SetDim(spatial_size(), DimIndex::Y, value);
    return *this;
  }
  BatchDescriptor& set_width(int64_t value) {
    SetDim(spatial_size(), DimIndex::X, value);
    return *this;
  }
  BatchDescriptor& set_spatial_dim(DimIndex dim, int64_t value) {
    SetDim(spatial_size(), dim, value);
    return *this;
  }
  BatchDescriptor& set_value_max(float value) {
    value_max_ = value;
    return *this;
  }
  BatchDescriptor& set_value_min(float value) {
    value_min_ = value;
    return *this;
  }
  BatchDescriptor& set_layout(DataLayout layout) {
    tensor_.set_data_layout(layout);
    return *this;
  }
  BatchDescriptor& set_quantized_activation_mode(
      QuantizedActivationMode quantized_activation_mode) {
    quantized_activation_mode_ = quantized_activation_mode;
    return *this;
  }

  // Return the number of nodes in a single feature map.
  int64_t NodesPerFeatureMap() const;

  // Return the number of nodes across all feature maps. Note that this is not
  // affected by the batch count.
  int64_t NodesAcrossFeatureMaps() const;

  // Returns the number of elements (e.g. RGB pixel values) required to hold a
  // given batch descriptor, given a no-padding assumption. Note that this is
  // affected by the batch count.
  int64_t ElementCount() const;

  // Return the number of weights required to fully connect a layer with
  // dimensions given by the 'input' descriptor with a layer with dimensions
  // given by the 'output' descriptor.
  static int64_t FullyConnectedWeightCount(const BatchDescriptor& input,
                                           const BatchDescriptor& output);

  // Return the number of biases required to fully connect to an output layer
  // with dimensions given the 'output' descriptor.
  static int64_t FullyConnectedBiasCount(const BatchDescriptor& output);

  // Return a BatchDescriptor for the output of a depth concatenation
  // with the given input descriptors. The inputs should have the same
  // dimensions, except possibly for feature_map_count(), though this
  // function does not verify that.
  static BatchDescriptor DepthConcatenateOutputDescriptor(
      absl::Span<const dnn::BatchDescriptor> inputs);

 private:
  absl::Span<const int64_t> spatial_size() const {
    return AsInt64Slice(tensor_.dimensions()).subspan(2);
  }

  absl::Span<int64_t> spatial_size() {
    return AsInt64Slice(tensor_.mutable_dimensions()).subspan(2);
  }

  TensorDescriptorProto tensor_;
  float value_max_;
  float value_min_;
  QuantizedActivationMode quantized_activation_mode_;
};

// Returns a string representation of the given filter layout.
std::string FilterLayoutString(FilterLayout layout);

// Describes a filter for the convolution. This is the "window" from
// height-by-width patches of each of the feature maps in the input layer to the
// cells within the output feature map.
//
// Uses the named argument construction form:
//
//  FilterDescriptor filter_dimensions;
//  filter_dimensions
//    .set_output_feature_map_count(42)
//    .set_input_feature_map_count(7)
//    ...
//
// Arguments:
// - output_feature_map_count: number of feature maps in the output layer.
// - input_feature_map_count: number of feature maps in the input layer (from
//      which the filter patch is taken).
// - input_filter_height: "height" number of neurons used in the sliding window
//      over the input layer.
// - input_filter_width: "width" number of neurons used in the sliding window
//      over the input layer.
//
// Sometimes names like "filter input height" are referred to by synonymous
// terminology, such as "kernel y size".
//
// If unspecified, layout defaults to kOutputInputYX.
class FilterDescriptor {
 public:
  // By default construction, all dimensions are set to zero, so they should all
  // be populated by the user via the named-argument helpers below. (See class
  // comment for details.)
  FilterDescriptor();
  explicit FilterDescriptor(int ndims);
  ~FilterDescriptor();

  // Named-argument helpers for avoiding user error during construction.
  FilterDescriptor& set_output_feature_map_count(int64_t value) {
    tensor_.set_dimensions(0, value);
    return *this;
  }
  FilterDescriptor& set_input_feature_map_count(int64_t value) {
    tensor_.set_dimensions(1, value);
    return *this;
  }
  FilterDescriptor& set_input_filter_height(int64_t value) {
    SetDim(input_filter_dims(), DimIndex::Y, value);
    return *this;
  }
  FilterDescriptor& set_input_filter_width(int64_t value) {
    SetDim(input_filter_dims(), DimIndex::X, value);
    return *this;
  }
  FilterDescriptor& set_layout(FilterLayout layout) {
    tensor_.set_filter_layout(layout);
    return *this;
  }
  FilterDescriptor& set_spatial_dim(DimIndex dim, int64_t value) {
    SetDim(input_filter_dims(), dim, value);
    return *this;
  }
  int ndims() const { return input_filter_dims().size(); }

  void CloneFrom(const FilterDescriptor& other);

  std::string ToString() const;
  std::string ToShortString() const;
  TensorDescriptorProto ToProto(DataType data_type) const;

  // Returns the number of weights required as parameters for a convolution
  // using this filter descriptor.
  int64_t ComputeWeightCount() const;

  // Returns the number of biases required as parameters for a convolution
  // using this filter descriptor.
  int64_t bias_count() const { return output_feature_map_count(); }

  int64_t output_feature_map_count() const { return tensor_.dimensions(0); }
  int64_t input_feature_map_count() const { return tensor_.dimensions(1); }
  int64_t input_filter_height() const {
    return GetDim(input_filter_dims(), DimIndex::Y);
  }
  int64_t input_filter_width() const {
    return GetDim(input_filter_dims(), DimIndex::X);
  }
  int64_t input_filter_dim(DimIndex dim) const {
    return GetDim(input_filter_dims(), dim);
  }

  FilterLayout layout() const { return tensor_.filter_layout(); }

  absl::Span<const int64_t> input_filter_dims() const {
    return AsInt64Slice(tensor_.dimensions()).subspan(2);
  }

  // Full dimensions of the underlying filter,
  // ordered according to a specific layout.
  std::vector<int64_t> full_dims(const FilterLayout& layout) const;

  // Full strides of the underlying filter,
  // ordered according to a specific layout.
  std::vector<int64_t> full_strides(const FilterLayout& layout) const;

  // Vectorized dimensions where users can specify the dimension that the number
  // of dimensions is reported rather than the full number of elements.
  std::vector<int64_t> vectorized_dims(const FilterLayout& layout,
                                       int vector_size, int vector_dim) const;

  // Vectorized strides correspond to the vectorized_dims.
  std::vector<int64_t> vectorized_strides(const FilterLayout& layout,
                                          int vector_size,
                                          int vector_dim) const;

 private:
  absl::Span<int64_t> input_filter_dims() {
    return AsInt64Slice(tensor_.mutable_dimensions()).subspan(2);
  }

  TensorDescriptorProto tensor_;
};

// Describes how padding should be aligned when the total number of pad
// elements is odd.
enum class PadAlignment : int64_t {
  kDefault = 0,        // default padding for the device.
  kCudnnPadding,       // cuDNN padding - prefer to pad at the start.
  kTensorFlowPadding,  // TensorFlow padding - prefer to pad at the end.
};

// Returns a string representation of the given padding alignment.
std::string PadAlignmentString(PadAlignment alignment);

// Print alignment to str. Needed to use CHECK_EQ between two PadAlignments.
std::ostream& operator<<(std::ostream& str, dnn::PadAlignment alignment);

// Describes a convolution.
//
// Uses the named argument construction form:
//
//  ConvolutionDescriptor convolution_dimensions;
//  convolution_dimensions
//    .set_vertical_filter_stride(2)
//    .set_horizontal_filter_stride(2)
//    ...
//
// Arguments:
// - zero_padding_height: padding of the "y dimension" of the input data. Note
//    that this is different from the height of the filter.
// - zero_padding_width: analogous to the height above, but in the "x
//    dimension".
// - vertical_filter_stride: the convolution slides a 2-dimensional window of
//    filter-height-by-filter-width over the input layer -- the center of that
//    window is moved in the "y dimension" according to this stride value.
// - horizontal_filter_stride: analogous to the vertical stride above, but in
//    the "x dimension".
// - vertical_dilation_rate: there will be (vertical_dilation_rate - 1) skipped
//   cells between each filter element in the "y dimension".
// - horizontal_dilation_rate: there will be (horizontal_dilation_rate - 1)
//   skipped cells between each filter element in the "x dimension".
// - convolution_not_crosscor: By default (convolution_not_crosscor == false),
//   we perform cross correlation rather than convolution. With the flag set,
//   we perform convolution. Convolution and cross correlation are related by
//   rotating the filter by 180 degrees (or equivalently flipping all spatial
//   dimensions).
class ConvolutionDescriptor {
 public:
  // By default construction, there is no zero-padding and the filter stride is
  // 1x1 (centering the filter on every cell in the input layer's
  // width-by-height area).
  ConvolutionDescriptor();
  explicit ConvolutionDescriptor(int ndims);
  ~ConvolutionDescriptor();

  std::string ToString() const;
  std::string ToShortString() const;
  ConvolutionDescriptorProto ToProto() const { return proto_; }

  ConvolutionDescriptor& set_zero_padding_height(int64_t value) {
    SetDim(padding(), DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_zero_padding_width(int64_t value) {
    SetDim(padding(), DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_zero_padding(DimIndex dim, int64_t value) {
    SetDim(padding(), dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_vertical_filter_stride(int64_t value) {
    SetDim(strides(), DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_horizontal_filter_stride(int64_t value) {
    SetDim(strides(), DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_filter_stride(DimIndex dim, int64_t value) {
    SetDim(strides(), dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_vertical_dilation_rate(int64_t value) {
    SetDim(dilations(), DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_horizontal_dilation_rate(int64_t value) {
    SetDim(dilations(), DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_dilation_rate(DimIndex dim, int64_t value) {
    SetDim(dilations(), dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_group_count(int group_count) {
    proto_.set_group_count(group_count);
    return *this;
  }
  ConvolutionDescriptor& set_convolution_not_crosscorr(bool conv) {
    proto_.set_convolution_mode(conv ? ConvolutionMode::CONVOLUTION
                                     : ConvolutionMode::CROSS_CORRELATION);
    return *this;
  }
  ConvolutionDescriptor& set_name(const std::string& name) {
    proto_.set_name(name);
    return *this;
  }
  int64_t zero_padding_height() const { return GetDim(padding(), DimIndex::Y); }
  int64_t zero_padding_width() const { return GetDim(padding(), DimIndex::X); }
  int64_t vertical_filter_stride() const {
    return GetDim(strides(), DimIndex::Y);
  }
  int64_t horizontal_filter_stride() const {
    return GetDim(strides(), DimIndex::X);
  }
  int64_t vertical_dilation_rate() const {
    return GetDim(dilations(), DimIndex::Y);
  }
  int64_t horizontal_dilation_rate() const {
    return GetDim(dilations(), DimIndex::X);
  }

  int zero_padding(DimIndex dim) const { return GetDim(padding(), dim); }
  int filter_stride(DimIndex dim) const { return GetDim(strides(), dim); }
  int dilation_rate(DimIndex dim) const { return GetDim(dilations(), dim); }
  // TODO(timshen): remove this function. No users of this class is setting a
  // non-default pad alignment.
  PadAlignment pad_alignment() const { return PadAlignment::kDefault; }
  int group_count() const { return proto_.group_count(); }
  int ndims() const { return padding().size(); }
  bool convolution_not_crosscorr() const {
    return proto_.convolution_mode() == ConvolutionMode::CONVOLUTION;
  }

  absl::Span<const int64_t> strides() const {
    return AsInt64Slice(proto_.strides());
  }

  absl::Span<const int64_t> dilations() const {
    return AsInt64Slice(proto_.dilations());
  }

  absl::Span<const int64_t> padding() const {
    return AsInt64Slice(proto_.paddings());
  }

  std::string name() const { return proto_.name(); }

 private:
  absl::Span<int64_t> strides() {
    return AsInt64Slice(proto_.mutable_strides());
  }

  absl::Span<int64_t> dilations() {
    return AsInt64Slice(proto_.mutable_dilations());
  }

  absl::Span<int64_t> padding() {
    return AsInt64Slice(proto_.mutable_paddings());
  }

  ConvolutionDescriptorProto proto_;

  // TODO(leary) cudnn provides these fields, but need to characterize what
  // their effect is -- they may be boolean rather than integral.
  // int64_t upscale_input_x;
  // int64_t upscale_input_y;
};

// A patch of values in the input can be pooled via either a max or an average
// operation.
// Specify int64_t so there's no padding in PoolingDescriptor.
enum class PoolingMode : int64_t {
  kMaximum,
  kAverage,
};

// Specify the dimension in which to concatenate inputs in space.
// Specify int64_t so there's no padding in SpaceConcatenateMode.
enum class SpaceConcatenateMode : int64_t {
  XDirection,
  YDirection,
};

// Returns a short name for the pooling mode, e.g. "Avg".
std::string ShortPoolingModeString(PoolingMode mode);

// Describes a pooling operation to be enqueued onto a stream via a platform's
// DnnSupport.
//
// TODO(broune): describe how padding works and what happens if the
// window height/width is not divisible by the vertical/horizontal
// stride.
//
// Arguments:
//  pooling_mode: pooling operator to use on the input patch
//  window_height: height of input window
//  window_width: width of input window
//  vertical_stride: vertical delta for center of the input patch
//  horizontal_stride: horizontal delta for center of the input patch
class PoolingDescriptor {
 public:
  PoolingDescriptor();
  explicit PoolingDescriptor(int ndims);

  PoolingDescriptor& set_pooling_mode(PoolingMode value) {
    mode_ = value;
    return *this;
  }
  PoolingDescriptor& set_window_height(int64_t value) {
    SetDim(&window_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_window_width(int64_t value) {
    SetDim(&window_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_window(DimIndex dim, int64_t value) {
    SetDim(&window_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_vertical_padding(int64_t value) {
    SetDim(&padding_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_horizontal_padding(int64_t value) {
    SetDim(&padding_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_padding(DimIndex dim, int64_t value) {
    SetDim(&padding_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_vertical_stride(int64_t value) {
    SetDim(&strides_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_horizontal_stride(int64_t value) {
    SetDim(&strides_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_stride(DimIndex dim, int64_t value) {
    SetDim(&strides_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_propagate_nans(bool value) {
    propagate_nans_ = value;
    return *this;
  }
  PoolingDescriptor& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }

  int ndims() const { return ndims_; }
  void CloneFrom(const PoolingDescriptor& other);

  std::string ToString() const;
  std::string ToShortString() const;

  PoolingMode mode() const { return mode_; }
  int64_t window_height() const { return GetDim(window_, DimIndex::Y); }
  int64_t window_width() const { return GetDim(window_, DimIndex::X); }
  int64_t window(DimIndex dim) const { return GetDim(window_, dim); }
  int64_t vertical_padding() const { return GetDim(padding_, DimIndex::Y); }
  int64_t horizontal_padding() const { return GetDim(padding_, DimIndex::X); }
  int64_t padding(DimIndex dim) const { return GetDim(padding_, dim); }
  int64_t vertical_stride() const { return GetDim(strides_, DimIndex::Y); }
  int64_t horizontal_stride() const { return GetDim(strides_, DimIndex::X); }
  int64_t stride(DimIndex dim) const { return GetDim(strides_, dim); }
  absl::Span<const int64_t> window() const { return window_; }
  absl::Span<const int64_t> padding() const { return padding_; }
  absl::Span<const int64_t> strides() const { return strides_; }
  bool propagate_nans() const { return propagate_nans_; }
  std::string name() const { return name_; }

 private:
  PoolingMode mode_;
  int ndims_;
  bool propagate_nans_;
  std::string name_;  // Name as in Tensorflow NodeDef, for debugging purposes.

  // Stored as: ..., y, x.
  std::vector<int64_t> window_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> strides_;
};

// Collects parameters for DNN algorithms
class AlgorithmDesc {
 public:
  typedef int64_t Index;
  AlgorithmDesc() : AlgorithmDesc(0, false, std::nullopt) {}
  explicit AlgorithmDesc(AlgorithmProto proto) : proto_(std::move(proto)) {}
  AlgorithmDesc(Index algo_id, bool use_tensor_ops)
      : AlgorithmDesc(algo_id, use_tensor_ops, std::nullopt) {}
  AlgorithmDesc(Index algo_id, bool use_tensor_ops,
                std::optional<uint64_t> workspace_size) {
    proto_.set_is_cudnn_frontend(false);
    proto_.set_algo_id(algo_id);
    proto_.set_math_type(use_tensor_ops ? AlgorithmProto::TENSOR_OP_MATH
                                        : AlgorithmProto::DEFAULT_MATH);
    if (workspace_size) {
      proto_.mutable_workspace_size()->set_value(*workspace_size);
    }
  }
  AlgorithmDesc(int64_t engine_id,
                const std::vector<std::pair<int64_t, int64_t>>& tuning_knobs,
                std::optional<uint64_t> workspace_size);
  bool is_cudnn_frontend() const { return proto_.is_cudnn_frontend(); }

  bool tensor_ops_enabled() const {
    return proto_.math_type() == AlgorithmProto::TENSOR_OP_MATH;
  }
  std::optional<uint64_t> workspace_size() const {
    if (proto_.has_workspace_size()) {
      return proto_.workspace_size().value();
    }
    return std::nullopt;
  }
  Index algo_id() const { return proto_.algo_id(); }

  std::vector<std::pair<int64_t, int64_t>> TuningKnobs() const;

  bool operator==(const AlgorithmDesc& other) const;

  uint64_t hash() const;

  AlgorithmProto ToProto() const { return proto_; }

  std::string ToString() const;

 private:
  AlgorithmProto proto_;
};

// Describes the result from a perf experiment.
//
// Arguments:
//  algorithm: returns the exact algorithm that was used.
//  elapsed_time_in_ms: returns the measured elapsed time in milliseconds.
class ProfileResult {
 public:
  bool is_valid() const {
    return algorithm_.has_value() &&
           elapsed_time_in_ms() != std::numeric_limits<float>::max();
  }

  AlgorithmDesc algorithm() const { return *algorithm_; }
  void set_algorithm(AlgorithmDesc val) { algorithm_ = val; }

  float elapsed_time_in_ms() const { return elapsed_time_in_ms_; }
  void set_elapsed_time_in_ms(float val) { elapsed_time_in_ms_ = val; }

  size_t scratch_size() const { return scratch_size_; }
  void set_scratch_size(size_t val) { scratch_size_ = val; }

 private:
  std::optional<AlgorithmDesc> algorithm_;
  float elapsed_time_in_ms_ = std::numeric_limits<float>::max();
  // The scratch size algorithm_ requires. Currently it's only populated by
  // convolutions.
  size_t scratch_size_ = 0;
};

// Backend-specific data shared between repeated launches of the same
// convolution.
template <typename Sig>
class OpRunner;

// An abstract class owning cached state for a particular op/configuration.
//
// The primary motivation for this is cuDNN backend ExecutionPlans, which are
// costly to recreate.
//
// All OpRunners must be outlived by their parent Stream.
template <typename... Args>
class OpRunner<void(Args...)> {
 public:
  virtual ~OpRunner() {}

  // Get a description of the runner, for uniqueness of autotune entries.
  //
  // Since this is used to determine whether runners are equivalent for the
  // purpose of scoring autotune entries, it shall be unique among runners of
  // the same op and parameters.
  virtual std::string ToString() const = 0;

  // Get the number of bytes of scratch space needed for `operator()`.
  //
  // If determining the workspace size can fail, runners should precompute and
  // cache it at construction time.
  virtual size_t GetWorkspaceSize() const = 0;

  // Convert to an AlgorithmDesc for AoT compilation or autotuning.
  virtual tsl::StatusOr<AlgorithmDesc> ToAlgorithmDesc() const = 0;

  // Launch the operation, with the signature determined by `Sig`.
  virtual tsl::Status operator()(Stream*, ProfileResult*,
                                 DeviceMemoryBase scratch_memory,
                                 Args... args) const = 0;
};

using ConvSignature = void(DeviceMemoryBase /* input_data */,
                           DeviceMemoryBase /* filter_data */,
                           DeviceMemoryBase /* output_data */);
using ConvRunner = OpRunner<ConvSignature>;

using FusedConvSignature = void(DeviceMemoryBase /* input_data */,
                                DeviceMemoryBase /* filter_data */,
                                DeviceMemoryBase /* side_input_data */,
                                DeviceMemoryBase /* bias_data */,
                                DeviceMemoryBase /* output_data */);
using FusedConvRunner = OpRunner<FusedConvSignature>;

using FusedMatmulSignature = void(DeviceMemoryBase /* a_data */,
                                  DeviceMemoryBase /* b_data */,
                                  DeviceMemoryBase /* bias_data */,
                                  DeviceMemoryBase /* c_data */);
using FusedMatmulRunner = OpRunner<FusedMatmulSignature>;

using FusedMHASoftmaxSignature = void(DeviceMemoryBase /*BMM1_inputA_data*/,
                                      DeviceMemoryBase /* BMM1_inputB_data */,
                                      DeviceMemoryBase /* BMM2_inputA_data */,
                                      DeviceMemoryBase /* output_data */);
using FusedMHASoftmaxRunner = OpRunner<FusedMHASoftmaxSignature>;

using FusedMHAMaskSignature = void(DeviceMemoryBase /*BMM1_inputA_data*/,
                                   DeviceMemoryBase /* BMM1_inputB_data */,
                                   DeviceMemoryBase /* mask_data */,
                                   DeviceMemoryBase /* BMM2_inputA_data */,
                                   DeviceMemoryBase /* output_data */);
using FusedMHAMaskRunner = OpRunner<FusedMHAMaskSignature>;

using FusedMHABiasMaskSignature = void(DeviceMemoryBase /*BMM1_inputA_data*/,
                                       DeviceMemoryBase /* BMM1_inputB_data */,
                                       DeviceMemoryBase /* mask_data */,
                                       DeviceMemoryBase /* bias_data */,
                                       DeviceMemoryBase /* BMM2_inputA_data */,
                                       DeviceMemoryBase /* output_data */);
using FusedMHABiasMaskRunner = OpRunner<FusedMHABiasMaskSignature>;

using FusedMHABiasSignature = void(DeviceMemoryBase /*BMM1_inputA_data*/,
                                   DeviceMemoryBase /* BMM1_inputB_data */,
                                   DeviceMemoryBase /* bias_data */,
                                   DeviceMemoryBase /* BMM2_inputA_data */,
                                   DeviceMemoryBase /* output_data */);
using FusedMHABiasRunner = OpRunner<FusedMHABiasSignature>;

// Describes the configuration for the algorithms that will used.
//
// Arguments:
//  algorithm: the primary algorithm that should be used.
//  algorithm_no_scratch: a secondary algorithm that should be used, if the
//    the allocation for the scratch memory fails.
//  scrach_size: specify the size of scratch memory in bytes needed for the
//    algorithm used.
//
// On CUDA platform with CUDNN library, algorithm and algorithm_no_scratch
// would be used. On ROCm platform with MIOpen library, algorithm and
// scratch_size would be used. The major difference between the two platforms
// are whether it's possible to get an algorithm without scratch memory. On
// CUDA + CUDNN it's possible, and algorithm_no_scratch can be used to track
// such information, whereas on ROCm + MIOpen there is no guarantee to getting
// one without scratch memory, and scratch_size field is used to track it.
class AlgorithmConfig {
 public:
  AlgorithmConfig() {}
  explicit AlgorithmConfig(AlgorithmDesc algorithm) : algorithm_(algorithm) {}
  AlgorithmConfig(AlgorithmDesc algorithm, size_t scratch_size)
      : algorithm_(algorithm), scratch_size_(scratch_size) {}
  AlgorithmConfig(AlgorithmDesc algorithm, AlgorithmDesc algorithm_no_scratch)
      : algorithm_(algorithm), algorithm_no_scratch_(algorithm_no_scratch) {}
  AlgorithmConfig(AlgorithmDesc algorithm, size_t scratch_size,
                  AlgorithmDesc algorithm_no_scratch)
      : algorithm_(algorithm),
        algorithm_no_scratch_(algorithm_no_scratch),
        scratch_size_(scratch_size) {}

  // TODO(ruochengw): After cl/380702564, add support for algorithm configs with
  // cuDNN Frontend APIs.
  explicit AlgorithmConfig(const AlgorithmConfigProto& algorithm_config_proto) {
    const AlgorithmProto& algorithm_proto = algorithm_config_proto.algorithm();
    algorithm_ = AlgorithmDesc(algorithm_proto);
    if (algorithm_config_proto.optional_scratch_size_case() !=
        /*ONEOF_NAME_NOT_SET=*/0) {
      scratch_size_ = algorithm_config_proto.scratch_size();
    }
    if (algorithm_config_proto.optional_algorithm_no_scratch_case() !=
        /*ONEOF_NAME_NOT_SET=*/0) {
      const AlgorithmProto& algorithm_no_scratch_proto =
          algorithm_config_proto.algorithm_no_scratch();
      algorithm_no_scratch_ = AlgorithmDesc(algorithm_no_scratch_proto);
    }
  }

  std::optional<AlgorithmDesc> algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmDesc val) { algorithm_ = val; }
  std::optional<AlgorithmDesc> algorithm_no_scratch() const {
    return algorithm_no_scratch_;
  }
  void set_algorithm_no_scratch(AlgorithmDesc val) {
    algorithm_no_scratch_ = val;
  }
  std::optional<size_t> scratch_size() const { return scratch_size_; }
  void set_scratch_size(size_t val) { scratch_size_ = val; }
  bool operator==(const AlgorithmConfig& other) const {
    return this->algorithm_ == other.algorithm_ &&
           this->algorithm_no_scratch_ == other.algorithm_no_scratch_ &&
           this->scratch_size_ == other.scratch_size_;
  }
  bool operator!=(const AlgorithmConfig& other) const {
    return !(*this == other);
  }
  std::string ToString() const;

  // TODO(ruochengw): After cl/380702564, add support for algorithm configs with
  // cuDNN Frontend APIs.
  AlgorithmConfigProto ToProto() const {
    AlgorithmConfigProto algorithm_config_proto;
    if (algorithm_.has_value()) {
      *algorithm_config_proto.mutable_algorithm() =
          algorithm_.value().ToProto();
    }
    if (algorithm_no_scratch_.has_value()) {
      *algorithm_config_proto.mutable_algorithm_no_scratch() =
          algorithm_no_scratch_.value().ToProto();
    }
    if (scratch_size_.has_value()) {
      algorithm_config_proto.set_scratch_size(scratch_size_.value());
    }
    return algorithm_config_proto;
  }

 private:
  std::optional<AlgorithmDesc> algorithm_;
  std::optional<AlgorithmDesc> algorithm_no_scratch_;
  std::optional<size_t> scratch_size_;
};

// Describes a local response normalization (LRN). LRN is used e.g. in
// dist_belief.
//
// Let V be the vector of feature maps at some (batch, y, x)
// coordinate. LRN applies independently to each vector V in the
// input, across all coordinates (batch, y, x), by mapping each V to
// another vector U of the same size using the formula
//
//   U_i = V_i / ((bias + alpha * (sum_j V_j^2)) ^ beta)
//
// where the sum is taken over j in the closed range [i - range, i + range].
//
// When calculating U_i the j in the sum can extend beyond the bounds
// of V. If wrap_around is true, then V_j = V_{j mod F} where F is the
// size of V, which is the number of feature maps. If wrap_around is
// false, then V_j = 0 for j outside [0, F-1].
//
// If segment_size <= F, where F is the number of feature_maps, then
// segment_size has no effect. Otherwise, each consecutive segment of
// segment_size entries in V are normalized separately.
//
// Not all StreamExecutors allow wrap_around == true or segment_size
// != 64. Some do not implement normalization at all.
class NormalizeDescriptor {
 public:
  NormalizeDescriptor();

  NormalizeDescriptor& set_bias(float bias) {
    bias_ = bias;
    return *this;
  }

  NormalizeDescriptor& set_range(int32_t range) {
    range_ = range;
    return *this;
  }

  NormalizeDescriptor& set_alpha(float alpha) {
    alpha_ = alpha;
    return *this;
  }

  NormalizeDescriptor& set_beta(float beta) {
    beta_ = beta;
    return *this;
  }

  NormalizeDescriptor& set_wrap_around(bool wrap_around) {
    wrap_around_ = wrap_around;
    return *this;
  }

  NormalizeDescriptor& set_segment_size(int32_t segment_size) {
    segment_size_ = segment_size;
    return *this;
  }

  void CloneFrom(const NormalizeDescriptor& other);

  std::string ToString() const;
  std::string ToShortString() const;

  float bias() const { return bias_; }
  int32_t range() const { return range_; }
  float alpha() const { return alpha_; }
  float beta() const { return beta_; }
  bool wrap_around() const { return wrap_around_; }
  int32_t segment_size() const { return segment_size_; }

 private:
  float bias_;
  int32_t range_;
  float alpha_;
  float beta_;
  bool wrap_around_;
  int32_t segment_size_;
};

// Returns a string representation of the given activation mode.
std::string ActivationModeString(ActivationMode mode);

// Describes the operation that DoElementwiseOperation should perform on its
// inputs.
enum class ElementwiseOperation { kAdd, kMultiply };

std::string ElementwiseOperationString(ElementwiseOperation op);

// A simple class representing the version of the backing library, to
// workaround the "too perfect forwarding" issue in gcc6+ compilers.
// See PR#16309 and issue #18402 for links discussing the issue.
class VersionInfo {
 public:
  explicit VersionInfo(int major = 0, int minor = 0, int patch = 0)
      : major_(major), minor_(minor), patch_(patch) {}
  explicit VersionInfo(DnnVersionInfoProto proto)
      : major_(proto.major()), minor_(proto.minor()), patch_(proto.patch()) {}

  DnnVersionInfoProto ToProto() const {
    DnnVersionInfoProto proto;
    proto.set_major(major_);
    proto.set_minor(minor_);
    proto.set_patch(patch_);
    return proto;
  }

  int major_version() const { return major_; }
  int minor_version() const { return minor_; }
  int patch() const { return patch_; }

  std::tuple<int, int, int> as_tuple() const {
    return std::make_tuple(major_, minor_, patch_);
  }

  friend bool operator<(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() < b.as_tuple();
  }
  friend bool operator>(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() > b.as_tuple();
  }
  friend bool operator<=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() <= b.as_tuple();
  }
  friend bool operator>=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() >= b.as_tuple();
  }
  friend bool operator==(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() == b.as_tuple();
  }
  friend bool operator!=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() != b.as_tuple();
  }

 private:
  int major_;
  int minor_;
  int patch_;
};

// Suite of operations typically used for implementing Deep/Convolutional Neural
// Nets. Note: A false return value of an operation indicates the
// implementation is not available.
//
// TODO(b/118763918): this class (or rather dispatch table) has several
// problems:
// * Some overloads are missing. Ideally we want to have template virtual
//   functions while the template arguments is a closed set. However, we don't
//   get that from the language.
// * The API is a union of cuDNN and another private backend. Only 10% of the
//   functions are actually implemented by both backends, the rest are
//   actually backend-specific. The massive interface creates extra mental
//   burden.
// * Poor error handling: the API should return tsl::Status objects.
//
// PrepareForConvolution is an example for how new APIs should be written.
class DnnSupport {
 public:
  DnnSupport() {}
  virtual ~DnnSupport() {}

  virtual tsl::Status Init() = 0;

  // Gets the version of the backing library, as a VersionInfo object.
  virtual tsl::StatusOr<VersionInfo> GetVersion() {
    return tsl::errors::Unimplemented(
        "DnnSupport::GetVersion not implemented on this platform.");
  }

  // Performs a single-precision forward batch normalization operation onto
  // the stream.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the batch normalization
  //    operation should be enqueued onto.
  //  x: input data.
  //  scale: scaling parameters.
  //  offset: offset parameters.
  //  estimated_mean: population mean estimated during training.
  //    Used for inference only; empty for training.
  //  estimated_variance: population variance estimated during training,
  //    used for inference only; empty for training.
  //  side_input: optional input that is element-wise added to the output of
  //    batch normalization.
  //  x_desc: dimensions of the input data, which is the same as the dimensions
  //    of the output and side input.
  //  scale_offset_desc: dimensions of scale and offset.
  //  epsilon: a small floating point number added to the variance of x.
  //  activation_mode: activation applied to the result of batch normalization
  //    (or after adding optional side input)
  //  y: output data.
  //  batch_mean: batch mean, to be used to compute the running mean.
  //  batch_variance: batch variance, to be used to compute
  //    the running variance.
  //  reserve_space_1: saved mean, to be reused in the backward gradient
  //    computation.
  //  reserve_space_2: saved inv_var (1/sqrt(epsilon + variance), to be reused
  //    in the backward gradient computation.
  //  is_training: Set to true for training, false for inference.
  virtual bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<float>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<float>& side_input, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* reserve_space_1,
      DeviceMemory<float>* reserve_space_2, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Performs a half-precision forwards batch normalization operation onto the
  // stream. See DoBatchNormalizationForward above for argument details.
  virtual bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::half>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<Eigen::half>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* reserve_space_1,
      DeviceMemory<float>* reserve_space_2, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Performs a bfloat16 forward batch normalization operation onto the
  // stream. See DoBatchNormalizationForward above for argument details.
  virtual bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::bfloat16>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<Eigen::bfloat16>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::bfloat16>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* reserve_space_1,
      DeviceMemory<float>* reserve_space_2, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Performs a single-precision backward batch normalization gradient
  // computation operation onto the stream.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the batch normalization
  //    gradient computation operation should be enqueued onto.
  //  y_backprop: gradient with regard to output y.
  //  x: input data.
  //  scale: scaling parameters.
  //  inv_var: 1/sqrt(epsilon + variance) of x.
  //  x_desc: dimensions of the input data, which is the same as the dimensions
  //    of the output.
  //  scale_offset_desc: dimensions of scale and offset.
  //  epsilon: a small floating point number added to the variance of x.
  //  x_backprop: gradient with respect to input x.
  //  scale_backprop: gradient with respect to scale.
  //  offset_backprop: gradient with respect to offset.
  virtual bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<float>& y_backprop,
      const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& inv_var, const DeviceMemory<float>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<float>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Performs a half-precision backward batch normalization gradient computation
  // operation onto the stream. See DoBatchNormalizationBackward above for
  // argument details.
  virtual bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
      const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& inv_var, const DeviceMemory<Eigen::half>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::half>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<Eigen::half>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Performs a bfloat16 backward batch normalization gradient computation
  // operation onto the stream. See DoBatchNormalizationBackward above for
  // argument details.
  virtual bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::bfloat16>& y_backprop,
      const DeviceMemory<Eigen::bfloat16>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& inv_var,
      const DeviceMemory<Eigen::bfloat16>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::bfloat16>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<Eigen::bfloat16>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Enqueues a fused convolution operation onto the stream.
  // We provide several variants with different types for inputs, biases and
  // scaling parameters.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'convolve' operation
  //    should be enqueued onto.
  //  conv_input_descriptor: dimensions of the convolution input layer.
  //  conv_input_data: un-owned device memory region which contains the
  //    convolution input.
  //  conv_input_scale: a floating point scale to multiply with each element
  //    of conv_input_data.
  //  filter_descriptor: dimensions of the convolution filter.
  //  filter_data: un-owned device memory region which contains the
  //    convolution filter weights.
  //  convolution_descriptor: stride of the convolution filter.
  //  biases: un-owned device memory region containing biases to add to the
  //    input.
  //  activation_mode: Type of activation to perform.
  //  side_input_data: un-owned device memory region which contains optional
  //    side input data. If 'side_input_scale' is non-zero, then this must
  //    point to data in the tensor shape specified by output_shape.
  //    It will be scaled by 'side_input_scale' and added to the convolution
  //    result and bias prior to applying the activation function.
  //  side_input_scale: a floating point scale to multiply with each element
  //    of side_input_data.
  //  output_descriptor: dimensions of the output layer.
  //  output_data: un-owned device memory region in which to place the
  //    convolution result.
  //  scratch_allocator: un-owned, may-be-null object that may allocate scratch
  //    space in order to speed up the convolution operation.
  //  algorithm_config: specifies which algorithm should be used for the
  //    operation.
  //  output_profile_result: the output profile result for this call. The
  //    profiling is only enabled when this is not nullptr.
  //
  // conv_input_descriptor, filter_descriptor, convolution_descriptor and
  // output_descriptor together specify exactly how the convolution is aligned
  // with the input data:
  //
  // * (input dimensions - filter size + 1) / filter stride == output dimensions
  //   corresponds to dist_belief padding = VALID, i.e. the input is not padded.
  // * input dimensions / filter stride == output dimensions
  //   corresponds to dist_belief padding = SAME, i.e. input and output are the
  //   same size - this requires padding the input.
  // * (input dimensions + filter size - 1) / filter stride == output dimensions
  //   corresponds to dist_belief padding = FULL, i.e. the output is sized so
  //   that if the inverse of the filter is applied to the output in VALID mode
  //   the result is the same size as the input - this requires even more
  //   padding of the input.
  virtual tsl::Status DoFusedConvolve(
      Stream* stream, DataType input_type, DataType side_input_type,
      DataType bias_type, DataType output_type,
      const dnn::BatchDescriptor& conv_input_descriptor,
      DeviceMemoryBase conv_input_data, double conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      DeviceMemoryBase side_input_data, double side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor, DeviceMemoryBase biases,
      dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) {
    return tsl::errors::Unimplemented(
        "DnnSupport::DoFusedConvolve not implemented on this platform.");
  }

  template <typename ElementType, typename OutputType>
  tsl::Status PrepareForConvolution(
      ConvolutionKind kind, Stream* stream,
      const BatchDescriptor& batch_descriptor,
      DeviceMemory<ElementType> input_data,
      const FilterDescriptor& filter_descriptor,
      DeviceMemory<ElementType> filter_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<OutputType> output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const AlgorithmConfig& algorithm_config,
      ScratchAllocator* scratch_allocator, AlgorithmDesc* algorithm_desc,
      DeviceMemory<uint8_t>* scratch_memory) {
    return DoPrepareForConvolution(
        kind, ToDataType<ElementType>::value, stream, batch_descriptor,
        input_data, filter_descriptor, filter_data, output_descriptor,
        output_data, convolution_descriptor, algorithm_config,
        scratch_allocator, algorithm_desc, scratch_memory);
  }

  // cuDNN-specific input transformation that allows running int8x32
  // convolutions faster using Tensor Core IMMA instruction.
  virtual tsl::Status CudnnReorderConvolutionFilterAndBias(
      Stream* stream, const FilterDescriptor& filter_descriptor,
      const DeviceMemory<int8_t>& filter_input,
      DeviceMemory<int8_t>* filter_output,
      std::optional<const DeviceMemory<float>> bias_input,
      std::optional<DeviceMemory<float>> bias_output) {
    return tsl::errors::Unimplemented(
        "DnnSupport::CudnnReorderConvolutionFilterAndBias is specific to CUDA "
        "convolution implementation.");
  }

  // Enqueues a single-precision convolution operation onto the stream.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'convolve' operation
  //    should be enqueued onto.
  //  input_descriptor: dimensions of the input layer.
  //  input_data: un-owned device memory region which contains the
  //    convolution input.
  //  filter_descriptor: dimensions of the convolution filter.
  //  convolution_descriptor: stride of the convolution filter.
  //  output_descriptor: dimensions of the output layer.
  //  output_data: un-owned device memory region in which to place the
  //    convolution result.
  //  algorithm_desc: specifies which algorithm should be used for the
  //    operation.
  //  scratch: un-owned device memory for scratch space in order to speed up
  //    the convolution operation.
  //  output_profile_result: the output profile result for this call. The
  //    profiling is only enabled when this is not nullptr.
  //
  // input_descriptor, filter_descriptor, convolution_descriptor and
  // output_descriptor together specify exactly how the convolution is aligned
  // with the input data:
  //
  // * (input dimensions - filter size + 1) / filter stride == output dimensions
  //   corresponds to dist_belief padding = VALID, i.e. the input is not padded.
  // * input dimensions / filter stride == output dimensions
  //   corresponds to dist_belief padding = SAME, i.e. input and output are the
  //   same size - this requires padding the input.
  // * (input dimensions + filter size - 1) / filter stride == output dimensions
  //   corresponds to dist_belief padding = FULL, i.e. the output is sized so
  //   that if the inverse of the filter is applied to the output in VALID mode
  //   the result is the same size as the input - this requires even more
  //   padding of the input.
  virtual tsl::Status DoConvolve(
      ConvolutionKind kind, DataType element_type, DataType output_type,
      Stream* stream, const BatchDescriptor& input_descriptor,
      DeviceMemoryBase input_data, const FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data, const BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
      ProfileResult* output_profile_result) = 0;

  // Return a list of algorithms supported by the forward convolution pass.
  // cc_major and cc_minor are the compute capabilities of the device.
  virtual bool GetConvolveAlgorithms(
      CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
      std::vector<AlgorithmDesc>* out_algorithms);

  virtual tsl::Status GetConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType input_type, dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, ScratchAllocator* scratch_allocator,
      std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_exec_plans);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
  ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor);

  virtual tsl::Status GetFusedConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType element_type, dnn::DataType bias_type,
      dnn::DataType output_type, double conv_input_scale,
      double side_input_scale, double leakyrelu_alpha, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, dnn::ActivationMode activation_mode,
      std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans);

  virtual tsl::Status GetFusedMatmulRunners(
      bool use_cudnn_frontend, dnn::DataType element_type,
      dnn::DataType bias_type, dnn::DataType output_type, Stream* stream,
      bool trans_a, bool trans_b, uint64_t m, uint64_t n, uint64_t k,
      int64_t lda, int64_t ldb, int64_t ldc,
      dnn::ActivationMode activation_mode, bool use_fallback,
      std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
          out_exec_plans);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
  FusedConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
      double side_input_scale, double leakyrelu_alpha,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::ActivationMode activation_mode);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::FusedMHASoftmaxRunner>>
  FusedMHASoftmaxRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::FusedMHAKind kind,
      const dnn::MatmulTensorDescriptor& bmm1_lhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm1_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm2_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& intermediate_bmm2_lhs_descriptor,
      const dnn::TensorDescriptor& output_descriptor,
      std::optional<double> dropout_rate, std::optional<int64_t> seed);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::FusedMHAMaskRunner>>
  FusedMHAScaleMaskSoftmaxRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::FusedMHAKind kind,
      const dnn::MatmulTensorDescriptor& bmm1_lhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm1_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm2_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& intermediate_bmm2_lhs_descriptor,
      const dnn::TensorDescriptor& output_descriptor,
      const dnn::TensorDescriptor& mask_descriptor, double scale,
      std::optional<double> dropout_rate, std::optional<int64_t> seed);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::FusedMHABiasMaskRunner>>
  FusedMHAScaleBiasMaskSoftmaxRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::FusedMHAKind kind,
      const dnn::MatmulTensorDescriptor& bmm1_lhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm1_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm2_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& intermediate_bmm2_lhs_descriptor,
      const dnn::TensorDescriptor& output_descriptor,
      const dnn::TensorDescriptor& mask_descriptor,
      const dnn::TensorDescriptor& bias_descriptor, double scale,
      std::optional<double> dropout_rate, std::optional<int64_t> seed);

  virtual tsl::StatusOr<std::unique_ptr<const dnn::FusedMHABiasRunner>>
  FusedMHAScaleBiasSoftmaxRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::FusedMHAKind kind,
      const dnn::MatmulTensorDescriptor& bmm1_lhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm1_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& bmm2_rhs_descriptor,
      const dnn::MatmulTensorDescriptor& intermediate_bmm2_lhs_descriptor,
      const dnn::TensorDescriptor& output_descriptor,
      const dnn::TensorDescriptor& bias_descriptor, double scale,
      std::optional<double> dropout_rate, std::optional<int64_t> seed);

  virtual bool GetMIOpenConvolveAlgorithms(
      dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      ScratchAllocator* scratch_allocator,
      std::vector<ProfileResult>* out_algorithms);

  // Returns a list of supported rnn algorithms.
  virtual bool GetRnnAlgorithms(std::vector<AlgorithmDesc>* out_algorithms);

  // Version of DoConvolve that uses pre-quantized 8 bit coefficients.
  // coefficient_scales specifies the scaling of each column of coefficients:
  // original float coefficient[row * num_columns + column] =
  //     quantized coefficient[row * num_columns + column] *
  //     coefficient_scales[column].
  virtual bool DoConvolveQuantized(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int8_t>& filter_coefficients,
      const DeviceMemory<float>& coefficient_scales,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) = 0;

  // Same as DoConvolveQuantized above, but int8 filter coefficients.
  virtual bool DoConvolveQuantized(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int16>& filter_coefficients,
      const DeviceMemory<float>& coefficient_scales,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) = 0;

  // Variation of the above with the weight matrix split into two matrices.
  // first_weights: Coefficients of the first matrix.
  // second_weights: Coefficients of the second matrix.
  // depth_multiplier: specifies the columns of the first matrix and rows
  // of the second one - first_weights columns = depth_multiplier,
  // second_weights rows = depth_multiplier *
  //                       filter_descriptor.input_feature_map_count().
  // see go/separable for documentation on separable convolutions.
  virtual bool DoSeparableConvolve(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const FilterDescriptor& filter_descriptor, int depth_multiplier,
      const DeviceMemory<float>& first_weights,
      const DeviceMemory<float>& second_weights,
      const ConvolutionDescriptor& convolution_descriptor,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) = 0;

  // Return a list of algorithms supported by the backward convolution pass for
  // data.
  virtual bool GetConvolveBackwardDataAlgorithms(
      CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
      std::vector<AlgorithmDesc>* out_algorithms);

  // Return a list of algorithms supported by the backward convolution pass for
  // filters.
  virtual bool GetConvolveBackwardFilterAlgorithms(
      CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
      std::vector<AlgorithmDesc>* out_algorithms);

  // Fully connects the "nodes" (float values) in input_data with
  // shape input_dimensions to output_data with output_dimensions
  // using provided weights. This is equivalent to computing a matrix
  // product, hence the name MatMul.
  //
  // A BatchDescriptor has four dimensions: batch, y, x, depth. Matrix products
  // happen in two dimensions. To get down to two dimensions, we consider the
  // input y, x and depth dimension as one combined dimension T. For now,
  // assume that the output height and width are 1 and let OD be the output
  // depth.
  //
  // There are three device memory buffers passed in to this
  // function. We can now view all three as matrices:
  //
  //   input_data: A batch x T matrix
  //   weights: A T x OD matrix
  //   output_data: A batch x OD matrix
  //
  // This function then computes the matrix product of input_data and
  // weights and writes the result into output_data.
  //
  // Here the weights buffer is in row major order, i.e. the first OD
  // entries in weights are the first row, the second OD entries in
  // weights are the second row and so on.
  //
  // The case for output width*height > 1 is more complicated. Let K =
  // OY * OX where OY is the output height and OX is the output
  // width. Then weights is divided into K sub-arrays W_i, for
  // i=0,...,k-1, that each represent a T x OD matrix. This function
  // then computes the K matrix multiplications of input_data with
  // each W_i. This creates K matrices with dimensions batch x
  // OD. These K matrices are concatenated horizontally to form one
  // larger matrix with dimensions batch x (K*OD); note that this is
  // not the same as concatenating the bytes of the matrices. The
  // combined matrix can then be interpreted as a tensor with
  // dimensions (batch, OY, OX, OD). If the output tensor format is
  // not kBatchYXDepth, this function would then need to arrange for
  // the output to be in the requested layout, if that is
  // supported. Note that the case K=1 is equivalent to the
  // description above. It is recommended to prefer the case K=1.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'fully connect' operation
  //    should be enqueued onto.
  //  output_data: un-owned device memory region in which to place the
  //    fully connected result.
  virtual bool DoMatMul(Stream* stream, const DeviceMemory<float>& input_data,
                        const DeviceMemory<float>& weights,
                        const dnn::BatchDescriptor& input_dimensions,
                        const dnn::BatchDescriptor& output_dimensions,
                        DeviceMemory<float>* output_data) = 0;

  // Version of DoMatMul that uses pre-quantized 8 bit weights.
  // weight_scales specifies the scaling of each column of weights:
  // original float weight[row * num_columns + column] =
  //     quantized_weight[row * nnum_columns + column] * weight_scales[column].
  virtual bool DoMatMulQuantized(Stream* stream,
                                 const DeviceMemory<float>& input_data,
                                 const DeviceMemory<int8_t>& quantized_weights,
                                 const DeviceMemory<float>& weight_scales,
                                 const dnn::BatchDescriptor& input_dimensions,
                                 const dnn::BatchDescriptor& output_dimensions,
                                 DeviceMemory<float>* output_data) = 0;

  // Version of DoMatMul that uses pre-quantized 16 bit weights.
  // weight_scales specifies the scaling of each column of weights:
  // original float weight[row * num_columns + column] =
  //     quantized_weight[row * nnum_columns + column] * weight_scales[column].
  virtual bool DoMatMulQuantized(Stream* stream,
                                 const DeviceMemory<float>& input_data,
                                 const DeviceMemory<int16>& quantized_weights,
                                 const DeviceMemory<float>& weight_scales,
                                 const dnn::BatchDescriptor& input_dimensions,
                                 const dnn::BatchDescriptor& output_dimensions,
                                 DeviceMemory<float>* output_data) = 0;

  // Adds biases to the feature maps in input_data producing
  // output_data. input_data can equal output_data, but must not
  // partially overlap it.
  //
  // Let K = count() * height() * width() and N = feature_map_count()
  // on dimensions. Then input_value contains K*N values and biases
  // contains N values. We can thus logically consider input_value to
  // contain K vectors of N elements each. This function adds biases
  // to each of those N vectors.
  //
  // TODO(broune): This works differently when width() * height() > 1
  // and the call to ThenBiasAdd() follows a call to ThenMatMul(). In
  // that case there should be width() * height() *
  // feature_map_count() biases, but this is not implemented on all
  // StreamExecutors.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'bias add' operation
  //    should be enqueued onto.
  //  input_data: un-owned device memory region containing the input.
  //  biases: un-owned device memory region containing biases to add to the
  //    input.
  //  dimensions: dimensions of input_data and output_data.
  //  output_data: un-owned device memory region in which to place the result.
  virtual bool DoBiasAdd(Stream* stream, const DeviceMemory<float>& input_data,
                         const DeviceMemory<float>& biases,
                         const dnn::BatchDescriptor& dimensions,
                         DeviceMemory<float>* output_data) = 0;

  // Performs a forward pooling operation on input_data, writing to
  // output_data. See PoolingDescriptor for how to configure the
  // pooling operation.
  //
  // Pooling happens as a window that moves across the Y and X
  // dimensions of input_data, where each position of the window
  // yields one output value. E.g. for max pooling, the computed value
  // is the maximum element in the window. The operation is applied
  // independently to each batch and at each feature map (depth), so
  // that the output depth and feature_map_count are the same as for
  // the input. The output width and height can be different.
  //
  // See PoolingDescriptor for how to configure the pooling operation.
  virtual tsl::Status DoPoolForward(
      DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
      const dnn::BatchDescriptor& output_dimensions,
      DeviceMemoryBase output_data, ScratchAllocator* workspace_allocator) = 0;

  // Performs differentiation of the pooling operation.
  virtual tsl::Status DoPoolBackward(
      DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
      const dnn::BatchDescriptor& output_dimensions,
      DeviceMemoryBase output_data, DeviceMemoryBase input_diff_data,
      DeviceMemoryBase output_diff_data,
      ScratchAllocator* workspace_allocator) = 0;

  // Applies local response normalization to the values from input_data and
  // writes the result to output_data.
  //
  // See comments on NormalizeDescriptor for a description of local response
  // normalization.
  virtual bool DoNormalizeWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
    return false;
  }

  // Performs backpropagation for the normalization operation
  //
  // Given raw data, its corresponding normalized output, and a gradient of some
  // unspecified function with respect to the normalized variables, computes the
  // gradient of that unspecified function with respect to the raw variables.
  //
  // The normalized data input array is expected to match the output that would
  // be obtained by running the raw data input array through the DoNormalize
  // method above.
  //
  // See comments on NormalizeDescriptor for a description of local response
  // normalization.
  virtual bool DoNormalizeBackwardWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceMemory<float>& raw_data,
      const DeviceMemory<float>& normalized_data,
      const DeviceMemory<float>& normalized_variable_gradient,
      DeviceMemory<float>* raw_variable_gradient,
      ScratchAllocator* workspace_allocator) {
    return false;
  }

  // Applies an activation function (see ActivationMode) to all of the values
  // held on the device in 'input_data', whose dimensions are described by
  // 'dimensions'.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'activate' operation
  //    should be enqueued onto.
  //  activation_mode: Type of activation to perform.
  //  input_data: un-owned device memory region which contains the
  //    activate input.
  //  output_data: un-owned device memory region in which to place the
  //    activate result.
  virtual bool DoActivate(Stream* stream, ActivationMode activation_mode,
                          const BatchDescriptor& dimensions,
                          const DeviceMemory<float>& input_data,
                          DeviceMemory<float>* output_data, uint64_t options) {
    return false;
  }

  // Concatenates several layers into one, by concatenating the depth of each
  // layer at matching x and y coordinates.
  // The inputs must all have the same width and height, the output will have
  // the same width and height as the inputs and its depth will be the sum of
  // the input depths.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'depth concatenate'
  // operation should be enqueued onto.
  //  input_dimensions: The dimensions of each input.
  //  input_data: un-owned device memory region which contains the
  //    input data for each input layer.
  //  output_data: un-owned device memory region in which to place the
  //    depth concatenate result.
  virtual bool DoDepthConcatenate(
      Stream* stream, absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float>* const> input_data,
      DeviceMemory<float>* output_data) = 0;

  // Concatenates several layers into one, by concatenating each in the
  // x-dimension or y-dimension, based on a user-specified flag.
  // For x-concatenation, layers are aligned at matching y and depth
  // coordinates, and for y-concatenation, they are aligned at matching x and
  // depth coordinates. The inputs must all have the same depth and batch size.
  // For x-concatenation, the inputs must have the same height (y-size), and the
  // output will have the same depth and height as the inputs and its width (x-
  // size) will be the sum of the input widths.  For y-concatenation, the inputs
  // must have the same width, and the output will have the same depth and width
  // as the inputs, and its height will be the sum of the input heights.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the 'space concatenate'
  //    operation should be enqueued onto.
  //  input_dimensions: the dimensions of each input.
  //  input_data: un-owned device memory region which contains the input data
  //    for each input layer.
  //  output_data: un-owned device memory region in which to place the space
  //    concatenate result.
  //  concat_direction:  either dnn:SpaceConcatenateMode::XDirection or
  //    dnn::SpaceConcatenateMode::YDirection.
  virtual bool DoSpaceConcatenate(
      Stream* stream, absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float>* const> input_data,
      DeviceMemory<float>* output_data,
      dnn::SpaceConcatenateMode concat_direction) {
    return false;
  }

  // Change the layout of the data by shrinking one dimension (or set of
  // dimensions) and growing another dimension (or set of dimensions), while
  // keeping the total number of data elements constant, and maintaining the
  // current data ordering.
  //
  // Currently, the only supported operation is depth into space by a power of
  // 2. E.g. (y, x, z) -> (y*2, x*2, z/4)
  //
  // Note that Reshape may not be a no-op, depending on the platform and which
  // dimensions are being changed.
  //
  // Example: forgetting about batch for the moment, let's take a tensor that's
  // 2x1x8 (y by x by z) and reshape to a tensor that's 4x2x2. The memory layout
  // is row-major order: y,x,z. I.e. z changes the fastest, then x, then y. The
  // elements of the tensor range from 0 to 15. The x,y,z indices are below each
  // element.
  //
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  // y0 y0 y0 y0 y0 y0 y0 y0 y1 y1 y1 y1 y1 y1 y1 y1
  // x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0
  // z0 z1 z2 z3 z4 z5 z6 z7 z0 z1 z2 z3 z4 z5 z6 z7
  //
  // reshape to 4x2x2
  //
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  // y0 y0 y0 y0 y1 y1 y1 y1 y2 y2 y2 y2 y3 y3 y3 y3
  // x0 x0 x1 x1 x0 x0 x1 x1 x0 x0 x1 x1 x0 x0 x1 x1
  // z0 z1 z0 z1 z0 z1 z0 z1 z0 z1 z0 z1 z0 z1 z0 z1
  virtual bool DoReshape(Stream* stream,
                         const dnn::BatchDescriptor& input_dimensions,
                         const DeviceMemory<float>& input_data,
                         const dnn::BatchDescriptor& output_dimensions,
                         DeviceMemory<float>* output_data) {
    return false;
  }

  // Depth to space takes an X by Y image with depth D*M^2 and changes it to an
  // MX x MY image with depth D. Each input location (x,y) with depth D*M^2 in
  // the input image is changed to an MxM contiguous area in the output image,
  // with the values being laid out in the raster order by DepthToSpaceLayout,
  // and will have a new depth of D.
  //
  // Example.
  // M=2, Din =8, Xin=2, Yin=2. Xout=4, Yout=4,  Dout=2
  // DepthHeightWidth layout
  // Values within a 'cell' are at different depths and same x & y.
  // Input:
  // abcdefgh  ijklmnop
  // qrstuvwx  yz012345
  // Output:
  // ae bf im jn
  // cg dh ko lp
  // qu rv y2 z3
  // sw tx 04 15
  //
  // sqrt_depth_reduction: 'M' in the comment above
  virtual bool DoDepthToSpace(Stream* stream,
                              const dnn::BatchDescriptor& input_dimensions,
                              const DeviceMemory<float>& input_data,
                              const DepthToSpaceLayout& depth_to_space_layout,
                              const int& sqrt_depth_reduction,
                              DeviceMemory<float>* output_data) {
    return false;
  }

  // Space to depth is the inverse of depth to space. Space to depth takes each
  // non-overlapping M by M patch (in the X and Y dimensions) with depth D of
  // the input, and transforms it to a 1 by 1 patch with depth D*M^2. If the
  // input has size (MX, MY, D), the output has size (X, Y, D*M^2). The number
  // of data elements is not changed.
  //
  // Example.
  // M=2, Din =2, Xin=4, Yin=4,  Dout=8
  // DepthHeightWidth layout
  // Values within a 'cell' are at different depths and same x & y.
  // Input:
  // ae bf im jn
  // cg dh ko lp
  // qu rv y2 z3
  // sw tx 04 15
  // Output:
  // abcdefgh  ijklmnop
  // qrstuvwx  yz012345
  //
  // sqrt_depth_increase: 'M' in the comment above
  virtual bool DoSpaceToDepth(Stream* stream,
                              const dnn::BatchDescriptor& input_dimensions,
                              const DeviceMemory<float>& input_data,
                              const DepthToSpaceLayout& space_to_depth_layout,
                              const int& sqrt_depth_increase,
                              DeviceMemory<float>* output_data) {
    return false;
  }

  // Computes the specified operation (e.g. addition or multiplication)
  // between corresponding elements in the inputs and stores the result in the
  // output element.
  // The inputs and output must all have the same dimensions, but may have
  // different quantization parameters (min_value and max_value).
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'elementwise operation'
  // should be enqueued onto.
  //  operation: The operation to perform.
  //  input_dimensions: The dimensions of each input.
  //  input_data: un-owned device memory region which contains the
  //    input data for each input layer.
  //  output_dimensions: The dimensions of the output.
  //  output_data: un-owned device memory region in which to place the
  //    operation result.
  virtual bool DoElementwiseOperate(
      Stream* stream, ElementwiseOperation operation,
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float>* const> input_data,
      const dnn::BatchDescriptor& output_dimensions,
      DeviceMemory<float>* output_data) = 0;

  // Computes the specified operation (e.g. addition or multiplication)
  // between corresponding elements in the inputs and stores the result in the
  // output element. Each input is multiplied by a scalar constant and the
  // result is divided by a scalar constant.
  // e.g. To perform Z = 0.9*X + 1.1*Y, set the input multiplicands to 9 and 11
  // and the output divisor to 10.
  // The inputs and output must all have the same dimensions, but may have
  // different quantization parameters (min_value and max_value).
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'elementwise operation'
  // should be enqueued onto.
  //  operation: The operation to perform.
  //  input_multiplicands: Amount to scale each input.
  //  output_divisor: Amount to divide the output.
  //  input_dimensions: The dimensions of each input.
  //  input_data: un-owned device memory region which contains the
  //    input data for each input layer.
  //  output_dimensions: The dimensions of the output.
  //  output_data: un-owned device memory region in which to place the
  //    operation result.
  virtual bool DoElementwiseOperateScaledQuantized(
      Stream* stream, ElementwiseOperation operation,
      absl::Span<const int> input_multiplicands, int output_divisor,
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float>* const> input_data,
      const dnn::BatchDescriptor& output_dimensions,
      DeviceMemory<float>* output_data) {
    return false;
  }

  // Pads the input with zeros in the X and Y dimensions. The feature_map
  // dimension is unchanged.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'elementwise operation'
  // should be enqueued onto.
  //  dimensions: The dimensions of the input.
  //  input_data: un-owned device memory region which contains the
  //    input data for the input layer.
  //  left_pad: Amount to pad the input on the left.
  //  right_pad: Amount to pad the input on the right.
  //  top_pad: Amount to pad the input at the top (low Y).
  //  bottom_pad: Amount to pad the input at the bottom (high Y).
  //  output_data: un-owned device memory region in which to place the
  //    padded result.
  virtual bool DoXYPad(Stream* stream, const dnn::BatchDescriptor& dimensions,
                       const DeviceMemory<float>& input_data, int64_t left_pad,
                       int64_t right_pad, int64_t top_pad, int64_t bottom_pad,
                       DeviceMemory<float>* output_data) = 0;

  // Extracts a slice of the input in the X and Y dimensions. The feature_map
  // dimension is unchanged.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'elementwise operation'
  // should be enqueued onto.
  //  dimensions: The dimensions of the input.
  //  input_data: un-owned device memory region which contains the
  //    input data for the input layer.
  //  left_trim: Amount to cut off the input on the left.
  //  right_trim: Amount to cut off the input on the right.
  //  top_trim: Amount to cut off the input at the top (low y).
  //  bottom_trim: Amount to cut off the input at the bottom (high Y).
  //  output_data: un-owned device memory region in which to place the
  //    padded result.
  virtual bool DoXYSlice(Stream* stream, const dnn::BatchDescriptor& dimensions,
                         const DeviceMemory<float>& input_data,
                         int64_t left_trim, int64_t right_trim,
                         int64_t top_trim, int64_t bottom_trim,
                         DeviceMemory<float>* output_data) = 0;

  // Grows the input tensor by replicating the X and Y dimensions. The batch and
  // depth/feature_map dimensions are unchanged. Currently, the input tensor is
  // limited to X=1 and Y=1.
  //
  // For example, the input has dimensions x=2, y=3, and replicate_x=3,
  // replicate_y=2. The diagonal elements of the output would be: [x0y0, x1y1,
  // x0y2, x1y0, x0y1, x1y2].
  // Here is the example as a picture. input:
  // AB
  // CD
  // EF
  // broadcast result:
  // ABABAB
  // CDCDCD
  // EFEFEF
  // ABABAB
  // CDCDCD
  // EFEFEF
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'elementwise operation'
  // should be enqueued onto.
  //  dimensions: The dimensions of the input.
  //  input_data: un-owned device memory region which contains the
  //    input data for the input layer.
  //  replicate_x: Amount to replicate the input's X dimension.
  //  replicate_y: Amount to replicate the input's Y dimension.
  //  output_data: un-owned device memory region in which to place the
  //    padded result.
  virtual bool DoXYBroadcast(Stream* stream,
                             const dnn::BatchDescriptor& dimensions,
                             const DeviceMemory<float>& input_data,
                             int64_t replicate_x, int64_t replicate_y,
                             DeviceMemory<float>* output_data) {
    return false;
  }

  // Enqueues an asynchronous memcpy of the *quantized* output of a layer (that
  // is, bytes instead of scaled floats) into 'host_dst' if they are available
  // for the underlying DNN implementation. If this quantized output is not
  // available, false is returned, which will place 'stream' into an error
  // state.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'quantized memcpy'
  //    operation should be enqueued onto.
  //  gpu_unquantized_src: the device memory that contains the unquantized data
  //    -- this data should also have a corresponding quantized representation
  //    on the device for this operation to succeed.
  //  mode: Type of quantization of the data to write into host_dst.
  //  host_dst: un-owned host memory region that is mutated in place,
  //    it is clobbered by the values in 'gpu_unquantized_src' when the enqueued
  //    (asynchronous) memcpy operation is performed.
  //  size: size in bytes of the host_dst host memory region.
  virtual bool DoMemcpyD2HQuantized(
      Stream* stream, const DeviceMemory<float>& gpu_unquantized_src,
      QuantizedActivationMode mode, void* host_dst, int64_t size) = 0;

  // Enqueues an asynchronous memcpy of 'host_dst' into the *quantized* input
  // of a layer (that is, bytes instead of scaled floats) if they are supported
  // by the underlying DNN implementation. If this quantized input is not
  // supported, false is returned, which will place 'stream' into an error
  // state.
  //
  // Arguments (all borrowed):
  //  stream: borrowed pointer to the stream that the 'quantized memcpy'
  //    operation should be enqueued onto.
  //  host_src: un-owned host memory region that contains the quantized data.
  //  size: size in bytes of the host_src host memory region.
  //  mode: Type of quantization of the data to read from host_src.
  //  gpu_unquantized_dst: the device memory that is clobbered by the values in
  //    'host_src' when the enqueued (asynchronous) memcpy operation is
  //    performed. -- this data should also have a corresponding quantized
  //    representation on the device for this operation to
  //    succeed.
  virtual bool DoMemcpyH2DQuantized(
      Stream* stream, const void* host_src, int64_t size,
      QuantizedActivationMode mode,
      DeviceMemory<float>* gpu_unquantized_dst) = 0;

  // Create an RNN descriptor based on model shapes and configurations.
  // The caller retains the ownership of the descriptor.
  //
  // Arguments:
  //  num_layers: the number of layers for a RNN model.
  //  hidden_size: the size of the hidden state.
  //  input_size: the size of the input state.
  //  cell_size: the size of the cell state
  //  input_mode: an enum to specify whether a linear transformation is added
  //    after the input state. If input_size is different from hidden_size, this
  //    is required.
  //  direction_mode: an enum to specify whether this model is unidirectional or
  //    bidirectional.
  //  rnn_mode: an enum to specify the type of model to build.
  //  data_type: an enum to specify the data types used in this model.
  //  dropout: the dropout threshold between layers. When it is 0., no dropout
  //    is added.
  //  seed: a seed for initializing the dropout layers.
  //  state_allocator: an memory allocator that will be used to store the state
  //    for dropout layer. The user has to maintain the memory until the model
  //    is no longer in use.
  //  use_padded_io: a bool to specify whether the input is using padded IO.
  virtual tsl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
  createRnnDescriptor(int num_layers, int hidden_size, int input_size,
                      int cell_size, int batch_size,
                      dnn::RnnInputMode input_mode,
                      dnn::RnnDirectionMode direction_mode,
                      dnn::RnnMode rnn_mode, dnn::DataType data_type,
                      const dnn::AlgorithmConfig& algorithm_config,
                      float dropout, uint64_t seed,
                      ScratchAllocator* state_allocator, bool use_padded_io) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "createRnnDescriptor is unimplemented");
  }

  // Create a RNN sequence descriptor that specifies either the input or output
  // sequence. The caller retains the ownership of the returned descriptor.
  //
  // Arguments:
  //  max_seq_length: the max length of the sequences.
  //  batch_size: the size of a minibatch.
  //  data_size: the size of the state.
  //  seq_lengths: the lengths of sequences in a batch.
  //  data_type: an enum to specify the type for the underlying data.
  virtual tsl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size, dnn::DataType data_type) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "createRnnSequenceTensorDescriptor is unimplemented");
  }

  virtual tsl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size,
                                    const absl::Span<const int>& seq_lengths,
                                    bool time_major, dnn::DataType data_type) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "createRnnSequenceTensorDescriptor is unimplemented");
  }

  // Create an RNN state descriptor that specifies the input or hidden state.
  // The caller retains the ownership of the returned descriptor.
  virtual tsl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  createRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "createRnnStateTensorDescriptor is unimplemented");
  }

  // Enqueue a forward operation of the RNN model onto the stream.
  //
  // Arguments:
  //  stream: pointer to the stream where this operation should be enqueued to.
  //  rnn_desc: a RNN descriptor created by createRnnDescriptor.
  //  input_desc: descriptor for the input sequence.
  //  input_data: the device memory region that contains the input data.
  //  input_h_desc: descriptor for the input "h" state.
  //  input_h_data: the device memory region that contains the input "h" data.
  //  input_c_desc: descriptor for the input "c" state.
  //  input_c_data: the device memory region that contains the input "c" data.
  //    This must be specified for LSTM models.
  //  params: the device memory region that contains the parameters used in this
  //    model.
  //  output_desc: descriptor for the output sequence.
  //  output_data: the memory region that stores the output sequence data.
  //  output_h_desc: descriptor for the output "h" state.
  //  output_h_data: the memory region that stores the output "h" data.
  //  output_c_desc: descriptor for the output "c" state.
  //  output_c_data: the memory region that stores the output "c" data. This
  //    must be specified for LSTM models.
  //  is_training: whether this is used in training or inference. That decides
  //    whether respace_space data need to be produced.
  //  reserve_space_allocator: if "is_training" is true, an memory allocator
  //    to create memory that holds the produced reserve_space. The caller is
  //  retains the data and feed it to the backward pass.
  //  workspace_allocator: an allocator to create temporary workspace used in
  //    this kernel. The caller is responsible for retaining the memory long
  //    enough for the lifespan of this operation, and recycles afterwards.
  virtual bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                            const dnn::RnnSequenceTensorDescriptor& input_desc,
                            const DeviceMemory<Eigen::half>& input_data,
                            const DeviceMemory<int>& seq_lengths_data,
                            const dnn::RnnStateTensorDescriptor& input_h_desc,
                            const DeviceMemory<Eigen::half>& input_h_data,
                            const dnn::RnnStateTensorDescriptor& input_c_desc,
                            const DeviceMemory<Eigen::half>& input_c_data,
                            const DeviceMemory<Eigen::half>& params,
                            const dnn::RnnSequenceTensorDescriptor& output_desc,
                            DeviceMemory<Eigen::half>* output_data,
                            const dnn::RnnStateTensorDescriptor& output_h_desc,
                            DeviceMemory<Eigen::half>* output_h_data,
                            const dnn::RnnStateTensorDescriptor& output_c_desc,
                            DeviceMemory<Eigen::half>* output_c_data,
                            bool is_training,
                            ScratchAllocator* reserve_space_allocator,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                            const dnn::RnnSequenceTensorDescriptor& input_desc,
                            const DeviceMemory<float>& input_data,
                            const DeviceMemory<int>& seq_lengths_data,
                            const dnn::RnnStateTensorDescriptor& input_h_desc,
                            const DeviceMemory<float>& input_h_data,
                            const dnn::RnnStateTensorDescriptor& input_c_desc,
                            const DeviceMemory<float>& input_c_data,
                            const DeviceMemory<float>& params,
                            const dnn::RnnSequenceTensorDescriptor& output_desc,
                            DeviceMemory<float>* output_data,
                            const dnn::RnnStateTensorDescriptor& output_h_desc,
                            DeviceMemory<float>* output_h_data,
                            const dnn::RnnStateTensorDescriptor& output_c_desc,
                            DeviceMemory<float>* output_c_data,
                            bool is_training,
                            ScratchAllocator* reserve_space_allocator,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                            const dnn::RnnSequenceTensorDescriptor& input_desc,
                            const DeviceMemory<double>& input_data,
                            const DeviceMemory<int>& seq_lengths_data,
                            const dnn::RnnStateTensorDescriptor& input_h_desc,
                            const DeviceMemory<double>& input_h_data,
                            const dnn::RnnStateTensorDescriptor& input_c_desc,
                            const DeviceMemory<double>& input_c_data,
                            const DeviceMemory<double>& params,
                            const dnn::RnnSequenceTensorDescriptor& output_desc,
                            DeviceMemory<double>* output_data,
                            const dnn::RnnStateTensorDescriptor& output_h_desc,
                            DeviceMemory<double>* output_h_data,
                            const dnn::RnnStateTensorDescriptor& output_c_desc,
                            DeviceMemory<double>* output_c_data,
                            bool is_training,
                            ScratchAllocator* reserve_space_allocator,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) {
    return false;
  }
  // Enqueue a backward operation of the RNN model onto the stream.
  //
  // Arguments:
  //  stream: pointer to the stream where this operation should be enqueued to.
  //  rnn_desc: a RNN descriptor created by createRnnDescriptor.
  //  input_desc: descriptor for the input sequence.
  //  input_data: the device memory region that contains the input data.
  //  input_h_desc: descriptor for the input "h" state.
  //  input_h_data: the device memory region that contains the input "h" data.
  //  input_c_desc: descriptor for the input "c" state.
  //  input_c_data: the device memory region that contains the input "c" data.
  //    This must be specified for LSTM models.
  //  params: the device memory region that contains the parameters used in this
  //    model.
  //  output_desc: descriptor for the output sequence.
  //  output_data: the memory region that stores the output sequence data.
  //  output_h_desc: descriptor for the output "h" state.
  //  output_h_data: the memory region that stores the output "h" data.
  //  output_c_desc: descriptor for the output "c" state.
  //  output_c_data: the memory region that stores the output "c" data. This
  //    must be specified for LSTM models.
  //  output_backprop_data: the device memory region that contains the backprop
  //    to the output sequence.
  //  output_h_backprop_data: the device memory region that contains the
  //    backprop to the output "h" state.
  //  output_c_backprop_data: the device memory region that contains the
  //    backprop to the output "c" state.
  //  input_backprop_data: the device memory region that stores the backprop
  //    to the input sequence.
  //  input_h_backprop_data: the device memory region that stores the backprop
  //    to the input "h" state.
  //  input_c_backprop_data: the device memory region that stores the backprop
  //    to the input "c" state.
  //  params_backprop_data: the device memory region that stores the backprop
  //    to the parameters.
  //  reserve_space_data: the reserve_space data that is produced by the forward
  //    operation. This memory region could be modified by this operation.
  //  workspace_allocator: a memory allocator that creates the temporary
  //    workspace memory used by this operation. The caller is responsible for
  //    keeping the memory alive long enough for this operation, and recylces
  //    afterwards.
  virtual bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<Eigen::half>& input_data,
      const DeviceMemory<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceMemory<Eigen::half>& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceMemory<Eigen::half>& input_c_data,
      const DeviceMemory<Eigen::half>& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      const DeviceMemory<Eigen::half>& output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      const DeviceMemory<Eigen::half>& output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      const DeviceMemory<Eigen::half>& output_c_data,
      const DeviceMemory<Eigen::half>& output_backprop_data,
      const DeviceMemory<Eigen::half>& output_h_backprop_data,
      const DeviceMemory<Eigen::half>& output_c_backprop_data,
      DeviceMemory<Eigen::half>* input_backprop_data,
      DeviceMemory<Eigen::half>* input_h_backprop_data,
      DeviceMemory<Eigen::half>* input_c_backprop_data,
      DeviceMemory<Eigen::half>* params_backprop_data,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<float>& input_data,
      const DeviceMemory<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceMemory<float>& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceMemory<float>& input_c_data,
      const DeviceMemory<float>& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      const DeviceMemory<float>& output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      const DeviceMemory<float>& output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      const DeviceMemory<float>& output_c_data,
      const DeviceMemory<float>& output_backprop_data,
      const DeviceMemory<float>& output_h_backprop_data,
      const DeviceMemory<float>& output_c_backprop_data,
      DeviceMemory<float>* input_backprop_data,
      DeviceMemory<float>* input_h_backprop_data,
      DeviceMemory<float>* input_c_backprop_data,
      DeviceMemory<float>* params_backprop_data,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<double>& input_data,
      const DeviceMemory<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceMemory<double>& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceMemory<double>& input_c_data,
      const DeviceMemory<double>& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      const DeviceMemory<double>& output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      const DeviceMemory<double>& output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      const DeviceMemory<double>& output_c_data,
      const DeviceMemory<double>& output_backprop_data,
      const DeviceMemory<double>& output_h_backprop_data,
      const DeviceMemory<double>& output_c_backprop_data,
      DeviceMemory<double>* input_backprop_data,
      DeviceMemory<double>* input_h_backprop_data,
      DeviceMemory<double>* input_c_backprop_data,
      DeviceMemory<double>* params_backprop_data,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  template <typename ElementType>
  tsl::Status PrepareForCtcLoss(Stream* stream,
                                const RnnStateTensorDescriptor& probs_desc,
                                DeviceMemory<ElementType> probs_data,
                                const RnnStateTensorDescriptor& grads_desc,
                                absl::Span<const int> labels_data,
                                absl::Span<const int> labels_lengths_data,
                                absl::Span<const int> input_lengths_data,
                                ScratchAllocator* workspace_allocator,
                                DeviceMemory<uint8_t>* scratch_memory,
                                int* ctc_loss_algo_id) {
    return DoPrepareForCtcLoss(
        stream, ToDataType<ElementType>::value, probs_desc, grads_desc,
        labels_data, labels_lengths_data, input_lengths_data,
        workspace_allocator, scratch_memory, ctc_loss_algo_id);
  }

  // Enqueue a CTC Loss operation onto the stream.
  //
  // Arguments:
  //  stream: pointer to the stream where this operation should be enqueued to.
  //  element_type: date type of the input tensors
  //  probs_desc: specifies the shape and the data layout of the input tensor.
  //  probs_data: the device memory region that contains the input tensor.
  //  labels_data: the device memory region that contains the labels_value
  //    tensor.
  //  labels_lengths_data: the device memory region that contains the
  //    labels_lengths tensor
  //  input_lengths_data: the device memory region that contains the seq_lengths
  //    tensor
  //  costs_data: the device memory region that contains the costs tensor.
  //  grads_desc: specifies the shape and the data layout of the grads tensor.
  //  grads_data: the device memory region that contains the grads tensor.
  //  ctc_loss_desc: a CTCLoss descriptor.
  //  workspace_allocator: a memory allocator that creates the temporary
  //    workspace memory used by this operation. The caller is responsible for
  //    keeping the memory alive long enough for this operation, and recylces
  //    afterwards.
  virtual tsl::Status DoCtcLoss(
      Stream* stream, dnn::DataType element_type,
      const RnnStateTensorDescriptor& probs_desc,
      const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
      const RnnStateTensorDescriptor& grads_desc, DeviceMemoryBase grads_data,
      DeviceMemory<uint8_t> scratch_memory, int ctc_loss_algo_id);

  template <typename ElementType>
  bool DoCtcLoss(Stream* stream,
                 const dnn::RnnStateTensorDescriptor& probs_desc,
                 const DeviceMemory<ElementType>& probs_data,
                 absl::Span<const int> labels_data,
                 absl::Span<const int> labels_lengths_data,
                 absl::Span<const int> input_lengths_data,
                 DeviceMemory<ElementType>* costs_data,
                 const dnn::RnnStateTensorDescriptor& grads_desc,
                 DeviceMemory<ElementType>* grads_data,
                 DeviceMemory<uint8_t>* scratch_memory, int ctc_loss_algo_id) {
    return IsStatusOk(
        DoCtcLoss(stream, ToDataType<ElementType>::value, probs_desc,
                  probs_data, labels_data, labels_lengths_data,
                  input_lengths_data, *costs_data, grads_desc, *grads_data,
                  *scratch_memory, ctc_loss_algo_id),
        false);
  }

  // Transforms a tensor into another tensor with a different layout and/or data
  // type.
  //
  // Arguments:
  //  stream: pointer to the stream where this operation should be enqueued to.
  //  input_desc: specifies the shape and the data layout of the input tensor.
  //  input_type: the data type of the input tensor.
  //  input_data: the device memory region that contains the input tensor.
  //  output_desc: specifies the shape and the data layout of the output tensor.
  //  output_type: the data type of the output tensor.
  //  scale: an element-wise scaling factor to apply.
  //  output_data: the device memory region that contains the output tensor.
  virtual bool DoTransformTensor(Stream* stream,
                                 const dnn::BatchDescriptor& input_desc,
                                 dnn::DataType input_type,
                                 const DeviceMemoryBase& input_data,
                                 const dnn::BatchDescriptor& output_desc,
                                 dnn::DataType output_type, float scale,
                                 DeviceMemoryBase* output_data) {
    return false;
  }

  // Enqueues a fused convolution+bias+activation operation onto the stream.
  //
  // Arguments (all borrowed):
  //
  //  stream: borrowed pointer to the stream that the 'fusion' operation should
  //  be enqueued onto.
  //
  //  conv_input_descriptor: dimensions of the convolution input layer.
  //  conv_input_data: device memory which contains the convolution input.
  //
  //  filter_descriptor: dimensions of the convolution filter.
  //  filter_data: device memory which contains the convolution filter weights.
  //
  //  convolution_descriptor: stride of the convolution filter.
  //
  //  bias_descriptor: dimensions of the bias layer
  //  biases: device memory region containing biases to add to the convolution
  //  output
  //
  //  activation_mode: Type of activation to perform.
  //
  //  output_descriptor: dimensions of the output layer.
  //  output_data: device memory region in which to place the fusion result.
  //
  //  output_profile_result: the output profile result for this call.
  //         The profiling is only enabled when this is not nullptr.
  //
  virtual bool DoFusedConvolutionBiasActivation(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<float>& conv_input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<float>& bias_data, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // Enqueues a fused batchnorm+activation (inference) operation onto the
  // stream.
  //
  // Arguments (all borrowed):
  //
  //  stream: borrowed pointer to the stream that the 'fusion' operation should
  //  be enqueued onto.
  //
  //  x_descriptor: dimensions of the batchnorm input layer.
  //  x_data: device memory which contains the batchnorm input.
  //
  //  scale_offset_mean_variance_descriptor:
  //      dimensions of the scale/offset/mean/variance tensor.
  //  scale_data: device memory which contains the scale input.
  //  offset_data: device memory which contains the offset input.
  //  mean_data: device memory which contains the mean input.
  //  variance_data: device memory which contains the variance input.
  //  epsilon : the epsilon value to use in batchnorm calculation
  //
  //  activation_mode: Type of activation to perform.
  //
  //  y_data: device memory region in which to place the fusion result.
  //
  //  output_profile_result: the output profile result for this call.
  //         The profiling is only enabled when this is not nullptr.
  //
  virtual bool DoFusedBatchNormActivationInference(
      Stream* stream, const dnn::BatchDescriptor& x_descriptor,
      const DeviceMemory<float>& x_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data,
      const DeviceMemory<float>& mean_data,
      const DeviceMemory<float>& variance_data, double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* y_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoFusedBatchNormActivationInference(
      Stream* stream, const dnn::BatchDescriptor& x_descriptor,
      const DeviceMemory<Eigen::half>& x_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data,
      const DeviceMemory<float>& mean_data,
      const DeviceMemory<float>& variance_data, double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // Enqueues a fused batchnorm+activation (training-fwd) operation onto the
  // stream.
  //
  // Arguments (all borrowed):
  //
  //  stream: borrowed pointer to the stream that the 'fusion' operation should
  //  be enqueued onto.
  //
  //  x_descriptor: dimensions of the batchnorm input layer.
  //  x_data: device memory which contains the batchnorm input.
  //
  //  scale_offset_mean_variance_descriptor:
  //      dimensions of the scale/offset/mean/variance tensor.
  //  scale_data: device memory which contains the scale input.
  //  offset_data: device memory which contains the offset input.
  //  epsilon : the epsilon value to use in batchnorm calculation
  //
  //  activation_mode: Type of activation to perform.
  //
  //  y_data: device memory region in which to place the fusion result.
  //  batch_mean_data: device memory in which to place the batch mean output.
  //  batch_var_data: device memory in which to place the batch variance output.
  //  saved_mean_data: device memory in which to save the mean for bwd pass.
  //  saved_var_data: device memory in which to save the variance for bwd pass.
  //
  //  output_profile_result: the output profile result for this call.
  //         The profiling is only enabled when this is not nullptr.
  //
  virtual bool DoFusedBatchNormActivationForward(
      Stream* stream, const dnn::BatchDescriptor& x_descriptor,
      const DeviceMemory<float>& x_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data, double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* y_data,
      DeviceMemory<float>* batch_mean_data, DeviceMemory<float>* batch_var_data,
      DeviceMemory<float>* saved_mean_data, DeviceMemory<float>* saved_var_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoFusedBatchNormActivationForward(
      Stream* stream, const dnn::BatchDescriptor& x_descriptor,
      const DeviceMemory<Eigen::half>& x_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data, double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y_data,
      DeviceMemory<float>* batch_mean_data, DeviceMemory<float>* batch_var_data,
      DeviceMemory<float>* saved_mean_data, DeviceMemory<float>* saved_var_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // Enqueues a fused batchnorm+activation (training-bwd) operation onto the
  // stream.
  //
  // Arguments (all borrowed):
  //
  //  stream: borrowed pointer to the stream that the 'fusion' operation should
  //  be enqueued onto.
  //
  //  y_act_backprop_descriptor: dimensions of the backprop input from the
  //  previous layer. y_act_backprop_data: device memory which contains the
  //  backprop input.
  //
  //  y_act_data: device memory which contains the actv-fwd output data.
  //
  //  activation_mode: actv-fwd type.
  //
  //  scale_offset_mean_variance_descriptor:
  //      dimensions of the scale/offset/mean/variance tensor.
  //  scale_data: device memory which contains the scale input.
  //  offset_data: device memory which contains the offset input.
  //  saved_mean_data: device memory which contains the saved mean from fwd
  //  pass. saved_var_data: device memory which contains the saved variance from
  //  fwd pass.
  //
  //  x_bn_backprop_data: device memory region in which to place the backprop
  //  data from this layer scale_backprop_data: device memory in which to place
  //  the scale backprop output. offset_backprop_data: device memory in which to
  //  place the offset backprop output.
  //
  //  output_profile_result: the output profile result for this call.
  //         The profiling is only enabled when this is not nullptr.
  //
  virtual bool DoFusedBatchNormActivationBackward(
      Stream* stream, const dnn::BatchDescriptor& y_act_backprop_descriptor,
      const DeviceMemory<float>& y_act_backprop_data,
      const DeviceMemory<float>& y_act_data,
      dnn::ActivationMode activation_mode, const DeviceMemory<float>& x_bn_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data,
      const DeviceMemory<float>& saved_mean_data,
      const DeviceMemory<float>& saved_var_data,
      DeviceMemory<float>* x_bn_backprop_data,
      DeviceMemory<float>* scale_backprop_data,
      DeviceMemory<float>* offset_backprop_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoFusedBatchNormActivationBackward(
      Stream* stream, const dnn::BatchDescriptor& y_act_backprop_descriptor,
      const DeviceMemory<Eigen::half>& y_act_backprop_data,
      const DeviceMemory<Eigen::half>& y_act_data,
      dnn::ActivationMode activation_mode,
      const DeviceMemory<Eigen::half>& x_bn_data,
      const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
      const DeviceMemory<float>& scale_data,
      const DeviceMemory<float>& offset_data,
      const DeviceMemory<float>& saved_mean_data,
      const DeviceMemory<float>& saved_var_data,
      DeviceMemory<Eigen::half>* x_bn_backprop_data,
      DeviceMemory<float>* scale_backprop_data,
      DeviceMemory<float>* offset_backprop_data,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // Notifies that a stream is being destroyed and should be invalidated from
  // any internal caching.  This exists to allow the CUDA implementation to
  // avoid redundant cudnnSetStream calls without risking problems when a stream
  // is destroyed and a new stream later created in the same memory.
  virtual void NotifyStreamDestroyed(Stream* stream) {}

 protected:
  // Returns whether status is 'ok', and potentially logs the error.
  static bool IsStatusOk(const tsl::Status& status, bool report_error);

 private:
  virtual tsl::Status DoPrepareForConvolution(
      ConvolutionKind kind, DataType element_type, Stream* stream,
      const BatchDescriptor& batch_descriptor, DeviceMemoryBase input_data,
      const FilterDescriptor& filter_descriptor, DeviceMemoryBase filter_data,
      const BatchDescriptor& output_descriptor, DeviceMemoryBase output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const AlgorithmConfig& algorithm_config,
      ScratchAllocator* scratch_allocator, AlgorithmDesc* algorithm_desc,
      DeviceMemory<uint8_t>* scratch_memory) {
    *algorithm_desc = {};
    *scratch_memory = {};
    return ::tsl::OkStatus();
  }

  virtual tsl::Status DoPrepareForCtcLoss(
      Stream* stream, DataType element_type,
      const RnnStateTensorDescriptor& probs_desc,
      const RnnStateTensorDescriptor& grads_desc,
      absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data,
      ScratchAllocator* scratch_allocator,
      DeviceMemory<uint8_t>* scratch_memory, int* ctc_loss_algo_id) {
    *scratch_memory = {};
    return ::tsl::OkStatus();
  }

  SE_DISALLOW_COPY_AND_ASSIGN(DnnSupport);
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DNN_H_
