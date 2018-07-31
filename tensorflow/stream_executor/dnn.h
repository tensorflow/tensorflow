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

#ifndef TENSORFLOW_STREAM_EXECUTOR_DNN_H_
#define TENSORFLOW_STREAM_EXECUTOR_DNN_H_

#include <functional>
#include <limits>
#include <memory>
#include <tuple>

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/array_slice.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace Eigen {
struct half;
}  // namespace Eigen

namespace stream_executor {

class HostBuffer;
class Stream;
class ScratchAllocator;

namespace dnn {

// Describes how an input or output layer's data is formatted.
// Specify int64 so there's no padding in BatchDescriptor.
enum class DataLayout : int64 {
  kYXDepthBatch = 0,  // Same as dist_belief::DF_DEPTH_MAJOR.
  kYXBatchDepth,      // Same as dist_belief::DF_BATCH_MAJOR.
  kBatchYXDepth,      // Same as run_brain output, and tensorflow's layout.
  kBatchDepthYX,      // cuDNN's NCHW layout, data laid out as image, feature
                      // maps, rows, columns.
  kBatchDepthYX4,     // cuDNN's NCHW_VECT_C layout, data laid out the same as
                      // kBatchDepthYX but each element is a vector of 4 feature
                      // maps.
};

// Specifies an index to use when accessing specific spatial dimensions.
enum class DimIndex : int {
  X = 0,
  Y = 1,
  Z = 2,
};

// Helper functions to make methods more readable.
inline int64 GetDim(const std::vector<int64>& data, DimIndex dim) {
  return data.rbegin()[static_cast<int64>(dim)];
}

inline void SetDim(std::vector<int64>* data, DimIndex dim, int64 value) {
  data->rbegin()[static_cast<int64>(dim)] = value;
}

// Returns a string representation of the given data layout.
string DataLayoutString(DataLayout layout);

// Specifies a quantization for activations in a given BatchDescriptor.
enum class QuantizedActivationMode {
  k8Bit = 1,
  k16Bit = 2,
  k32Bit = 4,
};

// Specifies the data type used by an operation.
enum class DataType {
  kFloat = 0,
  kDouble = 1,
  kHalf = 2,
  kInt8 = 3,
};

// A helper class to convert C/C++ types to the proper enums.
template <typename T>
struct ToDataType;
template <>
struct ToDataType<float> {
  static constexpr DataType value = DataType::kFloat;
};
template <>
struct ToDataType<double> {
  static constexpr DataType value = DataType::kDouble;
};
template <>
struct ToDataType<Eigen::half> {
  static constexpr DataType value = DataType::kHalf;
};
template <>
struct ToDataType<int8> {
  static constexpr DataType value = DataType::kInt8;
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
// In DepthToSpace, the D*M² values are read in and then, for DepthHeightWidth,
// written out to the output patch, by varying first width, then height, then
// depth. In C array format, it looks like [depth][height][width]. See
// DepthToSpace comment for more information.
enum class DepthToSpaceLayout { DepthHeightWidth };

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
    int64 offset;
    int64 size;
  };
  typedef std::vector<ParamsRegion> ParamsRegions;
  virtual ~RnnDescriptor() {}
  virtual int64 ParamsSizeInBytes() const { return -1; }
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
string QuantizedActivationModeString(QuantizedActivationMode mode);

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

  string ToString() const;
  string ToShortString() const;

  // Accessors.
  int64 count() const { return count_; }
  int64 feature_map_count() const { return feature_map_count_; }
  int64 height() const { return GetDim(spatial_size_, DimIndex::Y); }
  int64 width() const { return GetDim(spatial_size_, DimIndex::X); }
  int64 spatial_dim(DimIndex dim) const { return GetDim(spatial_size_, dim); }
  int ndims() const { return ndims_; }
  float value_max() const { return value_max_; }
  float value_min() const { return value_min_; }
  DataLayout layout() const { return layout_; }
  QuantizedActivationMode quantized_activation_mode() const {
    return quantized_activation_mode_;
  }
  // Full dimensions of the underlying data, ordered according to a specific
  // layout.
  std::vector<int64> full_dims(const DataLayout& layout) const;

  // Full strides of the underlying data, ordered according to a specific
  // layout.
  std::vector<int64> full_strides(const DataLayout& layout) const;

  // Named-argument helpers for avoiding user error during construction.
  BatchDescriptor& set_count(int64 value) {
    count_ = value;
    return *this;
  }
  BatchDescriptor& set_feature_map_count(int64 value) {
    feature_map_count_ = value;
    return *this;
  }
  BatchDescriptor& set_height(int64 value) {
    SetDim(&spatial_size_, DimIndex::Y, value);
    return *this;
  }
  BatchDescriptor& set_width(int64 value) {
    SetDim(&spatial_size_, DimIndex::X, value);
    return *this;
  }
  BatchDescriptor& set_spatial_dim(DimIndex dim, int64 value) {
    SetDim(&spatial_size_, dim, value);
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
    layout_ = layout;
    return *this;
  }
  BatchDescriptor& set_quantized_activation_mode(
      QuantizedActivationMode quantized_activation_mode) {
    quantized_activation_mode_ = quantized_activation_mode;
    return *this;
  }

  // Return the number of nodes in a single feature map.
  int64 NodesPerFeatureMap() const;

  // Return the number of nodes across all feature maps. Note that this is not
  // affected by the batch count.
  int64 NodesAcrossFeatureMaps() const;

  // Returns the number of elements (e.g. RGB pixel values) required to hold a
  // given batch descriptor, given a no-padding assumption. Note that this is
  // affected by the batch count.
  int64 ElementCount() const;

  // Return the number of weights required to fully connect a layer with
  // dimensions given by the 'input' descriptor with a layer with dimensions
  // given by the 'output' descriptor.
  static int64 FullyConnectedWeightCount(const BatchDescriptor& input,
                                         const BatchDescriptor& output);

  // Return the number of biases required to fully connect to an output layer
  // with dimensions given the 'output' descriptor.
  static int64 FullyConnectedBiasCount(const BatchDescriptor& output);

  // Return a BatchDescriptor for the output of a depth concatenation
  // with the given input descriptors. The inputs should have the same
  // dimensions, except possibly for feature_map_count(), though this
  // function does not verify that.
  static BatchDescriptor DepthConcatenateOutputDescriptor(
      port::ArraySlice<dnn::BatchDescriptor> inputs);

 private:
  int64 count_;
  int64 feature_map_count_;
  // Stored as: ..., y, x.
  std::vector<int64> spatial_size_;
  float value_max_;
  float value_min_;
  DataLayout layout_;
  int ndims_;
  QuantizedActivationMode quantized_activation_mode_;
};

// Describes how a filter is laid out in the memory.
// Specify int64 so there's no padding in FilterDescriptor.
enum class FilterLayout : int64 {
  kOutputInputYX = 0,  // cuDNN's default filter layout, laid out as:
                       // (major) output feature maps >> input feature maps >>
                       // rows >> columns (minor).
  kOutputYXInput,      // major to minor:
                       //   (output features, row, columns, input features)
  kOutputInputYX4,  // laid out the same as kOutputInputYX but each element is a
                    // vector of 4 feature maps.
  kInputYXOutput,   // Same as dist_belief's default filter layout.
  kYXInputOutput,   // Same as tensorflow's default filter layout.
};

// Returns a string representation of the given filter layout.
string FilterLayoutString(FilterLayout layout);

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
  FilterDescriptor& set_output_feature_map_count(int64 value) {
    output_feature_map_count_ = value;
    return *this;
  }
  FilterDescriptor& set_input_feature_map_count(int64 value) {
    input_feature_map_count_ = value;
    return *this;
  }
  FilterDescriptor& set_input_filter_height(int64 value) {
    SetDim(&input_filter_dims_, DimIndex::Y, value);
    return *this;
  }
  FilterDescriptor& set_input_filter_width(int64 value) {
    SetDim(&input_filter_dims_, DimIndex::X, value);
    return *this;
  }
  FilterDescriptor& set_layout(FilterLayout layout) {
    layout_ = layout;
    return *this;
  }
  FilterDescriptor& set_spatial_dim(DimIndex dim, int64 value) {
    SetDim(&input_filter_dims_, dim, value);
    return *this;
  }
  int ndims() const { return ndims_; }

  void CloneFrom(const FilterDescriptor& other);

  string ToString() const;
  string ToShortString() const;

  // Returns the number of weights required as parameters for a convolution
  // using this filter descriptor.
  int64 ComputeWeightCount() const;

  // Returns the number of biases required as parameters for a convolution
  // using this filter descriptor.
  int64 bias_count() const { return output_feature_map_count_; }

  int64 output_feature_map_count() const { return output_feature_map_count_; }
  int64 input_feature_map_count() const { return input_feature_map_count_; }
  int64 input_filter_height() const {
    return GetDim(input_filter_dims_, DimIndex::Y);
  }
  int64 input_filter_width() const {
    return GetDim(input_filter_dims_, DimIndex::X);
  }
  int64 input_filter_dim(DimIndex dim) const {
    return GetDim(input_filter_dims_, dim);
  }

  FilterLayout layout() const { return layout_; }
  std::vector<int64> input_filter_dims() const { return input_filter_dims_; }

 private:
  int64 output_feature_map_count_;
  int64 input_feature_map_count_;
  // Stored as: ..., y, x.
  std::vector<int64> input_filter_dims_;
  int ndims_;
  FilterLayout layout_;
};

// Describes how padding should be aligned when the total number of pad
// elements is odd.
enum class PadAlignment : int64 {
  kDefault = 0,        // default padding for the device.
  kCudnnPadding,       // cuDNN padding - prefer to pad at the start.
  kTensorFlowPadding,  // TensorFlow padding - prefer to pad at the end.
};

// Returns a string representation of the given padding alignment.
string PadAlignmentString(PadAlignment alignment);

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
class ConvolutionDescriptor {
 public:
  // By default construction, there is no zero-padding and the filter stride is
  // 1x1 (centering the filter on every cell in the input layer's
  // width-by-height area).
  ConvolutionDescriptor();
  explicit ConvolutionDescriptor(int ndims);
  ~ConvolutionDescriptor();

  string ToString() const;
  string ToShortString() const;

  ConvolutionDescriptor& set_zero_padding_height(int64 value) {
    SetDim(&zero_padding_, DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_zero_padding_width(int64 value) {
    SetDim(&zero_padding_, DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_zero_padding(DimIndex dim, int64 value) {
    SetDim(&zero_padding_, dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_vertical_filter_stride(int64 value) {
    SetDim(&filter_strides_, DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_horizontal_filter_stride(int64 value) {
    SetDim(&filter_strides_, DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_filter_stride(DimIndex dim, int64 value) {
    SetDim(&filter_strides_, dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_vertical_dilation_rate(int64 value) {
    SetDim(&dilation_rates_, DimIndex::Y, value);
    return *this;
  }
  ConvolutionDescriptor& set_horizontal_dilation_rate(int64 value) {
    SetDim(&dilation_rates_, DimIndex::X, value);
    return *this;
  }
  ConvolutionDescriptor& set_dilation_rate(DimIndex dim, int64 value) {
    SetDim(&dilation_rates_, dim, value);
    return *this;
  }
  ConvolutionDescriptor& set_pad_alignment(PadAlignment pad_alignment) {
    pad_alignment_ = pad_alignment;
    return *this;
  }
  ConvolutionDescriptor& set_group_count(int group_count) {
    group_count_ = group_count;
    return *this;
  }
  int64 zero_padding_height() const {
    return GetDim(zero_padding_, DimIndex::Y);
  }
  int64 zero_padding_width() const {
    return GetDim(zero_padding_, DimIndex::X);
  }
  int64 vertical_filter_stride() const {
    return GetDim(filter_strides_, DimIndex::Y);
  }
  int64 horizontal_filter_stride() const {
    return GetDim(filter_strides_, DimIndex::X);
  }
  int64 vertical_dilation_rate() const {
    return GetDim(dilation_rates_, DimIndex::Y);
  }
  int64 horizontal_dilation_rate() const {
    return GetDim(dilation_rates_, DimIndex::X);
  }

  int zero_padding(DimIndex dim) const { return GetDim(zero_padding_, dim); }
  int filter_stride(DimIndex dim) const { return GetDim(filter_strides_, dim); }
  int dilation_rate(DimIndex dim) const { return GetDim(dilation_rates_, dim); }
  PadAlignment pad_alignment() const { return pad_alignment_; }
  int group_count() const { return group_count_; }
  int ndims() const { return ndims_; }

  std::vector<int64> strides() const { return filter_strides_; }
  std::vector<int64> dilations() const { return dilation_rates_; }
  std::vector<int64> padding() const { return zero_padding_; }

 private:
  // Stored as: .. y, x.
  std::vector<int64> zero_padding_;
  std::vector<int64> filter_strides_;
  std::vector<int64> dilation_rates_;
  PadAlignment pad_alignment_;
  int group_count_;
  int ndims_;
  // TODO(leary) cudnn provides these fields, but need to characterize what
  // their effect is -- they may be boolean rather than integral.
  // int64 upscale_input_x;
  // int64 upscale_input_y;
};

// A patch of values in the input can be pooled via either a max or an average
// operation.
// Specify int64 so there's no padding in PoolingDescriptor.
enum class PoolingMode : int64 {
  kMaximum,
  kAverage,
};

// Specify the dimension in which to concatenate inputs in space.
// Specify int64 so there's no padding in SpaceConcatenateMode.
enum class SpaceConcatenateMode : int64 {
  XDirection,
  YDirection,
};

// Returns a short name for the pooling mode, e.g. "Avg".
string ShortPoolingModeString(PoolingMode mode);

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
  PoolingDescriptor& set_window_height(int64 value) {
    SetDim(&window_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_window_width(int64 value) {
    SetDim(&window_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_window(DimIndex dim, int64 value) {
    SetDim(&window_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_vertical_padding(int64 value) {
    SetDim(&padding_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_horizontal_padding(int64 value) {
    SetDim(&padding_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_padding(DimIndex dim, int64 value) {
    SetDim(&padding_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_vertical_stride(int64 value) {
    SetDim(&strides_, DimIndex::Y, value);
    return *this;
  }
  PoolingDescriptor& set_horizontal_stride(int64 value) {
    SetDim(&strides_, DimIndex::X, value);
    return *this;
  }
  PoolingDescriptor& set_stride(DimIndex dim, int64 value) {
    SetDim(&strides_, dim, value);
    return *this;
  }
  PoolingDescriptor& set_propagate_nans(bool value) {
    propagate_nans_ = value;
    return *this;
  }

  int ndims() const { return ndims_; }
  void CloneFrom(const PoolingDescriptor& other);

  string ToString() const;
  string ToShortString() const;

  PoolingMode mode() const { return mode_; }
  int64 window_height() const { return GetDim(window_, DimIndex::Y); }
  int64 window_width() const { return GetDim(window_, DimIndex::X); }
  int64 window(DimIndex dim) const { return GetDim(window_, dim); }
  int64 vertical_padding() const { return GetDim(padding_, DimIndex::Y); }
  int64 horizontal_padding() const { return GetDim(padding_, DimIndex::X); }
  int64 padding(DimIndex dim) const { return GetDim(padding_, dim); }
  int64 vertical_stride() const { return GetDim(strides_, DimIndex::Y); }
  int64 horizontal_stride() const { return GetDim(strides_, DimIndex::X); }
  int64 stride(DimIndex dim) const { return GetDim(strides_, dim); }
  std::vector<int64> window() const { return window_; }
  std::vector<int64> padding() const { return padding_; }
  std::vector<int64> strides() const { return strides_; }
  bool propagate_nans() const { return propagate_nans_; }

 private:
  PoolingMode mode_;
  int ndims_;
  bool propagate_nans_;

  // Stored as: ..., y, x.
  std::vector<int64> window_;
  std::vector<int64> padding_;
  std::vector<int64> strides_;
};

// Collects parameters for DNN algorithms
class AlgorithmDesc {
 public:
  typedef int64 Index;
  AlgorithmDesc() : algo_(kDefaultAlgorithm), tensor_ops_enabled_(true) {}
  AlgorithmDesc(Index a, bool use_tensor_ops)
      : algo_(a), tensor_ops_enabled_(use_tensor_ops) {}
  bool is_default() const { return algo_ == kDefaultAlgorithm; }
  bool tensor_ops_enabled() const { return tensor_ops_enabled_; }
  Index algo_id() const { return algo_; }
  bool operator==(const AlgorithmDesc& other) const {
    return this->algo_ == other.algo_ &&
           this->tensor_ops_enabled_ == other.tensor_ops_enabled_;
  }
  uint64 hash() const;

 private:
  enum { kDefaultAlgorithm = -1 };
  Index algo_;
  bool tensor_ops_enabled_;
};

// Describes the result from a perf experiment.
//
// Arguments:
//  algorithm: returns the exact algorithm that was used.
//  elapsed_time_in_ms: returns the measured elapsed time in milliseconds.
class ProfileResult {
 public:
  bool is_valid() const {
    return (!algorithm_.is_default() &&
            elapsed_time_in_ms_ != std::numeric_limits<float>::max());
  }
  AlgorithmDesc algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmDesc val) { algorithm_ = val; }
  float elapsed_time_in_ms() const { return elapsed_time_in_ms_; }
  void set_elapsed_time_in_ms(float val) { elapsed_time_in_ms_ = val; }

 private:
  AlgorithmDesc algorithm_;
  float elapsed_time_in_ms_ = std::numeric_limits<float>::max();
};

// Describes the configuration for the algorithms that will used.
//
// Arguments:
//  algorithm: the primary algorithm that should be used.
//  algorithm_no_scratch: a secondary algorithm that should be used, if the
//    the allocation for the scratch memory fails.
class AlgorithmConfig {
 public:
  AlgorithmConfig() {}
  explicit AlgorithmConfig(AlgorithmDesc algorithm) : algorithm_(algorithm) {}
  AlgorithmConfig(AlgorithmDesc algorithm, AlgorithmDesc algorithm_no_scratch)
      : algorithm_(algorithm), algorithm_no_scratch_(algorithm_no_scratch) {}
  AlgorithmDesc algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmDesc val) { algorithm_ = val; }
  AlgorithmDesc algorithm_no_scratch() const { return algorithm_no_scratch_; }
  void set_algorithm_no_scratch(AlgorithmDesc val) {
    algorithm_no_scratch_ = val;
  }
  bool operator==(const AlgorithmConfig& other) const {
    return this->algorithm_ == other.algorithm_ &&
           this->algorithm_no_scratch_ == other.algorithm_no_scratch_;
  }
  bool operator!=(const AlgorithmConfig& other) const {
    return !(*this == other);
  }
  string ToString() const;

 private:
  AlgorithmDesc algorithm_;
  AlgorithmDesc algorithm_no_scratch_;
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

  NormalizeDescriptor& set_range(int32 range) {
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

  NormalizeDescriptor& set_segment_size(int32 segment_size) {
    segment_size_ = segment_size;
    return *this;
  }

  void CloneFrom(const NormalizeDescriptor& other);

  string ToString() const;
  string ToShortString() const;

  float bias() const { return bias_; }
  int32 range() const { return range_; }
  float alpha() const { return alpha_; }
  float beta() const { return beta_; }
  bool wrap_around() const { return wrap_around_; }
  int32 segment_size() const { return segment_size_; }

 private:
  float bias_;
  int32 range_;
  float alpha_;
  float beta_;
  bool wrap_around_;
  int32 segment_size_;
};

// Describes a kind of non-linearity (threshold-like mathematical function).
enum class ActivationMode {
  kNone,
  kSigmoid,
  // Rectified linear activation: f(x) = x < 0 ? 0 : x
  kRelu,
  // Rectified linear activation, where upper maximum is 6.0.
  kRelu6,
  // Rectified linear activation, where upper maximum specified by
  // BatchDescriptor::value_max().
  kReluX,
  kTanh,
  // Like ReluX, but passes all values in the range [-X,X].
  kBandPass,
};

// Returns a string representation of the given activation mode.
string ActivationModeString(ActivationMode mode);

// Describes the operation that DoElementwiseOperation should perform on its
// inputs.
enum class ElementwiseOperation { kAdd, kMultiply };

string ElementwiseOperationString(ElementwiseOperation op);

// A simple class representing the version of the backing library, to
// workaround the "too perfect forwarding" issue in gcc6+ compilers.
// See PR#16309 and issue #18402 for links discussing the issue.
class VersionInfo {
 public:
  VersionInfo(int major = 0, int minor = 0, int patch = 0)
      : major_(major), minor_(minor), patch_(patch) {}
  int major_version() { return major_; }
  int minor_version() { return minor_; }
  int patch() { return patch_; }
 private:
  int major_;
  int minor_;
  int patch_;
};

// Suite of operations typically used for implementing Deep/Convolutional Neural
// Nets. Note: A false return value of an operation indicates the
// implementation is not available.
class DnnSupport {
 public:
  DnnSupport() {}
  virtual ~DnnSupport() {}

  virtual port::Status Init() = 0;

  // Gets the version of the backing library, as a VersionInfo object.
  virtual port::StatusOr<VersionInfo> GetVersion() {
    return port::UnimplementedError(
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
  //  x_desc: dimensions of the input data, which is the same as the dimensions
  //    of the output.
  //  scale_offset_desc: dimensions of scale and offset.
  //  epsilon: a small floating point number added to the variance of x.
  //  y: output data.
  //  batch_mean: batch mean, to be used to compute the running mean.
  //  batch_variance: batch variance, to be used to compute
  //    the running variance.
  //  reserve_space_1: saved mean, to be reused in the backward gradient
  //    computation.
  //  reserve_space_2: saved inv_var (1/sqrt(epsilon + variance), to be reused
  //    in the backward gradient computation.
  //  is_training: Set to true for training, false for inference.
  //  var_to_inv_var: a function to convert the variance to inverted variance
  //    for cuDNN v4 forward inference.
  //  inv_var_to_var: a function to convert the inverted variance to
  //    variance for cuDNN v4 forward training, to be used for TensorFlow
  //    to calculate the running variance.
  virtual bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<float>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<float>* y, DeviceMemory<float>* batch_mean,
      DeviceMemory<float>* batch_var, DeviceMemory<float>* reserve_space_1,
      DeviceMemory<float>* reserve_space_2, bool is_training,
      std::function<const DeviceMemory<float>&()> var_to_inv_var,
      std::function<void()> inv_var_to_var) {
    return false;
  }

  // Performs a half-precision forwards batch normalization operation onto the
  // stream. See DoBatchNormalizationForward above for argument details.
  virtual bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::half>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<Eigen::half>* y, DeviceMemory<float>* batch_mean,
      DeviceMemory<float>* batch_var, DeviceMemory<float>* reserve_space_1,
      DeviceMemory<float>* reserve_space_2, bool is_training,
      std::function<const DeviceMemory<float>&()> var_to_inv_var,
      std::function<void()> inv_var_to_var) {
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
      const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
      DeviceMemory<float>* offset_backprop) {
    return false;
  }

  // Performs a half-precision backward batch normalization gradient computation
  // operation onto the stream. See DoBatchNormalizationBackward above for
  // argument details.
  virtual bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
      const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<Eigen::half>* x_backprop,
      DeviceMemory<float>* scale_backprop,
      DeviceMemory<float>* offset_backprop) {
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
  virtual bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<double>& conv_input_data, double conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<double>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<double>& side_input_data, double side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<double>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<double>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // This is the float version of DoFusedConvolve.
  virtual bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<float>& conv_input_data, float conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<float>& side_input_data, float side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // This is the Eigen::half version of DoFusedConvolve.
  // The scaling parameters are still floats.
  virtual bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<Eigen::half>& conv_input_data, float conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<Eigen::half>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<Eigen::half>& side_input_data, float side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<Eigen::half>& biases,
      dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half>* output_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  // This is the int8 version of DoFusedConvolve.
  // The bias input and scaling parameters are floats.
  virtual bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<int8>& conv_input_data, float conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int8>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<int8>& side_input_data, float side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<int8>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) {
    return false;
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
  //  scratch_allocator: un-owned, may-be-null object that may allocate scratch
  //    space in order to speed up the convolution operation.
  //  algorithm_config: specifies which algorithm should be used for the
  //    operation.
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
  virtual bool DoConvolve(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Enqueues a double-precision convolution operation onto the stream.
  // See DoConvolve above for argument details.
  virtual bool DoConvolve(
      Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
      const DeviceMemory<double>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<double>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<double>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) = 0;

  // Enqueues a half-precision convolution operation onto the stream.
  // See DoConvolve above for argument details.
  virtual bool DoConvolve(
      Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
      const DeviceMemory<Eigen::half>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<Eigen::half>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half>* output_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Return a list of algorithms supported by the forward convolution pass.
  // cc_major and cc_minor are the compute capabilities of the device.
  virtual bool GetConvolveAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<AlgorithmDesc>* out_algorithms);

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
      const DeviceMemory<int8>& filter_coefficients,
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

  // Enqueues a single-precision backward convolution (for data) operation onto
  // the stream.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the 'convolve' operation
  //    should be enqueued onto.
  //  filter_descriptor: dimensions of the convolution filter.
  //  filter_data: coefficients for the convolution filter.
  //  output_descriptor: dimensions of the output gradients, which is the same
  //    as the dimensions of the output.
  //  backward_output_data: un-owned device memory region which contains the
  //    backprop of the output.
  //  convolution_descriptor: stride of the convolution filter.
  //  input_descriptor: dimensions of the input layer.
  //  backward_input_data: un-owned device memory region in which to place the
  //    backprop of the input.
  //  scratch_allocator: un-owned, may-be-null object that may allocate scratch
  //    space in order to speed up the convolution operation.
  virtual bool DoConvolveBackwardData(
      Stream* stream, const FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const BatchDescriptor& input_descriptor,
      DeviceMemory<float>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Return a list of algorithms supported by the backward convolution pass for
  // data.
  virtual bool GetConvolveBackwardDataAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<AlgorithmDesc>* out_algorithms);

  virtual bool DoConvolveBackwardData(
      Stream* stream, const FilterDescriptor& filter_descriptor,
      const DeviceMemory<double>& filter_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<double> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const BatchDescriptor& input_descriptor,
      DeviceMemory<double>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  virtual bool DoConvolveBackwardData(
      Stream* stream, const FilterDescriptor& filter_descriptor,
      const DeviceMemory<Eigen::half>& filter_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const BatchDescriptor& input_descriptor,
      DeviceMemory<Eigen::half>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Enqueues a single-precision backward convolution (for filter) operation
  // onto the stream.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the 'convolve' operation
  //    should be enqueued onto.
  //  input_descriptor: dimensions of the input layer.
  //  input_data: un-owned device memory region which contains the
  //    convolution input.
  //  output_descriptor: dimensions of the output gradients, which is the same
  //    as the dimensions of the output.
  //  backward_output_data: un-owned device memory region which contains the
  //    backprop of the output.
  //  convolution_descriptor: stride of the convolution filter.
  //  filter_descriptor: dimensions of the convolution filter.
  //  backward_filter_data: un-owned device memory region in which to place the
  //    backprop of the filter.
  //  scratch_allocator: un-owned, may-be-null object that may allocate scratch
  //    space in order to speed up the convolution operation.
  virtual bool DoConvolveBackwardFilter(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const FilterDescriptor& filter_descriptor,
      DeviceMemory<float>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Return a list of algorithms supported by the backward convolution pass for
  // filters.
  virtual bool GetConvolveBackwardFilterAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<AlgorithmDesc>* out_algorithms);

  virtual bool DoConvolveBackwardFilter(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<double>& input_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<double> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const FilterDescriptor& filter_descriptor,
      DeviceMemory<double>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  virtual bool DoConvolveBackwardFilter(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<Eigen::half>& input_data,
      const BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const ConvolutionDescriptor& convolution_descriptor,
      const FilterDescriptor& filter_descriptor,
      DeviceMemory<Eigen::half>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      ProfileResult* output_profile_result) = 0;

  // Enqueues a single-precision backward convolution (for bias) operation onto
  // the stream.
  //
  // Arguments:
  //  stream: borrowed pointer to the stream that the 'convolve' operation
  //    should be enqueued onto.
  //  input_descriptor: dimensions of the input layer.
  //  input_data: un-owned device memory region which contains the
  //    convolution input.
  //  bias_descriptor: dimensions of the bias tensor. Should be the same as the
  //    input dimensions, but with the spatial dimensions set to 1.
  //  backward_filter_data: un-owned device memory region in which to place the
  //    backprop of the bias.
  virtual bool DoConvolveBackwardBias(Stream* stream,
                                      const BatchDescriptor& input_descriptor,
                                      const DeviceMemory<float>& input_data,
                                      const BatchDescriptor& bias_descriptor,
                                      DeviceMemory<float>* backward_bias_data) {
    return false;
  }

  virtual bool DoConvolveBackwardBias(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<double>& input_data,
      const BatchDescriptor& bias_descriptor,
      DeviceMemory<double>* backward_bias_data) {
    return false;
  }

  virtual bool DoConvolveBackwardBias(
      Stream* stream, const BatchDescriptor& input_descriptor,
      const DeviceMemory<Eigen::half>& input_data,
      const BatchDescriptor& bias_descriptor,
      DeviceMemory<Eigen::half>* backward_bias_data) {
    return false;
  }

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
                                 const DeviceMemory<int8>& quantized_weights,
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
  virtual bool DoPoolForward(Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             const DeviceMemory<float>& input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemory<float>* output_data,
                             ScratchAllocator* workspace_allocator) = 0;

  virtual bool DoPoolForward(Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             const DeviceMemory<double>& input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemory<double>* output_data,
                             ScratchAllocator* workspace_allocator) {
    LOG(FATAL) << "DoPoolForward not implemented for double.";
    return false;
  }

  virtual bool DoPoolForward(Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             const DeviceMemory<Eigen::half>& input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemory<Eigen::half>* output_data,
                             ScratchAllocator* workspace_allocator) {
    LOG(FATAL) << "DoPoolForward not implemented for float16.";
    return false;
  }

  // Performs differentiation of the pooling operation.
  virtual bool DoPoolBackward(Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              const DeviceMemory<double>& input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              const DeviceMemory<double>& output_data,
                              const DeviceMemory<double>& input_diff_data,
                              DeviceMemory<double>* output_diff_data,
                              ScratchAllocator* workspace_allocator) {
    LOG(FATAL) << "DoPoolBackward not implemented.";
    return false;
  }

  virtual bool DoPoolBackward(Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              const DeviceMemory<float>& input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              const DeviceMemory<float>& output_data,
                              const DeviceMemory<float>& input_diff_data,
                              DeviceMemory<float>* output_diff_data,
                              ScratchAllocator* workspace_allocator) {
    LOG(FATAL) << "DoPoolBackward not implemented.";
    return false;
  }

  virtual bool DoPoolBackward(Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              const DeviceMemory<Eigen::half>& input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              const DeviceMemory<Eigen::half>& output_data,
                              const DeviceMemory<Eigen::half>& input_diff_data,
                              DeviceMemory<Eigen::half>* output_diff_data,
                              ScratchAllocator* workspace_allocator) {
    LOG(FATAL) << "DoPoolBackward not implemented.";
    return false;
  }

  // Applies local response normalization to the values from
  // input_data and writes the result to output_data. See comments on
  // NormalizeDescriptor for a description of local response
  // normalization.
  virtual bool DoNormalize(Stream* stream,
                           const dnn::NormalizeDescriptor& normalize_descriptor,
                           const DeviceMemory<float>& input_data,
                           DeviceMemory<float>* output_data) = 0;

  // Applies local response normalization to the values from input_data and
  // writes the result to output_data.
  //
  // Similar to DoNormalize, but normalizes across feature maps and allows for
  // specifying the dimensions of the tensor.
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
                          DeviceMemory<float>* output_data, uint64 options) {
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
      Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
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
      Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
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

  // Depth to space takes an X by Y image with depth D*M² and changes it to an
  // MX x MY image with depth D. Each input location (x,y) with depth D*M² in
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
  // the input, and transforms it to a 1 by 1 patch with depth D*M². If the
  // input has size (MX, MY, D), the output has size (X, Y, D*M²). The number of
  // data elements is not changed.
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
      port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
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
      port::ArraySlice<int> input_multiplicands, int output_divisor,
      port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
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
  virtual bool DoXYPad(Stream* stream, const dnn::BatchDescriptor &dimensions,
                       const DeviceMemory<float> &input_data,
                       int64 left_pad, int64 right_pad, int64 top_pad,
                       int64 bottom_pad, DeviceMemory<float> *output_data) = 0;

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
  virtual bool DoXYSlice(Stream* stream, const dnn::BatchDescriptor &dimensions,
                    const DeviceMemory<float> &input_data,
                    int64 left_trim, int64 right_trim, int64 top_trim,
                    int64 bottom_trim, DeviceMemory<float> *output_data) = 0;

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
                             int64 replicate_x, int64 replicate_y,
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
      QuantizedActivationMode mode, void* host_dst, int64 size) = 0;

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
      Stream* stream, const void* host_src, int64 size,
      QuantizedActivationMode mode,
      DeviceMemory<float>* gpu_unquantized_dst) = 0;

  // Enqueues an asynchronous copy of the contents of buffer_src to
  // gpu_unquantized_dst.
  virtual bool DoCopyHostBuffer2Device(
      Stream* stream, HostBuffer* buffer_src,
      DeviceMemory<float>* gpu_unquantized_dst) {
    return false;
  }

  // Enqueues an asynchronous copy of the contents of gpu_unquantized_src to
  // buffer_dst.
  virtual bool DoCopyDevice2HostBuffer(
      Stream* stream, const DeviceMemory<float>& gpu_unquantized_src,
      HostBuffer* buffer_dst) {
    return false;
  }

  // Create an RNN descriptor based on model shapes and configurations.
  // The caller retains the ownership of the descriptor.
  //
  // Arguments:
  //  num_layers: the number of layers for a RNN model.
  //  hidden_size: the size of the hidden state.
  //  input_size: the size of the input state.
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
  virtual port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
  createRnnDescriptor(int num_layers, int hidden_size, int input_size,
                      int batch_size, dnn::RnnInputMode input_mode,
                      dnn::RnnDirectionMode direction_mode,
                      dnn::RnnMode rnn_mode, dnn::DataType data_type,
                      const dnn::AlgorithmConfig& algorithm_config,
                      float dropout, uint64 seed,
                      ScratchAllocator* state_allocator) {
    return port::Status(port::error::UNIMPLEMENTED,
                        "createRnnDescriptor is unimplemented");
  }

  // Create a RNN sequence descriptor that specifies either the input or output
  // sequence. The caller retains the ownership of the returned descriptor.
  //
  // Arguments:
  //  seq_length: the length of the sequence.
  //  batch_size: the size of a minibatch.
  //  data_size: the size of the state.
  //  data_type: an enum to specify the type for the underlying data.
  virtual port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size, dnn::DataType data_type) {
    return port::Status(port::error::UNIMPLEMENTED,
                        "createRnnSequenceTensorDescriptor is unimplemented");
  }

  // Create an RNN state descriptor that specifies the input or hidden state.
  // The caller retains the ownership of the returned descriptor.
  virtual port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  createRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) {
    return port::Status(port::error::UNIMPLEMENTED,
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
      DeviceMemory<uint8>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<float>& input_data,
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
      DeviceMemory<uint8>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
  }

  virtual bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<double>& input_data,
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
      DeviceMemory<uint8>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) {
    return false;
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

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(DnnSupport);
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DNN_H_
