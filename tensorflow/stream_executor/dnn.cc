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

#include "tensorflow/stream_executor/dnn.h"

#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
namespace dnn {

bool DnnSupport::GetConvolveAlgorithms(
    std::vector<AlgorithmType>* out_algorithms) {
  return false;
}

bool DnnSupport::GetConvolveBackwardDataAlgorithms(
    std::vector<AlgorithmType>* out_algorithms) {
  return false;
}

bool DnnSupport::GetConvolveBackwardFilterAlgorithms(
    std::vector<AlgorithmType>* out_algorithms) {
  return false;
}

string QuantizedActivationModeString(QuantizedActivationMode mode) {
  switch (mode) {
    case dnn::QuantizedActivationMode::k8Bit:
      return "uint8";
    case dnn::QuantizedActivationMode::k16Bit:
      return "uint16";
    case dnn::QuantizedActivationMode::k32Bit:
      return "int32";
    default:
      LOG(FATAL) << "Unknown quantized_activation_mode "
                 << static_cast<int32>(mode);
  }
  return "unknown quantized_activation_mode";
}

string ActivationModeString(ActivationMode mode) {
  switch (mode) {
    case ActivationMode::kSigmoid:
      return "sigmoid";
    case ActivationMode::kRelu:
      return "relu";
    case ActivationMode::kRelu6:
      return "relu6";
    case ActivationMode::kReluX:
      return "reluX";
    case ActivationMode::kTanh:
      return "tanh";
    case ActivationMode::kBandPass:
      return "bandpass";
    default:
      LOG(FATAL) << "Unknown activation_mode " << static_cast<int32>(mode);
  }
  return "unknown activation_mode";
}

string ElementwiseOperationString(ElementwiseOperation op) {
  switch (op) {
    case ElementwiseOperation::kAdd:
      return "add";
    case ElementwiseOperation::kMultiply:
      return "multiply";
    default:
      LOG(FATAL) << "Unknown elementwise op " << static_cast<int32>(op);
  }
  return "unknown element wise op";
}

string DataLayoutString(DataLayout layout) {
  switch (layout) {
    case DataLayout::kYXDepthBatch:
      return "YXDepthBatch";
    case DataLayout::kYXBatchDepth:
      return "YXBatchDepth";
    case DataLayout::kBatchYXDepth:
      return "BatchYXDepth";
    case DataLayout::kBatchDepthYX:
      return "BatchDepthYX";
    default:
      LOG(FATAL) << "Unknown data layout " << static_cast<int32>(layout);
  }
  return "unknown data layout";
}

string FilterLayoutString(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX:
      return "OutputInputYX";
    case FilterLayout::kInputYXOutput:
      return "InputYXOutput";
    case FilterLayout::kYXInputOutput:
      return "YXInputOutput";
    default:
      LOG(FATAL) << "Unknown filter layout " << static_cast<int32>(layout);
  }
  return "unknown filter layout";
}

string PadAlignmentString(PadAlignment alignment) {
  switch (alignment) {
    case PadAlignment::kDefault:
      return "default";
    case PadAlignment::kCudnnPadding:
      return "cuDNN padding";
    case PadAlignment::kTensorFlowPadding:
      return "TensorFlow padding";
  }
}

string ShortPoolingModeString(PoolingMode mode) {
  switch (mode) {
    case PoolingMode::kMaximum:
      return "Max";
    case PoolingMode::kAverage:
      return "Avg";
    default:
      LOG(FATAL) << "Unknown filter layout " << static_cast<int32>(mode);
  }
  return "unknown filter layout";
}

std::tuple<int, int, int> GetDimIndices(const DataLayout& layout,
                                        const int data_dims) {
  int depth_idx, batch_idx, spatial_idx;
  switch (layout) {
    case DataLayout::kYXBatchDepth:
      depth_idx = data_dims - 1;
      batch_idx = data_dims - 2;
      spatial_idx = 0;
      break;

    case DataLayout::kYXDepthBatch:
      depth_idx = data_dims - 2;
      batch_idx = data_dims - 1;
      spatial_idx = 0;
      break;

    case DataLayout::kBatchYXDepth:
      depth_idx = data_dims - 1;
      batch_idx = 0;
      spatial_idx = 1;
      break;

    case DataLayout::kBatchDepthYX:
      depth_idx = 1;
      batch_idx = 0;
      spatial_idx = 2;
      break;
  }

  return std::make_tuple(depth_idx, batch_idx, spatial_idx);
}

std::vector<int64> ReorderDims(const std::vector<int64>& input,
                               const DataLayout& from, const DataLayout& to) {
  if (from == to) return input;

  int d_idx_from, b_idx_from, spatial_idx_from;
  int d_idx_to, b_idx_to, spatial_idx_to;

  std::tie(d_idx_from, b_idx_from, spatial_idx_from) =
      GetDimIndices(from, input.size());
  std::tie(d_idx_to, b_idx_to, spatial_idx_to) =
      GetDimIndices(to, input.size());

  std::vector<int64> reordered(input.size());
  reordered[b_idx_to] = input[b_idx_from];
  reordered[d_idx_to] = input[d_idx_from];

  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

// -- BatchDescriptor

BatchDescriptor::BatchDescriptor(int ndims)
    : count_(0),
      feature_map_count_(0),
      spatial_size_(ndims, 0),
      value_max_(0.0),
      value_min_(0.0),
      layout_(DataLayout::kYXDepthBatch),
      ndims_(ndims),
      quantized_activation_mode_(QuantizedActivationMode::k8Bit) {}

BatchDescriptor::BatchDescriptor() : BatchDescriptor(/*ndims=*/2) {}

std::vector<int64> BatchDescriptor::full_dims(const DataLayout& layout) const {
  std::vector<int64> bdyx_dims(ndims_ + 2);
  bdyx_dims[0] = count();
  bdyx_dims[1] = feature_map_count();
  std::copy(spatial_size_.begin(), spatial_size_.end(), bdyx_dims.begin() + 2);
  return ReorderDims(bdyx_dims, DataLayout::kBatchDepthYX, layout);
}

std::vector<int64> BatchDescriptor::full_strides(
    const DataLayout& layout) const {
  std::vector<int64> phys_dims = full_dims(layout_);
  std::vector<int64> phys_strides(phys_dims.size());
  phys_strides[ndims_ + 1] = 1;
  for (int i = ndims_; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, layout_, layout);
}

void BatchDescriptor::CloneFrom(const BatchDescriptor& other) {
  count_ = other.count_;
  feature_map_count_ = other.feature_map_count_;
  spatial_size_ = other.spatial_size_;
  value_max_ = other.value_max_;
  value_min_ = other.value_min_;
  layout_ = other.layout_;
  ndims_ = other.ndims_;
  quantized_activation_mode_ = other.quantized_activation_mode_;
}

string BatchDescriptor::ToString() const {
  string spatial;
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&spatial, "%lld ", spatial_size_[i]);
  }
  return port::Printf(
      "{count: %lld feature_map_count: %lld spatial: %s "
      "value_min: %f value_max: %f layout: %s}",
      count_, feature_map_count_, spatial.c_str(), value_min_, value_max_,
      DataLayoutString(layout_).c_str());
}

string BatchDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string depth = port::StrCat("d", feature_map_count());
  string batch = port::StrCat("b", count());

  string spatial = "s";
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&spatial, "%lld ", spatial_size_[i]);
  }

  string suffix;
  if (value_min() != value_max()) {
    port::StrAppend(&suffix, "[", value_min(), ";", value_max(), "]");
  }
  if (quantized_activation_mode() == QuantizedActivationMode::k16Bit) {
    suffix += "_16bit";
  }

  switch (layout()) {
    case DataLayout::kYXDepthBatch:
      return port::StrCat(spatial, depth, batch, suffix);
    case DataLayout::kYXBatchDepth:
      return port::StrCat(spatial, batch, depth, suffix);
    case DataLayout::kBatchYXDepth:
      return port::StrCat(batch, spatial, depth, suffix);
    case DataLayout::kBatchDepthYX:
      return port::StrCat(batch, depth, spatial, suffix);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64 BatchDescriptor::NodesPerFeatureMap() const {
  int64 ret = 1;
  for (int i = 0; i < ndims_; i++) {
    ret *= spatial_size_[i];
  }
  return ret;
}

int64 BatchDescriptor::NodesAcrossFeatureMaps() const {
  return NodesPerFeatureMap() * feature_map_count_;
}

int64 BatchDescriptor::ElementCount() const {
  return count_ * feature_map_count_ * NodesPerFeatureMap();
}

int64 BatchDescriptor::FullyConnectedWeightCount(
    const BatchDescriptor& input, const BatchDescriptor& output) {
  return input.NodesAcrossFeatureMaps() * output.NodesAcrossFeatureMaps();
}

int64 BatchDescriptor::FullyConnectedBiasCount(const BatchDescriptor& output) {
  return output.NodesAcrossFeatureMaps();
}

BatchDescriptor BatchDescriptor::DepthConcatenateOutputDescriptor(
    port::ArraySlice<dnn::BatchDescriptor> inputs) {
  if (inputs.empty()) {
    return BatchDescriptor();
  }
  int feature_map_count = 0;
  for (const auto& dimensions : inputs) {
    feature_map_count += dimensions.feature_map_count();
  }
  BatchDescriptor output = inputs[0];
  output.set_feature_map_count(feature_map_count);
  return output;
}

// -- FilterDescriptor

FilterDescriptor::FilterDescriptor(int ndims)
    : output_feature_map_count_(0),
      input_feature_map_count_(0),
      input_filter_dims_(ndims, 0),
      ndims_(ndims),
      layout_(FilterLayout::kOutputInputYX) {}

FilterDescriptor::FilterDescriptor() : FilterDescriptor(/*ndims=*/2) {}

FilterDescriptor::~FilterDescriptor() {}

void FilterDescriptor::CloneFrom(const FilterDescriptor& other) {
  set_output_feature_map_count(other.output_feature_map_count())
      .set_input_feature_map_count(other.input_feature_map_count())
      .set_layout(other.layout());
  input_filter_dims_ = other.input_filter_dims_;
  ndims_ = other.ndims_;
}

string FilterDescriptor::ToString() const {
  string desc = port::Printf(
      "{output_feature_map_count: %lld input_feature_map_count: %lld "
      "layout: %s shape: ",
      output_feature_map_count_, input_feature_map_count_,
      FilterLayoutString(layout_).c_str());
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&desc, "%lld ", input_filter_dims_[i]);
  }
  port::StrAppend(&desc, "}");

  return desc;
}

string FilterDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string od = port::StrCat("od", output_feature_map_count_);
  string id = port::StrCat("id", input_feature_map_count_);

  string spatial = "s";
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&spatial, "%lld ", input_filter_dims_[i]);
  }

  switch (layout_) {
    case FilterLayout::kOutputInputYX:
      return port::StrCat(od, id, spatial);
    case FilterLayout::kInputYXOutput:
      return port::StrCat(id, spatial, od);
    case FilterLayout::kYXInputOutput:
      return port::StrCat(spatial, id, od);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout_);
      return "";  // Avoid return warning (unreachable)
  }
}

int64 FilterDescriptor::ComputeWeightCount() const {
  int64 ret = output_feature_map_count_ * input_feature_map_count_;
  for (int i = 0; i < ndims_; i++) {
    ret *= input_filter_dims_[i];
  }
  return ret;
}

// -- ConvolutionDescriptor

ConvolutionDescriptor::ConvolutionDescriptor(int ndims)
    : zero_padding_(ndims, 0),
      filter_strides_(ndims, 1),
      pad_alignment_(PadAlignment::kDefault),
      ndims_(ndims) {}

ConvolutionDescriptor::ConvolutionDescriptor()
    : ConvolutionDescriptor(/*ndims=*/2) {}

ConvolutionDescriptor::~ConvolutionDescriptor() {}

string ConvolutionDescriptor::ToString() const {
  string padding;
  string strides;
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&padding, "%lld ", zero_padding_[i]);
    port::Appendf(&strides, "%lld ", filter_strides_[i]);
  }

  return port::Printf("{zero_padding: %s pad_alignment: %s filter_strides: %s}",
                      padding.c_str(),
                      PadAlignmentString(pad_alignment_).c_str(),
                      strides.c_str());
}

string ConvolutionDescriptor::ToShortString() const {
  string desc;
  for (int i = 0; i < ndims_; i++) {
    if (i > 0) port::Appendf(&desc, "_");
    port::Appendf(&desc, "p%d:%lld", i, zero_padding_[i]);
  }
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&desc, "_s%d:%lld", i, filter_strides_[i]);
  }
  return desc;
}

// -- PoolingDescriptor

PoolingDescriptor::PoolingDescriptor(int ndims)
    : mode_(dnn::PoolingMode::kMaximum),
      ndims_(ndims),
      window_(ndims, 0),
      padding_(ndims, 0),
      strides_(ndims, 1) {}

PoolingDescriptor::PoolingDescriptor() : PoolingDescriptor(/*ndims=*/2) {}

void PoolingDescriptor::CloneFrom(const PoolingDescriptor& other) {
  mode_ = other.mode_;
  ndims_ = other.ndims_;
  window_ = other.window_;
  padding_ = other.padding_;
  strides_ = other.strides_;
}

string PoolingDescriptor::ToString() const {
  const char* mode_string =
      mode_ == dnn::PoolingMode::kMaximum ? "kMaximum" : "kAverage";

  string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&window, "%lld ", window_[i]);
    port::Appendf(&strides, "%lld ", strides_[i]);
    port::Appendf(&padding, "%lld", padding_[i]);
  }

  return port::Printf("{mode: %s window: %s strides: %s padding: %s}",
                      mode_string, window.c_str(), strides.c_str(),
                      padding.c_str());
}

string PoolingDescriptor::ToShortString() const {
  string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    port::Appendf(&window, "_w%d:%lld", i, window_[i]);
    port::Appendf(&strides, "_s%d:%lld", i, strides_[i]);
    port::Appendf(&padding, "_p%d:%lld", i, padding_[i]);
  }
  return port::StrCat(mode_ == dnn::PoolingMode::kMaximum ? "max" : "avg",
                      window, strides, padding);
}

// -- NormalizeDescriptor

NormalizeDescriptor::NormalizeDescriptor()
    : bias_(0.0),
      range_(0),
      alpha_(0.0),
      beta_(0.0),
      wrap_around_(false),
      segment_size_(0) {}

void NormalizeDescriptor::CloneFrom(const NormalizeDescriptor& other) {
  bias_ = other.bias_;
  range_ = other.range_;
  alpha_ = other.alpha_;
  beta_ = other.beta_;
  wrap_around_ = other.wrap_around_;
  segment_size_ = other.segment_size_;
}

string NormalizeDescriptor::ToString() const {
  return port::Printf(
      "{bias: %f range: %d alpha: %f beta: %f wrap_around: %d "
      "segment_size: %d}",
      bias_, range_, alpha_, beta_, wrap_around_, segment_size_);
}

string NormalizeDescriptor::ToShortString() const {
  return port::StrCat("bias:", bias_, "_range:", range_, "_alpha:", alpha_,
                      "_beta:", beta_, "_wrap:", wrap_around_, "_size:",
                      segment_size_);
}

}  // namespace dnn
}  // namespace gputools
}  // namespace perftools
