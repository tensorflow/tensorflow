#include "tensorflow/stream_executor/dnn.h"

#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
namespace dnn {

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
    default:
      LOG(FATAL) << "Unknown activation_mode " << static_cast<int32>(mode);
  }
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
}

// -- BatchDescriptor

BatchDescriptor::BatchDescriptor()
    : count_(0),
      feature_map_count_(0),
      height_(0),
      width_(0),
      value_max_(0.0),
      value_min_(0.0),
      layout_(DataLayout::kYXDepthBatch),
      quantized_activation_mode_(QuantizedActivationMode::k8Bit) {}

void BatchDescriptor::CloneFrom(const BatchDescriptor& other) {
  count_ = other.count_;
  feature_map_count_ = other.feature_map_count_;
  height_ = other.height_;
  width_ = other.width_;
  value_max_ = other.value_max_;
  value_min_ = other.value_min_;
  layout_ = other.layout_;
  quantized_activation_mode_ = other.quantized_activation_mode_;
}

string BatchDescriptor::ToString() const {
  return port::Printf(
      "{count: %lld feature_map_count: %lld height: %lld width: %lld "
      "value_min: %f value_max: %f layout: %s}",
      count_, feature_map_count_, height_, width_, value_min_, value_max_,
      DataLayoutString(layout_).c_str());
}

string BatchDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string x = port::StrCat("x", width());
  string y = port::StrCat("y", height());
  string depth = port::StrCat("d", feature_map_count());
  string batch = port::StrCat("b", count());

  string suffix;
  if (value_min() != value_max()) {
    port::StrAppend(&suffix, "[", value_min(), ";", value_max(), "]");
  }
  if (quantized_activation_mode() == QuantizedActivationMode::k16Bit) {
    suffix += "_16bit";
  }

  switch (layout()) {
    case DataLayout::kYXDepthBatch:
      return port::StrCat(y, x, depth, batch, suffix);
    case DataLayout::kYXBatchDepth:
      return port::StrCat(y, x, batch, depth, suffix);
    case DataLayout::kBatchYXDepth:
      return port::StrCat(batch, y, x, depth, suffix);
    case DataLayout::kBatchDepthYX:
      return port::StrCat(batch, depth, y, x, suffix);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
  }
}

int64 BatchDescriptor::NodesPerFeatureMap() const { return width_ * height_; }

int64 BatchDescriptor::NodesAcrossFeatureMaps() const {
  return NodesPerFeatureMap() * feature_map_count_;
}

int64 BatchDescriptor::ElementCount() const {
  return count_ * feature_map_count_ * height_ * width_;
}

int64 BatchDescriptor::FullyConnectedWeightCount(
    const BatchDescriptor& input, const BatchDescriptor& output) {
  return input.NodesAcrossFeatureMaps() * output.NodesAcrossFeatureMaps();
}

int64 BatchDescriptor::FullyConnectedBiasCount(const BatchDescriptor& output) {
  return output.NodesAcrossFeatureMaps();
}

// -- FilterDescriptor

FilterDescriptor::FilterDescriptor()
    : output_feature_map_count_(0),
      input_feature_map_count_(0),
      input_filter_height_(0),
      input_filter_width_(0),
      layout_(FilterLayout::kOutputInputYX) {}

FilterDescriptor::~FilterDescriptor() {}

void FilterDescriptor::CloneFrom(const FilterDescriptor& other) {
  set_output_feature_map_count(other.output_feature_map_count())
      .set_input_feature_map_count(other.input_feature_map_count())
      .set_input_filter_height(other.input_filter_height())
      .set_input_filter_width(other.input_filter_width())
      .set_layout(other.layout());
}

string FilterDescriptor::ToString() const {
  return port::Printf(
      "{output_feature_map_count: %lld input_feature_map_count: %lld "
      "input_filter_height: %lld input_filter_width: %lld layout: %s}",
      output_feature_map_count_, input_feature_map_count_, input_filter_height_,
      input_filter_width_, FilterLayoutString(layout_).c_str());
}

string FilterDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string od = port::StrCat("od", output_feature_map_count_);
  string id = port::StrCat("id", input_feature_map_count_);
  string y = port::StrCat("y", input_filter_height_);
  string x = port::StrCat("x", input_filter_width_);

  switch (layout_) {
    case FilterLayout::kOutputInputYX:
      return port::StrCat(od, id, y, x);
    case FilterLayout::kInputYXOutput:
      return port::StrCat(id, y, x, od);
    case FilterLayout::kYXInputOutput:
      return port::StrCat(y, x, id, od);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout_);
  }
}

int64 FilterDescriptor::ComputeWeightCount() const {
  return output_feature_map_count_ * input_feature_map_count_ *
         input_filter_height_ * input_filter_width_;
}

// -- ConvolutionDescriptor

ConvolutionDescriptor::ConvolutionDescriptor()
    : zero_padding_height_(0),
      zero_padding_width_(0),
      vertical_filter_stride_(1),
      horizontal_filter_stride_(1) {}

ConvolutionDescriptor::~ConvolutionDescriptor() {}

string ConvolutionDescriptor::ToString() const {
  return port::Printf(
      "{zero_padding_height: %lld zero_padding_width: %lld "
      "vertical_filter_stride: %lld horizontal_filter_stride: %lld}",
      zero_padding_height_, zero_padding_width_, vertical_filter_stride_,
      horizontal_filter_stride_);
}

string ConvolutionDescriptor::ToShortString() const {
  return port::StrCat("py:", zero_padding_height_, "_px:", zero_padding_width_,
                      "_sy:", vertical_filter_stride_, "_sx:",
                      horizontal_filter_stride_);
}

// -- PoolingDescriptor

PoolingDescriptor::PoolingDescriptor()
    : mode_(dnn::PoolingMode::kMaximum),
      window_height_(0),
      window_width_(0),
      vertical_padding_(0),
      horizontal_padding_(0),
      vertical_stride_(0),
      horizontal_stride_(0) {}

void PoolingDescriptor::CloneFrom(const PoolingDescriptor& other) {
  mode_ = other.mode_;
  window_height_ = other.window_height_;
  window_width_ = other.window_width_;
  vertical_padding_ = other.vertical_padding_;
  horizontal_padding_ = other.horizontal_padding_;
  vertical_stride_ = other.vertical_stride_;
  horizontal_stride_ = other.horizontal_stride_;
}

string PoolingDescriptor::ToString() const {
  const char* mode_string =
      mode_ == dnn::PoolingMode::kMaximum ? "kMaximum" : "kAverage";
  return port::Printf(
      "{mode: %s window_height: %lld window_width: %lld vertical_stride: %lld "
      "horizontal_stride: %lld vertical padding: %lld horizontal padding: "
      "%lld}",
      mode_string, window_height_, window_width_, vertical_stride_,
      horizontal_stride_, vertical_padding_, horizontal_padding_);
}

string PoolingDescriptor::ToShortString() const {
  return port::StrCat(mode_ == dnn::PoolingMode::kMaximum ? "max" : "avg",
                      "_y:", window_height_, "_x:", window_width_, "_py:",
                      vertical_padding_, "_px:", horizontal_padding_, "_sy:",
                      vertical_stride_, "_sx:", horizontal_stride_);
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
