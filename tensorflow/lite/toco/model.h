/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_MODEL_H_
#define TENSORFLOW_LITE_TOCO_MODEL_H_

#include <complex>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_types.h"

namespace toco {

using tflite::QuantizationParams;

enum class OperatorType : uint8 {
  kNone,
  // General-purpose neural network operators.
  kAdd,
  kAddN,
  kAveragePool,
  kBatchMatMul,
  kBatchNormalization,
  kCeil,
  kConv,
  kConcatenation,
  kCos,
  kDepthwiseConv,
  kDepthToSpace,
  kSpaceToDepth,
  kDequantize,
  kDiv,
  kExp,
  kExpandDims,
  kFill,
  kFloorDiv,
  kFloorMod,
  kFullyConnected,
  kL2Normalization,
  kL2Pool,
  kLstmCell,
  kUnidirectionalSequenceLstm,
  kLocalResponseNormalization,
  kLog,
  kLogistic,
  kMaxPool,
  kFakeQuant,
  kMul,
  kOneHot,
  kRandomUniform,
  kRange,
  kRank,
  kRelu,
  kRelu1,
  kRelu6,
  kPRelu,
  kSoftmax,
  kLogSoftmax,
  kSub,
  kTanh,
  kTransposeConv,
  kCast,
  kFloor,
  kGather,
  kResizeBilinear,
  kSin,
  kSpaceToBatchND,
  kPack,
  kBatchToSpaceND,
  kPad,
  kPadV2,
  kReduceProd,  // Reduction product
  kStridedSlice,
  kSlice,
  kSqueeze,
  kMean,
  kArgMax,
  // The SVDF Op is a decomposition of a densely connected Op into
  // low rank filters. For details:
  // https://research.google.com/pubs/pub43813.html
  kSvdf,
  // Special operators used for importing TensorFlow nodes.
  // The general intent is to have some graph transformation either
  // drop them or rewrite them as general-purpose operators.
  kAll,
  kAssert,
  kConcat,
  kConcatV2,
  kGreater,
  kGreaterEqual,
  kIdentity,
  kLess,
  kLessEqual,
  kReduceMax,  //  Reduction Max
  kMaximum,    //  Element-wise Maximum
  kReduceMin,  //  Reduction Min
  kMinimum,    //  Element-wise Minimum
  kMatMul,
  kMerge,
  kNeg,
  kReshape,
  kRsqrt,
  kShape,
  kSplit,
  kSplitV,
  kSqrt,
  kSquare,
  kSquaredDifference,
  kSum,
  kSwitch,
  kTile,
  kTranspose,
  kTopK_V2,
  kDynamicPartition,
  kDynamicStitch,
  // An unsupported TF operation. It's only needed to be able to represent TF
  // graph internally and is expected to be dropped by graph transformations.
  kUnsupported,
  // Finally, TensorFlow uses different conventions for axes ordering,
  // see AxesOrder, and this cannot always be resolved at the time of importing
  // nodes, as TensorFlow parameters may be constant-expression subgraphs
  // instead of being given as plain constant arrays. So we need to insert
  // special nodes in the graph to shuffle axes.
  kReorderAxes,
  kSelect,
  kSparseToDense,
  kEqual,
  kNotEqual,
  kPow,
  kArgMin,
  kAny,
  kLogicalAnd,
  kLogicalNot,
  kLogicalOr,
  kCTCBeamSearchDecoder,
  kUnpack,
  kZerosLike,
  kResizeNearestNeighbor,
  kLeakyRelu,
  kAbs,
  kMirrorPad,
  kUnique,
  kUnidirectionalSequenceRnn,
  kBidirectionalSequenceLstm,
  kReverseV2,
  kBidirectionalSequenceRnn,
  kGatherNd,
  kWhere,
  kElu,
  kReverseSequence,
  kMatrixDiag,
  kMatrixSetDiag
};

// Helper to deal with TensorFlow arrays using a different ordering of
// dimensions
// ("axes") than our own.
// TODO(benoitjacob): Ultimately, we shouldn't have any "ordering" of axes,
// we should have associative arrays mapping symbolic axes identifiers (like
// "output_depth") to dimensions. We would then not need this anymore.
enum class AxesOrder {
  kOneAxis,  // one-dimensional array, one unique axis.
  kCR,       // column-major matrix storage order. Our standard.
  kRC,       // row-major matrix storage order. TensorFlow default.
  kOHWI,     // Our standard for conv weights
  kHWIO,     // TensorFlow conv weights
  k1HWO,     // Our standard for DepthwiseConv weights
  kHWIM,     // TensorFlow DepthwiseConv weights
  kNHWC,     // TensorFlow activations
  kHWOI,     // TensorFlow back-prop conv weights
};

// The type of the scalars in an array.
// Note that the type does not by itself tell whether the values in the array
// are non-quantized (can be accessed directly) or quantized (must be
// interpreted in conjunction with QuantizationParams).
//
// In practice though:
//   float values are never quantized
//   uint8 values are always quantized
//   int32 values are sometimes quantized (depending on whether
//   QuantizationParams are present).
//   complex values are never quantized
//   other types are never quantized at the moment.
//
// kNone means that we don't know the data type yet, or that we don't care
// because we'll be dropping the array anyway (e.g. some exotic array types
// may be involved only in debug-only subgraphs that we may not be interested
// in actually supporting).
enum class ArrayDataType : uint8 {
  kNone,  // 0
  kBool,
  kFloat,
  kInt8,
  kUint8,
  kInt16,  // 5
  kUint16,
  kInt32,
  kUint32,
  kInt64,
  kUint64,  // 10
  kString,
  kComplex64,
};

// Compile-time logic to map ArrayDataType to the corresponding C++ scalar type
template <ArrayDataType A>
struct DataTypeImpl {};
template <>
struct DataTypeImpl<ArrayDataType::kNone> {
  typedef int Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kBool> {
  typedef bool Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kFloat> {
  typedef float Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt8> {
  typedef int8 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint8> {
  typedef uint8 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt16> {
  typedef int16 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint16> {
  typedef uint16 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt32> {
  typedef int32 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint32> {
  typedef uint32 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt64> {
  typedef int64 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint64> {
  typedef uint64 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kString> {
  typedef string Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kComplex64> {
  typedef std::complex<float> Type;
};

template <ArrayDataType A>
using DataType = typename DataTypeImpl<A>::Type;

// Base class for type-specific buffer types.
struct GenericBuffer {
  // Non-default-constructible: only ArrayDataType-specific subclass
  // objects may be constructed.
  GenericBuffer() = delete;
  // Non-copyable-or-movable: we should only store pointers-to-Buffer
  // in containers, not Operators themselves, so there should be no
  // copy or move.
  GenericBuffer(const GenericBuffer&) = delete;
  GenericBuffer(const GenericBuffer&&) = delete;

  // We need a virtual destructor so we can store pointers-to-Buffer
  // in containers and have the containers call the right subclass destructor.
  virtual ~GenericBuffer() {}

  virtual int Length() const = 0;

  const ArrayDataType type;

 protected:
  // Constructor used by subclasses for specific ArrayDataType's.
  explicit GenericBuffer(ArrayDataType t) : type(t) {}
};

// Type-specific buffer, containing type-specific storage.
template <ArrayDataType A>
struct Buffer : GenericBuffer {
  Buffer() : GenericBuffer(A) {}

  int Length() const override { return data.size(); }

  std::vector<DataType<A>> data;
};

class Shape {
 public:
  // For Shape, we stick to half-way encapsulation for now:
  // we hide the raw dims_ member, but expose it raw by accessors
  // because from some brainstorming, it's not at all easy to
  // anticipate which flavor of more hermetic encapsulation would
  // actually buy us future-proof-ness without being needlessly
  // cumbersome.
  Shape() {}
  Shape(std::initializer_list<int> dim_list) : dims_(dim_list) {}

  void ReplaceDims(std::initializer_list<int> dim_list) {
    dims_ = std::vector<int>(dim_list);
  }

  const std::vector<int>& dims() const { return dims_; }
  std::vector<int>* mutable_dims() { return &dims_; }
  const int dimensions_count() const { return dims_.size(); }

  // We still have that one convenience accessor to avoid
  // the awkward double bracket issue:  shape.dims()[i].
  int dims(int i) const {
    // Always check for out-of-bounds accesses, even in optimized builds where
    // standard assertions are disabled. Out-of-bounds access here is a common
    // occurrence.
    CHECK_GE(i, 0);
    CHECK_GT(dims_.size(), i);
    return dims_[i];
  }

  bool operator==(const Shape& comp) const {
    return (this->dims_ == comp.dims());
  }

  bool operator!=(const Shape& comp) const { return !((*this) == comp); }

 private:
  std::vector<int> dims_;
};

// Base class for all operator classes.
struct Operator {
  // Non-default-constructible: only OperatorType-specific subclass
  // objects may be constructed.
  Operator() = delete;
  // Non-copyable-or-movable: we should only store pointers-to-Operator
  // in containers, not Operators themselves, so there should be no
  // copy or move.
  Operator(const Operator&) = delete;
  Operator(const Operator&&) = delete;

  // We need a virtual destructor so we can store pointers-to-Operator
  // in containers and have the containers call the right subclass destructor.
  virtual ~Operator() {}

  // The specific type of operator. Corresponds 1:1 to subclasses.
  const OperatorType type;

  // The activation function that may be fused into this operator,
  // or None if no activation function is fused.
  FusedActivationFunctionType fused_activation_function;

  // Input arrays: either activation arrays or constant array parameters.
  // We refer to them by their name, not by their address; the mapping of
  // names to addresses is given by the Model, which owns both Operator's and
  // Array's. Thus, an Operator on its own doesn't contain much information,
  // it is meant to be used in conjunction with the Model that owns it.
  std::vector<string> inputs;

  // Output activation arrays. Same comments as for inputs apply here too.
  std::vector<string> outputs;

  // If true, the operator has more outputs than are listed in the 'outputs'
  // member. These need to be resolved by some graph transformation.
  // This flag is only here to indicate that an operator should not be
  // discarded as unused, even if from its 'outputs' member alone it
  // looks unused.
  bool unresolved_outputs = false;

  // A serialized tensorflow::NodeDef string.
  // The field is filled only when importing from TensorFlow.
  // It's guaranteed to be filled for `TensorFlowUnsupportedOperator`.
  // It's not guaranteed to be filled for other ops. Ops created by graph
  // transformations won't have TensorFlow NodeDef.
  string tensorflow_node_def;

 protected:
  // Constructor used by subclasses for specific OperatorType's.
  explicit Operator(OperatorType t)
      : type(t),
        fused_activation_function(FusedActivationFunctionType::kNone) {}
};

// Padding types for Conv-like operators. This is how padding is typically
// specified in model files. But for inference, we will need to resolve this
// to a FixedPadding, see below.
enum class PaddingType { kNone, kSame, kValid };

// Padding as resolved for a specific layer shape, as needed for inference.
// For a given layer shape, a given padding type will resolve to a choice of
// a number of padding rows and columns, which we call the padding height and
// width respectively.
struct FixedPadding {
  int width = 0;
  int height = 0;
};

// "Universal" padding struct containing both a generic PaddingType (as
// represented in a model file), and a FixedPadding (as needed for inference).
// The latter is resolved during the PropagateFixedSizes pass.
struct Padding {
  FixedPadding& GetOrCreateFixedPadding() {
    if (!fixed) {
      FixedPadding* ptr = new FixedPadding;
      fixed = std::unique_ptr<FixedPadding>(ptr);
    }
    return *fixed;
  }

  Padding() : type(PaddingType::kNone) {}
  PaddingType type;
  std::unique_ptr<FixedPadding> fixed;
};

// "Convolutional" layer, as represented in model files.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the Conv weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// Outputs:
//   outputs[0]: required: the output activations array
//   outputs[1]: optional: the intermediate array of im2col-replicated input
//                         activations. Present when targeting implementations
//                         of Conv layers as Im2col+GEMM.
//
// TensorFlow equivalent: Conv2D
struct ConvOperator : Operator {
  ConvOperator() : Operator(OperatorType::kConv) {}
  Padding padding;
  int stride_width = 0;
  int stride_height = 0;
  // A dilation_rate of 0 is invalid and this field is an optional attribute.
  // Thus initializing it to 1 to allow default conv behavior when the
  // attribute is not present.
  int dilation_width_factor = 1;
  int dilation_height_factor = 1;
};

// CTCBeamSearchDecoder operator:
//
// Inputs:
//   inputs[0]: required: the logits.
//   inputs[1]: required: sequence length.
//   inputs[2]: optional: beam width.
//   inputs[3]: optional: top paths.
//   inputs[4]: optional: merge repeated.
//
//  Outputs:
//    outputs[0]: deocoded.
//    outputs[1]: log probability.
//
// TensorFlow equivalent: CTCBeamSearchDecoder
struct CTCBeamSearchDecoderOperator : Operator {
  CTCBeamSearchDecoderOperator()
      : Operator(OperatorType::kCTCBeamSearchDecoder) {}
  int beam_width;
  int top_paths;
  bool merge_repeated = true;
};

// Depthwise-separable convolution operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the DepthwiseConv weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// TensorFlow equivalent: DepthwiseConv2dNative
struct DepthwiseConvOperator : Operator {
  DepthwiseConvOperator() : Operator(OperatorType::kDepthwiseConv) {}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int depth_multiplier = 0;
  // A dilation_rate of 0 is invalid and this field is an optional attribute.
  // Thus initializing it to 1 to allow default conv behavior when the
  // attribute is not present.
  int dilation_width_factor = 1;
  int dilation_height_factor = 1;
};

// Depth-to-space transform operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: DepthToSpace
struct DepthToSpaceOperator : Operator {
  DepthToSpaceOperator() : Operator(OperatorType::kDepthToSpace) {}
  int block_size = 0;
};

// Space-to-depth transform operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: SpaceToDepth
struct SpaceToDepthOperator : Operator {
  SpaceToDepthOperator() : Operator(OperatorType::kSpaceToDepth) {}
  int block_size = 0;
};

// Fully-connected operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the FullyConnected weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// TensorFlow equivalent: a pair consisting of a Reshape node reshaping the
// input activations as a matrix, followed by a MatMul node.
struct FullyConnectedOperator : Operator {
  FullyConnectedOperator() : Operator(OperatorType::kFullyConnected) {}
  FullyConnectedWeightsFormat weights_format =
      FullyConnectedWeightsFormat::kDefault;
};

// Dequantization operator, converting a quantized array of integers with
// quantization parameters specifying how these integers correspond to real
// numbers
// (see QuantizationParams) to an output activations array of floating-point
// values.
//
// In floating-point image models, there is typically a Dequantization operator
// at the very beginning, converting the input image RGB data, consisting of
// uint8 integer values, to floating-point input activations. That is where
// image model parameters such as "mean_value" and "std_value" are typically
// handled.
//
// This is the only operator type that converts from quantized to
// floating-point,
// and there is at the moment no operator type at all to convert from
// floating-point
// to quantized. Every other operator does either float->float or
// quantized->quantized.
//
// Inputs:
//   inputs[0]: required: the input quantized activations array
//
// TensorFlow equivalent: Dequantize
struct DequantizeOperator : Operator {
  DequantizeOperator() : Operator(OperatorType::kDequantize) {}
};

// Batch-normalization operator.
//
// We only support batch-normalization using pre-learned moments, so this is
// just
// computing (input - mean) * multiplier + offset. As such, this can be
// expressed as a combination of Add and Mul nodes, and indeed this is how
// we break it down during tooling for the purpose of fusing it into
// other operators.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the learned mean array
//   inputs[2]: required: the learned multiplier array
//   inputs[3]: required: the learned offset array
//
// TensorFlow equivalent: a combination of Add and Mul nodes
struct BatchNormalizationOperator : Operator {
  BatchNormalizationOperator()
      : Operator(OperatorType::kBatchNormalization),
        global_normalization(false) {}
  bool global_normalization;
};

// L2-normalization operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: none. In TensorFlow, L2 normalization is implemented
// by a sub-graph of operators implementing L2-normalization
// from lower-level arithmetic nodes; during tooling, we identify such
// sub-graphs
// and replace them by L2NormalizationOperator's. See IdentifyL2Normalization.
struct L2NormalizationOperator : Operator {
  L2NormalizationOperator() : Operator(OperatorType::kL2Normalization) {}
};

// LSTM Cell operator.
//
// Inputs:
//   inputs[0]: required: the input data array
//   inputs[1]: required: the previous output activations array
//   inputs[2]: required: the learned weights array
//   inputs[3]: required: the learned biases array
//   inputs[4]: required: the previous output state
//   outputs[0]: required: the output activations array
//   outputs[1]: required: the new state array
//
// TensorFlow equivalent: none. In TensorFlow, an LSTM is implemented
// with a sub-graph of lower-level arithmetic nodes; during tooling, we identify
// such sub-graphs and replace them with LstmCells. See IdentifyLstmCell().
struct LstmCellOperator : Operator {
  enum Inputs {
    DATA_INPUT = 0,
    PREV_ACTIV_INPUT = 1,
    WEIGHTS_INPUT = 2,
    BIASES_INPUT = 3,
    PREV_STATE_INPUT = 4,
    NUM_INPUTS = 5
  };
  enum Outputs {
    ACTIV_OUTPUT = 0,
    STATE_OUTPUT = 1,
    CONCAT_TEMP = 2,
    ACTIV_TEMP = 3,
    NUM_OUTPUTS = 4
  };
  enum KernelType {
    KERNEL_BASIC = 0,
    KERNEL_FULL = 1,
  };

  LstmCellOperator()
      : Operator(OperatorType::kLstmCell), kernel_type(KERNEL_BASIC) {}

  KernelType kernel_type;
};

struct UnidirectionalSequenceLstmOperator : Operator {
  UnidirectionalSequenceLstmOperator()
      : Operator(OperatorType::kUnidirectionalSequenceLstm) {}
};

struct BidirectionalSequenceLstmOperator : Operator {
  BidirectionalSequenceLstmOperator()
      : Operator(OperatorType::kBidirectionalSequenceLstm) {}
  bool merge_outputs;
};

struct BidirectionalSequenceRnnOperator : Operator {
  BidirectionalSequenceRnnOperator()
      : Operator(OperatorType::kBidirectionalSequenceRnn) {}
  bool merge_outputs;
};

// Element-wise multiplication operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Mul
struct MulOperator : Operator {
  MulOperator() : Operator(OperatorType::kMul) {}
};

// Element-wise Abs operator:
//   x -> abs(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Relu
struct AbsOperator : Operator {
  AbsOperator() : Operator(OperatorType::kAbs) {}
};

// Elu
//   f(x) -> exp(x) - 1 for x < 0, x for x >= 0.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Elu
struct EluOperator : Operator {
  EluOperator() : Operator(OperatorType::kElu) {}
};

// Element-wise Relu operator:
//   x -> max(0, x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Relu
struct ReluOperator : Operator {
  ReluOperator() : Operator(OperatorType::kRelu) {}
};

// Element-wise Relu1 operator:
//   x -> min(max(x, -1), 1)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. We can construct the operator with Minimum
// and Maximum operations
struct Relu1Operator : Operator {
  Relu1Operator() : Operator(OperatorType::kRelu1) {}
};

// Element-wise Relu6 operator:
//   x -> max(0, min(6, x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Relu6
struct Relu6Operator : Operator {
  Relu6Operator() : Operator(OperatorType::kRelu6) {}
};

// PRelu
//   f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the alpha array
//
// Equivalent to keras.layers.PReLU.
struct PReluOperator : Operator {
  PReluOperator() : Operator(OperatorType::kPRelu) {}
};

// LeakyRelu
//   x -> max(x, alpha * x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: LeakyRelu
struct LeakyReluOperator : Operator {
  LeakyReluOperator() : Operator(OperatorType::kLeakyRelu) {}

  float alpha = 0.2f;  // 0.2 matches the default value for the TF op attribute.
};

// Element-wise Logistic operator:
//   x -> Logistic(x) = 1 / (1 + exp(-x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sigmoid
struct LogisticOperator : Operator {
  LogisticOperator() : Operator(OperatorType::kLogistic) {}
};

// Element-wise natural log operator:
//   x -> ln(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Log
struct LogOperator : Operator {
  LogOperator() : Operator(OperatorType::kLog) {}
};

// Element-wise Tanh operator:
//   x -> Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Tanh
struct TanhOperator : Operator {
  TanhOperator() : Operator(OperatorType::kTanh) {}
};

// Element-wise Sin operator:
//   x -> Sin(x) = sin(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sin
struct SinOperator : Operator {
  SinOperator() : Operator(OperatorType::kSin) {}
};

// Element-wise addition operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Add
struct AddOperator : Operator {
  AddOperator() : Operator(OperatorType::kAdd) {}
};

// Element-wise addition operator for N inputs.
//
// Inputs:
//   inputs[i]: The i-th array to add together to form the output.
//
// TensorFlow equivalent: AddN
struct AddNOperator : Operator {
  AddNOperator() : Operator(OperatorType::kAddN) {}
};

// Concatenation operator: concatenates its inputs
// along the axis.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to concatenate.
//
// TensorFlow equivalent: Concat.
struct ConcatenationOperator : Operator {
  ConcatenationOperator() : Operator(OperatorType::kConcatenation) {}
  int axis = 0;
};

// Reordering dimensions. Used only during tooling to transform graphs from
// the TensorFlow format.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. This is only useful to convert between formats.
struct ReorderAxesOperator : Operator {
  ReorderAxesOperator() : Operator(OperatorType::kReorderAxes) {}
  AxesOrder input_axes_order;
  AxesOrder output_axes_order;
};

// Average-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: AveragePool
struct AveragePoolOperator : Operator {
  AveragePoolOperator() : Operator(OperatorType::kAveragePool) {}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// Local response normalization operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: LRN
struct LocalResponseNormalizationOperator : Operator {
  LocalResponseNormalizationOperator()
      : Operator(OperatorType::kLocalResponseNormalization) {}

  int range = 0;
  float bias = 0.f;
  float alpha = 0.f;
  float beta = 0.f;
};

// Max-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: MaxPool
struct MaxPoolOperator : Operator {
  MaxPoolOperator() : Operator(OperatorType::kMaxPool) {}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// L2-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. Can be shimmed by squaring+avgpool+sqrt.
struct L2PoolOperator : Operator {
  L2PoolOperator() : Operator(OperatorType::kL2Pool) {}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// The expected [min, max] range of values in a given array.
// Used for quantization only.
// This information typically comes from special nodes found in quantized
// models, see FakeQuantOperator, and is used during quantization to resolve
// actual quantization parameters (see QuantizationParams).
struct MinMax {
  double min = 0.;
  double max = 0.;
};

inline bool operator==(const MinMax& m1, const MinMax& m2) {
  return m1.min == m2.min && m1.max == m2.max;
}

// Fake-quantization operator. This does two things:
//   - Annotate its input and output arrays with MinMax information,
//   - Arithmetic-wise, this operator rounds incoming activation values
//     to the nearest representable value on the scale of 256
//     values from the min to the max value dictated by its MinMax info.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: optional: the 'min' value, if it has not yet been resolved
//              to a constant.
//   inputs[2]: optional: the 'max' value, if it has not yet been resolved
//              to a constant.
//
// TensorFlow equivalent: FakeQuantWithMinMaxVars, FakeQuantWithMinMaxArgs.
struct FakeQuantOperator : Operator {
  FakeQuantOperator() : Operator(OperatorType::kFakeQuant) {}
  std::unique_ptr<MinMax> minmax;
  int num_bits = 8;
  bool narrow_range = false;
};

// Element-wise division operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Div
struct DivOperator : Operator {
  DivOperator() : Operator(OperatorType::kDiv) {}
};

// Element-wise identity (x->x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Identity
struct TensorFlowIdentityOperator : Operator {
  TensorFlowIdentityOperator() : Operator(OperatorType::kIdentity) {}
};

// Batch matrix multiplication operator. This comes from the (deprecated)
// tf.batch_matmul or a tf.matmul that has rank 3. dims(0) is the batch count
// and it can be trivially unrolled into a series of matmuls on each element.
//
// Inputs:
//   inputs[0]: required: the left-hand side matrix
//   inputs[1]: required: the right-hand side matrix
//
// TensorFlow equivalent: MatMul
struct BatchMatMulOperator : Operator {
  BatchMatMulOperator() : Operator(OperatorType::kBatchMatMul) {}
  bool adj_x = false;
  bool adj_y = false;
};

// General matrix multiplication operator. We don't want to support general
// matrix multiplication at inference time, so we resolve it during tooling
// to more specific operator types, namely, FullyConnected.
//
// Inputs:
//   inputs[0]: required: the left-hand side matrix
//   inputs[1]: required: the right-hand side matrix
//
// TensorFlow equivalent: MatMul
struct TensorFlowMatMulOperator : Operator {
  TensorFlowMatMulOperator() : Operator(OperatorType::kMatMul) {}
  bool transpose_a = false;
  bool transpose_b = false;
};

// Padding operator. Pads a tensor with zeros.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the padding array
//
// This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of
// `input` in that dimension.
//
// TensorFlow equivalent: Pad
struct PadOperator : Operator {
  PadOperator() : Operator(OperatorType::kPad) {}

  std::vector<int> left_padding;
  std::vector<int> right_padding;
};

// PaddingV2 operator. Pads a tensor with the given constant value.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the padding array
//   inputs[2]: required: the scalar constant_values
//
// This operation pads input according to the paddings and constant_values you
// specify. paddings is an integer tensor with shape [Dn, 2], where n is the
// rank of input. For each dimension D of input, paddings[D, 0] indicates how
// many padding values to add before the contents of input in that dimension,
// and paddings[D, 1] indicates how many padding values to add after the
// contents of input in that dimension. constant_values is a scalar tensor of
// the same type as input that indicates the value to use for padding input.
//
// TensorFlow equivalent: PadV2
struct PadV2Operator : Operator {
  PadV2Operator() : Operator(OperatorType::kPadV2) {}

  std::vector<int> left_padding;
  std::vector<int> right_padding;
};

// Strided slice operator.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the begin array
//   inputs[2]: required: the end array
//   inputs[3]: optional: the strides array
//
// TensorFlow equivalent: StridedSlice
struct StridedSliceOperator : Operator {
  StridedSliceOperator() : Operator(OperatorType::kStridedSlice) {}

  std::vector<int> start_indices;
  std::vector<int> stop_indices;
  std::vector<int> strides;

  int begin_mask;
  int ellipsis_mask;
  int end_mask;
  int new_axis_mask;
  int shrink_axis_mask;

  StridedSliceOperator(const StridedSliceOperator& other)
      : Operator(OperatorType::kStridedSlice) {
    inputs = other.inputs;
    outputs = other.outputs;

    start_indices = other.start_indices;
    stop_indices = other.stop_indices;
    strides = other.strides;

    begin_mask = other.begin_mask;
    ellipsis_mask = other.ellipsis_mask;
    end_mask = other.end_mask;
    new_axis_mask = other.new_axis_mask;
    shrink_axis_mask = other.shrink_axis_mask;
  }

  void PadIndices(int dim_count) {
    // Add indices and mask bits to fully include extra dimensions
    CHECK_GE(dim_count, start_indices.size());
    CHECK_EQ(start_indices.size(), stop_indices.size());
    CHECK_EQ(stop_indices.size(), strides.size());

    for (int i = start_indices.size(); i < dim_count; i++) {
      start_indices.push_back(0);
      stop_indices.push_back(0);
      strides.push_back(1);
      begin_mask |= 1 << i;
      end_mask |= 1 << i;
    }
  }

  void ReverseIndices() {
    CHECK_EQ(start_indices.size(), stop_indices.size());
    CHECK_EQ(stop_indices.size(), strides.size());

    std::reverse(start_indices.begin(), start_indices.end());
    std::reverse(stop_indices.begin(), stop_indices.end());
    std::reverse(strides.begin(), strides.end());

    begin_mask = toco::port::ReverseBits32(static_cast<uint32>(begin_mask)) >>
                 (32 - start_indices.size());
    ellipsis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(ellipsis_mask)) >>
        (32 - start_indices.size());
    end_mask = toco::port::ReverseBits32(static_cast<uint32>(end_mask)) >>
               (32 - start_indices.size());
    new_axis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(new_axis_mask)) >>
        (32 - start_indices.size());
    shrink_axis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(shrink_axis_mask)) >>
        (32 - start_indices.size());
  }
};

// Reshaping operator, reshaping its input array to a two-dimensional shape
// (a "matrix"). This is used in the TensorFlow format, in conjunction with
// MatMul nodes, to implement fully-connected layers.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Reshape --- except that we only support a special case
// here, where the output shape is a matrix (2D) shape.
struct TensorFlowReshapeOperator : Operator {
  TensorFlowReshapeOperator() : Operator(OperatorType::kReshape) {}
  std::vector<int> shape;
};

// Removes dimensions of size 1 from the shape of a tensor.
// https://www.tensorflow.org/api_docs/python/tf/squeeze
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Squeeze
struct SqueezeOperator : Operator {
  SqueezeOperator() : Operator(OperatorType::kSqueeze) {}

  std::vector<int> squeeze_dims;
};

// Inputs:
//   inputs[0]: required: the output shape
//   inputs[1]: required: the weights
//   inputs[2]: required: the input activations array
//   NOTE: The input activations is NOT the first input.
//
//
// Outputs:
//   outputs[0]: required: the output activations array
//
// TensorFlow equivalent: Conv2DBackpropInput
struct TransposeConvOperator : Operator {
  enum Inputs {
    OUTPUT_SHAPE = 0,
    WEIGHTS = 1,
    DATA_INPUT = 2,
  };

  TransposeConvOperator() : Operator(OperatorType::kTransposeConv) {}
  Padding padding;
  int stride_width = 0;
  int stride_height = 0;
  // Dilation is possible with transpose convolution, but Tensorflow does not
  // currently support it, so we omit it.
};

// Given a tensor input, this operation calculates element-wise exponential
// (y = e^x).
//
// Inputs:
//   inputs[0]: required: input tensor
//
// TensorFlow equivalent: Exp
struct ExpOperator : Operator {
  ExpOperator() : Operator(OperatorType::kExp) {}
};

// Given a tensor input, this operation calculates element-wise exponential
// (y = cos(x)).
//
// Inputs:
//   inputs[0]: required: input tensor
//
// TensorFlow equivalent: Cos
struct CosOperator : Operator {
  CosOperator() : Operator(OperatorType::kCos) {}
};

// Given a tensor input, this operation inserts a dimension of 1 at the
// dimension index axis of input's shape. The dimension index axis starts at
// zero; if you specify a negative number for axis it is counted backward from
// the end.
//
// Inputs:
//   inputs[0]: required: input tensor
//   inputs[1]: required: 0-D (scalar). Specifies the dimension index at which
//   to expand the shape of input
//
// TensorFlow equivalent: ExpandDims
struct ExpandDimsOperator : Operator {
  ExpandDimsOperator() : Operator(OperatorType::kExpandDims) {}
};

// Ceates a tensor of shape dims and fills it with the given scalar value.
// Output type will be the same as the given scalar value.
//
// Inputs:
//   inputs[0]: required: 1-D (int32) - the shape of the output tensor
//   inputs[1]: required: 0-D (scalar) - value to fill the tensor with
//
// TensorFlow equivalent: Fill
struct FillOperator : Operator {
  FillOperator() : Operator(OperatorType::kFill) {}
};

// Element-wise floor division operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: FloorDiv
struct FloorDivOperator : Operator {
  FloorDivOperator() : Operator(OperatorType::kFloorDiv) {}
};

// Element-wise floor mod operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: FloorMod
struct FloorModOperator : Operator {
  FloorModOperator() : Operator(OperatorType::kFloorMod) {}
};

struct RandomUniformOperator : Operator {
  RandomUniformOperator() : Operator(OperatorType::kRandomUniform) {}
  ArrayDataType dtype = ArrayDataType::kNone;
  int64 seed;
  int64 seed2;
};

// Creates a sequence of numbers that begins at start and extends by increments
// of delta up to but not including limit.
//
// The dtype of the resulting tensor is inferred from the inputs unless it is
// provided explicitly.
//
// Inputs:
//   inputs[0]: required: the start
//   inputs[1]: required: the limit
//   inputs[2]: required: the delta
//
// TensorFlow equivalent: Range
struct RangeOperator : Operator {
  RangeOperator() : Operator(OperatorType::kRange) {}
  ArrayDataType dtype = ArrayDataType::kNone;
};

// Rank operator. Extracts the rank of the tensor.
//
// Inputs:
//   inputs[0]: required: the input array
//
// This operation outputs a 0-D int32 Tensor representing the rank of input.
//
// TensorFlow equivalent: Rank.
struct TensorFlowRankOperator : Operator {
  TensorFlowRankOperator() : Operator(OperatorType::kRank) {}
  ArrayDataType output_data_type = ArrayDataType::kInt32;
};

// Element-wise negation (-x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Neg
struct NegOperator : Operator {
  NegOperator() : Operator(OperatorType::kNeg) {}
};

// Element-wise select operator choosing elements from inputs[1] or input[2]
//
// Inputs:
//  inputs[0]: required: boolean mask per index
//  inputs[1]: required: tensor of values if true
//  inputs[2]: required: tensor of values if false
//
//  TensorFlow equivalent: Select
struct SelectOperator : Operator {
  SelectOperator() : Operator(OperatorType::kSelect) {}
};

// Element-wise reciprocal-square-root (x^-0.5) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Rsqrt
struct TensorFlowRsqrtOperator : Operator {
  TensorFlowRsqrtOperator() : Operator(OperatorType::kRsqrt) {}
};

// Stacks a list of rank-R tensors into one rank-(R+1) tensor.
//
// Packs the list of tensors in values into a tensor with rank one higher than
// each tensor in values, by packing them along the axis dimension. Given a list
// of length N of tensors of shape (A, B, C);.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to merge.
//
// TensorFlow equivalent: Pack
struct PackOperator : Operator {
  PackOperator() : Operator(OperatorType::kPack) {}
  int values_count;
  int axis = 0;
  ArrayDataType dtype = ArrayDataType::kNone;
};

// Shape operator. Extracts the shape of the tensor.
//
// Inputs:
//   inputs[0]: required: the input array
//
// This operation outputs a 1-D integer tensor representing the shape of
// the input.
//
// TensorFlow equivalent: Shape.
struct TensorFlowShapeOperator : Operator {
  TensorFlowShapeOperator() : Operator(OperatorType::kShape) {}
  ArrayDataType output_data_type = ArrayDataType::kInt32;
};

// Element-wise square-root (x^0.5) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sqrt
struct TensorFlowSqrtOperator : Operator {
  TensorFlowSqrtOperator() : Operator(OperatorType::kSqrt) {}
};

// Element-wise square (x*x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Square
struct TensorFlowSquareOperator : Operator {
  TensorFlowSquareOperator() : Operator(OperatorType::kSquare) {}
};

// Element-wise squared difference ((x-y)*(x-y)) operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: SquaredDifference
struct SquaredDifferenceOperator : Operator {
  SquaredDifferenceOperator() : Operator(OperatorType::kSquaredDifference) {}
};

// Transposes a tensor.
//
// By default, this operation performs a regular matrix transpose on 2-D input
// tensors.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Transpose
struct TransposeOperator : Operator {
  TransposeOperator() : Operator(OperatorType::kTranspose) {}
  std::vector<int> perm;
};

// Element-wise subtraction operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Sub
struct SubOperator : Operator {
  SubOperator() : Operator(OperatorType::kSub) {}
};

// Sum reduction: computes the sum of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sum
struct TensorFlowSumOperator : Operator {
  TensorFlowSumOperator() : Operator(OperatorType::kSum) {}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Prod reduction: computes the product of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Prod
struct TensorFlowProdOperator : Operator {
  TensorFlowProdOperator() : Operator(OperatorType::kReduceProd) {}
  std::vector<int> axis;
  bool keep_dims = false;
};

// TensorFlow Tile equivalent. Refer to TensorFlow documentation for details.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: int array with length of rank(input[0])
struct TensorFlowTileOperator : Operator {
  TensorFlowTileOperator() : Operator(OperatorType::kTile) {}
};

// TensorFlow Slice equivalent. Refer to TensorFlow documentation for details.
struct SliceOperator : Operator {
  SliceOperator() : Operator(OperatorType::kSlice) {}

  std::vector<int> begin;
  std::vector<int> size;
};

// TensorFlow Split equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
struct TensorFlowSplitOperator : Operator {
  TensorFlowSplitOperator() : Operator(OperatorType::kSplit) {}
  int num_split = 0;
};

// TensorFlow SplitV equivalent. Refer to TensorFlow documentation for details.
struct TensorFlowSplitVOperator : Operator {
  TensorFlowSplitVOperator() : Operator(OperatorType::kSplitV) {}
  int num_split = 0;
};

// TensorFlow Concat equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Concretely, once the concat dim becomes known, if it is the depth
// dimension then we can change this op into a DepthConcatenation op.
// Otherwise, we hope for some other graph transformation to drop this node.
struct TensorFlowConcatOperator : Operator {
  TensorFlowConcatOperator() : Operator(OperatorType::kConcat) {}
};

// TensorFlow ConcatV2 equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Concretely, once the concat dim becomes known, if it is the depth
// dimension then we can change this op into a DepthConcatenation op.
// Otherwise, we hope for some other graph transformation to drop this node.
struct TensorFlowConcatV2Operator : Operator {
  TensorFlowConcatV2Operator() : Operator(OperatorType::kConcatV2) {}
};

// TensorFlow Merge equivalent. Refer to TensorFlow documentation for details.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to merge.
//
// It is expected that graph transformations will drop all but exactly one
// of the inputs, at which point the Merge node will be equivalent to an
// Identity node forwarding the remaining input.
//
// Note: We do not currently support runtime control flow: we only support
// control flow that can be resolved at tooling time (independently of input
// activations).
struct TensorFlowMergeOperator : Operator {
  TensorFlowMergeOperator() : Operator(OperatorType::kMerge) {}
};

// TensorFlow Switch equivalent. Refer to TensorFlow documentation for details.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the boolean predicate, given as an array of size 1
//     and of type kBool, will determine which output gets selected.
//
// Outputs: a TensorFlow Switch node always has exactly two outputs. Depending
// on the boolean value that the input predicate resolves to (see note below),
// one or the other of the outputs will be 'selected': the input array will be
// forwarded to the 'selected output' as if by a Identity node, while the other
// output will be discarded, and any graph edge connecting that discarded output
// will be dropped. The rule for selecting outputs is as follows:
//   outputs[0] will be selected if the input predicate resolves to 'true'.
//   outputs[1] will be selected if the input predicate resolves to 'false'.
//
// Note: We do not currently support runtime control flow: we only support
// control flow that can be resolved at tooling time (independently of input
// activations).
struct TensorFlowSwitchOperator : Operator {
  TensorFlowSwitchOperator() : Operator(OperatorType::kSwitch) {}
};

// TensorFlow All equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowAllOperator : Operator {
  TensorFlowAllOperator() : Operator(OperatorType::kAll) {}
};

// TensorFlow Assert equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, we just drop Assert nodes.
struct TensorFlowAssertOperator : Operator {
  TensorFlowAssertOperator() : Operator(OperatorType::kAssert) {}
};

// TensorFlow Less equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowLessOperator : Operator {
  TensorFlowLessOperator() : Operator(OperatorType::kLess) {}
};

// TensorFlow LessEqual equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowLessEqualOperator : Operator {
  TensorFlowLessEqualOperator() : Operator(OperatorType::kLessEqual) {}
};

// TensorFlow Less equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowGreaterOperator : Operator {
  TensorFlowGreaterOperator() : Operator(OperatorType::kGreater) {}
};

// TensorFlow GreaterEqual equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowGreaterEqualOperator : Operator {
  TensorFlowGreaterEqualOperator() : Operator(OperatorType::kGreaterEqual) {}
};

// TensorFlow Equal equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowEqualOperator : Operator {
  TensorFlowEqualOperator() : Operator(OperatorType::kEqual) {}
};

// TensorFlow Not Equal equivalent. Refer to TensorFlow documentation for
// details.
struct TensorFlowNotEqualOperator : Operator {
  TensorFlowNotEqualOperator() : Operator(OperatorType::kNotEqual) {}
};

// Max reduction: computes the max of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Max
struct TensorFlowMaxOperator : Operator {
  TensorFlowMaxOperator() : Operator(OperatorType::kReduceMax) {}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Min reduction: computes the min of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Min
struct TensorFlowMinOperator : Operator {
  TensorFlowMinOperator() : Operator(OperatorType::kReduceMin) {}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Element-wise maximum operator. Currently it only supports scalar as
// the second operand.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Maximum
struct TensorFlowMaximumOperator : Operator {
  TensorFlowMaximumOperator() : Operator(OperatorType::kMaximum) {}
};

// Element-wise minimum operator. Currently it only supports scalar as
// the second operand.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Minimum
struct TensorFlowMinimumOperator : Operator {
  TensorFlowMinimumOperator() : Operator(OperatorType::kMinimum) {}
};

// General TF operation, unsupported by tf.mini. Expected to be dropped by
// graph transformations.
struct TensorFlowUnsupportedOperator : Operator {
  TensorFlowUnsupportedOperator() : Operator(OperatorType::kUnsupported) {}

  // The original TF operation type. Used for diagnostic purposes.
  string tensorflow_op;
  // A boolean indicating if the unsupported op should be treated as quantized.
  bool quantized = false;
  // A boolean indicating if the unsupported op output should allow float values
  // in quantized mode.
  bool support_output_type_float_in_quantized_op = false;
  // Output data types
  std::vector<ArrayDataType> output_data_types;
  // Output shapes.
  std::vector<Shape> output_shapes;
};

// Softmax activation function.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Softmax
struct SoftmaxOperator : Operator {
  SoftmaxOperator() : Operator(OperatorType::kSoftmax) {}
  float beta = 0.f;
};

// LogSoftmax activation function.
//
// Inputs:
//   inputs[0]: required: the logits input array
//
// TensorFlow equivalent: LogSoftmax
struct LogSoftmaxOperator : Operator {
  LogSoftmaxOperator() : Operator(OperatorType::kLogSoftmax) {}

  // LogSoftmax can in principal have very large negative output, depending on
  // the input size.  However, input x_i that is less than x_max-10 is
  // accumulated as exp(x_i-x_max), which is truncated to zero.
  //
  // Since we effectively disregard smallish inputs in the normalizing factor,
  // we also drop them in the output (set to minimum output), and in doing so
  // make better use of the quantization range / resolution.
  static constexpr float kOutputRangeMin = -16.0;
};

// Cast operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Cast
struct CastOperator : Operator {
  CastOperator() : Operator(OperatorType::kCast) {}
  ArrayDataType src_data_type = ArrayDataType::kNone;
  ArrayDataType dst_data_type = ArrayDataType::kNone;
};

// Floor operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Floor
struct FloorOperator : Operator {
  FloorOperator() : Operator(OperatorType::kFloor) {}
};

// Ceil operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Ceil
struct CeilOperator : Operator {
  CeilOperator() : Operator(OperatorType::kCeil) {}
};

// Gather operator. It gathers slices from params according to indices.
// Only 1-D indices are supported at the moment.
//
// Inputs:
//   inputs[0]: required: the params array
//   inputs[1]: required: the indices to gather
//   inputs[2]: optional: axis
//
// TensorFlow equivalent: Gather
struct GatherOperator : Operator {
  GatherOperator() : Operator(OperatorType::kGather) {}
  // Axis is populated explicitly or implicitly from the axis input by
  // ResolveGatherAttributes. An empty axis indicates that the axis has not yet
  // be resolved.
  absl::optional<int> axis;

  // This field is not used by the standard TF Lite export but it is still need
  // for legacy Gather implementations.
  int input_rank = 0;
};

// GatherNd operator. It gathers slices from params according to indices.
//
// Inputs:
//   inputs[0]: required: the params array
//   inputs[1]: required: the indices to gather
//
// TensorFlow equivalent: GatherNd
struct GatherNdOperator : Operator {
  GatherNdOperator() : Operator(OperatorType::kGatherNd) {}
};

// ArgMax operator. It returns the index of the maximum value along axis.
//
// Inputs:
//   inputs[0]: required: the input tensor
//   inputs[1]: optional: 0-D (scalar) axis
//
// TensorFlow equivalent: ArgMax
struct ArgMaxOperator : Operator {
  ArgMaxOperator() : Operator(OperatorType::kArgMax) {}
  ArrayDataType output_data_type = ArrayDataType::kInt64;
};

// ArgMin operator. It returns the index of the minimum value along axis.
//
// Inputs:
//   inputs[0]: required: the input tensor
//   inputs[1]: optional: 0-D (scalar) axis
//
// TensorFlow equivalent: ArgMin
struct ArgMinOperator : Operator {
  ArgMinOperator() : Operator(OperatorType::kArgMin) {}
  ArrayDataType output_data_type = ArrayDataType::kInt64;
};

// ResizeBilinear operator. It resizes input images with bilinear interpolation.
// It does not support align_corners at the moment.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the new image size
//
// TensorFlow equivalent: ResizeBilinear
struct ResizeBilinearOperator : Operator {
  ResizeBilinearOperator() : Operator(OperatorType::kResizeBilinear) {}

  bool align_corners = false;
};

// ResizeNearestNeighborOperator operator. It resizes input images with nearest
// neighbor interpolation. It does not support align_corners at the moment.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the new image size
//
// TensorFlow equivalent: ResizeNearestNeighbor
struct ResizeNearestNeighborOperator : Operator {
  ResizeNearestNeighborOperator()
      : Operator(OperatorType::kResizeNearestNeighbor) {}

  bool align_corners = false;
};

// SpaceToBatchND operator. It divides spatial dimensions into a grid of
// blocks and interleaves these blocks with the batch dimension. Currently,
// only 2-d blocks are supported.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the block shape
//   inputs[2]: required: the paddings
//
// TensorFlow equivalent: SpaceToBatchND
struct SpaceToBatchNDOperator : Operator {
  SpaceToBatchNDOperator() : Operator(OperatorType::kSpaceToBatchND) {}

  std::vector<int> block_shape;
  std::vector<int> before_paddings;
  std::vector<int> after_paddings;
};

// BatchToSpaceND operator. Rearranges data from batch into blocks of
// spatial data. Currently, only 2-d blocks are supported.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the block shape
//   inputs[2]: required: the crops
//
// TensorFlow equivalent: BatchToSpaceND
struct BatchToSpaceNDOperator : Operator {
  BatchToSpaceNDOperator() : Operator(OperatorType::kBatchToSpaceND) {}

  std::vector<int> block_shape;
  std::vector<int> before_crops;
  std::vector<int> after_crops;
};

// Mean operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Mean
struct MeanOperator : Operator {
  MeanOperator() : Operator(OperatorType::kMean) {}

  std::vector<int> axis;
  bool keep_dims = false;
};

// Svdf operator:
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: weights_feature
//   inputs[2]: required: weights_time
//   inputs[3]: optional: bias
struct SvdfOperator : Operator {
  SvdfOperator() : Operator(OperatorType::kSvdf) {}
  int rank;
};

// TopKV2 operator.
//
// Inputs:
//    input tensor and top_k scalar.
struct TopKV2Operator : Operator {
  TopKV2Operator() : Operator(OperatorType::kTopK_V2) {}
};

// DynamicPartition operator:
//
// Inputs:
//  inputs[0]: required: data.
//  inputs[1]: required: partitions.
//
// TensorFlow equivalent: DynamicPartition
struct DynamicPartitionOperator : Operator {
  DynamicPartitionOperator() : Operator(OperatorType::kDynamicPartition) {}
  int num_partitions;
};

// DynamicStitch operator:
//
// Inputs:
//  inputs[0,N): required: indices.
//  inputs[N,2N): required: data.
//
// TensorFlow equivalent: DynamicStitch/ParallelDynamicStitch
struct DynamicStitchOperator : Operator {
  DynamicStitchOperator() : Operator(OperatorType::kDynamicStitch) {}
  int num_partitions;
};

// SparseToDense operator:
//
// Inputs:
// Inputs[0]: required: sparse_indices.
// Inputs[1]: required: output_shape.
// Inputs[2]: required: sparse_values.
//
// TensorFlow equivalent: SparseToDense.
struct SparseToDenseOperator : Operator {
  SparseToDenseOperator() : Operator(OperatorType::kSparseToDense) {}
  bool validate_indices;
};

// Pow operator:
//
// Inputs:
// Inputs[0]: required: A tensor.
// Inputs[1]: required: A tensor.
//
// TensorFlow equivalent: Pow.
struct PowOperator : Operator {
  PowOperator() : Operator(OperatorType::kPow) {}
};

// Any operator:
//
// Inputs:
// Inputs[0]: required: A boolean input tensor.
// Inputs[1]: required: reduction_indices.
//
// TensorFlow equivalent: tf.reduce_any.
struct TensorFlowAnyOperator : Operator {
  TensorFlowAnyOperator() : Operator(OperatorType::kAny) {}
  std::vector<int> axis;
  bool keep_dims = false;
};

// LogicalAnd operator:
//
// Inputs:
// Inputs[0]: required: A boolean tensor.
// Inputs[1]: required: A boolean tensor.
//
// TensorFlow equivalent: tf.logical_and.
struct LogicalAndOperator : Operator {
  LogicalAndOperator() : Operator(OperatorType::kLogicalAnd) {}
};

// LogicalNot operator:
//
// Inputs:
// Inputs[0]: required: A boolean tensor.
//
// TensorFlow equivalent: tf.logical_not.
struct LogicalNotOperator : Operator {
  LogicalNotOperator() : Operator(OperatorType::kLogicalNot) {}
};

// OneHot operator:
//
// Inputs:
// Inputs[0]: required: indices.
// Inputs[1]: required: depth.
// Inputs[2]: required: on_value.
// Inputs[3]: required: off_value.
//
// TensorFlow equivalent: OneHot.
struct OneHotOperator : Operator {
  enum Inputs {
    INDICES_INPUT = 0,
    DEPTH_INPUT = 1,
    ON_VALUE_INPUT = 2,
    OFF_VALUE_INPUT = 3,
  };

  OneHotOperator() : Operator(OperatorType::kOneHot) {}
  int axis = -1;
};

// LogicalOr operator:
//
// Inputs:
// Inputs[0]: required: A Bool tensor.
// Inputs[1]: required: A Bool tensor.
//
// TensorFlow equivalent: LogicalOr.
struct LogicalOrOperator : Operator {
  LogicalOrOperator() : Operator(OperatorType::kLogicalOr) {}
};

// Unpack operator:
//
// Inputs:
// Inputs[0]: required: A boolean input tensor.
// Inputs[1]: required: reduction_indices.
//
// TensorFlow equivalent: tf.unstack.
struct UnpackOperator : Operator {
  UnpackOperator() : Operator(OperatorType::kUnpack) {}
  int num;
  int axis;
  ArrayDataType dtype = ArrayDataType::kNone;
};

// ZerosLike operator:
//
// Inputs:
// inputs[0]: required: the input array
//
// TensorFlow equivalent: tf.zeros_like
struct TensorFlowZerosLikeOperator : Operator {
  TensorFlowZerosLikeOperator() : Operator(OperatorType::kZerosLike) {}
};

// ReverseV2 operator:
//
// Inputs:
// Inputs[0]: required: the input array.
//
// TensorFlow equivalent: ReverseV2.
struct ReverseV2Operator : Operator {
  ReverseV2Operator() : Operator(OperatorType::kReverseV2) {}
};

enum class MirrorPadMode { kNone, kSymmetric, kReflect };

// MirrorPad Operator:
//
// Inputs:
// Inputs[0]: required: input tensor to be padded.
// Inputs[1]: required: 2 Column matrix specifying padding sizes. The number of
// rows must be the same as the rank of the input.
// Inputs[2]: required: REFLECT or SYMMETRIC.
//
// TensorFlow equivalent: MirrorPad.
struct MirrorPadOperator : Operator {
  MirrorPadOperator() : Operator(OperatorType::kMirrorPad) {}
  // mode is either SYMMETRIC or REFLECT.
  MirrorPadMode mode;
};

// ReverseSequence operator:
//
// Inputs:
// Inputs[0]: required: the input array.
// Inputs[1]: required: the lengths of the elements to be reversed.
//
// TensorFlow equivalent: tf.reverse_sequence.
struct ReverseSequenceOperator : Operator {
  ReverseSequenceOperator() : Operator(OperatorType::kReverseSequence) {}
  int seq_dim;
  int batch_dim = 0;
};

// Unique Operator:
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Unique
struct UniqueOperator : Operator {
  UniqueOperator() : Operator(OperatorType::kUnique) {}
  ArrayDataType idx_out_type = ArrayDataType::kInt32;
};

struct UnidirectionalSequenceRnnOperator : Operator {
  UnidirectionalSequenceRnnOperator()
      : Operator(OperatorType::kUnidirectionalSequenceRnn) {}
  bool time_major;
  FusedActivationFunctionType fused_activation_function;
};

// Where Operator:
// Return the coordinates of the true values in condition tensor in row-major
// order.
//
// Inputs:
//  inputs[0]: required: boolean condition tensor
//
//  TensorFlow equivalent: Where
struct WhereOperator : Operator {
  WhereOperator() : Operator(OperatorType::kWhere) {}
};

// Matrix Diag Operator:
// Construct a batched diagonal tensor with given batched diagonal values.
// Inputs: A tensor of values that will be on the diagonal of the returned
//         tensor.
struct MatrixDiagOperator : Operator {
  MatrixDiagOperator() : Operator(OperatorType::kMatrixDiag) {}
};

// Matrix Set Diag Operator:
// Construct a batched diagonal tensor with given input and diagonal values.
// Input is a rank (k+1) tensor of values.
// diagonal is a rank (k) tensor of values that will be on the diagonal
// of the returned output. Output is rank k+1.
//         tensor.
struct MatrixSetDiagOperator : Operator {
  MatrixSetDiagOperator() : Operator(OperatorType::kMatrixSetDiag) {}
};

// Alloc's are used for transient arrays only. An Alloc specifies which interval
// of the "transient_data" workspace buffer passed to inference functions, is to
// be used for the transient array at hand. The 'start' and 'end' values are
// offsets from the start of the workspace buffer, expressed in bytes.
struct Alloc {
  int64 start = 0;
  int64 end = 0;
};

inline bool operator<(const Alloc& a, const Alloc& b) {
  return a.start < b.start;
}

// Array represents an array (either a constant parameter array or an
// activations array) in a Model.
struct Array {
  template <ArrayDataType A>
  const Buffer<A>& GetBuffer() const {
    DCHECK(buffer);
    DCHECK(buffer->type == A);
    return *static_cast<const Buffer<A>*>(buffer.get());
  }
  template <ArrayDataType A>
  Buffer<A>& GetMutableBuffer() {
    if (!buffer) {
      Buffer<A>* ptr = new Buffer<A>;
      buffer = std::unique_ptr<GenericBuffer>(ptr);
    }
    DCHECK(buffer);
    DCHECK(buffer->type == A);
    return *static_cast<Buffer<A>*>(buffer.get());
  }
  Alloc& GetOrCreateAlloc() {
    if (!alloc) {
      alloc = std::unique_ptr<Alloc>(new Alloc);
    }
    return *alloc;
  }
  MinMax& GetOrCreateMinMax() {
    if (!minmax) {
      minmax = std::unique_ptr<MinMax>(new MinMax);
    }
    return *minmax;
  }
  MinMax& GetMinMax() const {
    DCHECK(minmax);
    return *minmax;
  }
  QuantizationParams& GetOrCreateQuantizationParams() {
    if (!quantization_params) {
      quantization_params =
          std::unique_ptr<QuantizationParams>(new QuantizationParams);
    }
    return *quantization_params;
  }
  QuantizationParams& GetQuantizationParams() const {
    DCHECK(quantization_params);
    return *quantization_params;
  }

  // The data type of the actual elements of this array, that is:
  //  - If there is a buffer (see 'buffer' member), it must be of the same
  //    type.
  //  - If there is no buffer, meaning that this is a runtime (i.e. activations)
  //    array, then this specifies the type of elements that there will be
  //    at runtime.
  //
  // Note that this only specifies the storage type of elements; this does
  // not specify whether these are to be treated as 'real' or 'quantized'
  // values.
  // That is decided by whether the 'quantization_params' member is null.
  ArrayDataType data_type = ArrayDataType::kNone;
  // The final value that data_type should have at the end of graph
  // transformations
  ArrayDataType final_data_type = ArrayDataType::kNone;
  // The dimensions of this array --- this specifies both sizes and strides
  // (the storage layout).
  //
  // Issues with shape handling that remain include:
  //   - No way to distinguish between 0-dimensional dims and missing dims.
  //   - No way to describe dims that may be runtime-variable.
  //   - Addressing of dims by integer index differs in different graph formats
  //     (TensorFlow vs. other frameworks vs. what we have informally grown
  //     within toco).
  //     This is currently quite messy; see ReorderAxesOperator which is how we
  //     bridge some of these discrepancies at the moment. This is overdue for
  //     a redesign; I'm thinking that it would be nice to have more flexible
  //     dims that allow mapping 1:1, cleanly, dims as they are in various
  //     formats,
  //     then explicitly convert between different conventions.

  // Proto-style accessors
  bool has_shape() const { return array_shape != nullptr; }
  const Shape& shape() const {
    CHECK(has_shape());
    return *array_shape;
  }
  Shape* mutable_shape() {
    if (!array_shape) {
      array_shape.reset(new Shape);
    }
    return array_shape.get();
  }
  void copy_shape(const Shape& src_shape) { *mutable_shape() = src_shape; }
  void clear_shape() { array_shape = nullptr; }

  // The constant buffer backing this array. This is non-null if and only if
  // this is a constant parameter array. Conversely, this is null for
  // activations arrays.
  //
  // Note that this buffer is pure storage. In the case of quantized values,
  // it only stores the quantized values, it does not know by itself about the
  // quantization parameters necessary to interprete these values, that is
  // in the separate 'quantization_params' field. In fact, this 'buffer' field
  // does no even know whether values are quantized. It only has a data_type,
  // which must equal the 'data_type' member here, and which only describes
  // the storage type of element, does not tell whether they are quantized i.e.
  // whether they are to be interpreted with quantization_params.
  std::unique_ptr<GenericBuffer> buffer;
  // Only for activation arrays (i.e. when 'buffer' is null).
  // Only for code generation.
  //
  // Describes the allocation of this array within the workspace buffer
  // allocated
  // for all transient arrays.
  std::unique_ptr<Alloc> alloc;
  // Describes the [min, max] range of values
  // to be assumed when determining quantization_params.
  //
  // Only used for quantization. In fact, only used for determining
  // quantization_params.
  //
  // Used for both constant arrays (those having a 'buffer') and non-constant
  // arrays (activations). Indeed, it is important to use the same min-max range
  // as was used during training, even if that min-max range is slightly wrong
  // w.r.t. actual buffer elements. Doing otherwise would defeat the point of
  // re-training for quantization.
  std::unique_ptr<MinMax> minmax;
  // Quantization parameters. The non-null-ness of this pointer is what
  // defines whether this array is quantized or not.
  //
  // If this is non-null, then these quantization parameters are to be used
  // to assign a meaning as real numbers to the elements of this array.
  std::unique_ptr<QuantizationParams> quantization_params;
  // narrow_range is a detail of how toco handles FakeQuant operators with
  // narrow_range, see
  // https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_vars
  //
  // For more context about what that is useful for, see the big comment in
  // graph_transformations/ensure_uint8_weights_safe_for_fast_int8_kernels.cc
  //
  // The narrow_range flag applies only to quantized arrays, and changes
  // their quantization in the following way when it is set to 'true':
  // 1. The computation of {zero_point, scale} from {min, max} needs to be
  //    amended so that the real min value will get quantized to
  //    (min_quantized_value + 1) instead of just (min_quantized_value).
  //    E.g. for uint8 quantization, the real min value should get quantized to
  //    the uint8 value 1, not 0.
  // 2. Quantized values should get clamped to the interval
  //    [min_quantized_value + 1, max_value]. Equivalently, the
  //    min_quantized_value should get nudged to (min_quantized_value + 1).
  // The reason why 1. does not imply 2. is that real values may not belong to
  // the stated [min, max] interval. Concretely, weights recorded at the last
  // learning step may not fall in the [min, max] interval recorded over
  // previous learning steps, as the values evolve across learning steps.
  //
  // Rationale why this is directly a field on Array:
  // - This can't be just a field on FakeQuantOperator, because
  //   FakeQuantOperators are gone (DropFakeQuant) before we get to using that
  //   information (Quantize). We need a place to store that bit in the interim.
  // - This can't be in QuantizationParams because we need to record this
  //   ahead of quantization, and QuantizationParams are only created during
  //   quantization.
  // - This could be in MinMax, but that would be an abuse of what MinMax is
  //   about, and would break existing code that assumes that a MinMax is just
  //   a min and a max. Unlike MinMax which is agnostic as to the quantized
  //   data type, narrow_range refers to values in the quantized data type.
  bool narrow_range = false;

 private:
  std::unique_ptr<Shape> array_shape;
};

// Our Model struct, represents an entire model (our "top-level" struct).
// Owns everything.
class Model {
 public:
  using ArrayMap = std::unordered_map<string, std::unique_ptr<Array>>;

  bool HasArray(const string& name) const { return arrays.count(name) > 0; }
  Array& GetArray(const string& name) const {
    DCHECK(HasArray(name)) << "Array not found: " << name;
    return *arrays.at(name);
  }
  Array& GetOrCreateArray(const string& name) {
    // Make sure name is not used by an optional array
    DCHECK(!optional_arrays.count(name));
    if (!HasArray(name)) {
      Array* ptr = new Array;
      arrays[name] = std::unique_ptr<Array>(ptr);
    }
    Array& result = GetArray(name);
    return result;
  }
  void CreateOptionalArray(const string& name) {
    DCHECK(!arrays.count(name) && !optional_arrays.count(name));
    optional_arrays.insert(name);
  }
  bool IsOptionalArray(const string& name) const {
    return optional_arrays.count(name);
  }

  // Note that this invalidates all array iterators.
  void EraseArray(const string& name) { arrays.erase(name); }
  void EraseArrays(std::function<bool(const string&)> discardable) {
    for (auto it = arrays.begin(); it != arrays.end();) {
      if (discardable(it->first)) {
        it = arrays.erase(it);
      } else {
        ++it;
      }
    }
  }
  const ArrayMap& GetArrayMap() const { return arrays; }
  ArrayMap& GetMutableArrayMap() { return arrays; }

  int64 ArithmeticOpsCount() const { return ops_count; }

  // Optional arrays are used for optional tensors,
  // these tensors do not have data, but with reserved names as op inputs.
  std::set<string> optional_arrays;

  // The list of operators. Notice how it's a list of unique_ptr's, implying
  // that the Model is what owns Operator's and keeps them alive.
  std::vector<std::unique_ptr<Operator>> operators;

  // Generic flags, a place where we combine information passed to us via
  // command-line parameters (e.g. --input_width=N) with information that
  // we may or may not find in the input model file.
  ModelFlags flags;
  // For code-generation only: required size of the transient_data buffer
  std::size_t transient_data_size = 0;
  // For code-generation only: required alignment of the transient_data buffer
  std::size_t transient_data_alignment = 0;
  // Arithmetic operations performed in the model.
  int64 ops_count = 0;

 private:
  // The associative array mapping names to Array's.
  // Notice how it's a container of unique_ptr's, implying
  // that the Model is what owns Array's and keeps them alive.
  // The Operator's refer to these Array's by their name strings, not by their
  // addresses. See Operator::inputs, Operator::outputs.
  std::unordered_map<string, std::unique_ptr<Array>> arrays;
};

// OperatorSignature contains the information required to making versioning
// decisions.
struct OperatorSignature {
  // The operator.
  const Operator* op;

  // The model in which the operator resides.
  const Model* model;
};
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_MODEL_H_
