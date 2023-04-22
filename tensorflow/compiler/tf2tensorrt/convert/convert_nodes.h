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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_NODES_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_NODES_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

namespace convert {
using ::stream_executor::port::StatusOr;

struct EngineConnection {
  // Constructs a non-control edge.
  EngineConnection(const string& outside, int out_id, int out_port,
                   const string& inside, int in_id, int in_port,
                   bool input_edge, int port)
      : outside_node_name(outside),
        outside_id(out_id),
        outside_port(out_port),
        inside_node_name(inside),
        inside_id(in_id),
        inside_port(in_port),
        is_input_edge(input_edge),
        port_number(port) {}

  // Constructs a control edge.
  EngineConnection(const string& outside, int out_id, const string& inside,
                   int in_id, bool input_edge)
      : outside_node_name(outside),
        outside_id(out_id),
        outside_port(Graph::kControlSlot),
        inside_node_name(inside),
        inside_id(in_id),
        inside_port(Graph::kControlSlot),
        is_input_edge(input_edge),
        port_number(Graph::kControlSlot) {}

  bool is_control_edge() const { return port_number == Graph::kControlSlot; }

  const string outside_node_name;
  const int outside_id;
  const int outside_port;
  PartialTensorShape outside_shape;  // Only set for input edge.

  const string inside_node_name;
  const int inside_id;
  const int inside_port;
  PartialTensorShape inside_shape;  // Only set for output edge.

  DataType connection_type;
  const bool is_input_edge;

  // The port number of the TRT node connected with this edge.
  const int port_number;
};

struct EngineInfo {
  EngineInfo()
      : engine_type(EngineType::TRTStatic),
        max_workspace_size_bytes(0),
        max_batch_size(absl::nullopt),
        maximum_cached_engines(0),
        precision_mode(TrtPrecisionMode::FP32),
        use_calibration(true),
        allow_build_at_runtime(true) {}

  string engine_name;
  string device;
  GraphDef segment_graph_def;

  // Non-control input connections inside this vector are sorted in a way such
  // that, the segment nodes connecting to them are topological sorted.
  // In addition, for non-control connections, there must be no duplicates.
  std::vector<EngineConnection> connections;

  enum class EngineType { TRTStatic = 0, TRTDynamic = 1 };
  EngineType engine_type;
  int64 max_workspace_size_bytes;
  absl::optional<int> max_batch_size;
  int maximum_cached_engines;
  TrtPrecisionMode precision_mode;
  bool use_calibration;
  bool allow_build_at_runtime;
};

// Constructs a graphdef from the segment in the given graph. Adds _Arg
// nodes for input edges (InputPH_*) and _Retval nodes for output edges
// (OutputPH_*). This function needs to be called before TensorRT nodes
// inserted in order to correctly get sizes from the original graph.
//
// - subgraph_node_names: the node names of the subgraph.
// - subgraph_node_ids: the node ids of the subgraph, must be sorted in
//   topological order.
// - segment_def: the output GraphDef, whose non-input/output nodedefs will be
//   sorted in topological order.
// - scope_name: the name of the scope where the TRTEngineOp will be placed.
//
// TODO(aaroey): add tests to validate these properties.
Status ConvertSegmentToGraphDef(
    const Graph* graph, const grappler::GraphProperties& graph_properties,
    const std::vector<const Node*>& subgraph_nodes,
    std::vector<EngineConnection>* connections, GraphDef* segment_def,
    string* scope_name);

// Converts given subgraph to a TRT engine saved in 'engine'. Returns ok iff
// 'builder' successfully build the engine. If the result is not ok, 'engine'
// will be set to nullptr
// Once returned, 'builder' is not needed any more and can be safely destroyed.
//
// - convert_successfully: indicates whether the conversion to TensorRT network
//   is successful. This is different than successfully building the engine:
//   building can still fail afterwards.
Status ConvertGraphDefToEngine(
    const GraphDef& gdef, TrtPrecisionMode precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<PartialTensorShape>& input_shapes,
    nvinfer1::ILogger* logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, bool use_calibration,
    const bool use_implicit_batch, bool* convert_successfully,
    TrtShapeOptimizationProfile* profiles, absl::string_view engine_name);

// Helper class for the segmenter to determine whether an output edge from the
// TRT segment is valid.
class OutputEdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an output edge of the
  // TRT segment.
  bool operator()(const Edge* out_edge) const;
};

int64_t TrtTensorDimsNumElements(const nvinfer1::Dims& dims);

// Class to convert TF compile-time constants (e.g. Const nodes) to TRT weight.
class TRT_ShapedWeights {
 public:
  explicit TRT_ShapedWeights(
      nvinfer1::DataType type = nvinfer1::DataType::kFLOAT);

  // Copy from another weights.
  //
  // NOTE: this does not copy the underlying buffer but only increase its
  // reference count.
  TRT_ShapedWeights(const TRT_ShapedWeights& rhs);

  nvinfer1::Weights GetTrtWeights() const;

  const Tensor& GetTensor() const { return tensor_; }

  // Returns the raw pointer to the underlying buffer which holds the weights
  // value.
  void* GetValues() const {
    return const_cast<char*>(tensor_.tensor_data().data());
  }

  // Fills all the weight values with value.
  template <typename T>
  Status SetValues(T value);

  Status SetShape(nvinfer1::Dims dims);

  // Returns total number of elements. Returning 0 means either some dim is 0
  // or the number of dims is 0. Note that a TF scalar constant is marked as
  // Dims{0, {1}}, and has a count() == 1.
  int64_t count() const { return count(shape_); }

  // Returns the total number of elements in a weight with shape dims.
  static int64_t count(nvinfer1::Dims dims);

  size_t size_bytes() const;

  string DebugString() const;

  template <typename T>
  absl::Span<const T> GetSpan() const {
    return absl::Span<const T>(tensor_.flat<T>().data(), count());
  }

  template <typename T>
  std::vector<T> ToVector() const {
    auto span = GetSpan<T>();
    return std::vector<T>(span.data(), span.data() + span.size());
  }

  nvinfer1::DataType TrtDType() const { return type_; }

  // TODO(aaroey): make these private.
  // Before TRT 6, scalar weights are not supported. In that case a TF scalar
  // constant tensor is represented via TRT_ShapedWeights::shape_ = {1,{1}}.
  //
  // Starting TRT 6, scalar weights are supported, a scalar constant tensor is
  // represented via TRT_ShapedWeights::shape_ = {0, {1}}.
  nvinfer1::Dims shape_;  // Note: shape.type[] is not used.

 private:
  // This constructor is only used by TrtWeightStore, which creates the
  // underlying buffer.
  TRT_ShapedWeights(nvinfer1::DataType type, nvinfer1::Dims dims,
                    Tensor tensor);

  nvinfer1::DataType type_;

  // All weights should be stored inside TrtWeightStore to make sure lifetime of
  // all the underlying tensors are available until the engine is built. For
  // this reason, tensor_ should never be reassigned to a different value that
  // is not already present in the TrtWeightStore.
  Tensor tensor_;

  friend class TrtWeightStore;
};

// Container for TRT_ShapedWeights. We need this container because, TRT doesn't
// manage the lifetime of the weights buffer, it only keeps a pointer to it and
// requires that the data referenced by the pointer be available until the
// building of engine is complete. For more information see
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
//
// TODO(laigd): consider adding garbage collection to the unused weights.
class TrtWeightStore {
 public:
  // Get a TRT_ShapedWeights with 'type' and 'dims'.
  TRT_ShapedWeights GetTempWeights(nvinfer1::DataType trt_type,
                                   const nvinfer1::Dims& dims);

  // Get a TRT_ShapedWeights with the same data type and dimensions as
  // 'weights'.
  TRT_ShapedWeights GetTempWeights(const TRT_ShapedWeights& weights) {
    return GetTempWeights(weights.TrtDType(), weights.shape_);
  }

 private:
  // The backend storage of the TRT_ShapedWeights.
  std::vector<Tensor> store_;
};

// Represents a TRT-style input to a TF node, it can be either a
// ITensorProxyPtr (representing nvinfer1::ITensor* or SimpleITensor),
// or TRT_ShapedWeights which is compile-time constant.
//
// TODO(laigd): maybe rename it to TrtArgument, or mimic XlaCompiler::Argument.
class TRT_TensorOrWeights {
 public:
  TRT_TensorOrWeights() {}
  TRT_TensorOrWeights(ITensorProxyPtr);
  TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size);

  // Constructor that makes it an ITensor, doesn't take ownership of 'tensor'.
  // This is used by Converter when building the TRT network, where the ITensor
  // is owned by the TRT network being built. See comment for 'trt_tensor_'
  // in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::ITensor* tensor, int batch_size = -1);

  // Constructor that makes it an ITensor by creating one using provided data
  // type and shape, and takes ownership of the created ITensor. This is used by
  // TrtNodeValidator to encapsulate the type and shape information for
  // validation of graph nodes, and the created ITensor is fake and temporary,
  // and should not be used to build any TRT network. See comment for
  // 'simple_tensor_' in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                               const nvinfer1::Dims& trt_dims, int batch_size);

  // Constructor that makes it a TRT_TensorOrWeights.
  explicit TRT_TensorOrWeights(const TRT_ShapedWeights& weights);

  TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs);

  void operator=(const TRT_TensorOrWeights& rhs);

  bool is_tensor() const { return initialized_ && is_tensor_; }
  bool is_weights() const { return initialized_ && !is_tensor_; }

  ITensorProxyPtr tensor() const;

  TRT_ShapedWeights& weights() {
    CHECK(is_weights());
    return weights_;
  }

  const TRT_ShapedWeights& weights() const {
    CHECK(is_weights());
    return weights_;
  }

  nvinfer1::Dims GetTrtDims() const;

  Status GetTfType(DataType* tf_type) const;

  int batch_size() const { return batch_size_; }

  string DebugString() const;

 private:
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }

  // First dimension of the TF tensor (NOT tensor_) that is represented by
  // tensor_ is treated as the "batch dimension" by TRT, and tensor_'s
  // dimensions (obtained via tensor_->getDimensions()) do not contain the batch
  // dimension. For example, when a TF tensor with shape (A,B,C) is represented
  // in TRT, tensor_->getDimensions() will be (B,C) and batch_size_ will be A.
  //
  // This requires that all tensors in the subgraph that is converted to a TRT
  // engine have the same batch size are represented by the first dimension of
  // their shape, and Converter will verify this during conversion. The drawback
  // is that currently it cannot convert a graph that doesn't have the batch
  // size represented in the shapes or the batch sizes are different. See
  // b/118387490 for more details.
  //
  // If use_implicit_batch is false, batch_size_ is unused and
  // tensor_->getDimensions() will contain the entire shape (A,B,C).
  ITensorProxyPtr tensor_proxy_ptr_ = nullptr;
  int batch_size_ = -1;

  TRT_ShapedWeights weights_;
  bool initialized_ = false;
  bool is_tensor_ = false;

  friend class Converter;
};

class Converter;

// Parameters for each op converter.
struct OpConverterParams {
  // Constructor used for validation only.
  OpConverterParams(const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store,
                    TrtPrecisionMode precision_mode, bool use_calibration,
                    bool use_implicit_batch);

  // Constructor used for conversion.
  OpConverterParams(Converter* converter, const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store);

  Converter* converter = nullptr;
  const NodeDef& node_def;
  const std::vector<TRT_TensorOrWeights>& inputs;
  std::vector<TRT_TensorOrWeights>* outputs;
  const bool validation_only;
  TrtWeightStore* weight_store;
  const TrtPrecisionMode precision_mode;
  const bool use_calibration;
  const bool use_implicit_batch;
};

using OpConverter = std::function<Status(OpConverterParams*)>;

// Class to verify if specific TF node is supported by TRT.
class TrtNodeValidator {
 public:
  // 'graph_properties' is the GraphProperties of the graph whose nodes will be
  // checked by IsTensorRTCandidate() later. It is used to get the shape and
  // data type information of a tensor for validation purpose.
  TrtNodeValidator(const grappler::GraphProperties& graph_properties,
                   TrtPrecisionMode precision_mode, bool use_calibration,
                   bool use_implicit_batch);

  // Returns OK iff 'node' is a TF-TRT conversion candidate, which will be added
  // to TRT subgraph and later converted into TRT engine.
  Status IsTensorRTCandidate(const Node* node);

 private:
  static const std::set<string>* quantize_ops;

  void RegisterOpValidators();

  // Convert a Const node to a TRT_TensorOrWeights.
  Status ConvertConstToWeights(const NodeDef& const_node_def,
                               const std::vector<TRT_TensorOrWeights>& inputs,
                               TRT_TensorOrWeights* output);

  // Convert the output tensor at 'output_port' of 'node_def' to a
  // TRT_TensorOrWeights which will be later used as an input to other nodes and
  // passed to ValidateNode() below.
  Status ConvertToTensorOrWeights(const NodeDef& node_def, int output_port,
                                  TRT_TensorOrWeights* tensor_or_weights);

  // Stores all the validators by op type. If no validator is registered for
  // specific op, it means no validation is needed and ValidateNode() will
  // return OK.
  std::unordered_map<string, OpConverter> op_validators_;

  // Store the weights added during validation. Some validations (e.g.
  // validation for Const node) may produce weights.
  TrtWeightStore weight_store_;

  // GraphProperties of the graph whose nodes are to be validated by
  // IsTensorRTCandidate().
  const grappler::GraphProperties& graph_properties_;

  // Quantization ops are only converted when using quantized precisions.
  const TrtPrecisionMode precision_mode_;

  const bool use_calibration_;

  const bool use_implicit_batch_;

  friend class ValidatorTest;
  friend class OpConverterTest;
};

// Class to convert TF nodes to TRT network.
class Converter {
 public:
  // Used for Converter::RenameAndMarkOutputTensors()
  struct EngineOutputInfo {
    // The TRT tensor name which produces the output.
    string source_tensor_name;
    // The TensorFlow node name which is receiving the output from the TRT
    // engine. This should always be the Identity node created in
    // ConvertSegmentToGraphDef.
    string dest_node_name;
    // Output type. TensorRT requires this to be explicitly set for engine
    // outputs.
    nvinfer1::DataType trt_dtype;
  };

  static StatusOr<std::unique_ptr<Converter>> Create(
      TrtPrecisionMode precision_mode, bool use_calibration,
      nvinfer1::ILogger* trt_logger, const bool use_implicit_batch,
      absl::string_view engine_name);

  //////////////////////////////////////////////////////////////////////////////
  // Methods used by the TRT engine builder to build a TRT network from a TF
  // function/subgraph.

  // Convert the node to TRT network.
  Status ConvertNode(const NodeDef& node_def);

  // Add input tensor to the TRT network with given 'name', 'dtype', 'dims' and
  // 'batch_size'.
  Status AddInputTensor(const string& name, nvinfer1::DataType dtype,
                        const nvinfer1::Dims& dims, int batch_size);

  // Mark the tensors with names specified by source_tensor_name as output of
  // the TRT network, and set their names in the TRT network as dest_node_name.
  Status RenameAndMarkOutputTensors(
      const std::vector<EngineOutputInfo>& output_tensors);

  // Build a TRT engine using the created network.
  Status BuildCudaEngine(TrtUniquePtrType<nvinfer1::ICudaEngine>* engine,
                         int max_batch_size, size_t max_workspace_size_bytes,
                         nvinfer1::IGpuAllocator* allocator,
                         TRTInt8Calibrator* calibrator,
                         TrtShapeOptimizationProfile* profiles);

  //////////////////////////////////////////////////////////////////////////////
  // Methods used by op converters to convert individual TF node and add layers
  // to the TRT network.

  // Op converters (e.g. ConvertReshape) need to access the TRT network in order
  // to add TRT layers.
  nvinfer1::INetworkDefinition* network() { return trt_network_.get(); }

  // What precision are we targeting?
  TrtPrecisionMode precision_mode() const { return precision_mode_; }

  // Calibration will be or was previously performed on this network?
  bool use_calibration() const { return use_calibration_; }

  // Whether implicit batch mode is enabled
  bool use_implicit_batch() const { return use_implicit_batch_; }

  // This should be called on the inputs and outputs of any layer we create
  // where we know that the quantization range does not change during that
  // operation. (e.g. Reshape, Transpose, Identity, MaxPool).
  void MarkQuantizationRangesAsInferrable(ITensorProxyPtr* input,
                                          ITensorProxyPtr* output);

  // This function should be called when we know the quantization range of a
  // tensor, either from a quantize/dequantize node or when the output is a
  // fixed range (e.g. SoftMax, Relu6, Sigmoid).
  void ProvideQuantizationRange(ITensorProxyPtr* tensor, float min_range,
                                float max_range);

  // Should be called when full TRT network has been constructed and before
  // building the engine.
  void MaybeApplyQuantizationRanges();

  // Below are helper methods for op converters to add different layers to the
  // TRT network.

  // Transpose 'input_tensor' with given permutation 'order_with_batch_dim' to
  // 'output_tensor'. The permutation 'order_with_batch_dim' contains the batch
  // dimension which should always be 0. If this is for adding a transpose layer
  // to support the conversion of 'node_def', callers need to provide a
  // non-empty 'sub_op_name' appended to the name of 'node_def' to avoid layer
  // name conflicts.
  Status TransposeTensor(ITensorProxyPtr input_tensor,
                         const std::vector<int>& order_with_batch_dim,
                         ITensorProxyPtr* output_tensor,
                         const NodeDef& node_def,
                         absl::string_view sub_op_name = "");

  // Reshapes a dynamic shape tensor by removing or adding dimensions of size 1,
  // and/or permuting the dimensions. The new shape is derived from the shape of
  // the input tensor according to the slices and size_for_added_dims arguments.
  //
  // If there would be at most one unknown dimension, we could set the new shape
  // using IShuffleLayer::setReshapeDimensions, which treats -1 as a special
  // value (the same way as TF). In general, we can have more than one unknown
  // dimensions, and we have to manipulate the shape tensors during runtime to
  // define the new shape. This helper function defines the necessary shape
  // inference layers and calls reshape using the calculated new shape.
  //
  // Example:
  //
  // Assume that we want to reshape a tensor from shape {A,B,C,D} to {C,D,A,B}
  // (no transpose, just change the shape). In dynamic shape mode, the A,B,C,D
  // values are not necessarily known at conversion time, they can be all -1. We
  // can only define the new shape at runtime, when the actual shape is already
  // known. To define the new shape:
  // - We use an IShapeLayer to retrieve a shape tensor with the {A,B,C,D}
  //   values.
  // - Create two slices {C,D} and {A,B} of the shape tensor.
  // - Concatenate these slices {C,D,A,B},
  // - Set the {C,D,A,B} shape tensor as an input shape tensor for
  // IShuffleLayer.
  //
  // This can be achieved by calling DynamicReshape(input, {{2,4},{0,2}},
  // params).
  //
  // Before each slice we can insert new dims if the corresponding
  // size_for_added_dims element is not negative. The size_for_added_dims array
  // can have more than slices.size() elements, in order to insert a dimension
  // after the last slice. For example, to add two leading 1 dimensions, and
  // three trailing 1 dimensions, call DynamicReshape(input, {{0,nbDims}},
  // {2, 3}).
  //
  // Parameters:
  // input - input tensor
  // slices - [start, end) pairs of slices
  // params - conversion parameters
  // output - reshaped tensor
  // size_for_added_dims - size of dimension inserted right before slice[i]. We
  //   only insert a new dim if size_for_added_dims[i] >= 0.
  Status DynamicReshape(ITensorProxyPtr input,
                        std::vector<std::pair<int, int>> slices,
                        OpConverterParams* params, ITensorProxyPtr* output,
                        std::vector<int> size_for_added_dims = {},
                        absl::optional<int> op_instance = absl::nullopt);

  // Inserts a singleton dimension at axis for a dynamic shape tensor.
  Status DynamicExpandDims(ITensorProxyPtr input, const nvinfer1::Dims& dims,
                           int axis, OpConverterParams* params,
                           ITensorProxyPtr* output,
                           absl::optional<int> op_instance = absl::nullopt);

  // Helper function to add a squeeze op to the network.
  //
  // The input_dims argument stores the TRT dimensions of the input tensor,
  // where the dimensions to be squeezed are replaced by 0.
  Status SqueezeTensor(ITensorProxyPtr input, std::vector<int>* input_dims,
                       OpConverterParams* params, ITensorProxyPtr* output);

  // Creates an IConstantLayer using 'weights' whose dimensions are specified by
  // 'dims', and returns the output ITensor.
  ITensorProxyPtr CreateConstantLayer(const TRT_ShapedWeights& weights,
                                      const nvinfer1::Dims& dims);

  // Gets the min and max value in a TRT_ShapedWeights
  Status GetWeightRange(const TRT_ShapedWeights& weights, float* out_min,
                        float* out_max) const;

  // Constructs a name and passed it to the TensorRT layer to support xprof.
  void SetLayerName(
      nvinfer1::ILayer* layer, const NodeDef& node_def,
      absl::string_view sub_op_name = "",
      absl::optional<int> sub_op_instance = absl::nullopt,
      absl::optional<std::string> origin_node_name = absl::nullopt);

  void SetLayerName(nvinfer1::ILayer* layer, absl::string_view main_op_name,
                    absl::string_view sub_op_name,
                    absl::optional<int> sub_op_instance = absl::nullopt);

 private:
  Converter(TrtPrecisionMode precision_mode, bool use_calibration,
            nvinfer1::ILogger* trt_logger, const bool use_implicit_batch,
            absl::string_view engine_name);

  Status Init(nvinfer1::ILogger* trt_logger);

  // Verify the provided batch_size is consistent with batch_size_ and update it
  // if necessary.
  Status MaybeUpdateBatchSize(int batch_size);

  // Add the provided tensor/weights to the map trt_tensors_.
  Status AddTensorOrWeights(const string& name, TRT_TensorOrWeights input);

  // Get the tensor/weights from trt_tensors_ by 'name'.
  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output);

  // Get the inputs of 'node_def' from trt_tensors_.
  Status GetInputs(const NodeDef& node_def,
                   std::vector<TRT_TensorOrWeights>* inputs) const;

  void RegisterOpConverters();

  void PropagateQuantizationRanges();

  // Registered op converters by op type.
  std::unordered_map<string, OpConverter> op_registry_;

  // Tensors/weights added during construction of trt_network_.
  std::unordered_map<string, TRT_TensorOrWeights> trt_tensors_;

  // The TRT builder used to create the network and build the engine. Not owned.
  TrtUniquePtrType<nvinfer1::IBuilder> trt_builder_;

  // The TRT network being built.
  TrtUniquePtrType<nvinfer1::INetworkDefinition> trt_network_;

  // Store the weights added during construction of trt_network_.
  TrtWeightStore weight_store_;

  // During conversion, this table is populated with quantization ranges per
  // tensor. MaybeApplyQuantizationRanges() will use this table to set the TRT
  // quantization ranges. Since TRT only supports symmetric ranges, we will
  // store the range as a single float = max(abs(min_range), abs(max_range)).
  // Range refers to the floating point values, e.g. min_range = 0.0f, max_range
  // = 6.0f for Relu6.
  std::unordered_map<ITensorProxyPtr*, float> quantization_ranges_proxy_;
  std::unordered_map<nvinfer1::ITensor*, float> quantization_ranges_;

  // Edges where quantization ranges can be inferred (copied) across ops - from
  // first tensor to second tensor. PropagateQuantizationRanges() will propagate
  // known ranges from quantization_ranges_ across these edges, adding the new
  // ranges to quantization_ranges_ so that they can be applied in
  // MaybeApplyQuantizationRanges().
  std::vector<std::pair<ITensorProxyPtr*, ITensorProxyPtr*>>
      quantization_infer_proxy_;
  std::vector<std::pair<nvinfer1::ITensor*, nvinfer1::ITensor*>>
      quantization_infer_;

  const TrtPrecisionMode precision_mode_;

  const bool use_calibration_;

  // If this is false, all dimensions including the batch dimension are
  // set explicitely.
  const bool use_implicit_batch_;

  // Batch size of inputs to trt_network_ added by AddInputTensor(). During
  // network construction it will update this, use it to verify the batch
  // size of all inputs are compatible, and make sure individual TF node is
  // acceptable by TRT.
  int batch_size_ = -1;

  // Assign a ID to each constant layer we create, so that we can assign a
  // unique name to the layer.
  int next_constant_layer_id_ = 0;

  // The name of the TRTEngineOp node.
  absl::string_view engine_name_;

  friend class ConverterTest;
  friend class OpConverterTest;
};

// Converts 'input' of 'node_def' into 'tensor' with shape specified by 'dims'
// (which doesn't contain the batch dimension).
//
// If validation_only is true, it doesn't do the conversion but only do some
// minimum validation for the eligibility of the conversion, and *tensor will
// be set to nullptr.
// If validation_only is false converter must not be nullptr.
Status PrepareTensorForShape(
    Converter* converter, const TRT_TensorOrWeights& input,
    const nvinfer1::Dims& dims, const bool validation_only,
    ITensorProxyPtr* tensor, const NodeDef& node_def,
    absl::optional<int> op_instance = absl::nullopt,
    absl::optional<std::string> origin_node_name = absl::nullopt);

// Return OK if the broadcast scheme is supported and compute the shapes after
// broadcasting. check_feasibility can be set to false in cases where dimensions
// do not need to match exactly (as in the case of BatchMatMulV2).
Status GetTrtBroadcastShape(const TRT_TensorOrWeights& operand_l,
                            const TRT_TensorOrWeights& operand_r,
                            const bool check_feasibility,
                            const bool use_implicit_batch,
                            nvinfer1::Dims* operand_l_new_dims,
                            nvinfer1::Dims* operand_r_new_dims);

// Map of all supported UnaryOperations
const std::unordered_map<string, nvinfer1::UnaryOperation>* UnaryOperationMap();
// Map of all supported ActivationTypes
const std::unordered_map<string, nvinfer1::ActivationType>* ActivationTypeMap();
// Map of all supported BinaryOperations
const std::unordered_map<string, nvinfer1::ElementWiseOperation>*
BinaryOperationMap();

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_NODES_H_
