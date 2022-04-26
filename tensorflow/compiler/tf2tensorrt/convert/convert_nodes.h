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
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"
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

        allow_build_at_runtime(true),
        use_explicit_precision(false) {}

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
  bool use_explicit_precision;
};

// Constructs a graphdef from the segment in the given graph and stores it to
// the engine_info. Adds _Arg nodes for input edges (InputPH_*) and _Retval
// nodes for output edges (OutputPH_*). Maintains the topological order of the
// non-input/output nodes in the graphdef. This function needs to be called
// before TensorRT layers are created because it prepares the original graph
// for TensorRT conversion.
//
// - subgraph_node_names: the node names of the subgraph.
// - subgraph_node_ids: the node ids of the subgraph, must be sorted in
//   topological order.
// - engine_info: a data structure that records the information about the
//   engine containing the subgraph.
//
// TODO(aaroey): add tests to validate these properties.
Status ConvertSegmentToGraphDef(
    const Graph* graph, const grappler::GraphProperties& graph_properties,
    const std::vector<const Node*>& subgraph_nodes, EngineInfo* engine_info);

// Converts given subgraph to a TRT engine saved in 'engine'. Returns ok iff
// 'builder' successfully build the engine. If the result is not ok, 'engine'
// will be set to nullptr
// Once returned, 'builder' is not needed any more and can be safely destroyed.
//
// - convert_successfully: indicates whether the conversion to TensorRT network
//   is successful. This is different than successfully building the engine:
//   building can still fail afterwards.
// Note: When 'cluster' is not null, it contains the graph to be converted.
//       We may perform additional optimizations to the graph before converting
//       the graph.
Status ConvertGraphDefToEngine(
    const GraphDef& gdef, TrtPrecisionMode precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<PartialTensorShape>& input_shapes,
    nvinfer1::ILogger* logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, bool use_calibration,
    const bool use_implicit_batch, bool* convert_successfully,
    TrtShapeOptimizationProfile* profiles, absl::string_view engine_name,
    bool use_explicit_precision,
    tensorflow::grappler::Cluster* cluster = nullptr);

// Helper class for the segmenter to determine whether an output edge from the
// TRT segment is valid.
class OutputEdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an output edge of the
  // TRT segment.
  bool operator()(const Edge* out_edge) const;
};

// Class to verify if specific TF node is supported by TRT.
class TrtNodeValidator {
 public:
  // 'graph_properties' is the GraphProperties of the graph whose nodes will be
  // checked by IsTensorRTCandidate() later. It is used to get the shape and
  // data type information of a tensor for validation purpose.
  TrtNodeValidator(const grappler::GraphProperties& graph_properties,
                   TrtPrecisionMode precision_mode, bool use_calibration,
                   bool use_implicit_batch, bool use_explicit_precision);

  // Returns OK iff 'node' is a TF-TRT conversion candidate, which will be added
  // to TRT subgraph and later converted into TRT engine.
  Status IsTensorRTCandidate(const Node* node);

  static const std::set<string>* quantize_ops;

  // Returns validator by op type. If no validator is registered for
  // specific op, it means no validation is needed and ValidateNode() will
  // return OK.
  StatusOr<OpConverter> GetValidator(const std::string& op);

 private:
  // Convert a Const node to a TRT_TensorOrWeights.
  Status ConvertConstToWeights(const NodeDef& const_node_def,
                               const std::vector<TRT_TensorOrWeights>& inputs,
                               TRT_TensorOrWeights* output);

  // Convert the output tensor at 'output_port' of 'node_def' to a
  // TRT_TensorOrWeights which will be later used as an input to other nodes and
  // passed to ValidateNode() below.
  Status ConvertToTensorOrWeights(const NodeDef& node_def, int output_port,
                                  TRT_TensorOrWeights* tensor_or_weights);

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

  const bool use_explicit_precision_;

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
      absl::string_view engine_name, bool use_explicit_precision = false);

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

  // This function should be called when we know the quantization range of a
  // tensor from a quantize/dequantize node.
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
                       OpConverterParams* params, ITensorProxyPtr* output,
                       absl::optional<int> op_instance = absl::nullopt);

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

  std::unordered_map<string, TRT_TensorOrWeights>& TensorsMap() {
    return trt_tensors_;
  }

  bool UseExplicitPrecision() const { return use_explicit_precision_; }

 private:
  Converter(TrtPrecisionMode precision_mode, bool use_calibration,
            nvinfer1::ILogger* trt_logger, const bool use_implicit_batch,
            absl::string_view engine_name, bool use_explicit_precision);

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

  // Indicates whether to use explicit precision in TensorRT (Q/DQ support).
  bool use_explicit_precision_;

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
    const DimsAdapter& dims, const bool validation_only,
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

template <typename T>
using OperationMap = std::unordered_map<std::string, T>;

// Map from Tensorflow operation names to TensorRT unary operations.
using UnaryOperationMapType = OperationMap<nvinfer1::UnaryOperation>;
const UnaryOperationMapType* UnaryOperationMap();

// Map from Tensorflow boolean operation names to TensorRT unary operations.
const UnaryOperationMapType* UnaryBooleanOperationMap();

// Map of all supported ActivationTypes.
const OperationMap<nvinfer1::ActivationType>* ActivationTypeMap();

// Map from Tensorflow binary operation names to TensorRT binary operations types.
using BinaryOperationMapType = OperationMap<nvinfer1::ElementWiseOperation>;
const BinaryOperationMapType* BinaryOperationMap();

// Map from Tensorflow boolean binary operation names to TensorRT binary operations types.
const BinaryOperationMapType* BinaryBooleanOperationMap();

template <typename T>
absl::InlinedVector<std::string, 10> GetOperationNames(const T& set) {
  absl::InlinedVector<std::string, 10> result;
  absl::c_transform(set, std::back_inserter(result),
                    [](const auto x) { return x.first; });
  return result;
}

// Adds a matrix multiplication operation to the TensorRT graph. The "params"
// pointer is only used to access the TRT network builder. The inputs and
// parameters for the op are fully specified by input_[a|b] and transpose_[a|b].
StatusOr<ITensorProxyPtr> ConvertMatMulImpl(OpConverterParams* params,
                                            TRT_TensorOrWeights input_a,
                                            TRT_TensorOrWeights input_b,
                                            bool transpose_a, bool transpose_b);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_NODES_H_
