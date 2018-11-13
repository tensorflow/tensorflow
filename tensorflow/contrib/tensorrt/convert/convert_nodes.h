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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
extern const char* const kInputPHName;
extern const char* const kOutputPHName;

namespace convert {

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
  tensorflow::PartialTensorShape outside_shape;  // Only set for input edge.

  const string inside_node_name;
  const int inside_id;
  const int inside_port;
  tensorflow::PartialTensorShape inside_shape;  // Only set for output edge.

  tensorflow::DataType connection_type;
  const bool is_input_edge;

  // The port number of the TRT node connected with this edge.
  const int port_number;
};

struct EngineInfo {
  EngineInfo()
      : engine_type(EngineType::TRTStatic),
        max_workspace_size_bytes(0),
        precision_mode(FP32MODE) {}

  string engine_name;
  string device;
  tensorflow::GraphDef segment_graph_def;

  // Non-control input connections inside this vector are sorted in a way such
  // that, the segment nodes connecting to them are topological sorted.
  // In addition, for non-control connections, there must be no duplicates.
  std::vector<EngineConnection> connections;

  enum class EngineType { TRTStatic = 0, TRTDynamic = 1 };
  EngineType engine_type;
  int64 max_workspace_size_bytes;
  int maximum_cached_engines;
  std::vector<int> cached_engine_batches;
  int precision_mode;
};

// Constructs a graphdef from the segment in the given graph. Adds placeholder
// nodes for input edges (InputPH_*) and identity nodes for output edges
// (OutputPH_*). This function needs to be called before TensorRT nodes
// inserted in order to correctly get sizes from the original graph.
//
// - subgraph_node_names: the node names of the subgraph.
// - subgraph_node_ids: the node ids of the subgraph, must be sorted in
//   topological order.
// - segment_def: the output GraphDef, whose non-input/output nodedefs will be
//   sorted in topological order.
//
// TODO(aaroey): add tests to validate these properties.
tensorflow::Status ConvertSegmentToGraphDef(
    const tensorflow::Graph* graph,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::set<string>& subgraph_node_names,
    const std::vector<int>& subgraph_node_ids,
    std::vector<EngineConnection>* connections,
    tensorflow::GraphDef* segment_def, string* common_scope);

// Converts given subgraph to a TRT engine saved in 'engine'. Returns ok iff
// 'builder' successfully build the engine. If the result is not ok, 'engine'
// will be set to nullptr
// Once returned, 'builder' is not needed any more and can be safely detroyed.
//
// - convert_successfully: indicates whether the converson to TensorRT network
//   is successful. This is different than successfully building the engine:
//   building can still fail afterwards.
tensorflow::Status ConvertGraphDefToEngine(
    const tensorflow::GraphDef& gdef, int precision_mode, int max_batch_size,
    size_t max_workspace_size_bytes,
    const std::vector<tensorflow::PartialTensorShape>& input_shapes,
    Logger* logger, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator,
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine,
    bool* convert_successfully);

// Helper class for the segmenter to determine whether an output edge from the
// TRT segment is valid.
class OutputEdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an output edge of the
  // TRT segment.
  bool operator()(const tensorflow::Edge* out_edge) const;
};

string DebugString(const nvinfer1::Dims& dims);
string DebugString(const nvinfer1::ITensor& tensor);
int64_t TrtDimsNumElements(const nvinfer1::Dims& dims);

// Class to convert TF compile-time constants (e.g. Const nodes) to TRT weight.
class TRT_ShapedWeights {
 public:
  explicit TRT_ShapedWeights(DataType type = DT_FLOAT);

  // Copy from another weights.
  //
  // NOTE: this does not copy the underlying buffer but only increase its
  // reference count.
  TRT_ShapedWeights(const TRT_ShapedWeights& rhs);

  nvinfer1::Weights GetTrtWeights() const;

  void* GetValues() const {
    return const_cast<char*>(tensor_.tensor_data().data());
  }

  int64_t count() const;

  size_t size_bytes() const;

  string DebugString() const;

  // TODO(aaroey): make these private.
  nvinfer1::Dims shape_;  // Note: shape.type[] is not used.
  tensorflow::DataType type_;

 private:
  // This constructor is only used by TrtWeightStore, which creates the
  // underlying buffer.
  TRT_ShapedWeights(DataType type, nvinfer1::Dims dims, Tensor tensor);

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
  TRT_ShapedWeights GetTempWeights(tensorflow::DataType type,
                                   const nvinfer1::Dims& dims);

  // Get a TRT_ShapedWeights with the same data type and dimensions as
  // 'weights'.
  TRT_ShapedWeights GetTempWeights(const TRT_ShapedWeights& weights) {
    return GetTempWeights(weights.type_, weights.shape_);
  }

 private:
  // The backend storage of the TRT_ShapedWeights.
  std::vector<Tensor> store_;
};

// Represents a TRT-style input to a TF node, it can be either a
// nvinfer1::ITensor, or TRT_ShapedWeights which is compile-time constant.
//
// TODO(laigd): maybe rename it to TrtArgument, or mimic XlaCompiler::Argument.
class TRT_TensorOrWeights {
 public:
  TRT_TensorOrWeights() {}

  // Constructor that makes it an ITensor, doesn't take ownership of 'tensor'.
  // This is used by Converter when building the TRT network, where the ITensor
  // is owned by the TRT network being built. See comment for 'tensor_' below.
  explicit TRT_TensorOrWeights(nvinfer1::ITensor* tensor, int batch_size = -1);

  // Constructor that makes it an ITensor by creating one using provided data
  // type and shape, and takes ownership of the created ITensor. This is used by
  // TrtNodeValidator to encapsulate the type and shape information for
  // validation of graph nodes, and the created ITensor is fake and temporary,
  // and should not be used to build any TRT network. See comment for
  // 'simple_itensor_' below.
  explicit TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                               const nvinfer1::Dims& trt_dims, int batch_size);

  // Constructor that makes it a TRT_TensorOrWeights.
  explicit TRT_TensorOrWeights(const TRT_ShapedWeights& weights);

  TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs);

  void operator=(const TRT_TensorOrWeights& rhs);

  bool is_tensor() const { return initialized_ && is_tensor_; }
  bool is_weights() const { return initialized_ && !is_tensor_; }

  nvinfer1::ITensor* tensor();

  const nvinfer1::ITensor* tensor() const;

  TRT_ShapedWeights& weights() {
    CHECK(is_weights());
    return weights_;
  }

  const TRT_ShapedWeights& weights() const {
    CHECK(is_weights());
    return weights_;
  }

  nvinfer1::Dims GetTrtDims() const;

  int batch_size() const { return batch_size_; }

  string DebugString() const;

 private:
  class SimpleITensor;

  void set_batch_size(int batch_size) { batch_size_ = batch_size; }

  // When it represents an ITensor, the ITensor can be either passed by the
  // caller via the constructor that takes an ITensor* as parameter, or be
  // created as a SimpleITensor.
  //
  // In the first case, the ITensor pointer is stored in 'tensor_' below, and
  // the ITensor itself is not owned by this class. This method is used by
  // Converter (e.g. AddInputTensor) and op converters during TRT network
  // construction, where the TRT network owns the ITensor.
  //
  // In the second case, the created SimpleITensor is stored in
  // 'simple_itensor_' below and is owned by this class. SimpleITensor is a fake
  // implementation of ITensor and is used only by TrtNodeValidator to validate
  // the graph nodes.
  nvinfer1::ITensor* tensor_ = nullptr;  // Not owned.
  std::shared_ptr<SimpleITensor> simple_itensor_ = nullptr;

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
  int batch_size_ = -1;

  TRT_ShapedWeights weights_;
  bool initialized_ = false;
  bool is_tensor_ = false;

  friend class Converter;
};

class Converter;

// Parameters for each op converter.
struct OpConverterParams {
  OpConverterParams(Converter* arg_converter,
                    const tensorflow::NodeDef& arg_node_def,
                    const std::vector<TRT_TensorOrWeights>& arg_inputs,
                    std::vector<TRT_TensorOrWeights>* arg_outputs,
                    bool arg_validation_only, TrtWeightStore* arg_weight_store)
      : converter(arg_converter),
        node_def(arg_node_def),
        inputs(arg_inputs),
        outputs(arg_outputs),
        validation_only(arg_validation_only),
        weight_store(arg_weight_store) {}

  Converter* converter;
  const tensorflow::NodeDef& node_def;
  const std::vector<TRT_TensorOrWeights>& inputs;
  std::vector<TRT_TensorOrWeights>* outputs;
  const bool validation_only;
  TrtWeightStore* weight_store;
};

using OpConverter = std::function<Status(OpConverterParams*)>;

// Class to verify if specific TF node is supported by TRT.
class TrtNodeValidator {
 public:
  TrtNodeValidator();

  // Validate the node, and return ok if it's supported by TRT.
  //
  // - 'node_def' is the node to validate.
  // - 'input_node_and_ports' are the input NodeDefs and their output ports that
  //   are connected to 'node_def' in the TF graph.
  // - 'graph_properties' is the GraphProperties of the graph where 'node_def'
  //   belongs. It is used to get the shape and data type information of a
  //   tensor for validation purpose.
  Status ValidateNode(
      const NodeDef& node_def,
      const std::vector<std::pair<const NodeDef*, int>>& input_node_and_ports,
      const grappler::GraphProperties& graph_properties);

 private:
  void RegisterOpValidators();

  // Convert a Const node to a TRT_TensorOrWeights.
  Status ConvertConstToWeights(const NodeDef& const_node_def,
                               const std::vector<TRT_TensorOrWeights>& inputs,
                               TRT_TensorOrWeights* output);

  // Convert the output tensor at 'output_port' of 'node_def' to a
  // TRT_TensorOrWeights which will be later used as an input to other nodes and
  // passed to ValidateNode() below.
  Status ConvertToTensorOrWeights(
      const NodeDef& node_def, int output_port,
      const grappler::GraphProperties& graph_properties,
      TRT_TensorOrWeights* tensor_or_weights);

  // Stores all the validators by op type. If no validator is registered for
  // specific op, it means no validation is needed and ValidateNode() will
  // return OK.
  std::unordered_map<string, OpConverter> op_validators_;

  // Store the weights added during validation. Some validations (e.g.
  // validation for Const node) may produce weights.
  TrtWeightStore weight_store_;

  friend class ValidatorTest;
  friend class OpConverterTest;
};

// Class to convert TF nodes to TRT network.
class Converter {
 public:
  Converter(nvinfer1::INetworkDefinition* trt_network, bool is_fp16);

  //////////////////////////////////////////////////////////////////////////////
  // Methods used by the TRT engine builder to build a TRT network from a TF
  // function/subgraph.

  // Convert the node to TRT network.
  Status ConvertNode(const tensorflow::NodeDef& node_def);

  // Add input tensor to the TRT network with given 'name', 'dtype', 'dims' and
  // 'batch_size'.
  Status AddInputTensor(const string& name, nvinfer1::DataType dtype,
                        const nvinfer1::Dims& dims, int batch_size);

  // Mark the tensors with names specified by output_tensors[i].first as output
  // of the TRT network, and set their names in the TRT network as
  // output_tensors[i].second. The tensor names (output_tensors[i].first) are
  // standard TF tensor names, i.e. node names followed by output slot number
  // (or just the node name if the tensor is the first output of the node).
  Status RenameAndMarkOutputTensors(
      const std::vector<std::pair<string, string>>& output_tensors);

  //////////////////////////////////////////////////////////////////////////////
  // Methods used by op converters to convert individual TF node and add layers
  // to the TRT network.

  // Op converters (e.g. ConvertReshape) need to access the TRT network in order
  // to add TRT layers.
  nvinfer1::INetworkDefinition* network() { return trt_network_; }

  // Is the converter operating in fp16 mode?
  bool is_fp16() const { return is_fp16_; }

  // Below are helper methods for op converters to add different layers to the
  // TRT network.

  // Transpose 'input_tensor' with given permutation 'order_with_batch_dim' to
  // 'output_tensor'. The permutation 'order_with_batch_dim' contains the batch
  // dimension which should always be 0.
  Status TransposeTensor(nvinfer1::ITensor* input_tensor,
                         const std::vector<int>& order_with_batch_dim,
                         const nvinfer1::ITensor** output_tensor);

  // Converts 'input' into 'tensor' with shape specified by 'dims'.
  Status PrepareTensorForShape(const TRT_TensorOrWeights& input,
                               const nvinfer1::Dims& dims,
                               const nvinfer1::ITensor** tensor);

 private:
  // Verify the provided batch_size is consistent with batch_size_ and update it
  // if necessary.
  Status MaybeUpdateBatchSize(int batch_size);

  // Add the provided tensor/weights to the map trt_tensors_.
  Status AddTensorOrWeights(const string& name, TRT_TensorOrWeights input);

  // Get the tensor/weights from trt_tensors_ by 'name'.
  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output);

  // Get the inputs of 'node_def' from trt_tensors_.
  Status GetInputs(const tensorflow::NodeDef& node_def,
                   std::vector<TRT_TensorOrWeights>* inputs) const;

  void RegisterOpConverters();

  // Registered op converters by op type.
  std::unordered_map<string, OpConverter> op_registry_;

  // Tensors/weights added during construction of trt_network_.
  std::unordered_map<string, TRT_TensorOrWeights> trt_tensors_;

  // Special op converter for custom plugins.
  OpConverter plugin_converter_;

  // The TRT networking being built.
  nvinfer1::INetworkDefinition* trt_network_;

  // Store the weights added during construction of trt_network_.
  TrtWeightStore weight_store_;

  const bool is_fp16_;

  // Batch size of inputs to trt_network_ added by AddInputTensor(). During
  // network construction it will update this, use it to verify the batch
  // size of all inputs are compatible, and make sure individual TF node is
  // acceptable by TRT.
  int batch_size_ = -1;

  friend class ConverterTest;
  friend class OpConverterTest;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
