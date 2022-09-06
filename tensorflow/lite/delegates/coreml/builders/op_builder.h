/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_

#include <functional>
#include <string>

#include "mlmodel/format/Model.pb.h"
#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace coreml {
class OpBuilder;

// A class represents an ID in the coreML graph.
// A node is represented by a pair (node_id, and output_index)
// API is experimental and subject to change.
class TensorID {
 public:
  TensorID() {}
  TensorID(int node, int output_id) : node_(node), output_id_(output_id) {}

  std::string ToString() const;

  int NodeID() const;

  int OutputID() const;

 private:
  int node_ = -1;
  int output_id_ = -1;
};

// Builder for the whole graph.
// All op builders should be added using AddBuilder
// and then BuildModel should be called to return the CoreML generated.
//
// API is experimental and subject to change.
class GraphBuilder {
 public:
  explicit GraphBuilder(int coreml_version) : coreml_version_(coreml_version) {}

  // Returns pointer to the created builder. Ownership still belongs
  // to the GraphBuilder.
  OpBuilder* AddBuilder(int builtin_code, const TfLiteNode* node);

  // Returns pointer to the created builder with op builder function provided.
  OpBuilder* AddBuilder(const std::function<OpBuilder*(GraphBuilder*)>& builder,
                        const TfLiteNode* node);

  // Builds Model instance and returns it.
  CoreML::Specification::Model* BuildModel();

  // Returns string representing tensor 'tensor_id' in coreML.
  // tensor_id should have been added before calling this method.
  std::string GetTensorName(int tensor_id);

  // Returns Core ML Tensor ID for TFL 'tensor_id'.
  // tensor_id should have been added before calling this method.
  const TensorID GetTensorID(int tensor_id);

  void AddTensorWithID(int tf_tensor_id, const TensorID& tensor_id);

  // Return true if this tensor was added before to the graph.
  bool HasTensor(int tflite_tensor_index);
  // Return if this tensor is used in the graph (not as data).
  // This information is used to mark constant tensors that are used as input.
  bool IsTensorUsed(int tflite_tensor_index);

  const int coreml_version_;

 private:
  std::vector<std::unique_ptr<OpBuilder>> builders_;
  // Index in the vector is the tflite_tensor_index, the value
  // is the ID in the coreml graph.
  std::vector<TensorID> tensors_;
  std::vector<bool> used_tensor_;
};

// Interface for all op layers
// API is experimental and subject to change.
class OpBuilder {
 public:
  explicit OpBuilder(GraphBuilder* graph_builder)
      : graph_builder_(graph_builder) {}
  virtual ~OpBuilder() {}

  // Returns the Layer this builder responsible for.
  // Ownership is transferred to caller.
  virtual CoreML::Specification::NeuralNetworkLayer* Build();

  // Associates TfLite input tensors to Core ML layer's inputs and properties.
  // Verification for input constraints should happen here.
  virtual TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                                      TfLiteContext* context) = 0;

  // Associates TFLite output tensor with the node's output. If the OpBuilder
  // has subgraphs, The final output of that subgraph should be associated with
  // the output tensor.
  virtual TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                                       TfLiteContext* context) = 0;

  // Adds additional required OpBuilders, and populate builder_output_ with
  // Actual output that corresponds to output tensor of TFL Node.
  // Clients need to override this in cases where the nodes can be used for
  // composing other ops. For example, Relu6 in TfLite can be converted to
  // Relu -> Threshold -> Neg.
  // TODO(b/147211734): have this called automatically when necessary.
  virtual TfLiteStatus PopulateSubgraph(TfLiteContext* context);

  virtual const std::string& DebugName() = 0;

  void SetBuiltinData(void* builtin_data);

  void SetNodeID(int id);

  void SetTfLiteNode(const TfLiteNode* node);

  int GetID() const;

  // Adds input with tensor name.
  void AddInput(const std::string& input_name);

  // Adds input with CoreML tensor ID.
  void AddInput(const TensorID& input_id);

  // Adds input with TF Lite tensor ID.
  // TODO(taeheej): cleanup AddInput use cases and used tensor tracking.
  void AddInput(int tf_input_id);

  // Simply adds new output to the underlying layer.
  TensorID AddOutput();

  // Should set builder_output_ (if unset) and return it as the output of
  // this node. To be used by clients that needs the output of the node.
  virtual TensorID GetOutput(TfLiteContext* context);

 protected:
  // Sets layer's name.
  void SetDebugName(const char* layer_name, int id);

  GraphBuilder* graph_builder_ = nullptr;
  // Data needed by this node.
  void* builtin_data_ = nullptr;
  int node_id_ = -1;
  int num_outputs_ = 0;
  const TfLiteNode* tflite_node_ = nullptr;
  TensorID builder_output_;
  std::string debug_name_;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> layer_;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_OP_BUILDER_H_
