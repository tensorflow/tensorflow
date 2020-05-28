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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"

namespace tflite {
namespace delegates {
namespace hexagon {

struct OpNode {
  std::vector<hexagon_nn_input> inputs;
  std::vector<hexagon_nn_output> outputs;
  // Value from the Enum of Ops in hexagon_nn_ops
  int op_type;
  hexagon_nn_padding_type padding_type = NN_PAD_NA;
  // Id of node in the Hexagon graph.
  int node_id = -1;
  // Index/ID of node in the tflite graph.
  // This ID can be duplicate if one TFLite node creates multiple Hexagon op
  // nodes.
  int tflite_node_index = -1;
};

class GraphBuilder;

class OpBuilder {
 public:
  OpBuilder(GraphBuilder* graph_builder, int hexagon_op_type)
      : graph_builder_(graph_builder) {
    op_node_.op_type = hexagon_op_type;
  }
  // A tensor is identified in the graph using a pair of IDs
  // (Node ID, output Tensor ID)
  // Node producing this tensor, and the index of the tensor in this
  // node output list.
  using TensorID = std::pair<int, int>;

  virtual ~OpBuilder() {}

  // TODO(karimnosseir): Do we need to have builder pattern, or they are few not
  // worth it ?
  void SetOpType(int op_type) { op_node_.op_type = op_type; }

  void SetNodeId(int node_id) { op_node_.node_id = node_id; }

  void SetTFLiteNodeId(int node_index) {
    op_node_.tflite_node_index = node_index;
  }

  void SetConstNode() { op_node_.op_type = OP_Const; }

  void SetPaddingType(hexagon_nn_padding_type padding_type) {
    op_node_.padding_type = padding_type;
  }

  void SetBuiltinData(void* builtin_data) { builtin_data_ = builtin_data; }

  bool IsConstNode() const { return op_node_.op_type == OP_Const; }

  void print() {}

  const OpNode* Build();

  void AddInput(const TensorID& tensor_id) { input_ids_.push_back(tensor_id); }

  TensorID AddOutput(const TfLiteIntArray* dims);

  TensorID AddOutput(int elementsize, int rank,
                     const std::vector<int>& max_sizes);

  int GetID() const { return op_node_.node_id; }

  int GetTFLiteNodeID() const { return op_node_.tflite_node_index; }

  int GetOpType() const { return op_node_.op_type; }

  void SetTfLiteNode(const TfLiteNode* node) { tflite_node_ = node; }

  virtual TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                        const TfLiteIntArray* outputs,
                                        TfLiteContext* context) {
    return kTfLiteOk;
  }

  virtual TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                                       TfLiteContext* context) {
    return kTfLiteOk;
  }

  static TfLiteStatus ComputeMinAndMaxQuantValues(const TfLiteTensor& tensor,
                                                  float* min, float* max) {
    if (tensor.type == kTfLiteUInt8) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<uint8_t>::min(),
                                         std::numeric_limits<uint8_t>::max());
    } else if (tensor.type == kTfLiteInt8) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max());
    } else if (tensor.type == kTfLiteInt32) {
      return ComputeMinAndMaxQuantValues(tensor, min, max,
                                         std::numeric_limits<int32_t>::min(),
                                         std::numeric_limits<int32_t>::max());
    }
    return kTfLiteError;
  }

 protected:
  // Helper method to fetch dimensions.
  // TODO(karimnosseir): Move to a shared place.
  void GetDims(int* batch_size, int* height_size, int* width_size,
               int* depth_size, const TfLiteIntArray* dims) {
    int* dim[] = {batch_size, height_size, width_size, depth_size};
    for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
    for (int i = 4 - dims->size; i < 4; ++i) {
      *dim[i] = dims->data[i - (4 - dims->size)];
    }
  }

  template <typename T>
  static TfLiteStatus ComputeMinAndMaxQuantValues(const TfLiteTensor& tensor,
                                                  float* min, float* max,
                                                  T min_value, T max_value) {
    *min = 0;
    *max = 0;
    const TfLiteQuantization& quant = tensor.quantization;
    if (quant.type != TfLiteQuantizationType::kTfLiteAffineQuantization) {
      printf("Tensor not quantized: %s\n", tensor.name);
      return kTfLiteError;
    }
    const TfLiteAffineQuantization* params =
        static_cast<const TfLiteAffineQuantization*>(quant.params);
    float scale = params->scale->data[0];
    float zero_point = static_cast<float>(params->zero_point->data[0]);
    *min = scale * (static_cast<float>(min_value) - zero_point);
    *max = scale * (static_cast<float>(max_value) - zero_point);

    return kTfLiteOk;
  }

  OpNode op_node_;
  // inputs to the current op. Each pair identifies a single output from
  // another node (node_id, output_id).
  std::vector<TensorID> input_ids_;
  // Pointer to the graph builder.
  GraphBuilder* graph_builder_ = nullptr;
  // Data needed by this node.
  void* builtin_data_ = nullptr;
  // TODO(karimnosseir): Currently we only use it for getting output
  // size. Can we avoid passing it ?
  const TfLiteNode* tflite_node_ = nullptr;
};

class GraphBuilder {
 public:
  GraphBuilder(const HexagonNN* hexagon_nn, TfLiteContext* context,
               int graph_id)
      : hexagon_nn_(hexagon_nn), context_(context), graph_id_(graph_id) {}

  // Returns per OP builder. 'op_type' is the TfLite builtinOperator.
  OpBuilder* AddNodeFromTfLiteOp(int op_type, TfLiteNode* node,
                                 int tflite_node_index);

  // Add node to the graph. The caller responsible for setting correct
  // data in the Op.
  // 'tflite_node_index' is the node index in TFLite that creates this op.
  OpBuilder* AddNode(int tflite_node_index = -1);

  // Add const node that provides the data held by 'tensor'.
  // If `int8_to_uint8` is true, then the data will be casted to uint8 from
  // int8.
  OpBuilder* AddConstNodeWithData(int tensor_id, const TfLiteTensor& tensor,
                                  bool int8_to_uint8 = false);

  // Same as above but takes shape of the tensor that will holds the data.
  OpBuilder* AddConstNodeWithData(const int shape[], char* data, int data_size);

  OpBuilder* CreateOpBuilderFromTfLiteOp(int op_type, TfLiteNode* node);

  // Construct Input node with 'input_tensors' as output.
  TfLiteStatus AddInputTensors(const TfLiteIntArray* input_tensors,
                               TfLiteContext* context);

  // Construct Output node with 'output_tensors' as input.
  TfLiteStatus AddOutputTensors(const TfLiteIntArray* output_tensors,
                                TfLiteContext* context);

  // Adds BatchSeqConfig node to the graph. This is configuration
  // for a dynamic batch size for the graph.
  // A graph can have only one node of this type.
  void AddBatchSeqConfig(int max_size_for_batch,
                         TfLiteIntArray* input_batch_dimensions,
                         TfLiteIntArray* output_batch_dimensions);

  // Returns tensor id inside Hexagon graph.
  OpBuilder::TensorID GetHexagonTensorId(int tflite_tensor_index) {
    if (!HasTensor(tflite_tensor_index)) {
      // Return invalid ID.
      return OpBuilder::TensorID(-1, -1);
    }
    return tensors_[tflite_tensor_index];
  }

  // Return true if this tensor was added before to the graph.
  bool HasTensor(int tflite_tensor_index) {
    if (tensors_.size() <= tflite_tensor_index) {
      return false;
    }
    // the first field is node ID and id = 0 is reserved
    // so anything > 0 is correctly initialized.
    return tensors_[tflite_tensor_index].first != 0;
  }

  void AddDebugNode() {}

  void Build() {
    for (int i = 0; i < builders_.size(); ++i) {
      if (builders_[i]->IsConstNode()) {
        continue;
      }
      const OpNode* op_node = builders_[i]->Build();
      int error = hexagon_nn_->hexagon_nn_append_node(
          graph_id_, op_node->node_id, op_node->op_type, op_node->padding_type,
          op_node->inputs.data(), op_node->inputs.size(),
          op_node->outputs.data(), op_node->outputs.size());
      if (error != 0) {
        printf("Error adding node: id:%d, op_type:%d\n", op_node->node_id,
               op_node->op_type);
      }
    }
  }

  void print() {
    printf("------------------------------\n");
    std::vector<unsigned char> buf(10000);
    hexagon_nn_->hexagon_nn_snpprint(graph_id_, buf.data(), buf.size());
    printf("%s", buf.data());
    printf("------------------------------\n");
    fflush(stdout);
  }

  // Add new tensor mapping to the tensor list.
  bool AddTensorWithID(int tflite_tensor_id, int hexagon_node_id,
                       int hexagon_node_output_id, bool overwrite = false) {
    if (!overwrite && HasTensor(tflite_tensor_id)) {
      return false;
    }
    if (tensors_.size() <= tflite_tensor_id) {
      tensors_.resize(tflite_tensor_id + 1);
    }
    tensors_[tflite_tensor_id] =
        OpBuilder::TensorID(hexagon_node_id, hexagon_node_output_id);
    return true;
  }

  int GetOpTypeId(int node_id) {
    if (node_id > builders_.size()) {
      return -1;
    }
    return builders_[node_id - 1]->GetOpType();
  }

  int GetTFLiteNodeID(int node_id) const {
    if (node_id > builders_.size()) {
      return -1;
    }
    return builders_[node_id - 1]->GetTFLiteNodeID();
  }

  // Returns true if the graph supports dynamic batch. False otherwise.
  bool GraphHasDynamicBatch() const { return max_size_for_batch_ != -1; }

  // Returns the maximum value for batch dimension the graph supports.
  // -1 if the graph doesn't support dynamic batch.
  int GetMaxBatchSize() const { return max_size_for_batch_; }

 private:
  // Helper method to fetch dimensions.
  // TODO(karimnosseir): Move this method to shared place.
  void GetDims(int* batch_size, int* height_size, int* width_size,
               int* depth_size, const TfLiteIntArray* dims) {
    int* dim[] = {batch_size, height_size, width_size, depth_size};
    for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
    for (int i = 4 - dims->size; i < 4; ++i) {
      *dim[i] = dims->data[i - (4 - dims->size)];
    }
  }

  // Adds a Cast op to convert a tensor from int8 to uint8 (or vice versa).
  TfLiteStatus AddCastOp(TfLiteContext* context, int op_type, int tensor_id);

  const HexagonNN* hexagon_nn_ = nullptr;
  TfLiteContext* context_ = nullptr;
  int graph_id_ = -1;
  std::vector<std::unique_ptr<OpBuilder>> builders_;
  // Index in the vector is the tflite_tensor_index, the value
  // is the ID in the hexgon graph.
  std::vector<OpBuilder::TensorID> tensors_;

  // If the graph being built supports dynamic batch, this represents
  // the maximum value for batch.
  int max_size_for_batch_ = -1;
};

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_BUILDERS_OP_BUILDER_H_
