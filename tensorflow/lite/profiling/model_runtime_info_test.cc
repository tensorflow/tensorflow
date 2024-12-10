/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/model_runtime_info.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/proto/model_runtime_info.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace profiling {

// A model that runs a pad op followed by a conv_2d op.
// XNNPACK will fuse the pad op into the conv op while TFLite on CPU will not.
class PadAndConv2DModel : public MultiOpModel {
 public:
  explicit PadAndConv2DModel(TfLiteDelegate* delegate = nullptr) {
    input_ = AddInput({TensorType_FLOAT32, {1, 3, 3, 1}});
    int pad_out = AddInnerTensor<float>({TensorType_FLOAT32, {1, 5, 5, 1}});
    output_ = AddOutput({TensorType_FLOAT32, {1, 5, 5, 1}});

    int padding_in_ =
        AddConstInput({TensorType_INT32, {4, 2}}, {0, 0, 1, 1, 1, 1, 0, 0});
    int conv_filter_ =
        AddConstInput({TensorType_FLOAT32, {1, 2, 2, 1}}, {0, 1, 1, 0});
    int conv_bias_ = AddConstInput({TensorType_FLOAT32, {1}}, {3});

    AddBuiltinOp(tflite::BuiltinOperator_PAD, tflite::BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union(), {input_, padding_in_},
                 {pad_out});

    AddBuiltinOp(
        tflite::BuiltinOperator_CONV_2D, tflite::BuiltinOptions_Conv2DOptions,
        CreateConv2DOptions(builder_, tflite::Padding_SAME, 1, 1).Union(),
        {pad_out, conv_filter_, conv_bias_}, {output_});

    SetDelegate(delegate);
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/delegate != nullptr,
                     /*allocate_and_delegate=*/false);
    SetSubgraphNames();
  }

  void SetSubgraphNames() {
    for (int i = 0; i < interpreter_->subgraphs_size(); ++i) {
      interpreter_->subgraph(i)->SetName(
          std::string("subgraph_" + std::to_string(i)).c_str());
    }
  }
  int input() const { return input_; }
  int output() const { return output_; }
  void SetProfiler(Profiler* profiler) { interpreter_->SetProfiler(profiler); }
  Interpreter* interpreter() const { return interpreter_.get(); }

  void Initialize(Profiler* profiler) {
    if (profiler != nullptr) {
      SetProfiler(profiler);
    }
    AllocateAndDelegate(true);
  }

  void ResetProfilerAndInvoke(profiling::BufferedProfiler* profiler) {
    profiler->Reset();
    profiler->StartProfiling();
    PopulateTensor(input(),
                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    ASSERT_EQ(kTfLiteOk, Invoke());
    profiler->StopProfiling();
  }

 private:
  int input_;
  int output_;
};

bool AreRepeatedIntFieldsEqual(const google::protobuf::RepeatedField<int32_t>& field_1,
                               const google::protobuf::RepeatedField<int32_t>& field_2) {
  return std::equal(field_1.begin(), field_1.end(), field_2.begin(),
                    field_2.end());
}

bool AreEdgesEqual(const Edge& edge_1, const Edge& edge_2) {
  auto proto_to_tuple = [](const Edge& edge) {
    return std::make_tuple(edge.id(), edge.name(), edge.data_type(),
                           edge.size(), edge.layout_type(),
                           edge.allocation_type());
  };
  return proto_to_tuple(edge_1) == proto_to_tuple(edge_2) &&
         AreRepeatedIntFieldsEqual(edge_1.shape(), edge_2.shape());
}

bool AreNodesEqual(const Node& node_1, const Node& node_2) {
  auto proto_to_tuple = [](const Node& node) {
    return std::make_tuple(node.id(), node.name(), node.type());
  };
  return proto_to_tuple(node_1) == proto_to_tuple(node_2) &&
         AreRepeatedIntFieldsEqual(node_1.inputs(), node_2.inputs()) &&
         AreRepeatedIntFieldsEqual(node_1.outputs(), node_2.outputs()) &&
         AreRepeatedIntFieldsEqual(node_1.intermediates(),
                                   node_2.intermediates()) &&
         AreRepeatedIntFieldsEqual(node_1.temporaries(), node_2.temporaries());
}

bool AreRuntimeSubgraphsEqual(const RuntimeSubgraph& subgraph_1,
                              const RuntimeSubgraph& subgraph_2) {
  auto proto_to_tuple = [](const RuntimeSubgraph& subgraph) {
    return std::make_tuple(subgraph.subgraph_id(), subgraph.subgraph_type(),
                           subgraph.execution_plan().size(),
                           subgraph.nodes_size(), subgraph.edges_size(),
                           subgraph.name());
  };

  if (proto_to_tuple(subgraph_1) == proto_to_tuple(subgraph_2) &&
      AreRepeatedIntFieldsEqual(subgraph_1.execution_plan(),
                                subgraph_2.execution_plan())) {
    for (size_t i = 0; i < subgraph_1.nodes_size(); ++i) {
      if (!AreNodesEqual(subgraph_1.nodes(i), subgraph_2.nodes(i))) {
        return false;
      }
    }
    for (size_t i = 0; i < subgraph_1.edges_size(); ++i) {
      if (!AreEdgesEqual(subgraph_1.edges(i), subgraph_2.edges(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool AreModelRuntimeDetailsEqual(const ModelRuntimeDetails& model_details_1,
                                 const ModelRuntimeDetails& model_details_2) {
  auto proto_to_tuple = [](const ModelRuntimeDetails& model_details) {
    return std::make_tuple(model_details.model_name(),
                           model_details.subgraphs_size());
  };

  if (proto_to_tuple(model_details_1) == proto_to_tuple(model_details_2)) {
    for (size_t i = 0; i < model_details_1.subgraphs_size(); ++i) {
      if (!AreRuntimeSubgraphsEqual(model_details_1.subgraphs(i),
                                    model_details_2.subgraphs(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

ModelRuntimeDetails CreateExpectedModelRuntimeDetails(
    bool is_xnnpack_delegate) {
  ModelRuntimeDetails expected_model_runtime_details;

  RuntimeSubgraph* subgraph = expected_model_runtime_details.add_subgraphs();
  subgraph->set_subgraph_id(0);
  subgraph->set_name("subgraph_0");
  subgraph->set_subgraph_type(RuntimeSubgraph::TFLITE_SUBGRAPH);
  if (is_xnnpack_delegate) {
    subgraph->add_execution_plan(2);
  } else {
    subgraph->add_execution_plan(0);
    subgraph->add_execution_plan(1);
  }

  Node* node = subgraph->add_nodes();
  node->set_id(0);
  node->set_name("PAD");
  node->set_type("34");
  node->add_inputs(0);
  node->add_inputs(3);
  node->add_outputs(1);

  Node* node_2 = subgraph->add_nodes();
  node_2->set_id(1);
  node_2->set_name("CONV_2D");
  node_2->set_type("3");
  node_2->add_inputs(1);
  node_2->add_inputs(4);
  node_2->add_inputs(5);
  node_2->add_outputs(2);
  node_2->add_temporaries(6);

  if (is_xnnpack_delegate) {
    node->set_delegated_to_node_id(2);
    node_2->set_delegated_to_node_id(2);

    Node* node_3 = subgraph->add_nodes();
    node_3->set_id(2);
    node_3->set_name("TfLiteXNNPackDelegate");
    node_3->set_type("TfLiteXNNPackDelegate");
    node_3->add_inputs(0);
    node_3->add_inputs(3);
    node_3->add_inputs(4);
    node_3->add_inputs(5);
    node_3->add_outputs(2);

    DelegateNodeDetails* delegate_node_details =
        node_3->mutable_delegate_node_details();
    delegate_node_details->set_delegate_name("TfLiteXNNPackDelegate");
    delegate_node_details->add_tflite_node_ids_replaced(0);
    delegate_node_details->add_tflite_node_ids_replaced(1);
  }

  Edge* edge = subgraph->add_edges();
  edge->set_id(0);
  edge->set_name("");
  edge->set_data_type(Edge::FLOAT32);
  edge->set_size(36);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(1);
  edge->add_shape(3);
  edge->add_shape(3);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteArenaRw");

  edge = subgraph->add_edges();
  edge->set_id(1);
  edge->set_name("");
  edge->set_data_type(Edge::FLOAT32);
  edge->set_size(100);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(1);
  edge->add_shape(5);
  edge->add_shape(5);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteArenaRw");

  edge = subgraph->add_edges();
  edge->set_id(2);
  edge->set_name("");
  edge->set_data_type(Edge::FLOAT32);
  edge->set_size(100);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(1);
  edge->add_shape(5);
  edge->add_shape(5);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteArenaRw");

  edge = subgraph->add_edges();
  edge->set_id(3);
  edge->set_name("");
  edge->set_data_type(Edge::INT32);
  edge->set_size(32);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(4);
  edge->add_shape(2);
  edge->set_allocation_type("kTfLiteMmapRo");

  edge = subgraph->add_edges();
  edge->set_id(4);
  edge->set_name("");
  edge->set_data_type(Edge::FLOAT32);
  edge->set_size(16);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(1);
  edge->add_shape(2);
  edge->add_shape(2);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteMmapRo");

  edge = subgraph->add_edges();
  edge->set_id(5);
  edge->set_name("");
  edge->set_data_type(Edge::FLOAT32);
  edge->set_size(4);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteMmapRo");

  edge = subgraph->add_edges();
  edge->set_id(6);
  edge->set_data_type(Edge::FLOAT32);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->set_allocation_type("kTfLiteArenaRwPersistent");

#if __ANDROID__ && (__aarch64__ || __arm__ || __aarch32__)
  //  On Android Arm builds, the Conv2D op uses im2col.
  edge->set_name("");
  edge->set_size(is_xnnpack_delegate ? 0 : 400);
  edge->add_shape(1);
  edge->add_shape(5);
  edge->add_shape(5);
  edge->add_shape(4);
  edge->set_allocation_type("kTfLiteArenaRw");
#else
  edge->set_name("Conv_hwcn_weights");
  edge->set_size(is_xnnpack_delegate ? 0 : 16);
  edge->add_shape(4);
  edge->add_shape(1);
  edge->set_allocation_type("kTfLiteArenaRwPersistent");
#endif

  return expected_model_runtime_details;
}

TEST(MODEL_RUNTIME_INFO_TEST, PadAndConv2DNoDelegate) {
  auto profiler = std::make_unique<profiling::BufferedProfiler>(1024, false);

  PadAndConv2DModel model(nullptr);
  model.Initialize(profiler.get());
  model.ResetProfilerAndInvoke(profiler.get());

#ifdef __ANDROID__
  std::string file_name = "/data/local/tmp/test_file.textproto";
#else
  std::string file_name = "/tmp/test_file.textproto";
#endif

  auto status = GenerateModelRuntimeInfo(*model.interpreter(), file_name);
  ASSERT_TRUE(status == kTfLiteOk);
  ModelRuntimeDetails model_runtime_details;

  std::ifstream file(file_name, std::ios::binary);
  ASSERT_TRUE(file.good());
  model_runtime_details.ParseFromIstream(&file);
  file.close();

  ModelRuntimeDetails expected_model_runtime_details =
      CreateExpectedModelRuntimeDetails(/*is_xnnpack_delegate=*/false);

  ASSERT_TRUE(AreModelRuntimeDetailsEqual(model_runtime_details,
                                          expected_model_runtime_details));
}

TEST(MODEL_RUNTIME_INFO_TEST, PadAndConv2DWithXnnpackDelegate) {
  auto profiler = std::make_unique<profiling::BufferedProfiler>(1024, false);

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  PadAndConv2DModel xnnpack_model(xnnpack_delegate.get());
  xnnpack_model.Initialize(profiler.get());
  xnnpack_model.ResetProfilerAndInvoke(profiler.get());

#ifdef __ANDROID__
  std::string file_name = "/data/local/tmp/test_file.textproto";
#else
  std::string file_name = "/tmp/test_file.textproto";
#endif

  auto status =
      GenerateModelRuntimeInfo(*xnnpack_model.interpreter(), file_name);
  ASSERT_TRUE(status == kTfLiteOk);
  ModelRuntimeDetails model_runtime_details;

  std::ifstream file(file_name, std::ios::binary);
  ASSERT_TRUE(file.good());
  model_runtime_details.ParseFromIstream(&file);
  file.close();
  ModelRuntimeDetails expected_model_runtime_details =
      CreateExpectedModelRuntimeDetails(/*is_xnnpack_delegate=*/true);

  ASSERT_TRUE(AreModelRuntimeDetailsEqual(model_runtime_details,
                                          expected_model_runtime_details));
}

}  // namespace profiling
}  // namespace tflite
