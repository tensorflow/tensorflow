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

#include "tensorflow/cc/tools/freeze_saved_model.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class FreezeTest : public ::testing::Test {
 protected:
  void GraphDefEqual(const GraphDef& actual, const GraphDef& expected) {
    EXPECT_EQ(actual.ShortDebugString(), expected.ShortDebugString());
  }

  // Builds a SignatureDef with the provided `inputs` and `outputs`.
  SignatureDef BuildSignatureDef(const std::unordered_set<string>& inputs,
                                 const std::unordered_set<string>& outputs) {
    SignatureDef signature_def;
    for (const string& input : inputs) {
      (*signature_def.mutable_inputs())[input].set_name(input);
    }
    for (const string& output : outputs) {
      (*signature_def.mutable_outputs())[output].set_name(output);
    }
    return signature_def;
  }

  // Adds `signature_def` to `saved_model_bundle` under `key`.
  void AddSignatureDefToSavedModelBundle(const SignatureDef& signature_def,
                                         const string& key,
                                         SavedModelBundle* saved_model_bundle) {
    MetaGraphDef* meta_graph_def = &saved_model_bundle->meta_graph_def;
    (*meta_graph_def->mutable_signature_def())[key] = signature_def;
  }

  // Adds an initialized session to `saved_model_bundle` using `graph_def` and
  // initializing with `init_node`.
  Status InitializeSavedModelBundleSession(
      const GraphDef& graph_def, const string& init_node,
      SavedModelBundle* saved_model_bundle) {
    SessionOptions session_options;
    saved_model_bundle->session.reset(NewSession(session_options));
    TF_RETURN_IF_ERROR(saved_model_bundle->session->Create(graph_def));
    if (!init_node.empty()) {
      std::vector<Tensor> outputs;
      return saved_model_bundle->session->Run(
          /* inputs */ {}, /* output_tensors */ {}, {init_node}, &outputs);
    }
    return Status::OK();
  }

  // Adds `graph_def` to `saved_model_bundle` and intializes a session with
  // `init_node`.
  Status AddGraphDefToSavedModelBundle(const GraphDef& graph_def,
                                       const string& init_node,
                                       SavedModelBundle* saved_model_bundle) {
    MetaGraphDef* meta_graph_def = &saved_model_bundle->meta_graph_def;
    *meta_graph_def->mutable_graph_def() = graph_def;
    return InitializeSavedModelBundleSession(graph_def, init_node,
                                             saved_model_bundle);
  }

  // Adds `graph_def` and `outputs` as the GraphDef and SignatureDef in
  // `saved_model_bundle` and initializes a session with `init_node`.
  Status AddGraphDefWithOutputsToSavedModelBundle(
      const GraphDef& graph_def, const std::unordered_set<string>& outputs,
      const string& init_node, SavedModelBundle* saved_model_bundle) {
    SignatureDef signature_def =
        BuildSignatureDef(std::unordered_set<string>(), outputs);
    AddSignatureDefToSavedModelBundle(signature_def, "signature_def",
                                      saved_model_bundle);
    return AddGraphDefToSavedModelBundle(graph_def, init_node,
                                         saved_model_bundle);
  }

  // Runs and compares the outputs of `tensor_name` on both the
  // `unfrozen_session` and the `frozen_graph_def.
  void RunAndCompareFrozenAndUnfrozenGraphs(Session* unfrozen_session,
                                            const GraphDef& frozen_graph_def,
                                            const string& tensor_name) {
    std::vector<Tensor> unfrozen_outputs;
    TF_ASSERT_OK(unfrozen_session->Run(/* inputs */ {}, {tensor_name},
                                       /* targets */ {}, &unfrozen_outputs));

    SessionOptions session_options;
    std::unique_ptr<Session> frozen_session(NewSession(session_options));
    TF_ASSERT_OK(frozen_session->Create(frozen_graph_def));
    std::vector<Tensor> frozen_outputs;
    TF_ASSERT_OK(frozen_session->Run(/* inputs */ {}, {tensor_name},
                                     /* targets */ {}, &frozen_outputs));

    test::ExpectTensorEqual<float>(unfrozen_outputs[0], frozen_outputs[0]);
  }
};

TEST_F(FreezeTest, InputsAndOutputsSingleSignatureDef) {
  // Test that inputs and outputs get correctly populated for a single
  // SignatureDef.
  SavedModelBundle saved_model_bundle;
  std::unordered_set<string> expected_inputs = {"input0:0", "input1:0"};
  std::unordered_set<string> expected_outputs = {"output0:0", "output1:0"};
  SignatureDef signature_def =
      BuildSignatureDef(expected_inputs, expected_outputs);
  AddSignatureDefToSavedModelBundle(signature_def, "signature_def",
                                    &saved_model_bundle);
  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));
  EXPECT_EQ(expected_inputs, inputs);
  EXPECT_EQ(expected_outputs, outputs);
}

TEST_F(FreezeTest, InputsAndOutputsMultipleSignatureDefs) {
  // Test that inputs and outputs get correctly merged and populated when
  // multiple SignatureDefs are provided.
  SavedModelBundle saved_model_bundle;
  SignatureDef signature_def_0 = BuildSignatureDef({"input0:0"}, {"output0:0"});
  SignatureDef signature_def_1 = BuildSignatureDef({"input1:0"}, {"output1:0"});
  AddSignatureDefToSavedModelBundle(signature_def_0, "signature_def_0",
                                    &saved_model_bundle);
  AddSignatureDefToSavedModelBundle(signature_def_1, "signature_def_1",
                                    &saved_model_bundle);
  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));
  std::unordered_set<string> expected_inputs = {"input0:0", "input1:0"};
  std::unordered_set<string> expected_outputs = {"output0:0", "output1:0"};
  EXPECT_EQ(expected_inputs, inputs);
  EXPECT_EQ(expected_outputs, outputs);
}

TEST_F(FreezeTest, GraphDefVersionsAndLibrary) {
  // Test that GraphDef versions and library are copied correctly into the
  // frozen graph.
  SavedModelBundle saved_model_bundle;
  GraphDef graph_def;
  graph_def.mutable_versions()->set_producer(1234);
  graph_def.mutable_versions()->set_min_consumer(1234);
  *graph_def.mutable_library()->add_function() = test::function::NonZero();
  TF_ASSERT_OK(
      AddGraphDefToSavedModelBundle(graph_def, "", &saved_model_bundle));

  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));

  GraphDefEqual(frozen_graph_def, graph_def);
}

TEST_F(FreezeTest, GraphDefWithNoVariables) {
  // Test freezing a graph with no variables.
  SavedModelBundle saved_model_bundle;
  GraphDef graph_def;
  Scope scope = Scope::NewRootScope();
  Output a = ops::Const(scope.WithOpName("a"), 10.0f, {});
  Output b = ops::Const(scope.WithOpName("b"), 10.0f, {});
  Output c = ops::Mul(scope.WithOpName("c"), a, b);
  TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  TF_ASSERT_OK(AddGraphDefWithOutputsToSavedModelBundle(graph_def, {"c:0"}, "",
                                                        &saved_model_bundle));

  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));

  GraphDefEqual(frozen_graph_def, graph_def);
}

TEST_F(FreezeTest, GraphDefWithVariablesNotNeededByOutputs) {
  // Test freezing a graph with variables that are not needed by the outputs in
  // the SignatureDef. The resulting graph shouldn't be frozen, but
  // non-dependent nodes should be pruned.
  SavedModelBundle saved_model_bundle;
  GraphDef graph_def;
  Scope scope = Scope::NewRootScope();
  Output a = ops::Const(scope.WithOpName("a"), 10.0f, {});
  Output b = ops::Const(scope.WithOpName("b"), 10.0f, {});
  Output c = ops::Mul(scope.WithOpName("c"), a, b);
  Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
  Output assign = ops::Assign(scope.WithOpName("assign"), var, a);
  TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  // "c" isnt dependent on the variable, so nothing should be frozen.
  TF_ASSERT_OK(AddGraphDefWithOutputsToSavedModelBundle(
      graph_def, {"c:0"}, assign.name(), &saved_model_bundle));

  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));

  GraphDef expected_graph_def;
  Scope expected_scope = Scope::NewRootScope();
  Output expected_a = ops::Const(expected_scope.WithOpName("a"), 10.0f, {});
  Output expected_b = ops::Const(expected_scope.WithOpName("b"), 10.0f, {});
  Output expected_c =
      ops::Mul(expected_scope.WithOpName("c"), expected_a, expected_b);
  TF_ASSERT_OK(expected_scope.ToGraphDef(&expected_graph_def));

  GraphDefEqual(frozen_graph_def, expected_graph_def);

  RunAndCompareFrozenAndUnfrozenGraphs(saved_model_bundle.session.get(),
                                       frozen_graph_def, "c:0");
}

TEST_F(FreezeTest, GraphDefWithVariablesNeededByOutputs) {
  // Test freezing a graph with variables that are needed by outputs in the
  // SignatureDef. The variables should be frozen.
  SavedModelBundle saved_model_bundle;
  GraphDef graph_def;
  Scope scope = Scope::NewRootScope();
  Output a = ops::Const(scope.WithOpName("a"), 10.0f, {});
  Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
  Output c = ops::Mul(scope.WithOpName("c"), a, var);
  Output assign = ops::Assign(scope.WithOpName("assign"), var, a);
  TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  // "c" isnt dependent on the variable, so nothing should be frozen.
  TF_ASSERT_OK(AddGraphDefWithOutputsToSavedModelBundle(
      graph_def, {"c:0"}, assign.name(), &saved_model_bundle));

  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));

  // There should be 3 nodes in the resulting graph_def, and none should be
  // variables.
  EXPECT_EQ(frozen_graph_def.node_size(), 3);
  for (const NodeDef& node : frozen_graph_def.node()) {
    EXPECT_NE(node.op(), "Variable") << node.name();
    EXPECT_NE(node.op(), "VariableV2") << node.name();
  }

  RunAndCompareFrozenAndUnfrozenGraphs(saved_model_bundle.session.get(),
                                       frozen_graph_def, "c:0");
}

TEST_F(FreezeTest, GraphDefWithVariablesNeededAndNotNeededByOutputs) {
  // Test freezing a graph with some variables that are needed and not needed by
  // the outputs in the SignatureDef. The resulting graph should only freeze
  // dependent variables.
  SavedModelBundle saved_model_bundle;
  GraphDef graph_def;
  Scope scope = Scope::NewRootScope();
  Output a = ops::Const(scope.WithOpName("a"), 10.0f, {});
  Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
  Output c = ops::Mul(scope.WithOpName("c"), a, var);
  Output assign = ops::Assign(scope.WithOpName("assign"), var, a);
  Output var_1 =
      ops::Variable(scope.WithOpName("var_1"), {}, DataType::DT_FLOAT);
  Output assign_1 = ops::Assign(scope.WithOpName("assign_1"), var, a);
  TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
  // "c" isnt dependent on the variable, so nothing should be frozen.
  TF_ASSERT_OK(AddGraphDefWithOutputsToSavedModelBundle(
      graph_def, {"c:0"}, assign.name(), &saved_model_bundle));

  GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_ASSERT_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs,
                                &outputs));

  // There should be 3 nodes in the resulting graph_def, and none should be
  // variables.
  EXPECT_EQ(frozen_graph_def.node_size(), 3);
  for (const NodeDef& node : frozen_graph_def.node()) {
    EXPECT_NE(node.op(), "Variable") << node.name();
    EXPECT_NE(node.op(), "VariableV2") << node.name();
  }

  RunAndCompareFrozenAndUnfrozenGraphs(saved_model_bundle.session.get(),
                                       frozen_graph_def, "c:0");
}

}  // namespace
}  // namespace tensorflow
