/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/grappler/grappler.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

void optimize_func(void* optimizer, const TF_Buffer* graph_buf,
                   const TF_GrapplerItem* item, TF_Buffer* optimized_graph_buf,
                   TF_Status* tf_status) {}

void PopulateDefaultParam(TP_OptimizerRegistrationParams* params) {
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  params->optimizer->create_func = nullptr;
  params->optimizer->optimize_func = optimize_func;
  params->optimizer->destroy_func = nullptr;
}

TEST(Grappler, SuccessfulRegistration) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Success";
    params->optimizer_configs->remapping = TF_TriState_Off;
  };

  TF_ASSERT_OK(InitGraphPlugin(plugin_init));
  ASSERT_EQ(PluginGraphOptimizerRegistry::CreateOptimizers(
                std::set<string>{"Success"})
                .size(),
            1);
  ConfigList config = PluginGraphOptimizerRegistry::GetPluginConfigs(
      true, std::set<string>{"Success"});
  ASSERT_EQ(config.toggle_config["remapping"], RewriterConfig::OFF);
}

TEST(Grappler, MultiplePluginRegistration) {
  auto plugin_init_0 = [](TP_OptimizerRegistrationParams* const params,
                          TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Device0";
  };
  auto plugin_init_1 = [](TP_OptimizerRegistrationParams* const params,
                          TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "Device1";
  };

  TF_ASSERT_OK(InitGraphPlugin(plugin_init_0));
  TF_ASSERT_OK(InitGraphPlugin(plugin_init_1));
  ASSERT_EQ(PluginGraphOptimizerRegistry::CreateOptimizers(
                std::set<string>{"Device0", "Device1"})
                .size(),
            2);
}

TEST(Grappler, DeviceTypeNotSet) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = nullptr;
  };

  tensorflow::Status status = InitGraphPlugin(plugin_init);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(
      status.error_message(),
      "'device_type' field in TP_OptimizerRegistrationParams must be set.");
}

TEST(Grappler, OptimizeFuncNotSet) {
  auto plugin_init = [](TP_OptimizerRegistrationParams* const params,
                        TF_Status* const status) -> void {
    TF_SetStatus(status, TF_OK, "");
    PopulateDefaultParam(params);
    params->device_type = "FuncNotSet";
    params->optimizer->optimize_func = nullptr;
  };

  tensorflow::Status status = InitGraphPlugin(plugin_init);
  ASSERT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  ASSERT_EQ(status.error_message(),
            "'optimize_func' field in TP_Optimizer must be set.");
}

TEST(TF_GrapplerItem, NodesToPreserve) {
  GrapplerItem item;
  item.fetch = std::vector<string>{"Conv", "BiasAdd"};
  std::unordered_set<string> nodes_preserved = item.NodesToPreserve();
  TF_GrapplerItem* c_item = reinterpret_cast<TF_GrapplerItem*>(&item);

  int list_total_size = 0;
  for (const string& s : nodes_preserved) {
    list_total_size += s.size();
  }

  size_t storage_size = 0;
  int num_values = 0;
  TF_Status* status = TF_NewStatus();
  TF_GetNodesToPreserveListSize(c_item, &num_values, &storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(nodes_preserved.size(), num_values);
  EXPECT_EQ(list_total_size, storage_size);

  std::unique_ptr<char*[]> values(new char*[nodes_preserved.size()]);
  std::unique_ptr<size_t[]> lens(new size_t[nodes_preserved.size()]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetNodesToPreserveList(c_item, values.get(), lens.get(),
                            nodes_preserved.size(), storage.get(), storage_size,
                            status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (size_t i = 0; i < nodes_preserved.size(); ++i) {
    EXPECT_EQ(nodes_preserved.find(string(static_cast<const char*>(values[i]),
                                          lens[i])) != nodes_preserved.end(),
              true);
  }
  TF_DeleteStatus(status);
}

TEST(TF_GrapplerItem, FetchNodes) {
  GrapplerItem item;
  item.fetch = std::vector<string>{"Conv", "BiasAdd"};
  TF_GrapplerItem* c_item = reinterpret_cast<TF_GrapplerItem*>(&item);

  int list_total_size = 0;
  for (const string& s : item.fetch) {
    list_total_size += s.size();
  }

  size_t storage_size = 0;
  int num_values = 0;
  TF_Status* status = TF_NewStatus();
  TF_GetFetchNodesListSize(c_item, &num_values, &storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(item.fetch.size(), num_values);
  EXPECT_EQ(list_total_size, storage_size);

  std::unique_ptr<char*[]> values(new char*[item.fetch.size()]);
  std::unique_ptr<size_t[]> lens(new size_t[item.fetch.size()]);
  std::unique_ptr<char[]> storage(new char[storage_size]);
  TF_GetFetchNodesList(c_item, values.get(), lens.get(), item.fetch.size(),
                       storage.get(), storage_size, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (size_t i = 0; i < item.fetch.size(); ++i) {
    EXPECT_EQ(item.fetch[i].size(), lens[i]) << i;
    EXPECT_EQ(item.fetch[i],
              string(static_cast<const char*>(values[i]), lens[i]))
        << i;
  }
  TF_DeleteStatus(status);
}

TEST(TF_GraphProperties, InputProperties) {
  std::unique_ptr<SingleMachine> cluster(new SingleMachine(5 * 60, 3, 0));
  TF_ASSERT_OK(cluster->Provision());

  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_Status* status = TF_NewStatus();
  TF_GraphProperties* graph_properties =
      TF_NewGraphProperties(reinterpret_cast<TF_GrapplerItem*>(&item));
  TF_InferStatically(graph_properties, true, false, false, false, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "AddN") {
      int num_values = 0;
      TF_GetInputPropertiesListSize(graph_properties, node.name().c_str(),
                                    &num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      EXPECT_EQ(num_values, 1);

      std::vector<TF_Buffer*> in_props_buf(num_values, TF_NewBuffer());

      TF_GetInputPropertiesList(graph_properties, node.name().c_str(),
                                in_props_buf.data(), num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      tensorflow::OpInfo::TensorProperties in_props;
      Status s = tensorflow::BufferToMessage(in_props_buf[0], &in_props);
      TF_ASSERT_OK(s);

      EXPECT_EQ(DT_FLOAT, in_props.dtype());
      EXPECT_FALSE(in_props.shape().unknown_rank());
      EXPECT_EQ(2, in_props.shape().dim_size());
      EXPECT_EQ(10, in_props.shape().dim(0).size());
      EXPECT_EQ(1, in_props.shape().dim(1).size());

      for (int i = 0; i < in_props_buf.size(); i++)
        TF_DeleteBuffer(in_props_buf[i]);
    }
  }
  TF_DeleteGraphProperties(graph_properties);
  TF_DeleteStatus(status);
  TF_ASSERT_OK(cluster->Shutdown());
}

TEST(TF_GraphProperties, OutputProperties) {
  std::unique_ptr<SingleMachine> cluster(new SingleMachine(5 * 60, 3, 0));
  TF_ASSERT_OK(cluster->Provision());

  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_Status* status = TF_NewStatus();
  TF_GraphProperties* graph_properties =
      TF_NewGraphProperties(reinterpret_cast<TF_GrapplerItem*>(&item));
  TF_InferStatically(graph_properties, true, false, false, false, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "AddN") {
      int num_values = 0;
      TF_GetOutputPropertiesListSize(graph_properties, node.name().c_str(),
                                     &num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      EXPECT_EQ(num_values, 1);

      std::vector<TF_Buffer*> out_props_buf(num_values, TF_NewBuffer());

      TF_GetOutputPropertiesList(graph_properties, node.name().c_str(),
                                 out_props_buf.data(), num_values, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      tensorflow::OpInfo::TensorProperties out_props;
      Status s = tensorflow::BufferToMessage(out_props_buf[0], &out_props);
      TF_ASSERT_OK(s);

      EXPECT_EQ(DT_FLOAT, out_props.dtype());
      EXPECT_FALSE(out_props.shape().unknown_rank());
      EXPECT_EQ(2, out_props.shape().dim_size());
      EXPECT_EQ(10, out_props.shape().dim(0).size());
      EXPECT_EQ(1, out_props.shape().dim(1).size());

      for (int i = 0; i < out_props_buf.size(); i++)
        TF_DeleteBuffer(out_props_buf[i]);
    }
  }
  TF_DeleteStatus(status);
  TF_DeleteGraphProperties(graph_properties);
  TF_ASSERT_OK(cluster->Shutdown());
}

TEST(TF_FunctionLibraryDefinition, LookUpOpDef) {
  TF_Buffer* g_buf = TF_NewBuffer();
  TF_Buffer* op_buf = TF_NewBuffer();
  TF_Status* status = TF_NewStatus();
  GraphDef g_def;
  Status s = MessageToBuffer(g_def, g_buf);
  TF_ASSERT_OK(s);
  TF_FunctionLibraryDefinition* func =
      TF_NewFunctionLibraryDefinition(g_buf, status);

  TF_LookUpOpDef(func, "Add", op_buf, status);
  string actual_string(reinterpret_cast<const char*>(op_buf->data),
                       op_buf->length);
  ASSERT_EQ(TF_OK, TF_GetCode(status));

  const OpDef* expected_op_def;
  TF_ASSERT_OK(OpRegistry::Global()->LookUpOpDef("Add", &expected_op_def));
  string expected_serialized;
  expected_op_def->SerializeToString(&expected_serialized);
  EXPECT_EQ(expected_serialized, actual_string);
  TF_DeleteBuffer(g_buf);
  TF_DeleteBuffer(op_buf);
  TF_DeleteStatus(status);
  TF_DeleteFunctionLibraryDefinition(func);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
