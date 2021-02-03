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

void optimize_func(void* optimizer, TF_Buffer* graph_buf,
                   TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {}

void PopulateDefaultParam(TP_OptimizerRegistrationParams* params) {
  params->struct_size = TP_OPTIMIZER_REGISTRARION_PARAMS_STRUCT_SIZE;
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
  ASSERT_EQ(
      PluginGraphOptimizerRegistry::CreateOptimizer(std::set<string>{"Success"})
          .size(),
      1);
  ConfigsList config = PluginGraphOptimizerRegistry::GetPluginConfigs(
      true, std::set<string>{"Success"});
  ASSERT_EQ(config.remapping, RewriterConfig::OFF);
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
  ASSERT_EQ(PluginGraphOptimizerRegistry::CreateOptimizer(
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

}  // namespace

}  // namespace grappler
}  // namespace tensorflow