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
#include "tensorflow/lite/delegates/flex/delegate_data.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace flex {
namespace {

using ::tensorflow::protobuf::TextFormat;
using ::tensorflow::protobuf::util::MessageDifferencer;

TEST(DelegateDataTest, Basic) {
  DelegateData data;
  // We only check for success because it is hard to make initialization fail.
  // It only happens if we manage to not link the CPU device factory into the
  // binary.
  tensorflow::SessionOptions session_options;
  session_options.config.set_intra_op_parallelism_threads(2);
  EXPECT_TRUE(data.Prepare(session_options).ok());

  TfLiteContext dummy_context1 = {};
  TfLiteContext dummy_context2 = {};
  ASSERT_NE(data.GetEagerContext(), nullptr);
  EXPECT_NE(data.GetBufferMap(&dummy_context1), nullptr);
  EXPECT_NE(data.GetBufferMap(&dummy_context1),
            data.GetBufferMap(&dummy_context2));
}

TEST(DelegateDataTest, CheckFunctionDef) {
  tensorflow::StaticDeviceMgr device_mgr(tensorflow::DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  tensorflow::EagerContext* eager_context = new tensorflow::EagerContext(
      tensorflow::SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, &device_mgr, /*device_mgr_owned*/ false, nullptr,
      nullptr);

  auto select_subgraphs_to_register =
      [](const std::vector<std::unique_ptr<Subgraph>>& subgraphs,
         std::set<std::string>* result) {
        result->insert("add_subgraph");
        result->insert("mul_subgraph");
        return tensorflow::Status::OK();
      };

  // Builds a TF Lite primary graph with two subgraphs.
  subgraph_test_util::SubgraphBuilder builder;
  std::unique_ptr<ErrorReporter> error_reporter =
      absl::make_unique<TestErrorReporter>();
  auto add_subgraph = absl::make_unique<Subgraph>(
      error_reporter.get(), /*external_contexts=*/nullptr,
      /*subgraphs=*/nullptr, /*resources=*/nullptr);
  add_subgraph->SetName("add_subgraph");
  auto mul_subgraph = absl::make_unique<Subgraph>(
      error_reporter.get(), /*external_contexts=*/nullptr,
      /*subgraphs=*/nullptr, /*resources=*/nullptr);
  mul_subgraph->SetName("mul_subgraph");
  builder.BuildAddSubgraph(add_subgraph.get());
  builder.BuildMulSubgraph(mul_subgraph.get());
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  subgraphs.push_back(std::move(add_subgraph));
  subgraphs.push_back(std::move(mul_subgraph));
  Subgraph main_subgraph(error_reporter.get(), nullptr, &subgraphs, nullptr);
  main_subgraph.SetName("main");
  TF_ASSERT_OK(RegisterFunctionDefForSubgraphs(
      main_subgraph, select_subgraphs_to_register,
      eager_context->HostCPU()->resource_manager(), eager_context));

  const string add_fdef_txt = R"pb(
    signature {
      name: "add_subgraph"
      input_arg { name: "args_0" type: DT_INT32 }
      input_arg { name: "args_1" type: DT_INT32 }
      output_arg { name: "res_0" type: DT_INT32 }
      is_stateful: true
    }
    node_def {
      name: "SubgraphResourceKey"
      op: "Const"
      attr {
        key: "dtype"
        value { type: DT_STRING }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {}
            string_val: "add_subgraph"
          }
        }
      }
    }
    node_def {
      name: "InvokeTfLite"
      op: "TfLiteSubgraphExecute"
      input: "SubgraphResourceKey:output:0"
      input: "args_0"
      input: "args_1"
      attr {
        key: "Tin"
        value { list { type: DT_INT32 type: DT_INT32 } }
      }
      attr {
        key: "Tout"
        value { list { type: DT_INT32 } }
      }
    }
    ret { key: "res_0" value: "InvokeTfLite:output:0" })pb";

  const string mul_fdef_txt = R"pb(
    signature {
      name: "mul_subgraph"
      input_arg { name: "args_0" type: DT_INT32 }
      input_arg { name: "args_1" type: DT_INT32 }
      output_arg { name: "res_0" type: DT_INT32 }
      is_stateful: true
    }
    node_def {
      name: "SubgraphResourceKey"
      op: "Const"
      attr {
        key: "dtype"
        value { type: DT_STRING }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {}
            string_val: "mul_subgraph"
          }
        }
      }
    }
    node_def {
      name: "InvokeTfLite"
      op: "TfLiteSubgraphExecute"
      input: "SubgraphResourceKey:output:0"
      input: "args_0"
      input: "args_1"
      attr {
        key: "Tin"
        value { list { type: DT_INT32 type: DT_INT32 } }
      }
      attr {
        key: "Tout"
        value { list { type: DT_INT32 } }
      }
    }
    ret { key: "res_0" value: "InvokeTfLite:output:0" })pb";

  tensorflow::FunctionDef add_fdef, mul_fdef;
  ASSERT_TRUE(TextFormat::ParseFromString(add_fdef_txt, &add_fdef));
  ASSERT_TRUE(TextFormat::ParseFromString(mul_fdef_txt, &mul_fdef));
  EXPECT_EQ(eager_context->GetFunctionDef("main"), nullptr);
  ASSERT_NE(eager_context->GetFunctionDef("add_subgraph"), nullptr);
  ASSERT_NE(eager_context->GetFunctionDef("mul_subgraph"), nullptr);
  EXPECT_TRUE(MessageDifferencer::Equals(
      *(eager_context->GetFunctionDef("add_subgraph")), add_fdef));
  EXPECT_TRUE(MessageDifferencer::Equals(
      *(eager_context->GetFunctionDef("mul_subgraph")), mul_fdef));

  eager_context->Unref();
}

TEST(DelegateDataTest, CheckFunctionDefWithOnlyMainGraph) {
  tensorflow::StaticDeviceMgr device_mgr(tensorflow::DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  tensorflow::EagerContext* eager_context = new tensorflow::EagerContext(
      tensorflow::SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, &device_mgr, /*device_mgr_owned*/ false, nullptr,
      nullptr);

  auto select_subgraphs_to_register =
      [](const std::vector<std::unique_ptr<Subgraph>>& subgraphs,
         std::set<std::string>* result) {
        result->insert("add_subgraph");
        result->insert("mul_subgraph");
        return tensorflow::Status::OK();
      };

  // Builds a TF Lite primary graph with two subgraphs.
  subgraph_test_util::SubgraphBuilder builder;
  std::unique_ptr<ErrorReporter> error_reporter =
      absl::make_unique<TestErrorReporter>();
  Subgraph main_subgraph(error_reporter.get(), /*external_contexts=*/nullptr,
                         /*subgraphs=*/nullptr, /*resources=*/nullptr);
  main_subgraph.SetName("main");
  TF_ASSERT_OK(RegisterFunctionDefForSubgraphs(
      main_subgraph, select_subgraphs_to_register,
      eager_context->HostCPU()->resource_manager(), eager_context));

  EXPECT_EQ(eager_context->GetFunctionDef("main"), nullptr);

  eager_context->Unref();
}

}  // namespace
}  // namespace flex
}  // namespace tflite
