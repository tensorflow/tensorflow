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

#include "tensorflow/compiler/jit/xla_kernel_creator.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

std::shared_ptr<NodeProperties> ToNodeProperties(const string& text) {
  NodeDef node_def;
  DataTypeVector dummy;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &node_def));
  return std::make_shared<NodeProperties>(nullptr, std::move(node_def), dummy,
                                          dummy);
}

// Create a FunctionDef that takes one resource and one regular param
FunctionDef XTimesY() {
  return FunctionDefHelper::Define(
      // Name
      "XTimesY",
      // Args
      {"x: float", "y: resource"},
      // Return values
      {"z: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"y0"}, "ReadVariableOp", {"y"}, {{"dtype", DT_FLOAT}}},
          {{"z"}, "Mul", {"x", "y0"}, {{"T", DT_FLOAT}}},
      });
}

class XlaKernelCreatorTest : public ::testing::Test {
 protected:
  void Init(const std::vector<FunctionDef>& flib) {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 1});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    FunctionDefLibrary proto;
    for (const auto& fdef : flib) {
      *(proto.add_function()) = fdef;
    }
    lib_def_ = absl::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), proto);
    OptimizerOptions opts;
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts,
        /*default_thread_pool=*/nullptr, /*cluster_flr=*/nullptr);
    flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  }

  FunctionLibraryRuntime* flr_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;

  std::unique_ptr<OpKernel> kernel_;
};

AttrValue BoolAttr(bool b) {
  AttrValue v;
  v.set_b(b);
  return v;
}

TEST_F(XlaKernelCreatorTest, OneFloatOneResourceArgument) {
  FunctionDef fdef = XTimesY();
  (*fdef.mutable_attr())["_XlaMustCompile"] = BoolAttr(true);
  Init({fdef});
  XlaKernelCreator xla_kernel_creator;
  auto callsite =
      ToNodeProperties(R"pb(
        name: 'XTimesY' op: 'XTimesY' input: 'a' input: 'b'
      )pb");
  (*(callsite->node_def.mutable_attr()))["_XlaMustCompile"] = BoolAttr(true);

  // Note: need to set attribute on the created node.
  Status status = xla_kernel_creator.CreateKernel(flr_, callsite, &kernel_);
  ASSERT_TRUE(status.ok()) << status.ToString();

  EXPECT_EQ("XTimesY", kernel_->name());
  EXPECT_EQ("XTimesY", kernel_->type_string());

  EXPECT_EQ(2, kernel_->num_inputs());
  EXPECT_EQ(DT_FLOAT, kernel_->input_type(0));
  EXPECT_EQ(DT_RESOURCE, kernel_->input_type(1));
  EXPECT_EQ(DEVICE_MEMORY, kernel_->input_memory_types()[0]);
  EXPECT_EQ(HOST_MEMORY, kernel_->input_memory_types()[1]);

  EXPECT_EQ(1, kernel_->num_outputs());
  EXPECT_EQ(DT_FLOAT, kernel_->output_type(0));
  EXPECT_EQ(DEVICE_MEMORY, kernel_->output_memory_types()[0]);
}

TEST_F(XlaKernelCreatorTest, FailsIfXlaCompileAttrNotSet) {
  FunctionDef fdef = XTimesY();
  Init({fdef});
  XlaKernelCreator xla_kernel_creator;

  Status status =
      xla_kernel_creator.CreateKernel(flr_, ToNodeProperties(R"proto(
                                        name: 'XTimesY'
                                        op: 'XTimesY'
                                        input: 'a'
                                        input: 'b'
                                      )proto"),
                                      &kernel_);
  EXPECT_TRUE(errors::IsInternal(status)) << status.ToString();
}

TEST_F(XlaKernelCreatorTest, FailsIfXlaCompileAttrIsSetToFalse) {
  FunctionDef fdef = XTimesY();
  (*fdef.mutable_attr())["_XlaMustCompile"] = BoolAttr(false);
  Init({fdef});
  XlaKernelCreator xla_kernel_creator;

  Status status =
      xla_kernel_creator.CreateKernel(flr_, ToNodeProperties(R"proto(
                                        name: 'XTimesY'
                                        op: 'XTimesY'
                                        input: 'a'
                                        input: 'b'
                                      )proto"),
                                      &kernel_);
  EXPECT_TRUE(errors::IsInternal(status)) << status.ToString();
}

}  // namespace tensorflow
