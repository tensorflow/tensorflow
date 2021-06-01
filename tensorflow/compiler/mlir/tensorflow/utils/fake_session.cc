/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

#include "absl/strings/match.h"
#include "llvm/Support/CommandLine.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace mlir {
namespace TF {
namespace test_util {
namespace {
using ::tensorflow::Status;
using ::tensorflow::Tensor;

const char kDeviceNamePrefix[] = "/job:worker/replica:0/task:1";
const char kDeviceName[] = "/job:worker/replica:0/task:1/device:CPU:0";

// Struct holding options for FakeSession which are configuered through
// command line flags.
struct FakeSessionOptions {
  llvm::cl::opt<bool> fail_to_fetch_local_device_manager{
      "fail-to-fetch-local-device-manager",
      llvm::cl::desc("Fail to fetch local device manager."),
      llvm::cl::init(false)};
};
FakeSessionOptions* kSessionOptions = []() { return new FakeSessionOptions; }();
}  // namespace

FakeSession::FakeSession() {
  BuildDeviceManager();
  InitVariables();
}

void FakeSession::BuildDeviceManager() {
  auto device =
      tensorflow::DeviceFactory::NewDevice("CPU", {}, kDeviceNamePrefix);
  device_mgr_ =
      absl::make_unique<tensorflow::StaticDeviceMgr>(std::move(device));
}

void FakeSession::InitVariables() {
  tensorflow::Device* device = nullptr;
  auto status = device_mgr_->LookupDevice(kDeviceName, &device);
  if (status != Status::OK()) return;
  auto container = device->resource_manager()->default_container();

  // Create 2 resources and initialize them with dummy values.
  (void)device->resource_manager()->Create(
      container, "var1", new tensorflow::Var(tensorflow::DataType::DT_FLOAT));
  (void)device->resource_manager()->Create(
      container, "var2", new tensorflow::Var(tensorflow::DataType::DT_FLOAT));
}

Status FakeSession::Create(const tensorflow::GraphDef& graph) {
  return tensorflow::errors::Unimplemented("not available");
}
Status FakeSession::Extend(const tensorflow::GraphDef& graph) {
  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::Close() {
  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::ListDevices(
    std::vector<tensorflow::DeviceAttributes>* response) {
  return tensorflow::errors::Unimplemented("not available");
}

Status FakeSession::LocalDeviceManager(
    const tensorflow::DeviceMgr** deviceMgrPtr) {
  if (kSessionOptions->fail_to_fetch_local_device_manager)
    return Status(tensorflow::error::UNKNOWN, "No Local Device Manager");
  *deviceMgrPtr = device_mgr_.get();
  return Status::OK();
}

Status FakeSession::Run(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes,
    std::vector<Tensor>* outputs) {
  tensorflow::RunMetadata run_metadata;
  return Run(tensorflow::RunOptions(), inputs, output_names, target_nodes,
             outputs, &run_metadata);
}

Status FakeSession::Run(
    const tensorflow::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    tensorflow::RunMetadata* run_metadata) {
  return Run(run_options, inputs, output_names, target_nodes, outputs,
             run_metadata, tensorflow::thread::ThreadPoolOptions());
}

Status FakeSession::Run(
    const tensorflow::RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes, std::vector<Tensor>* outputs,
    tensorflow::RunMetadata* run_metadata,
    const tensorflow::thread::ThreadPoolOptions& thread_pool_options) {
  for (const std::string& output_name : output_names) {
    Tensor output;
    if (output_name == "dense/bias") {
      Tensor t = Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "dense/kernel") {
      Tensor t =
          Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({100, 50}));
      t.flat<float>().setZero();
      outputs->push_back(t);
    } else if (output_name == "var1") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var1");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var2") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var2");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "var3") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("var3");
      t.scalar<tensorflow::ResourceHandle>()().set_device(kDeviceName);

      outputs->push_back(t);
    } else if (output_name == "invalid_var") {
      Tensor t = Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({1}));
      t.scalar<tensorflow::ResourceHandle>()().set_name("invalid_var");
      t.scalar<tensorflow::ResourceHandle>()().set_device("invalid_device");

      outputs->push_back(t);
    } else if (absl::StartsWith(output_name, "var")) {
      return Status(tensorflow::error::NOT_FOUND,
                    "Can't find variable " + output_name + " in session");
    } else {
      // Create a scalar float tensor.
      Tensor t = Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
      t.flat<float>()(0) = 1.0f;
      outputs->push_back(t);
    }
  }
  return Status::OK();
}

}  // namespace test_util
}  // namespace TF
}  // namespace mlir
