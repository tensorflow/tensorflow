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

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/threadpool_options.h"

namespace mlir {
namespace TF {
namespace test_util {
using ::tensorflow::Status;
using ::tensorflow::Tensor;

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
  // This method returns a null device manager without making an error.
  // Users of this method will be notified since it will have a fake data.
  *deviceMgrPtr = nullptr;
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
