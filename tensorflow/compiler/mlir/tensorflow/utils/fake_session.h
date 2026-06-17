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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_FAKE_SESSION_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_FAKE_SESSION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace TF {
namespace test_util {
// FakeSession is for testing only.
class FakeSession : public tensorflow::Session {
 public:
  FakeSession();

  absl::Status Create(const tensorflow::GraphDef& graph) override;
  absl::Status Extend(const tensorflow::GraphDef& graph) override;

  absl::Status Close() override;

  absl::Status ListDevices(
      std::vector<tensorflow::DeviceAttributes>* response) override;

  absl::Status LocalDeviceManager(
      const tensorflow::DeviceMgr** deviceMgrPtr) override;

  absl::Status Run(
      const std::vector<std::pair<std::string, ::tensorflow::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::tensorflow::Tensor>* outputs) override;

  absl::Status Run(
      const tensorflow::RunOptions& run_options,
      const std::vector<std::pair<std::string, ::tensorflow::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::tensorflow::Tensor>* outputs,
      tensorflow::RunMetadata* run_metadata) override;

  absl::Status Run(
      const tensorflow::RunOptions& run_options,
      const std::vector<std::pair<std::string, ::tensorflow::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<::tensorflow::Tensor>* outputs,
      tensorflow::RunMetadata* run_metadata,
      const tensorflow::thread::ThreadPoolOptions& thread_pool_options)
      override;

 private:
  void InitVariables();
  void BuildDeviceManager();
  void Initialize();

  std::unique_ptr<tensorflow::DeviceMgr> device_mgr_;
  bool initialized_ = false;
};

}  // namespace test_util
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_FAKE_SESSION_H_
