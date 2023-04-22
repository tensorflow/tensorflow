/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

using ::tensorflow::DeviceMgr;
using ::tensorflow::Session;
using ::tensorflow::Status;
using ::tensorflow::Tensor;

// FakeSession is for testing only.
class FakeSession : public tensorflow::Session {
 public:
  FakeSession() {}
  ~FakeSession() override = default;

  Status Create(const tensorflow::GraphDef& graph) override {
    return tensorflow::errors::Unimplemented("not available");
  }
  Status Extend(const tensorflow::GraphDef& graph) override {
    return tensorflow::errors::Unimplemented("not available");
  }

  Status Close() override {
    return tensorflow::errors::Unimplemented("not available");
  }

  Status ListDevices(
      std::vector<tensorflow::DeviceAttributes>* response) override {
    return tensorflow::errors::Unimplemented("not available");
  }

  Status LocalDeviceManager(
      const tensorflow::DeviceMgr** deviceMgrPtr) override {
    // This method returns a null device manager without making an error.
    // Users of this method will be notified since it will have a fake data.
    *deviceMgrPtr = nullptr;
    return Status::OK();
  }

  Status Run(const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs) override {
    tensorflow::RunMetadata run_metadata;
    return Run(tensorflow::RunOptions(), inputs, output_names, target_nodes,
               outputs, &run_metadata);
  }

  Status Run(const tensorflow::RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs,
             tensorflow::RunMetadata* run_metadata) override {
    return Run(run_options, inputs, output_names, target_nodes, outputs,
               run_metadata, tensorflow::thread::ThreadPoolOptions());
  }

  Status Run(const tensorflow::RunOptions& run_options,
             const std::vector<std::pair<std::string, Tensor>>& inputs,
             const std::vector<std::string>& output_names,
             const std::vector<std::string>& target_nodes,
             std::vector<Tensor>* outputs,
             tensorflow::RunMetadata* run_metadata,
             const tensorflow::thread::ThreadPoolOptions& thread_pool_options)
      override {
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
};

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesTestPass
    : public PassWrapper<LiftVariablesTestPass, OperationPass<ModuleOp>> {
 public:
  LiftVariablesTestPass() { session_ = new FakeSession(); }

  ~LiftVariablesTestPass() override { delete session_; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(LiftVariables(module, session_))) signalPassFailure();
  }

 private:
  Session* session_;
};

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesInvalidSessionTestPass
    : public PassWrapper<LiftVariablesInvalidSessionTestPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Pass an invalid session argument, which is a nullptr.
    if (failed(LiftVariables(module, /*session=*/nullptr))) signalPassFailure();
  }
};

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LIFT_VARIABLES_TEST_PASS_H_
