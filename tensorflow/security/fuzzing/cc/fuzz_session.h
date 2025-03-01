/* Copyright 2016 Google Inc. All Rights Reserved.

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
#ifndef TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_SESSION_H_
#define TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_SESSION_H_

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <limits>

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/lib/core/errors.h"

#define SINGLE_INPUT_OP_FUZZER(dtype, opName)                             \
  class Fuzz##opName : public FuzzSession<Tensor> {                       \
    void BuildGraph(const Scope& scope) override {                        \
      auto op_node =                                                      \
          tensorflow::ops::Placeholder(scope.WithOpName("input"), dtype); \
      tensorflow::ops::opName(scope.WithOpName("output"), op_node);       \
    }                                                                     \
    void FuzzImpl(const Tensor& input_tensor) final {                     \
      RunInputs({{"input", input_tensor}});                               \
    }                                                                     \
  }

#define BINARY_INPUT_OP_FUZZER(dtype1, dtype2, opName)                         \
  class Fuzz##opName : public FuzzSession<Tensor, Tensor> {                    \
    void BuildGraph(const Scope& scope) override {                             \
      auto op_node1 =                                                          \
          tensorflow::ops::Placeholder(scope.WithOpName("input1"), dtype1);    \
      auto op_node2 =                                                          \
          tensorflow::ops::Placeholder(scope.WithOpName("input2"), dtype2);    \
      tensorflow::ops::opName(scope.WithOpName("output"), op_node1, op_node2); \
    }                                                                          \
    void FuzzImpl(const Tensor& input_tensor1,                                 \
                  const Tensor& input_tensor2) final {                         \
      RunInputs({{"input1", input_tensor1}, {"input2", input_tensor2}});       \
    }                                                                          \
  }

namespace tensorflow {
namespace fuzzing {

template <typename T = uint8_t>
inline auto AnyTensor() {
  return fuzztest::Map(
      [](const std::vector<T>& v) {
        if (v.empty()) {
          return Tensor(DataTypeToEnum<T>::v(), TensorShape({0}));
        }

        TensorShape shape({static_cast<int64_t>(v.size())});
        if (!shape.IsValid()) {
           throw std::runtime_error("Invalid tensor shape.");
        }

        Tensor tensor(DataTypeToEnum<T>::v(), shape);
        auto flat_tensor = tensor.flat<T>();

        for (size_t i = 0; i < v.size(); ++i) {
          flat_tensor(i) = v[i];
        }
        return tensor;
      },
      fuzztest::Arbitrary<std::vector<T>>());
}


template <typename... T>
class FuzzSession {
 public:
  FuzzSession() : initialized_(false), session_options_(CreateSessionOptions()) {}
  virtual ~FuzzSession() = default;

  virtual void BuildGraph(const Scope& scope) = 0;
  virtual void FuzzImpl(const T&...) = 0;

  absl::Status InitIfNeeded() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) {
      return absl::OkStatus();
    }
    initialized_ = true;

    Scope root = Scope::NewRootScope().ExitOnError();
    BuildGraph(root);

    GraphDef graph_def;
    absl::Status graph_status = root.ToGraphDef(&graph_def);
    if (!graph_status.ok()) {
        LOG(ERROR) << "Failed to build graph def: " << graph_status.message();
        return graph_status;
    }

    absl::Status session_status = CreateSession(graph_def);
    if (!session_status.ok()) {
      LOG(ERROR) << "Could not create session: " << session_status.message();
      return session_status;
    }

    return absl::OkStatus();
  }

  void RunInputs(const std::vector<std::pair<string, Tensor>>& inputs) {
    absl::Status status = RunInputsWithStatus(inputs);
    if (!status.ok()) {
        LOG(WARNING) << "RunInputs failed: " << status.message();
    }
  }

  absl::Status RunInputsWithStatus(const std::vector<std::pair<string, Tensor>>& inputs) {
      if (!session_) {
          return absl::InternalError("Session not initialized.");
      }

      std::vector<Tensor> outputs;
      absl::Status status = session_->Run(inputs, {"output"}, {}, &outputs);
      if (!status.ok()) {
        LOG(WARNING) << "Session run failed: " << status.message();
      }
      return status;
  }

  void Fuzz(const T&... args) {
      absl::Status status = InitIfNeeded();
      if (!status.ok()) {
          LOG(ERROR) << "Fuzzer graph initialization failed: " << status.message();
          return;
      }

      try {
          FuzzImpl(args...);
      } catch (const std::exception& e) {
          LOG(ERROR) << "FuzzImpl threw an exception: " << e.what();
      } catch (...) {
          LOG(ERROR) << "FuzzImpl threw an unknown exception.";
      }
  }


 private:
  SessionOptions CreateSessionOptions() {
    SessionOptions options;
    options.config.set_intra_op_parallelism_threads(1);
    options.config.set_inter_op_parallelism_threads(1);
    options.config.set_use_per_session_threads(true);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    return options;
  }


  absl::Status CreateSession(const GraphDef& graph_def) {
      session_ = std::unique_ptr<Session>(NewSession(session_options_));
      if (!session_) {
          return absl::InternalError("Failed to create session.");
      }

      absl::Status status = session_->Create(graph_def);
      if (!status.ok()) {
        session_.reset();
        return status;
      }
      return absl::OkStatus();
  }

  bool initialized_;
  SessionOptions session_options_;
  std::unique_ptr<Session> session_;
  std::mutex init_mutex_;
};

}  // end namespace fuzzing
}  // end namespace tensorflow

#endif  // TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_SESSION_H_
