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

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

// Standard builder for hooking one placeholder to one op.
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

#define BINARY_INPUT_OP_FUZZER(dtype, opName)                                  \
  class Fuzz##opName : public FuzzSession<Tensor, Tensor> {                    \
    void BuildGraph(const Scope& scope) override {                             \
      auto op_node1 =                                                          \
          tensorflow::ops::Placeholder(scope.WithOpName("input1"), dtype);     \
      auto op_node2 =                                                          \
          tensorflow::ops::Placeholder(scope.WithOpName("input2"), dtype);     \
      tensorflow::ops::opName(scope.WithOpName("output"), op_node1, op_node2); \
    }                                                                          \
    void FuzzImpl(const Tensor& input_tensor1,                                 \
                  const Tensor& input_tensor2) final {                         \
      RunInputs({{"input1", input_tensor1}, {"input2", input_tensor2}});       \
    }                                                                          \
  }

namespace tensorflow {
namespace fuzzing {

// Used by GFT to map a known domain (vector<T>) to an unknown
// domain (Tensor of datatype). T and datatype should match/be compatible.
template <typename T = uint8_t>
inline auto AnyTensor() {
  return fuzztest::Map(
      [](auto v) {
        Tensor tensor(DataTypeToEnum<T>::v(),
                      TensorShape({static_cast<int64_t>(v.size())}));
        auto flat_tensor = tensor.flat<T>();
        for (int i = 0; i < v.size(); ++i) {
          flat_tensor(i) = v[i];
        }
        return tensor;
      },
      fuzztest::Arbitrary<std::vector<T>>());
}

// Create a TensorFlow session using a specific GraphDef created
// by BuildGraph(), and make it available for fuzzing.
// Users must override BuildGraph and FuzzImpl to specify
// (1) which operations are being fuzzed; and
// (2) How to translate the uint8_t* buffer from the fuzzer
//     to a Tensor or Tensors that are semantically appropriate
//     for the op under test.
// For the simple cases of testing a single op that takes a single
// input Tensor, use the SINGLE_INPUT_OP_BUILDER(dtype, opName) macro in place
// of defining BuildGraphDef.
//
// Typical use:
// SINGLE_INPUT_OP_FUZZER(DT_UINT8, Identity);
// FUZZ_TEST_F(FuzzIdentity, Fuzz).WithDomains(AnyTensor());
template <typename... T>
class FuzzSession {
 public:
  FuzzSession() : initialized_(false) {}
  virtual ~FuzzSession() {}

  // Constructs a Graph using the supplied Scope.
  // By convention, the graph should have inputs named "input1", ...
  // "inputN", and one output node, named "output".
  // Users of FuzzSession should override this method to create their graph.
  virtual void BuildGraph(const Scope& scope) = 0;

  // Implements the logic that converts an opaque byte buffer
  // from the fuzzer to Tensor inputs to the graph.  Users must override.
  virtual void FuzzImpl(const T&...) = 0;

  // Initializes the FuzzSession.  Not safe for multithreading.
  // Separate init function because the call to virtual BuildGraphDef
  // can't be put into the constructor.
  Status InitIfNeeded() {
    if (initialized_) {
      return OkStatus();
    }
    initialized_ = true;

    Scope root = Scope::DisabledShapeInferenceScope().ExitOnError();
    SessionOptions options;
    session_ = std::unique_ptr<Session>(NewSession(options));

    BuildGraph(root);

    GraphDef graph_def;
    TF_CHECK_OK(root.ToGraphDef(&graph_def));

    Status status = session_->Create(graph_def);
    if (!status.ok()) {
      // This is FATAL, because this code is designed to fuzz an op
      // within a session.  Failure to create the session means we
      // can't send any data to the op.
      LOG(FATAL) << "Could not create session: "  // Crash OK
                 << status.error_message();
    }
    return status;
  }

  // Runs the TF session by pulling on the "output" node, attaching
  // the supplied input_tensor to the input node(s), and discarding
  // any returned output.
  // Note: We are ignoring Status from Run here since fuzzers don't need to
  // check it (as that will slow them down and printing/logging is useless).
  void RunInputs(const std::vector<std::pair<string, Tensor> >& inputs) {
    RunInputsWithStatus(inputs).IgnoreError();
  }

  // Same as RunInputs but don't ignore status
  Status RunInputsWithStatus(
      const std::vector<std::pair<string, Tensor> >& inputs) {
    return session_->Run(inputs, {}, {"output"}, nullptr);
  }

  // Dispatches to FuzzImpl;  small amount of sugar to keep the code
  // of the per-op fuzzers tiny.
  void Fuzz(const T&... args) {
    Status status = InitIfNeeded();
    TF_CHECK_OK(status) << "Fuzzer graph initialization failed: "
                        << status.error_message();
    // No return value from fuzzing:  Success is defined as "did not
    // crash".  The actual application results are irrelevant.
    FuzzImpl(args...);
  }

 private:
  bool initialized_;
  std::unique_ptr<Session> session_;
};

}  // end namespace fuzzing
}  // end namespace tensorflow

#endif  // TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_SESSION_H_
