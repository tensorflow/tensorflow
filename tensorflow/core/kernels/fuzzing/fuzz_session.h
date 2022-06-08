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

#ifndef TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_
#define TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

// Standard invoking function macro to dispatch to a fuzzer class.
#ifndef PLATFORM_WINDOWS
#define STANDARD_TF_FUZZ_FUNCTION(FuzzerClass)                              \
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { \
    static FuzzerClass* fuzzer = new FuzzerClass();                         \
    return fuzzer->Fuzz(data, size);                                        \
  }
#else
// We don't compile this for Windows, MSVC doesn't like it as pywrap in Windows
// links all the code into one big object file and there are conflicting
// function names.
#define STANDARD_TF_FUZZ_FUNCTION(FuzzerClass)
#endif

// Standard builder for hooking one placeholder to one op.
#define SINGLE_INPUT_OP_BUILDER(dtype, opName)                          \
  void BuildGraph(const Scope& scope) override {                        \
    auto op_node =                                                      \
        tensorflow::ops::Placeholder(scope.WithOpName("input"), dtype); \
    (void)tensorflow::ops::opName(scope.WithOpName("output"), op_node); \
  }

namespace tensorflow {
namespace fuzzing {

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
// class FooFuzzer : public FuzzSession {
//   SINGLE_INPUT_OP_BUILDER(DT_INT8, Identity);
//   void FuzzImpl(const uint8_t* data, size_t size) {
//      ... convert data and size to a Tensor, pass it to:
//      RunInputs({{"input", input_tensor}});
//
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
  virtual void FuzzImpl(const uint8_t* data, size_t size) = 0;

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
      LOG(FATAL) << "Could not create session: " << status.error_message();
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
  int Fuzz(const uint8_t* data, size_t size) {
    Status status = InitIfNeeded();
    TF_CHECK_OK(status) << "Fuzzer graph initialization failed: "
                        << status.error_message();
    // No return value from fuzzing:  Success is defined as "did not
    // crash".  The actual application results are irrelevant.
    FuzzImpl(data, size);
    return 0;
  }

 private:
  bool initialized_;
  std::unique_ptr<Session> session_;
};

// A specialized fuzz implementation for ops that take
// a single string.  Caller must still define the op
// to plumb by overriding BuildGraph or using
// a plumbing macro.
class FuzzStringInputOp : public FuzzSession {
  void FuzzImpl(const uint8_t* data, size_t size) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() =
        string(reinterpret_cast<const char*>(data), size);
    RunInputs({{"input", input_tensor}});
  }
};

}  // end namespace fuzzing
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_
