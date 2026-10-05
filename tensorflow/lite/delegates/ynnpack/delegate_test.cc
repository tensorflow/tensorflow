/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace ynnpack {
namespace {

using DelegatePtr =
    std::unique_ptr<TfLiteDelegate, decltype(&TfLiteYNNPackDelegateDelete)>;

// Large enough that YNNPACK parallelizes the elementwise ops over the delegate
// thread pool.
constexpr int32_t kRows = 64;
constexpr int32_t kCols = 1024;

// Builds a float model with two delegatable unary ops: ABS -> NEG.
std::vector<char> CreateMultiOpModel() {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 2> operator_codes{{
      CreateOperatorCode(builder, BuiltinOperator_ABS),
      CreateOperatorCode(builder, BuiltinOperator_NEG),
  }};

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 2> shape{{kRows, kCols}};
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
  }};

  const std::array<int32_t, 1> op0_inputs{{0}};
  const std::array<int32_t, 1> op0_outputs{{1}};
  const std::array<int32_t, 1> op1_inputs{{1}};
  const std::array<int32_t, 1> op1_outputs{{2}};
  const std::array<flatbuffers::Offset<Operator>, 2> ops{{
      CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op0_inputs.data(), op0_inputs.size()),
          builder.CreateVector<int32_t>(op0_outputs.data(),
                                        op0_outputs.size())),
      CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector<int32_t>(op1_inputs.data(), op1_inputs.size()),
          builder.CreateVector<int32_t>(op1_outputs.data(),
                                        op1_outputs.size())),
  }};

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(ops.data(), ops.size()));

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Multi-op model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

DelegatePtr CreateDelegate(int num_threads) {
  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = num_threads;
  return DelegatePtr(TfLiteYNNPackDelegateCreate(&options),
                     TfLiteYNNPackDelegateDelete);
}

std::unique_ptr<Interpreter> CreateDelegatedInterpreter(
    const Model* model, const OpResolver& resolver, TfLiteDelegate* delegate) {
  std::unique_ptr<Interpreter> interpreter;
  EXPECT_EQ(InterpreterBuilder(model, resolver)(&interpreter), kTfLiteOk);
  if (interpreter == nullptr) return nullptr;
  EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
  // The whole graph must have been replaced by a single delegate node,
  // otherwise this test is not exercising the delegate.
  EXPECT_EQ(interpreter->execution_plan().size(), 1);
  return interpreter;
}

void FillInput(Interpreter* interpreter) {
  float* input = interpreter->typed_input_tensor<float>(0);
  const int64_t n = interpreter->input_tensor(0)->bytes / sizeof(float);
  for (int64_t i = 0; i < n; ++i) {
    input[i] = static_cast<float>(i % 7) - 3.0f;
  }
}

void CheckOutput(Interpreter* interpreter) {
  const float* input = interpreter->typed_input_tensor<float>(0);
  const float* output = interpreter->typed_output_tensor<float>(0);
  const int64_t n = interpreter->output_tensor(0)->bytes / sizeof(float);
  for (int64_t i = 0; i < n; ++i) {
    ASSERT_EQ(output[i], -std::abs(input[i])) << "at " << i;
  }
}

}  // namespace

TEST(Delegate, CreateWithoutParams) {
  DelegatePtr delegate(TfLiteYNNPackDelegateCreate(nullptr),
                       TfLiteYNNPackDelegateDelete);
  ASSERT_NE(delegate, nullptr);
}

TEST(Delegate, CreateWithNumThreadsParam) {
  DelegatePtr delegate = CreateDelegate(/*num_threads=*/4);
  ASSERT_NE(delegate, nullptr);
}

// The delegate owns the thread pool that the kernels' runtimes use. Destroying
// the delegate before the interpreter must not break destroying the
// interpreter (and its kernels) afterwards.
TEST(Delegate, DeleteDelegateBeforeInterpreter) {
  std::vector<char> buffer = CreateMultiOpModel();
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  DelegatePtr delegate = CreateDelegate(/*num_threads=*/4);
  std::unique_ptr<Interpreter> interpreter =
      CreateDelegatedInterpreter(model, resolver, delegate.get());
  ASSERT_NE(interpreter, nullptr);
  FillInput(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  CheckOutput(interpreter.get());

  delegate.reset();
  interpreter.reset();
}

namespace {

// A profiler that runs a callback the first time an operator is invoked. The
// interpreter begins this event after its last access to the `TfLiteDelegate`
// and immediately before calling the delegate kernel's `Eval`, so the callback
// runs while the kernel is "in flight".
class OnFirstOpInvokeProfiler : public Profiler {
 public:
  explicit OnFirstOpInvokeProfiler(std::function<void()> callback)
      : callback_(std::move(callback)) {}

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    if (event_type == EventType::OPERATOR_INVOKE_EVENT && callback_) {
      std::function<void()> callback = std::move(callback_);
      callback_ = nullptr;
      callback();
    }
    return 0;
  }
  void EndEvent(uint32_t event_handle) override {}

 private:
  std::function<void()> callback_;
};

}  // namespace

// Destroy the delegate (on another thread) while the delegate kernel is being
// invoked. The in-flight `Eval` must still be able to use the thread pool.
TEST(Delegate, DeleteDelegateDuringInvoke) {
  std::vector<char> buffer = CreateMultiOpModel();
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  DelegatePtr delegate = CreateDelegate(/*num_threads=*/4);
  std::unique_ptr<Interpreter> interpreter =
      CreateDelegatedInterpreter(model, resolver, delegate.get());
  ASSERT_NE(interpreter, nullptr);
  FillInput(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  CheckOutput(interpreter.get());

  OnFirstOpInvokeProfiler profiler([&]() {
    std::thread delete_thread([&]() { delegate.reset(); });
    delete_thread.join();
  });
  interpreter->SetProfiler(&profiler);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  interpreter->SetProfiler(nullptr);
  ASSERT_EQ(delegate, nullptr);
  CheckOutput(interpreter.get());
}

// Multiple interpreters share one delegate (and therefore one thread pool), and
// are created, invoked and destroyed concurrently with another interpreter
// invoking in a loop.
TEST(Delegate, ConcurrentInvokeAndInterpreterCreateDestroy) {
  std::vector<char> buffer = CreateMultiOpModel();
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  DelegatePtr delegate = CreateDelegate(/*num_threads=*/4);
  std::unique_ptr<Interpreter> invoke_interpreter =
      CreateDelegatedInterpreter(model, resolver, delegate.get());
  ASSERT_NE(invoke_interpreter, nullptr);
  FillInput(invoke_interpreter.get());
  ASSERT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);

  std::atomic<bool> stop{false};
  std::thread invoke_thread([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      EXPECT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);
    }
  });

  std::thread create_destroy_thread([&]() {
    for (int i = 0; i < 20; ++i) {
      std::unique_ptr<Interpreter> interpreter =
          CreateDelegatedInterpreter(model, resolver, delegate.get());
      ASSERT_NE(interpreter, nullptr);
      FillInput(interpreter.get());
      EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
      CheckOutput(interpreter.get());
    }
    stop.store(true, std::memory_order_relaxed);
  });

  create_destroy_thread.join();
  invoke_thread.join();
  CheckOutput(invoke_interpreter.get());

  // Tear down the delegate and the interpreter concurrently.
  std::thread destroy_thread([&]() { invoke_interpreter.reset(); });
  delegate.reset();
  destroy_thread.join();
}

}  // namespace ynnpack
}  // namespace tflite
