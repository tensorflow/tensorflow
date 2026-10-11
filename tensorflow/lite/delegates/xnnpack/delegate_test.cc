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

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {
namespace {

std::vector<char> CreateMultiOpModel() {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 2> operator_codes{{
      CreateOperatorCode(builder, BuiltinOperator_ABS),
      CreateOperatorCode(builder, BuiltinOperator_NEG),
  }};

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 2> shape{{1, 16}};
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

}  // namespace

TEST(Delegate, CreateWithoutParams) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithDefaultParams) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithNumThreadsParam) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, GetThreadPool) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(xnnpack_delegate.get()));
  ASSERT_TRUE(threadpool);
  ASSERT_EQ(2, pthreadpool_get_threads_count(threadpool));
}

TEST(Delegate, ConcurrentInvokeAndSubgraphCreateDestroy) {
  std::vector<char> buffer = CreateMultiOpModel();
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::unique_ptr<Interpreter> invoke_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&invoke_interpreter),
            kTfLiteOk);
  ASSERT_NE(invoke_interpreter, nullptr);
  ASSERT_EQ(invoke_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(invoke_interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);
  ASSERT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);

  std::atomic<bool> stop{false};
  std::thread invoke_thread([&]() {
    int step = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      EXPECT_EQ(invoke_interpreter->ResizeInputTensor(
                    invoke_interpreter->inputs()[0], {1, 16 + step}),
                kTfLiteOk);
      EXPECT_EQ(invoke_interpreter->AllocateTensors(), kTfLiteOk);
      EXPECT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);
      ++step;
    }
  });

  std::thread create_destroy_thread([&]() {
    for (int i = 0; i < 50; ++i) {
      std::unique_ptr<Interpreter> interpreter;
      EXPECT_EQ(InterpreterBuilder(model, resolver)(&interpreter), kTfLiteOk);
      ASSERT_NE(interpreter, nullptr);
      EXPECT_EQ(interpreter->ResizeInputTensor(interpreter->inputs()[0],
                                               {1, 16 * (i + 1)}),
                kTfLiteOk);
      EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
      EXPECT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
                kTfLiteOk);
      EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
    }
    stop.store(true, std::memory_order_relaxed);
  });

  create_destroy_thread.join();
  invoke_thread.join();

  std::thread destroy_thread([&]() { invoke_interpreter.reset(); });
  xnnpack_delegate.reset();
  destroy_thread.join();
}

}  // namespace xnnpack
}  // namespace tflite
