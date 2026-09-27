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

// Thread-safety tests for sharing one XNNPACK delegate between interpreters.
//
// A single `TfLiteDelegate` may legitimately be handed to several interpreters:
// that is how callers share one thread pool, one weights cache and one
// workspace instead of paying for them per model. The delegate carries a
// `workspace_mutex_` precisely because it expects that usage.
//
// XNNPACK itself does no locking. Every runtime built from a shared workspace
// is threaded onto that workspace's intrusive `first_user` list, and all three
// operations on that list are plain unsynchronized memory accesses:
//
//   * `xnn_create_runtime_v4` pushes onto the head and bumps a non-atomic
//     refcount (third_party/XNNPACK/src/runtime.c:765-768);
//   * `xnn_delete_runtime` unlinks, with a search loop that has no bound
//     (runtime.c:1244-1255);
//   * `xnn_reshape_runtime` walks the whole list after growing the workspace,
//     shifting every sharer's tensor pointers by however far the buffer moved
//     (runtime.c:308-372) -- having already freed the old buffer
//     (runtime.c:267).
//
// The third one is why a lost list entry is not merely untidy. The shift is
// relative, so it has to be applied exactly once to each attached runtime. A
// runtime the walk misses keeps pointers into memory that has already been
// handed back to the allocator.
//
// The delegate therefore has to serialize all four of its entry points --
// Create, Prepare, Invoke and destruction -- on `workspace_mutex_`. These tests
// exercise the interleavings that would break if any one of them stopped taking
// it. Under ThreadSanitizer they report the missing synchronization directly
// and deterministically; without TSan they are smoke tests that concurrent
// delegation neither crashes nor deadlocks.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
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

// Two interpreters is the minimum that can exhibit the race and is what the
// production scenario looks like. TSan reports unordered accesses regardless of
// how the two threads actually interleave, so more threads would not make the
// finding more reliable, only the test slower.
constexpr int kNumThreads = 2;

// The shapes are deliberately not tiny. The intermediate activation between the
// two layers is the tensor that gets placed in the shared workspace, and the
// workspace only has to grow -- which is what triggers the pointer rebasing
// loop -- when there is a meaningful amount to allocate.
constexpr int32_t kBatchSize = 8;
constexpr int32_t kInputChannels = 32;
constexpr int32_t kHiddenChannels = 64;
constexpr int32_t kOutputChannels = 16;

constexpr size_t kInputSize = kBatchSize * kInputChannels;
constexpr size_t kOutputSize = kBatchSize * kOutputChannels;

// Builds a two layer float model:
//
//   input ──FC──> hidden ──FC──> output
//
// `hidden` is neither a subgraph input nor output, so XNNPACK allocates it in
// the delegate's workspace. That is the shared state this test is about.
std::vector<char> CreateTwoLayerModel() {
  flatbuffers::FlatBufferBuilder builder;

  // Deterministic weights. The values themselves do not matter; only that the
  // buffers are static, so the delegate packs them at Create time.
  auto rng = std::mt19937(0);
  auto weight_rng = std::bind(
      std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float> filter1_data(kHiddenChannels * kInputChannels);
  std::generate(filter1_data.begin(), filter1_data.end(), std::ref(weight_rng));
  std::vector<float> bias1_data(kHiddenChannels);
  std::generate(bias1_data.begin(), bias1_data.end(), std::ref(weight_rng));
  std::vector<float> filter2_data(kOutputChannels * kHiddenChannels);
  std::generate(filter2_data.begin(), filter2_data.end(), std::ref(weight_rng));
  std::vector<float> bias2_data(kOutputChannels);
  std::generate(bias2_data.begin(), bias2_data.end(), std::ref(weight_rng));

  const auto float_buffer = [&](const std::vector<float>& data) {
    return CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(data.data()),
                             sizeof(float) * data.size()));
  };

  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_FULLY_CONNECTED)}};

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      // Buffer 0 is conventionally empty and shared by all dynamic tensors.
      CreateBuffer(builder, builder.CreateVector({})),
      float_buffer(filter1_data),
      float_buffer(bias1_data),
      float_buffer(filter2_data),
      float_buffer(bias2_data),
  }};

  const std::vector<int32_t> input_shape = {kBatchSize, kInputChannels};
  const std::vector<int32_t> filter1_shape = {kHiddenChannels, kInputChannels};
  const std::vector<int32_t> bias1_shape = {kHiddenChannels};
  const std::vector<int32_t> hidden_shape = {kBatchSize, kHiddenChannels};
  const std::vector<int32_t> filter2_shape = {kOutputChannels, kHiddenChannels};
  const std::vector<int32_t> bias2_shape = {kOutputChannels};
  const std::vector<int32_t> output_shape = {kBatchSize, kOutputChannels};

  const auto float_tensor = [&](const std::vector<int32_t>& shape,
                                uint32_t buffer) {
    return CreateTensor(
        builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
        TensorType_FLOAT32, buffer);
  };

  std::vector<flatbuffers::Offset<Tensor>> tensors{{
      float_tensor(input_shape, /*buffer=*/0),    // 0: input
      float_tensor(filter1_shape, /*buffer=*/1),  // 1: filter 1
      float_tensor(bias1_shape, /*buffer=*/2),    // 2: bias 1
      float_tensor(hidden_shape, /*buffer=*/0),   // 3: hidden (in workspace)
      float_tensor(filter2_shape, /*buffer=*/3),  // 4: filter 2
      float_tensor(bias2_shape, /*buffer=*/4),    // 5: bias 2
      float_tensor(output_shape, /*buffer=*/0),   // 6: output
  }};

  const auto fully_connected = [&](const std::array<int32_t, 3>& op_inputs,
                                   const std::array<int32_t, 1>& op_outputs) {
    flatbuffers::Offset<FullyConnectedOptions> options =
        CreateFullyConnectedOptions(builder, ActivationFunctionType_NONE,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT,
                                    /*keep_num_dims=*/false,
                                    /*asymmetric_quantize_inputs=*/false);
    return CreateOperator(
        builder, /*opcode_index=*/0,
        builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
        builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
        BuiltinOptions_FullyConnectedOptions, options.Union());
  };

  std::vector<flatbuffers::Offset<Operator>> operators{{
      fully_connected({{0, 1, 2}}, {{3}}),
      fully_connected({{3, 4, 5}}, {{6}}),
  }};

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{6}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Two layer model for delegate sharing"),
      builder.CreateVector(buffers.data(), buffers.size()));
  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

// Releases every thread at once. Starting the threads one by one would let the
// first finish `Create` before the second begins, which is exactly the
// interleaving the test must avoid.
class StartLine {
 public:
  explicit StartLine(int num_participants)
      : num_participants_(num_participants) {}

  // Called by each worker. Blocks until every worker has arrived.
  void Wait() {
    arrived_.fetch_add(1, std::memory_order_relaxed);
    while (arrived_.load(std::memory_order_relaxed) < num_participants_) {
      std::this_thread::yield();
    }
  }

 private:
  const int num_participants_;
  std::atomic<int> arrived_{0};
};

std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
CreateSharedDelegate() {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  // One worker thread per delegate keeps the intra-op thread pool out of the
  // picture. The race under test is between the two calling threads, not
  // between XNNPACK's own workers.
  delegate_options.num_threads = 1;
  return std::unique_ptr<TfLiteDelegate,
                         decltype(&TfLiteXNNPackDelegateDelete)>(
      TfLiteXNNPackDelegateCreate(&delegate_options),
      TfLiteXNNPackDelegateDelete);
}

std::unique_ptr<Interpreter> BuildInterpreter(const Model* model) {
  std::unique_ptr<Interpreter> interpreter;
  // Default delegates are disabled so that the only delegate applied is the one
  // this test shares on purpose.
  if (InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter) != kTfLiteOk) {
    return nullptr;
  }
  return interpreter;
}

// Applies one shared delegate to two interpreters from two threads at once.
//
// Both threads run `Subgraph::Create`, and therefore `xnn_create_runtime_v4`,
// against the same workspace. Before `Subgraph::Create` took
// `workspace_mutex_`, TSan reported races here on the workspace's `first_user`
// list head and on its non-atomic `ref_count`, with one stack in
// `xnn_create_runtime_v4` and the other in either the same function or
// `initialize_workspace_values`.
//
// Both findings must stay gone.
TEST(XnnpackDelegateThreadSafety, ConcurrentDelegationToOneSharedDelegate) {
  const std::vector<char> buffer = CreateTwoLayerModel();
  const Model* model = GetModel(buffer.data());

  // One delegate, and therefore one workspace, one weights cache and one thread
  // pool, for both interpreters.
  auto shared_delegate = CreateSharedDelegate();
  ASSERT_NE(shared_delegate, nullptr);

  std::array<std::unique_ptr<Interpreter>, kNumThreads> interpreters;
  for (int i = 0; i < kNumThreads; i++) {
    interpreters[i] = BuildInterpreter(model);
    ASSERT_NE(interpreters[i], nullptr) << "interpreter " << i;
    ASSERT_EQ(interpreters[i]->AllocateTensors(), kTfLiteOk)
        << "interpreter " << i;
  }

  // Phase 1: delegate concurrently. This is where `Subgraph::Create` runs.
  StartLine start_line(kNumThreads);
  std::array<TfLiteStatus, kNumThreads> delegation_status;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([&, i] {
      start_line.Wait();
      delegation_status[i] =
          interpreters[i]->ModifyGraphWithDelegate(shared_delegate.get());
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  threads.clear();

  for (int i = 0; i < kNumThreads; i++) {
    ASSERT_EQ(delegation_status[i], kTfLiteOk) << "interpreter " << i;
    // If the delegate declined the nodes there would be no XNNPACK runtime and
    // hence nothing sharing the workspace, making the test vacuous.
    ASSERT_EQ(interpreters[i]->execution_plan().size(), 1)
        << "interpreter " << i
        << ": the delegate did not take the graph, so no XNNPACK runtime was "
           "created and the workspace is not actually shared";
  }

  // Phase 2: invoke concurrently. `Subgraph::Invoke` holds the mutex, so this
  // phase on its own is expected to be clean; it is here because the damage
  // done in phase 1 only becomes observable once the runtimes are used.
  auto value_rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                             std::mt19937(1));
  for (int i = 0; i < kNumThreads; i++) {
    std::generate_n(interpreters[i]->typed_input_tensor<float>(0), kInputSize,
                    std::ref(value_rng));
  }

  StartLine invoke_start_line(kNumThreads);
  std::array<TfLiteStatus, kNumThreads> invoke_status;
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([&, i] {
      invoke_start_line.Wait();
      invoke_status[i] = interpreters[i]->Invoke();
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(invoke_status[i], kTfLiteOk) << "interpreter " << i;
  }

  // Both interpreters ran the same weights on different inputs, so the outputs
  // are expected to differ; all that is checked is that the results are
  // finite. A rebased-away or stale workspace pointer typically shows up here
  // as NaN or as a sanitizer report above.
  for (int i = 0; i < kNumThreads; i++) {
    const float* output = interpreters[i]->typed_output_tensor<float>(0);
    for (size_t k = 0; k < kOutputSize; k++) {
      EXPECT_TRUE(std::isfinite(output[k]))
          << "interpreter " << i << ", element " << k
          << ": non-finite output suggests the runtime read through a stale "
             "workspace pointer";
    }
  }
}

// Number of rounds each worker performs in the create-vs-invoke test. A data
// race only becomes visible to TSan when the two accesses land close enough in
// time, so the two workers are run repeatedly rather than once.
constexpr int kRounds = 20;

// Runs `Subgraph::Create` and `~Subgraph` on one thread while
// `Subgraph::Prepare` and `Subgraph::Invoke` run on another, both against the
// same `Delegate` and therefore the same `xnn_workspace_t`.
//
// This is the interleaving the first test does not reach. There, both threads
// delegate together and only then both invoke, so no thread is ever inside
// `Subgraph::Create` while another is inside `Subgraph::Invoke`. The
// distinction matters because these are the two ends of the mechanism: one
// thread pushes onto and unlinks from the sharer list while the other walks it
// rebasing pointers.
//
// The production consequence of getting this wrong is a runtime that is dropped
// from the sharer list and therefore never rebased when the workspace is freed
// and reallocated underneath it, leaving its tensors pointing into memory the
// allocator has already reused.
//
// This test also exercises the deadlock risk the fix introduces: the creating
// thread destroys each transient interpreter, and `~Subgraph` takes the same
// lock. A hang here means the lock is being taken twice on one path.
TEST(XnnpackDelegateThreadSafety, DelegationConcurrentWithInvocation) {
  const std::vector<char> buffer = CreateTwoLayerModel();
  const Model* model = GetModel(buffer.data());

  auto shared_delegate = CreateSharedDelegate();
  ASSERT_NE(shared_delegate, nullptr);

  // The long-lived interpreter, delegated up front and single-threaded so far.
  // The invoking worker below reuses it for every round; this stands in for an
  // application that keeps one model warm.
  std::unique_ptr<Interpreter> resident = BuildInterpreter(model);
  ASSERT_NE(resident, nullptr);
  ASSERT_EQ(resident->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(resident->ModifyGraphWithDelegate(shared_delegate.get()),
            kTfLiteOk);
  // Without this the delegate declined the graph, there is no XNNPACK runtime
  // attached to the workspace, and the test would prove nothing.
  ASSERT_EQ(resident->execution_plan().size(), 1)
      << "the delegate did not take the resident graph";

  auto value_rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                             std::mt19937(2));
  std::generate_n(resident->typed_input_tensor<float>(0), kInputSize,
                  std::ref(value_rng));

  // Statuses are collected rather than asserted inside the threads because
  // gunit assertions are not safe to run off the main thread.
  std::atomic<int> creation_failures{0};
  std::atomic<int> undelegated_creations{0};
  std::atomic<int> invocation_failures{0};

  StartLine start_line(2);

  // Worker A: keeps building and delegating fresh interpreters, so it spends
  // its time inside the unlocked `Subgraph::Create`.
  std::thread creator([&] {
    start_line.Wait();
    for (int round = 0; round < kRounds; round++) {
      std::unique_ptr<Interpreter> transient = BuildInterpreter(model);
      if (transient == nullptr || transient->AllocateTensors() != kTfLiteOk ||
          transient->ModifyGraphWithDelegate(shared_delegate.get()) !=
              kTfLiteOk) {
        creation_failures.fetch_add(1);
        continue;
      }
      if (transient->execution_plan().size() != 1) {
        undelegated_creations.fetch_add(1);
      }
      // `transient` is destroyed here, which also releases its reference to the
      // shared workspace. Destruction racing against creation is part of what
      // is under test.
    }
  });

  // Worker B: keeps re-shaping and invoking the resident interpreter.
  //
  // The resize is the whole point. Invoking repeatedly at a fixed shape does
  // not re-enter `Subgraph::Prepare`, so `xnn_reshape_runtime` never runs, the
  // workspace never grows, and the rebasing loop this test is aiming at is
  // never executed. Alternating the batch size forces TFLite to re-prepare the
  // delegate node on every round, which reaches
  // `initialize_workspace_values` (XNNPACK runtime.c:240-376) and, when the
  // arena has to grow, frees the old buffer and walks `workspace->first_user`
  // rewriting every sharing runtime's tensor pointers.
  std::thread invoker([&] {
    start_line.Wait();
    for (int round = 0; round < kRounds; round++) {
      const int batch = (round % 2 == 0) ? kBatchSize : kBatchSize * 2;
      if (resident->ResizeInputTensor(resident->inputs()[0],
                                      {batch, kInputChannels}) != kTfLiteOk ||
          resident->AllocateTensors() != kTfLiteOk) {
        invocation_failures.fetch_add(1);
        continue;
      }
      std::fill_n(resident->typed_input_tensor<float>(0),
                  static_cast<size_t>(batch) * kInputChannels, 0.5f);
      if (resident->Invoke() != kTfLiteOk) {
        invocation_failures.fetch_add(1);
      }
    }
  });

  creator.join();
  invoker.join();

  EXPECT_EQ(creation_failures.load(), 0)
      << "delegating a fresh interpreter failed while another thread was "
         "invoking";
  EXPECT_EQ(undelegated_creations.load(), 0)
      << "the delegate declined a fresh graph, so those rounds did not "
         "actually share the workspace";
  EXPECT_EQ(invocation_failures.load(), 0)
      << "invoking failed while another thread was delegating";

  const float* output = resident->typed_output_tensor<float>(0);
  for (size_t k = 0; k < kOutputSize; k++) {
    EXPECT_TRUE(std::isfinite(output[k]))
        << "element " << k
        << ": non-finite output suggests the resident runtime read through a "
           "workspace pointer that was rebased or freed underneath it";
  }
}

// Destroys the delegate before the interpreter that is using it.
//
// This is not an exotic ordering; it is the common one. Callers routinely
// declare the interpreter first and the delegate second, so the delegate is
// destroyed first and the delegated kernels follow. TFLite tolerates this
// because the XNNPACK workspace is reference counted and outlives the delegate
// object.
//
// The lock guarding that workspace has to outlive the delegate too. When it was
// a plain `std::mutex` member of `Delegate`, `~Subgraph` locked an already
// destroyed mutex and hung; holding it by `shared_ptr` from each `Subgraph` is
// what makes this ordering safe.
//
// A regression shows up as a hang, not a failure, so this test is deliberately
// cheap: if it stops finishing, the lifetime coupling has been broken again.
TEST(XnnpackDelegateThreadSafety, DelegateDestroyedBeforeInterpreter) {
  const std::vector<char> buffer = CreateTwoLayerModel();
  const Model* model = GetModel(buffer.data());

  // Declared first, so destroyed last -- the interpreter outlives the delegate.
  std::unique_ptr<Interpreter> interpreter = BuildInterpreter(model);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  {
    auto delegate = CreateSharedDelegate();
    ASSERT_NE(delegate, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
    // Without a delegated node there is no `Subgraph`, hence no destructor to
    // run, and the test would prove nothing.
    ASSERT_EQ(interpreter->execution_plan().size(), 1)
        << "the delegate did not take the graph";

    std::fill_n(interpreter->typed_input_tensor<float>(0), kInputSize, 0.25f);
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  }
  // The delegate is gone here; the delegated kernel is not. Destroying the
  // interpreter now runs `~Subgraph` after its `Delegate` has been freed.
  interpreter.reset();
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
