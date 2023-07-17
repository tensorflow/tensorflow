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
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "third_party/protobuf/text_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/tfrt/fallback/device_with_custom_allocator.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/async_handle.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/attribute_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/builtin_kernels.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel_runner_utils.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "tfrt/concurrency/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {
namespace {

struct MapFnOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.map_fn";
  // Tensor list or flow in inputs starts after max_iteration
  static constexpr int kTensorListFlowInStartIndex = 1;

  int32_t max_iteration() const {
    const auto& tensor =
        arguments()[0].Get<tensorflow::tfrt_stub::FallbackTensor>().tensor();
    DCHECK(TensorShapeUtils::IsScalar(tensor.shape()));

    return tensor.scalar<int32_t>()();
  }
  mlrt::RegisterValueSpan<tensorflow::tfrt_stub::FallbackTensor>
  tensor_list_or_flow_in() const {
    int num_args = arguments().size();
    return arguments()
        .drop_back(num_args - kTensorListFlowInStartIndex -
                   num_tensor_list_or_flow_in())
        .drop_front();
  }
  mlrt::bc::Span<uint8_t> tensor_list_or_flow_in_last_use() const {
    int num_args = last_uses().size();
    return last_uses().drop_front().drop_back(
        num_args - kTensorListFlowInStartIndex - num_tensor_list_or_flow_in());
  }

  int32_t body_func_index() const { return attributes().GetAs<int32_t>(0); }
  int32_t num_tensor_list_or_flow_in() const {
    return attributes().GetAs<int32_t>(1);
  }

  Context& context() { return execution_context().GetUserContext<Context>(); }

  void Invoke();
};

void MapFnOp::Invoke() {
  auto function = execution_context()
                      .loaded_executable()
                      .executable()
                      .functions()[body_func_index()];

  tsl::profiler::TraceMe trace_me("tf_mlrt.map_fn");
  trace_me.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode(
        {{"max_iteration", max_iteration()}, {"name", function.name().Get()}});
  });

  if (max_iteration() <= 0) {
    auto results = this->results();
    auto in_tensor_list_last_use = tensor_list_or_flow_in_last_use();
    DCHECK_EQ(results.size(), num_tensor_list_or_flow_in());
    auto in_tensor_list = tensor_list_or_flow_in();
    for (int i = 0; i < num_tensor_list_or_flow_in(); ++i) {
      if (in_tensor_list_last_use[i]) {
        results[i].Set(std::move(in_tensor_list[i]));
      } else {
        results[i].Set(in_tensor_list[i]);
      }
    }
    return;
  }

  DCHECK_GE(arguments().size(), 2);
  DCHECK_GE(results().size(), 1);

  std::vector<mlrt::AsyncHandle> handles;
  std::vector<mlrt::ExecutionContext*> execution_contexts;
  handles.reserve(max_iteration());
  execution_contexts.reserve(max_iteration());

  std::vector<mlrt::Promise> initializer_promises;
  initializer_promises.reserve(num_tensor_list_or_flow_in());

  std::vector<mlrt::Future> last_iter_futures;
  last_iter_futures.reserve(num_tensor_list_or_flow_in());
  for (int i = 0; i < num_tensor_list_or_flow_in(); ++i) {
    initializer_promises.push_back(
        mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>());
    last_iter_futures.push_back(initializer_promises.back().GetFuture());
  }

  std::vector<mlrt::Value> body_args;
  std::vector<uint8_t> body_arg_last_uses;
  body_args.resize(arguments().size() + 1 + num_tensor_list_or_flow_in());
  body_arg_last_uses.resize(body_args.size(), false);
  std::fill(body_arg_last_uses.begin(),
            body_arg_last_uses.begin() + 2 * num_tensor_list_or_flow_in() + 2,
            true);

  // Copy the invairant arguments (after max_iteration +
  // tensor_list_or_flow_ins)
  auto arg_iter = body_args.begin() + 2 * num_tensor_list_or_flow_in() + 2;
  for (int j = num_tensor_list_or_flow_in() + 1; j < arguments().size();
       ++j, ++arg_iter) {
    *arg_iter = arguments()[j];
  }

  auto* work_queue = execution_context().work_queue();
  DCHECK(work_queue);

  for (int i = 0; i < max_iteration(); ++i) {
    auto [promise, handle] = mlrt::AsyncHandle::Allocate(execution_context());

    auto& thread_execution_context = handle.execution_context();
    handles.push_back(std::move(handle));
    execution_contexts.push_back(&thread_execution_context);

    thread_execution_context.set_exit_handler(
        [&execution_context = thread_execution_context,
         promise = std::move(promise)]() mutable {
          std::move(promise).Finish(execution_context.status());
        });

    auto arg_iter = body_args.begin();
    for (int j = 0; j < last_iter_futures.size(); ++j) {
      *arg_iter = std::move(last_iter_futures[j]);
      ++arg_iter;
      auto tensor_promise =
          mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();

      // Iteration n's future provide a continuation token for the next
      // iteration.
      last_iter_futures[j] = tensor_promise.GetFuture();
      *arg_iter = std::move(tensor_promise);
      ++arg_iter;
    }

    // The current loop count is the next argument.
    tensorflow::Tensor loop_counter_tensor(DT_INT32, {});
    loop_counter_tensor.scalar<int32_t>()() = i;
    *arg_iter =
        tensorflow::tfrt_stub::FallbackTensor(std::move(loop_counter_tensor));
    ++arg_iter;

    tensorflow::Tensor element_index_tensor(DT_INT32, {});
    element_index_tensor.scalar<int32_t>()() = i;
    *arg_iter =
        tensorflow::tfrt_stub::FallbackTensor(std::move(element_index_tensor));
    ++arg_iter;

    thread_execution_context.Call(function, body_arg_last_uses,
                                  absl::MakeSpan(body_args),
                                  absl::Span<mlrt::Value>());
  }

  //  Kick off task by setting first future
  auto in_tensor_list = tensor_list_or_flow_in();
  auto in_tensor_list_last_use = tensor_list_or_flow_in_last_use();
  for (int j = 0; j < num_tensor_list_or_flow_in(); j++) {
    if (in_tensor_list_last_use[j]) {
      std::move(initializer_promises[j])
          .Set<tensorflow::tfrt_stub::FallbackTensor>(
              std::move(in_tensor_list[j]));
    } else {
      std::move(initializer_promises[j])
          .Set<tensorflow::tfrt_stub::FallbackTensor>(in_tensor_list[j]);
    }
  }

  int num_threads = work_queue->GetParallelismLevel();
  int batch_size = (max_iteration() + num_threads - 1) / num_threads;
  int num_batch = (max_iteration() + batch_size - 1) / batch_size;
  DCHECK_GE(num_batch, 1);
  int epilog_size = max_iteration() % batch_size;
  if (epilog_size == 0) {
    epilog_size = batch_size;
  }
  int prolog_size = num_batch > 1 ? batch_size : epilog_size;
  DCHECK_GT(batch_size, 0);
  DCHECK_GT(epilog_size, 0);
  DCHECK_LE(epilog_size, batch_size);
  DCHECK_GT(prolog_size, 0);
  DCHECK_LE(prolog_size, batch_size);

  auto run_batch = [execution_contexts = absl::MakeSpan(execution_contexts)](
                       int begin, int end) {
    DCHECK_LE(end, execution_contexts.size());
    for (int i = begin; i < end; ++i) {
      Execute(*execution_contexts[i]);
    }
  };

  // Run the first batch inline while the rest iterations are enqueued to the
  // thread pool.
  for (int batch_id = 1; batch_id < num_batch - 1; ++batch_id) {
    work_queue->AddTask([=]() {
      run_batch(batch_id * batch_size, batch_id * batch_size + batch_size);
    });
  }

  // epilog
  if (num_batch > 1) {
    work_queue->AddTask([=]() {
      int batch_id = num_batch - 1;
      run_batch(batch_id * batch_size, batch_id * batch_size + epilog_size);
    });
  }

  // prolog
  run_batch(0, prolog_size);

  mlrt::Future await_all = mlrt::AwaitAll(absl::MakeSpan(handles));

  // Need a separate promise to make this blocking call.
  // Do not use wait on last_iter_future b/c when last_iter_future is ready,
  // the body function may not return in theory yet.
  auto all_done_promise = mlrt::Promise::Allocate<mlrt::Control>();
  auto all_done_future = all_done_promise.GetFuture();

  std::move(await_all).Then([results = results(),
                             last_iter_futures = std::move(last_iter_futures),
                             execution_contexts = std::move(execution_contexts),
                             done_promise = std::move(all_done_promise)](
                                absl::Status status) mutable {
    DCHECK_EQ(results.size(), last_iter_futures.size());
    // TODO(deqiangc): future.then outside so that we can avoid this copy.
    for (int j = 0; j < last_iter_futures.size(); j++) {
      CHECK(last_iter_futures[j].IsReady());  // Crash OK

      if (last_iter_futures[j].IsError()) {
        // Error code and source location will reflect the first error if handle
        // does not report error.
        if (status.ok()) {
          status = absl::Status(
              /*code=*/last_iter_futures[j].GetError().code(),
              /*msg=*/
              absl::StrCat(last_iter_futures[j].GetError().message(),
                           ". First Error Index=", j, " of ",
                           last_iter_futures.size()),
              absl::SourceLocation());
          for (const auto& location :
               last_iter_futures[j].GetError().GetSourceLocations()) {
            status.AddSourceLocation(location);
          }
        }
      } else {
        results[j].Set(
            last_iter_futures[j].Get<tensorflow::tfrt_stub::FallbackTensor>());
      }
    }
    if (!status.ok()) {
      std::move(done_promise).SetError(std::move(status));
    } else {
      std::move(done_promise).Set<mlrt::Control>(mlrt::Control{});
    }
  });
  execution_context().Await(std::move(all_done_future));
}

struct CancelOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.cancel";
  void Invoke();
};

void CancelOp::Invoke() {
  if (execution_context().GetUserContext<Context>().IsCancelled()) {
    execution_context().FailOnCancellation();
  }
}

struct CreateOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.createop";

  absl::string_view node_def_text() const {
    return attributes().GetAs<mlrt::bc::String>(0).Get();
  }

  int32_t op_key() const { return attributes().GetAs<int32_t>(1); }

  Context& context() { return execution_context().GetUserContext<Context>(); }

  void Invoke();
};

void CreateOp::Invoke() {
  auto& fallback_request_state = context().fallback_request_state();

  tensorflow::NodeDef node_def;
  if (!proto2::TextFormat::ParseFromString(node_def_text(), &node_def)) {
    execution_context().Fail(absl::InternalError(
        absl::StrCat("CreateOp: failed to parse NodeDef: ", node_def_text())));
    return;
  }

  auto runner = tfrt_stub::OpKernelRunner::Create(
                    node_def.op(), node_def.name(), node_def.device(),
                    node_def.input().size(),
                    [&](tensorflow::AttrValueMap* attr_value_map) {
                      *attr_value_map = node_def.attr();
                      return OkStatus();
                    },
                    fallback_request_state.device_manager(),
                    fallback_request_state.process_function_library_runtime())
                    .value();

  if (!fallback_request_state.runner_table()->Insert(op_key(),
                                                     std::move(runner))) {
    execution_context().Fail(absl::InternalError(absl::StrCat(
        "CreateOp: OpKernelRunner already exists: ", node_def.op())));
  }
}

template <bool IsAsync, typename Frame>
void ExecuteOpInternal(Frame& frame) {
  int32_t op_key = frame.op_key();

  auto& context = frame.context();
  const auto& fallback_request_state = context.fallback_request_state();

  // Start recording the op execution time, given a non-null cost recorder.
  auto* cost_recorder = fallback_request_state.cost_recorder();
  uint64_t run_start_time = 0;
  if (cost_recorder != nullptr) run_start_time = tfrt::GetCpuClockCycle();

  auto* kernel_runner =
      fallback_request_state.runner_table()->GetUnsafe(op_key);
  DCHECK(kernel_runner);

  ExecuteKernelRunner<IsAsync>(frame, context, fallback_request_state,
                               *kernel_runner);

  // Finish recording the op execution time, given a non-null
  // cost recorder.
  //
  // TODO(b/259602527): Measure async op costs more accurately with whole
  // execution time. (It's not urgent because async ops are rare.)
  if (cost_recorder != nullptr) {
    const uint64_t run_finish_time = tfrt::GetCpuClockCycle();
    cost_recorder->RecordCost(op_key, run_finish_time - run_start_time);
  }
}

struct ExecuteOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  // TODO(chky, deqiangc): Consider changing "executeop" to "execute_op" so that
  // the naming convention is consistent with other kernels.
  static constexpr char kName[] = "tf_mlrt.executeop";
  static constexpr bool kUseCustomDevice = false;

  mlrt::RegisterValueSpan<tfrt_stub::FallbackTensor> args() const {
    return arguments();
  }

  absl::string_view node_def_text() const {
    return attributes().GetAs<mlrt::bc::String>(0).Get();
  }

  int32_t op_key() const { return attributes().GetAs<int32_t>(1); }

  Context& context() const {
    return execution_context().GetUserContext<Context>();
  }

  tensorflow::Device* device() const {
    return context().fallback_request_state().cpu_device();
  }

  void Invoke() { ExecuteOpInternal</*IsAsync=*/false>(*this); }
};

struct AsyncExecuteOp : ExecuteOp {
  using ExecuteOp::ExecuteOp;

  static constexpr char kName[] = "tf_mlrt.async_executeop";

  void Invoke() {
    static_assert(!AsyncExecuteOp::kUseCustomDevice);
    if (execution_context().GetUserContext<Context>().IsCancelled()) {
      execution_context().FailOnCancellation();
      return;
    }

    ExecuteOpInternal</*IsAsync=*/true>(*this);
  }
};

struct ExecuteOpDevice : ExecuteOp {
  using Base = ExecuteOp;
  using Base::Base;

  static constexpr char kName[] = "tf_mlrt.executeop.device";
  static constexpr bool kUseCustomDevice = true;

  mlrt::RegisterValueSpan<tfrt_stub::FallbackTensor> args() const {
    return arguments().drop_front();
  }

  mlrt::bc::Span<uint8_t> last_uses() const {
    return Base::last_uses().drop_front();
  }

  tensorflow::Device* device() const {
    return arguments()[0].Get<std::unique_ptr<tensorflow::Device>>().get();
  }

  void Invoke() { ExecuteOpInternal</*IsAsync=*/false>(*this); }
};

struct AsyncExecuteOpDevice : ExecuteOpDevice {
  using ExecuteOpDevice::ExecuteOpDevice;

  static constexpr char kName[] = "tf_mlrt.async_executeop.device";

  void Invoke() {
    static_assert(AsyncExecuteOpDevice::kUseCustomDevice);
    if (execution_context().GetUserContext<Context>().IsCancelled()) {
      execution_context().FailOnCancellation();
      return;
    }

    ExecuteOpInternal</*IsAsync=*/true>(*this);
  }
};

void SetResource(mlrt::KernelFrame frame) {
  auto& resource_tensor = frame.arguments()[0].Get<tfrt_stub::FallbackTensor>();
  int64_t index = frame.attributes().GetAs<int64_t>(0);
  auto& context = frame.execution_context().GetUserContext<Context>();
  const auto& fallback_request_state = context.fallback_request_state();

  auto* resource_array = fallback_request_state.resource_array();
  if (!resource_array) {
    frame.execution_context().Fail(
        absl::InternalError("Fallback resource_array is null"));
    return;
  }

  resource_array->SetResource(
      index,
      tensorflow::tfrt_stub::ImmutableTensor::Create(resource_tensor.tensor()));
}

void GetResource(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.get_resource");
  auto& context = frame.execution_context().GetUserContext<Context>();
  const auto& fallback_request_state = context.fallback_request_state();

  auto* resource_array = fallback_request_state.resource_array();
  if (!resource_array) {
    frame.execution_context().Fail(
        absl::InternalError("Fallback resource_array is null"));
    return;
  }

  mlrt::bc::Vector<int64_t> indices(frame.attributes()[0].data());

  auto results = frame.results();

  for (int i = 0; i < indices.size(); ++i) {
    results[i].Emplace<tensorflow::tfrt_stub::FallbackTensor>(
        resource_array->GetResourceAsFallbackTensor(indices[i]));
  }
}

void TensorToInt32(mlrt::KernelFrame frame) {
  const auto& tensor = frame.arguments()[0]
                           .Get<tensorflow::tfrt_stub::FallbackTensor>()
                           .tensor();
  if (TensorShapeUtils::IsScalar(tensor.shape()) &&
      tensor.dtype() == DT_INT32) {
    frame.results()[0].Set(tensor.scalar<int32_t>()());
  } else {
    frame.execution_context().Fail(absl::InvalidArgumentError(absl::StrCat(
        DataTypeString(tensor.dtype()), " cannot be converted to a int32")));
  }
}

absl::StatusOr<bool> PredicateInternal(const tensorflow::Tensor& tensor) {
  if (TensorShapeUtils::IsScalar(tensor.shape())) {
    switch (tensor.dtype()) {
#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    return tensor.scalar<T>()() != 0;

      CASE(float);
      CASE(double);
      CASE(uint8_t);
      CASE(int8_t);
      CASE(int16_t);
      CASE(int32_t);
      CASE(int64_t);
      CASE(bool);
#undef CASE
      case DT_STRING:
        return !tensor.scalar<tstring>()().empty();
      default:
        return absl::InvalidArgumentError(
            absl::StrCat(DataTypeString(tensor.dtype()),
                         " cannot be converted to a boolean"));
    }
  }

  return tensor.NumElements() > 0;
}

void Predicate(mlrt::KernelFrame frame) {
  const auto& tensor = frame.arguments()[0]
                           .Get<tensorflow::tfrt_stub::FallbackTensor>()
                           .tensor();
  auto result = PredicateInternal(tensor);
  if (ABSL_PREDICT_FALSE(!result.ok())) {
    frame.execution_context().Fail(result.status());
    return;
  }

  frame.results()[0].Set(*result);
}

void AllocateTensorFutures(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.allocate_futures");
  uint32_t num = frame.attributes().GetAs<uint32_t>(0);

  DCHECK_EQ(num * 2, frame.results().size());
  for (int i = 0; i < num; ++i) {
    auto promise =
        mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();
    frame.results()[num + i].Set<mlrt::Future>(promise.GetFuture());
    frame.results()[i].Set<mlrt::Promise>(std::move(promise));
  }
}

void AwaitTensor(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.await");
  auto& future = frame.arguments()[0].Get<mlrt::Future>();
  if (frame.last_uses()[0]) {
    frame.execution_context().Await<tensorflow::tfrt_stub::FallbackTensor>(
        std::move(future), &frame.results()[0]);
    frame.arguments()[0].Destroy<mlrt::Future>();
  } else {
    frame.execution_context().Await<tensorflow::tfrt_stub::FallbackTensor>(
        future, &frame.results()[0]);
  }
}

void AwaitAllTensor(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.await_all");
  mlrt::RegisterValueSpan<mlrt::Future> futures(frame.arguments());
  frame.execution_context().AwaitAll<tensorflow::tfrt_stub::FallbackTensor>(
      futures, frame.results());

  DCHECK_EQ(frame.last_uses().size(), futures.size());
  auto last_use_iter = frame.last_uses().begin();

  for (int i = 0; i < futures.size(); ++i) {
    if (*last_use_iter++) {
      futures.Destroy(i);
    }
  }
}

void PromiseTensor(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.promise");
  auto& promise = frame.arguments()[0].Get<mlrt::Promise>();
  auto& tensor =
      frame.arguments()[1].Get<tensorflow::tfrt_stub::FallbackTensor>();
  if (frame.last_uses()[1]) {
    std::move(promise).Set<tensorflow::tfrt_stub::FallbackTensor>(
        std::move(tensor));
  } else {
    std::move(promise).Set<tensorflow::tfrt_stub::FallbackTensor>(tensor);
  }

  frame.arguments()[0].Destroy<mlrt::Promise>();
}

void PromiseFuture(mlrt::KernelFrame frame) {
  tsl::profiler::TraceMe trace_me("tf_mlrt.promise_future");
  auto& promise = frame.arguments()[0].Get<mlrt::Promise>();
  auto incoming_future = frame.arguments()[1].Get<mlrt::Future>();
  std::move(incoming_future)
      .Then([promise = std::move(promise)](
                absl::StatusOr<tensorflow::tfrt_stub::FallbackTensor>
                    value) mutable {
        if (value.ok()) {
          std::move(promise).Set<tensorflow::tfrt_stub::FallbackTensor>(
              *std::move(value));
        } else {
          std::move(promise).SetError(std::move(value).status());
        }
      });
}

struct PromiseReturnOp : mlrt::PromiseReturnOpBase<PromiseReturnOp> {
  using PromiseReturnOpBase::PromiseReturnOpBase;

  static constexpr char kName[] = "tf_mlrt.promise_return";

  mlrt::Promise& promise() const { return arguments()[0].Get<mlrt::Promise>(); }

  tensorflow::tfrt_stub::FallbackTensor& value() const {
    return arguments()[1].Get<tensorflow::tfrt_stub::FallbackTensor>();
  }

  bool value_last_use() const { return last_uses()[1]; }
};

}  // namespace

mlrt::KernelRegistry& GetTfMlrtOptionalKernelRegistry() {
  static auto* const registry = new mlrt::KernelRegistry;
  return *registry;
}

void RegisterTfMlrtKernels(mlrt::KernelRegistry& registry) {
  mlrt::RegisterBuiltinKernels(registry);
  // TODO(chky,rohitju): These kernels should be unified with the corresponding
  // tfrt_fallback_sync kernels, e.g. tfrt_fallback_sync.executeop.
  registry.Register<CancelOp>();
  registry.Register<CreateOp>();
  registry.Register<CreateOp>("tfrt_fallback_sync.createop");
  registry.Register<ExecuteOp>();
  registry.Register<ExecuteOp>("tfrt_fallback_sync.executeop");
  registry.Register<AsyncExecuteOp>();
  registry.Register<ExecuteOpDevice>();
  registry.Register<AsyncExecuteOpDevice>();
  registry.Register("tf_mlrt.set_resource", &SetResource);
  registry.Register("tfrt_fallback_sync.set_resource", &SetResource);
  registry.Register("tf_mlrt.get_resource", &GetResource);
  registry.Register("tfrt_fallback_sync.get_resource", &GetResource);
  registry.Register("tf_mlrt.predicate", &Predicate);
  registry.Register("tf_mlrt.tensor_to_int32", &TensorToInt32);
  registry.Register("tf_mlrt.allocate_futures", &AllocateTensorFutures);
  registry.Register("tf_mlrt.await", &AwaitTensor);
  registry.Register("tf_mlrt.await_all", &AwaitAllTensor);
  registry.Register<MapFnOp>();
  registry.Register("tf_mlrt.promise", &PromiseTensor);
  registry.Register("tf_mlrt.promise_future", &PromiseFuture);
  registry.Register<PromiseReturnOp>();

  registry.Merge(GetTfMlrtOptionalKernelRegistry());
}

}  // namespace tf_mlrt
}  // namespace tensorflow
