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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/op_logger.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/device_with_custom_allocator.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tsl/platform/errors.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
const char kOpKernelRunnerCacheResourceName[] =
    "OpKernelRunnerCacheResourceName";

namespace {

using ::tensorflow::tfrt_stub::OpKernelRunner;
using ::tensorflow::tfrt_stub::OpKernelRunState;
using ::tfrt::AsyncValue;
using ::tfrt::AsyncValueRef;
using ::tfrt::Chain;
using ::tfrt::RCReference;
using ::tfrt::string_view;

void KernelFallbackEmitError(
    const tfrt::ExecutionContext& exec_ctx,
    const KernelFallbackCompatRequestState* fallback_request_state,
    tfrt::string_view op_name, tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const tensorflow::Status& status) {
  // Set all results to error, with the correct TFRT error code according to the
  // error propagated from runtime fallback execution.
  auto model_info =
      fallback_request_state == nullptr
          ? "(missing model info) "
          : tfrt::StrCat(
                fallback_request_state->session_metadata().name(), " (",
                fallback_request_state->session_metadata().version(), ") ");
  auto error = EmitErrorAsync(
      exec_ctx,
      absl::Status(
          status.code(),
          tfrt::StrCat(model_info, "error running kernel fallback kernel ",
                       op_name, ": ", status.message())));
  std::fill(results.begin(), results.end(), error);
  if (op_chain) *op_chain = std::move(error);
}

}  // namespace

static llvm::Expected<gtl::InlinedVector<tensorflow::Tensor, 4>>
ConvertInputTensors(llvm::ArrayRef<tfrt::Tensor*> arguments) {
  gtl::InlinedVector<tensorflow::Tensor, 4> input_tf_tensors;
  input_tf_tensors.reserve(arguments.size());
  for (tfrt::Tensor* argument : arguments) {
    auto expected_tf_tensor = tfrt::TFRTTensorToTFTensor(*argument);
    if (!expected_tf_tensor) {
      return tfrt::MakeStringError(
          tfrt::StrCat(expected_tf_tensor.takeError()));
    }
    input_tf_tensors.push_back(std::move(expected_tf_tensor.get()));
  }

  return input_tf_tensors;
}

static Status ValidateInputTypes(
    tfrt::string_view op_name,
    const gtl::InlinedVector<tensorflow::Tensor, 4>& input_tf_tensors,
    const DataTypeVector& input_types) {
  const size_t n_inputs = input_tf_tensors.size();

  if (input_types.size() != n_inputs) {
    return tensorflow::errors::InvalidArgument("expected ", input_types.size(),
                                               " inputs, got ", n_inputs);
  }

  for (size_t i = 0; i < n_inputs; ++i) {
    if (input_tf_tensors[i].dtype() != input_types[i]) {
      return tensorflow::errors::InvalidArgument(
          "cannot compute ", op_name.str(), " as input #", i, "(zero-based)",
          " was expected to be a ", DataTypeString(input_types[i]),
          " tensor but is a ", DataTypeString(input_tf_tensors[i].dtype()),
          " tensor");
    }
  }

  return absl::OkStatus();
}

namespace {

// Keep states needed by kernel execution in a thread local storage to avoid
// repeated reallocation and destruction of them.
OpKernelRunState& GetThreadLocalOpKernelRunState() {
  thread_local OpKernelRunState run_state;
  return run_state;
}

}  // namespace

// Execute a tensorflow::OpKernel Asynchronously. `kernel_runner` and
// `input_tf_tensors` are expected to be alive during the call to this function.
// Set result AsyncValues in `results` and return a Chain that indicates the
// execution completion of error otherwise.
template <typename TensorType>
static void KernelFallbackExecuteCompatAsyncInternal(
    const tfrt::ExecutionContext& exec_ctx, OpKernelRunState* run_state,
    const OpKernelRunner& kernel_runner,
    tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results) {
  auto chain = tfrt::MakeUnconstructedAsyncValueRef<tfrt::Chain>();
  if (op_chain) *op_chain = chain.CopyRef();

  // Allocate unconstructed result tensors and set them in the output `results`.
  llvm::SmallVector<AsyncValueRef<TensorType>, 4> result_refs;
  result_refs.reserve(results.size());
  for (auto& result : results) {
    result_refs.emplace_back(
        tfrt::MakeUnconstructedAsyncValueRef<TensorType>());
    result = result_refs.back().CopyRef();
  }

  struct AsyncState {
    explicit AsyncState(const OpKernelRunState& rs, int num_outputs)
        : run_state(rs.input_tf_tensor_values, rs.params),
          context(&run_state.params, num_outputs) {}

    OpKernelRunState run_state;
    OpKernelContext context;

    tfrt::AsyncValueRef<tfrt::Chain> chain;
    llvm::SmallVector<tfrt::AsyncValueRef<TensorType>, 4> result_refs;
  };

  DCHECK_EQ(results.size(), kernel_runner.op_kernel()->num_outputs());
  auto async_state = std::make_shared<AsyncState>(*run_state, results.size());
  async_state->chain = chain.CopyRef();
  async_state->result_refs = std::move(result_refs);

  auto* context_ptr = &async_state->context;

  auto done_callback = [async_state = std::move(async_state), exec_ctx]() {
    auto& context = async_state->context;

    if (!context.status().ok()) {
      auto diag = tfrt::EmitError(
          exec_ctx,
          absl::Status(context.status().code(),
                       tfrt::StrCat("error running kernel fallback kernel ",
                                    context.op_kernel().name(), ": ",
                                    context.status().message())));
      for (auto& result : async_state->result_refs)
        result.SetError(diag.status);
      async_state->chain.SetError(diag.status);
      return;
    }

    // Set payload and mark async values available in TFRT's thread.
    tfrt::EnqueueWork(exec_ctx, [async_state = std::move(async_state)]() {
      auto& context = async_state->context;
      for (int i = 0; i < context.num_outputs(); ++i) {
        async_state->result_refs[i].emplace(
            std::move(*context.mutable_output(i)));
      }
      async_state->chain.emplace();
    });
  };

  kernel_runner.RunAsync(context_ptr, std::move(done_callback));
}

// Execute a tensorflow::OpKernel synchronously. `kernel_runner` and
// `input_tf_tensors` are expected to be alive during the call to this function.
// Set result AsyncValues in `results` and return OK status on successfully
// finishing the execution. TensorType is expected to be convert-constructible
// from tensorflow::Tensor.
template <typename TensorType>
static void KernelFallbackExecuteCompatSyncInternal(
    const tfrt::ExecutionContext& exec_ctx,
    const KernelFallbackCompatRequestState* fallback_request_state,
    OpKernelRunState* run_state, const OpKernelRunner& kernel_runner,
    tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results) {
  DCHECK_EQ(results.size(), kernel_runner.op_kernel()->num_outputs());
  OpKernelContext context(&run_state->params, results.size());
  kernel_runner.Run(&context);

  if (!context.status().ok()) {
    KernelFallbackEmitError(exec_ctx, fallback_request_state,
                            kernel_runner.op_kernel()->name(), op_chain,
                            results, context.status());
    return;
  }

  for (int i = 0; i < context.num_outputs(); ++i) {
    results[i] = tfrt::MakeAvailableAsyncValueRef<TensorType>(
        std::move(*context.mutable_output(i)));
  }

  if (op_chain) *op_chain = tfrt::MakeAvailableAsyncValueRef<tfrt::Chain>();
}

tfrt::AsyncValueRef<tfrt::Chain> KernelFallbackExecuteCompatCoreRuntimeDispatch(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, llvm::ArrayRef<tfrt::Tensor*> arguments,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const OpKernelRunner& op_kernel_runner) {
  auto op_chain = tfrt::GetReadyChain();
  tensorflow::Status status;

  auto expected_input_tf_tensors = ConvertInputTensors(arguments);
  if (!expected_input_tf_tensors) {
    status = tensorflow::errors::Internal(
        tfrt::StrCat(expected_input_tf_tensors.takeError()));
    KernelFallbackEmitError(exec_ctx, &fallback_request_state, op_name,
                            &op_chain, results, status);
    return op_chain;
  }

  auto& run_state = GetThreadLocalOpKernelRunState();
  auto clean_up_inputs =
      gtl::MakeCleanup([&]() { run_state.input_tf_tensors.clear(); });

  auto& input_tf_tensors = run_state.input_tf_tensors;
  input_tf_tensors = std::move(expected_input_tf_tensors.get());

  // Check if input tensor dtypes are valid.
  status = ValidateInputTypes(op_name, input_tf_tensors,
                              op_kernel_runner.op_kernel()->input_types());

  // TODO(b/176997538): Skip checking dtypes for tf._BatchFunctionFallback op
  // due to b/176997538. Remove the skipping once the SavedModel lowering
  // problem is fixed.
  if (!status.ok() && !op_name.equals("_BatchFunctionFallback")) {
    KernelFallbackEmitError(exec_ctx, &fallback_request_state, op_name,
                            &op_chain, results, status);
    return op_chain;
  }

  auto& input_tf_tensor_values = run_state.input_tf_tensor_values;
  input_tf_tensor_values.resize(input_tf_tensors.size());
  for (int i = 0; i < input_tf_tensors.size(); ++i) {
    input_tf_tensor_values[i].tensor = &input_tf_tensors[i];
  }

  auto* device =
      GetDeviceFromFallbackState(fallback_request_state, op_kernel_runner);

  SetUpParams(op_kernel_runner, fallback_request_state, device, run_state);

  if (op_kernel_runner.IsAsync()) {
    KernelFallbackExecuteCompatAsyncInternal<KernelFallbackTensor>(
        exec_ctx, &run_state, op_kernel_runner, &op_chain, results);
  } else {
    KernelFallbackExecuteCompatSyncInternal<KernelFallbackTensor>(
        exec_ctx, &fallback_request_state, &run_state, op_kernel_runner,
        &op_chain, results);
  }

  return op_chain;
}

static absl::string_view StripTfPrefix(tfrt::string_view op_name) {
  return absl::StripPrefix(ToAbslStringView(op_name), "tf.");
}

// Generate metadata for an execution op event
std::string GetTracingMetadata(llvm::ArrayRef<tfrt::AsyncValue*> args,
                               const tfrt::ExecutionContext& exec_ctx,
                               const OpKernelRunner& kernel_runner) {
  auto request_id = exec_ctx.request_ctx()->id();
  // Get Long Name
  auto debug_info = exec_ctx.location().GetDebugInfo();
  auto long_name = debug_info.has_value() ? debug_info.value().info : "";

  if (!tsl::profiler::TfOpDetailsEnabled()) {
    return tsl::profiler::TraceMeEncode(
        {{"id", request_id}, {"long_name", ToAbslStringView(long_name)}});
  }

  // Get Input Tensors
  std::string input_string;
  llvm::raw_string_ostream input_string_stream(input_string);

  for (size_t i = 0; i < args.size(); ++i) {
    const auto& tensor = args[i]->get<Tensor>();
    input_string_stream << DataTypeString(tensor.dtype())
                        << tensor.shape().DebugString();
    input_string_stream << ";";
  }

  // Get Attributes
  std::string attr_string;
  llvm::raw_string_ostream attr_string_stream(attr_string);

  for (const auto& entry : kernel_runner.op_kernel()->def().attr()) {
    attr_string_stream << entry.first << ": {" << entry.second.DebugString();
    if (!attr_string.empty() && attr_string[attr_string.size() - 1] == '\n') {
      attr_string[attr_string.size() - 1] = '}';
    }
    attr_string_stream << ";\n";
  }

  return tsl::profiler::TraceMeEncode({
      {"id", request_id},
      {"long_name", ToAbslStringView(long_name)},
      {"inputs", input_string},
      {"attributes", attr_string},
  });
}

namespace {

class FallbackKernelAttributeFrame {
 public:
  explicit FallbackKernelAttributeFrame(tfrt::AsyncKernelFrame* frame)
      : frame_(frame) {
    DCHECK(frame_);
  }

  tfrt::StringAttr device() const {
    return tfrt::StringAttr(frame_->GetAttribute(kDeviceAttrPosition));
  }

  tfrt::AggregateAttr op_attr() const {
    return tfrt::AggregateAttr(frame_->GetAttribute(kOpAttrPosition));
  }

  tfrt::AggregateAttr op_func_attr() const {
    return tfrt::AggregateAttr(frame_->GetAttribute(kOpFuncAttrPosition));
  }

  tfrt::I64Attr op_key() const {
    return tfrt::I64Attr(frame_->GetAttribute(kOpKeyAttrPosition));
  }

  tfrt::StringAttr op_name() const {
    return tfrt::StringAttr(frame_->GetAttribute(kOpNameAttrPosition));
  }

 private:
  static constexpr int kDeviceAttrPosition = 0;
  static constexpr int kOpAttrPosition = 1;
  static constexpr int kOpFuncAttrPosition = 2;
  static constexpr int kOpKeyAttrPosition = 3;
  static constexpr int kOpNameAttrPosition = 4;

  tfrt::AsyncKernelFrame* frame_ = nullptr;
};

// The BEF kernel for kernel fallback compat mode. The arguments and results are
// expected to tensorflow::tfrt_stub::FallbackTensor.
TF_ATTRIBUTE_ALWAYS_INLINE static void KernelFallbackExecuteOpInternal(
    llvm::ArrayRef<tfrt::AsyncValue*> args,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    const FallbackKernelAttributeFrame& frame,
    const tfrt::ExecutionContext& exec_ctx,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const OpKernelRunner& kernel_runner, bool is_async,
    tensorflow::Device* device) {
  tsl::profiler::TraceMe trace_me([&]() -> std::string {
    if (kernel_runner.op_kernel()) {
      return tsl::profiler::TraceMeOp(
          kernel_runner.op_kernel()->name_view(),
          kernel_runner.op_kernel()->type_string_view());
    }
    return std::string(ToAbslStringView(frame.op_name().GetValue()));
  });

  trace_me.AppendMetadata(
      [&]() { return GetTracingMetadata(args, exec_ctx, kernel_runner); });

  if (fallback_request_state.log_device_placement() || VLOG_IS_ON(1)) {
    string msg =
        strings::StrCat("Executing op ", frame.op_name().GetValue().str(),
                        " in device ", frame.device().GetValue().str());
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }

  auto& run_state = GetThreadLocalOpKernelRunState();
  auto clean_up_inputs =
      gtl::MakeCleanup([&]() { run_state.input_tf_tensors.clear(); });

  // Prepare the input tensors.
  auto& input_tf_tensors = run_state.input_tf_tensors;
  auto& input_tf_tensor_values = run_state.input_tf_tensor_values;
  DCHECK(input_tf_tensors.empty());
  input_tf_tensor_values.resize(args.size());
  for (int i = 0; i < args.size(); ++i) {
    auto* arg = args[i];
    auto& fallback_tensor = arg->get<tensorflow::tfrt_stub::FallbackTensor>();
    // If the argument is immutable or unique, we can just keep the reference
    // without copying that invovles expensive atomic reference counting. And if
    // the argument is unique but mutable, then tensorflow optimizations like
    // buffer forwarding can be utilized. Otherwise, we conservatively copy the
    // tensor.
    if (!fallback_tensor.is_immutable() && !arg->IsUnique()) {
      input_tf_tensors.push_back(fallback_tensor.tensor());
    }
    input_tf_tensor_values[i].tensor = &fallback_tensor.tensor();
  }

  SetUpParams(kernel_runner, fallback_request_state, device, run_state);

  if (is_async) {
    KernelFallbackExecuteCompatAsyncInternal<
        tensorflow::tfrt_stub::FallbackTensor>(
        exec_ctx, &run_state, kernel_runner, op_chain, results);
  } else {
    KernelFallbackExecuteCompatSyncInternal<
        tensorflow::tfrt_stub::FallbackTensor>(
        exec_ctx, &fallback_request_state, &run_state, kernel_runner, op_chain,
        results);
  }
}

TF_ATTRIBUTE_ALWAYS_INLINE static void KernelFallbackExecuteOp(
    llvm::ArrayRef<tfrt::AsyncValue*> args,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    const FallbackKernelAttributeFrame& frame,
    const tfrt::ExecutionContext& exec_ctx) {
  const auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    KernelFallbackEmitError(
        exec_ctx, /*fallback_request_state=*/nullptr,
        frame.op_name().GetValue(), op_chain, results,
        tensorflow::errors::NotFound(
            "KernelFallbackCompatRequestState not found in RequestContext."));
    return;
  }

  // Start recording the op execution time, given a non-null cost recorder.
  auto* cost_recorder = fallback_request_state->cost_recorder();
  uint64_t run_start_time = 0;
  tfrt::AsyncValueRef<tfrt::Chain> cost_chain;
  if (cost_recorder != nullptr) {
    run_start_time = tfrt::GetCpuClockCycle();
    if (op_chain == nullptr) op_chain = &cost_chain;
  }

  auto* runner_table = fallback_request_state->runner_table();
  DCHECK(runner_table);

  auto* kernel_runner = runner_table->Get(frame.op_key().GetValue());
  DCHECK(kernel_runner);
  DCHECK_EQ(kernel_runner->op_kernel()->name(),
            StripTfPrefix(frame.op_name().GetValue()));

  auto* device =
      GetDeviceFromFallbackState(*fallback_request_state, *kernel_runner);

  KernelFallbackExecuteOpInternal(args, results, op_chain, frame, exec_ctx,
                                  *fallback_request_state, *kernel_runner,
                                  kernel_runner->IsAsync(), device);

  // Finish recording the op execution time, given a non-null
  // cost recorder.
  //
  // TODO(b/259602527): Measure async op costs more accurately with whole
  // execution time. (It's not urgent because async ops are rare.)
  if (cost_recorder != nullptr) {
    op_chain->AndThen(
        [cost_recorder, run_start_time, op_key = frame.op_key().GetValue()] {
          const uint64_t run_finish_time = tfrt::GetCpuClockCycle();
          cost_recorder->RecordCost(op_key, run_finish_time - run_start_time);
        });
  }
}

// The BEF kernel for creating tensorflow::OpKernel to be used in kernel
// fallback compat mode.
tfrt::AsyncValueRef<tfrt::Chain> KernelFallbackCreateOp(
    const tfrt::Chain& in_ch, tfrt::StringAttr device, tfrt::I64Attr num_args,
    tfrt::AggregateAttr op_attr_array, tfrt::AggregateAttr op_func_attr_array,
    tfrt::I64Attr op_key, tfrt::StringAttr op_name_attr,
    const tfrt::ExecutionContext& exec_ctx) {
  const auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    return tfrt::EmitErrorAsync(
        exec_ctx,
        "KernelFallbackCompatRequestState not found in RequestContext.");
  }

  auto* runner_table = fallback_request_state->runner_table();
  DCHECK(runner_table);

  auto attr_builder = [op_attr_array, op_func_attr_array](
                          tensorflow::AttrValueMap* attr_value_map) {
    return SetUpAttrValueMap(op_attr_array, op_func_attr_array, attr_value_map);
  };

  auto op_name = StripTfPrefix(op_name_attr.GetValue());

  auto statusor_runner = OpKernelRunner::Create(
      op_name, ToAbslStringView(device.GetValue()), num_args.GetValue(),
      attr_builder, fallback_request_state->device_manager(),
      fallback_request_state->process_function_library_runtime());
  if (!statusor_runner.ok())
    return tfrt::EmitErrorAsync(exec_ctx, statusor_runner.status());

  if (!runner_table->Insert(op_key.GetValue(),
                            std::move(statusor_runner).value())) {
    return tfrt::EmitErrorAsync(
        exec_ctx,
        absl::StrCat("KernelFallbackCreateOp: OpKernelRunner already exists: ",
                     op_name_attr.GetValue().str()));
  }

  return tfrt::MakeAvailableAsyncValueRef<tfrt::Chain>();
}

// FallbackSetResource is the fallback kernel that sets the tensor value in the
// fallback's resource array.
llvm::Expected<tfrt::Chain> FallbackSetResource(
    tfrt::Argument<tfrt::Chain> in_ch,
    tfrt::Argument<tensorflow::tfrt_stub::FallbackTensor> arg,
    tfrt::StringAttr device, tfrt::I64Attr index_attr,
    const tfrt::ExecutionContext& exec_ctx) {
  const auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    return tfrt::MakeStringError(
        "KernelFallbackCompatRequestState not found in RequestContext.");
  }

  auto* resource_array = fallback_request_state->resource_array();
  DCHECK(resource_array);

  int64_t index = index_attr.GetValue();

  // Setting the resource tensor to be immutable, so that we don't need
  // reference counting on it and that it cannot be buffer-forwarded.
  resource_array->SetResource(
      index,
      tensorflow::tfrt_stub::ImmutableTensor::Create(arg.get().tensor()));

  return tfrt::Chain();
}

// FallbackGetResource is the fallback kernel that retrieves the tensor value in
// the fallback's resource array.
void FallbackGetResource(tfrt::Argument<tfrt::Chain> in_ch,
                         tfrt::Result<tfrt::Chain> out_ch,
                         tfrt::RemainingResults results,
                         tfrt::StringAttr device, tfrt::ArrayAttr indices_attr,
                         const tfrt::ExecutionContext& exec_ctx) {
  tsl::profiler::TraceMe trace_me("tfrt_fallback_async.get_resource");
  trace_me.AppendMetadata([request_id = exec_ctx.request_ctx()->id()]() {
    return tsl::profiler::TraceMeEncode({{"id", request_id}});
  });

  const auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    tfrt::RCReference<tfrt::AsyncValue> error = tfrt::EmitErrorAsync(
        exec_ctx,
        "KernelFallbackCompatRequestState not found in RequestContext.");
    out_ch.Set(std::move(error));
    return;
  }

  auto* resource_array = fallback_request_state->resource_array();
  DCHECK(resource_array);

  llvm::ArrayRef<int64_t> indices = indices_attr.GetValue<int64_t>();

  for (int i = 0; i < indices.size(); ++i) {
    results[i] = resource_array->GetResource(indices[i]).CopyRef();
  }

  out_ch.Set(in_ch);
}

// The implementation of tfrt_fallback_async.executeop kernel. It executes a
// non-side-effecting TF op with the name of `op_name` in fallback. All relevant
// TF attributes are passed in `op_attr_array`.
void FallbackAsyncExecuteOp(tfrt::AsyncKernelFrame* frame) {
  FallbackKernelAttributeFrame attr_frame(frame);
#ifndef NDEBUG
  frame->GetExecutionContext()
      .host()
      ->GetOrCreateSharedContext<OpLogger>()
      .LogOp(attr_frame.op_name().GetValue());
#endif
  KernelFallbackExecuteOp(frame->GetArguments(), frame->GetResults(),
                          /*op_chain=*/nullptr, attr_frame,
                          frame->GetExecutionContext());
}

// The implementation of tfrt_fallback_async.executeop.seq kernel. It executes a
// side-effecting TF op with the name of `op_name` in fallback. All relevant
// TF attributes are passed in `op_attr_array`. `in_op_chain` and `out_op_chain`
// are used for side-effect visibility.
void FallbackAsyncExecuteOpSeq(tfrt::AsyncKernelFrame* frame) {
  auto all_args = frame->GetArguments();
  DCHECK_GT(all_args.size(), 0);
  tfrt::AsyncValueRef<tfrt::Chain> op_chain(tfrt::FormRef(all_args[0]));
  llvm::ArrayRef<tfrt::AsyncValue*> args = all_args.drop_front();

  auto all_results = frame->GetResults();
  DCHECK_GT(all_results.size(), 0);
  auto& out_op_chain = all_results[0];
  llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results =
      all_results.drop_front();

  KernelFallbackExecuteOp(args, results, &op_chain,
                          FallbackKernelAttributeFrame(frame),
                          frame->GetExecutionContext());
  out_op_chain = std::move(op_chain);
}

void KernelFallbackExecuteOpCustomAllocatorInternal(
    llvm::ArrayRef<tfrt::AsyncValue*> args,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    tfrt::AsyncValueRef<tfrt::Chain>* op_chain,
    const tfrt::ExecutionContext& exec_ctx,
    const FallbackKernelAttributeFrame& attr_frame) {
  const auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    KernelFallbackEmitError(
        exec_ctx, /*fallback_request_state=*/nullptr,
        attr_frame.op_name().GetValue(), op_chain, results,
        tensorflow::errors::NotFound(
            "KernelFallbackCompatRequestState not found in RequestContext."));
    return;
  }

  auto* runner_table = fallback_request_state->runner_table();
  DCHECK(runner_table);

  auto* kernel_runner = runner_table->Get(attr_frame.op_key().GetValue());
  DCHECK(kernel_runner);
  DCHECK_EQ(kernel_runner->op_kernel()->name(),
            StripTfPrefix(attr_frame.op_name().GetValue()));

  DCHECK_GT(args.size(), 0);
  auto* allocator = args.front()->get<tensorflow::Allocator*>();
  args = args.drop_front();

  auto* device =
      GetDeviceFromFallbackState(*fallback_request_state, *kernel_runner);

  if (!kernel_runner->IsAsync()) {
    tfrt_stub::DeviceWithCustomAllocator device_with_custom_allocator(
        device, allocator);

    KernelFallbackExecuteOpInternal(args, results,
                                    /*op_chain=*/op_chain, attr_frame, exec_ctx,
                                    *fallback_request_state, *kernel_runner,
                                    /*is_async=*/false,
                                    &device_with_custom_allocator);
  } else {
    auto device_with_custom_allocator =
        std::make_unique<tfrt_stub::DeviceWithCustomAllocator>(device,
                                                               allocator);

    tfrt::AsyncValueRef<tfrt::Chain> op_ch;
    if (op_chain == nullptr) {
      op_chain = &op_ch;
    }

    KernelFallbackExecuteOpInternal(args, results,
                                    /*op_chain=*/op_chain, attr_frame, exec_ctx,
                                    *fallback_request_state, *kernel_runner,
                                    /*is_async=*/true,
                                    device_with_custom_allocator.get());

    DCHECK(op_chain);
    op_chain->AndThen([d = std::move(device_with_custom_allocator)]() {});
  }
}

void FallbackAsyncExecuteOpWithAllocator(tfrt::AsyncKernelFrame* frame) {
  auto args = frame->GetArguments();
  auto results = frame->GetResults();
  FallbackKernelAttributeFrame attr_frame(frame);
  KernelFallbackExecuteOpCustomAllocatorInternal(
      args, results, /*op_chain=*/nullptr, frame->GetExecutionContext(),
      attr_frame);
}

void FallbackAsyncExecuteOpSeqWithAllocator(tfrt::AsyncKernelFrame* frame) {
  auto args = frame->GetArguments();
  DCHECK_GT(args.size(), 0);
  tfrt::AsyncValueRef<tfrt::Chain> op_chain(tfrt::FormRef(args.front()));
  args = args.drop_front();

  auto results = frame->GetResults();
  DCHECK_GT(results.size(), 0);
  auto& out_op_chain = results.front();
  results = results.drop_front();

  FallbackKernelAttributeFrame attr_frame(frame);
  KernelFallbackExecuteOpCustomAllocatorInternal(
      args, results, &op_chain, frame->GetExecutionContext(), attr_frame);

  out_op_chain = std::move(op_chain);
}

void FallbackCopyTensorIfSmall(
    tfrt::Argument<tensorflow::tfrt_stub::FallbackTensor> arg,
    tfrt::RemainingResults results) {
  const auto& fallback_tensor = arg.get();
  const auto& tensor = fallback_tensor.tensor();

  if (!fallback_tensor.is_immutable()) {
    // Create a new TensorBuffer which contains a new atomic counter for each
    // result, to avoid downstream threads contending the original atomic
    // counter.
    for (int i = 0; i < results.size(); ++i) {
      auto immutable_tensor =
          tensorflow::tfrt_stub::ImmutableTensor::Create(tensor);
      results[i] = tfrt::MakeAvailableAsyncValueRef<
          tensorflow::tfrt_stub::FallbackTensor>(
          std::move(immutable_tensor.tensor()));
    }
  } else {
    // For immutable tensors, we just need to copy the pointer. Note that we
    // still create a new AsyncValue in this case, to avoid atomic contention on
    // AsyncValue's refcount.
    for (int i = 0; i < results.size(); ++i) {
      results[i] = tfrt::MakeAvailableAsyncValueRef<
          tensorflow::tfrt_stub::FallbackTensor>(fallback_tensor);
    }
  }
}

llvm::Expected<tensorflow::tfrt_stub::FallbackTensor> ConstStringTensor(
    tfrt::ArrayAttr shape, tfrt::AggregateAttr value,
    const tfrt::ExecutionContext& context) {
  llvm::SmallVector<int64_t> dims;
  auto tfrt_tensor_shape = tfrt::TensorShape(shape.GetValue<int64_t>());
  tfrt_tensor_shape.GetDimensions(&dims);
  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape(dims));
  auto len = tensor.NumElements();
  auto from = value;
  auto to = tensor.flat<tensorflow::tstring>();
  if (from.GetNumElements() == 1) {
    // All elements are the same, and only one element is saved in BEF.
    for (size_t i = 0; i < len; ++i) {
      to(i) = ToAbslStringView(
          from.GetAttributeOfType<tfrt::StringAttr>(0).GetValue());
    }
  } else {
    assert(len == from.GetNumElements());
    for (size_t i = 0; i < len; ++i) {
      to(i) = ToAbslStringView(
          from.GetAttributeOfType<tfrt::StringAttr>(i).GetValue());
    }
  }
  return tensorflow::tfrt_stub::FallbackTensor(tensor);
}

llvm::Expected<tensorflow::tfrt_stub::FallbackTensor> ConstTensorProto(
    tfrt::StringAttr serialized_tensor_proto) {
  tensorflow::TensorProto tensor_proto;
  if (!tensor_proto.ParseFromString(serialized_tensor_proto.GetValue().str())) {
    return tfrt::MakeStringError("Failed to parse const tensor proto");
  }

  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return tfrt::MakeStringError("Failed to create tensor from tensor proto: ",
                                 tensor_proto.ShortDebugString());
  }

  return tensorflow::tfrt_stub::FallbackTensor(std::move(tensor));
}

// Returns true if the tensorflow::DataType is trivially copyable.
bool IsTriviallyCopyableTensorflowDataType(tensorflow::DataType dtype) {
  static const auto* const non_trivially_copyable_dtypes =
      new absl::flat_hash_set<tensorflow::DataType>{
          tensorflow::DataType::DT_STRING, tensorflow::DataType::DT_RESOURCE,
          tensorflow::DataType::DT_VARIANT};
  return !non_trivially_copyable_dtypes->contains(dtype);
}

llvm::Expected<tensorflow::tfrt_stub::FallbackTensor> ConstDenseTensor(
    tfrt::DenseAttr value, const tfrt::ExecutionContext& context) {
  auto dtype = GetTfDataType(tfrt::DType(value.dtype()));
  // The data type must be trivially copyable so that we can use memcpy.
  DCHECK(IsTriviallyCopyableTensorflowDataType(dtype));
  tensorflow::Tensor tensor(dtype, tensorflow::TensorShape(value.shape()));
  std::memcpy(tensor.data(), value.GetElements(), tensor.TotalBytes());
  return tensorflow::tfrt_stub::FallbackTensor(tensor);
}

llvm::Expected<bool> Predicate(
    const tensorflow::tfrt_stub::FallbackTensor& input,
    const tfrt::ExecutionContext& exec_ctx) {
  const auto& tensor = input.tensor();
  if (TensorShapeUtils::IsScalar(tensor.shape())) {
    switch (tensor.dtype()) {
#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    return tensor.scalar<T>()() != 0;

      CASE(float);
      CASE(double);
      CASE(uint8);
      CASE(int8);
      CASE(int16);
      CASE(int32);
      CASE(int64_t);
      CASE(bool);
#undef CASE
      case DT_STRING:
        return !tensor.scalar<tstring>()().empty();
      default:
        return tfrt::MakeStringError(DataTypeString(tensor.dtype()),
                                     " cannot be converted to a boolean");
    }
  }

  return tensor.NumElements() > 0;
}

void BatchFunction(
    tfrt::RepeatedArguments<tensorflow::tfrt_stub::FallbackTensor> args,
    tfrt::RemainingResults results, tfrt::StringAttr device,
    tfrt::Attribute<tfrt::Function> f, tfrt::AggregateAttr op_attr_array,
    const tfrt::ExecutionContext& exec_ctx) {
  constexpr char kTfKernelNameToFallback[] = "_BatchFunctionFallback";
  auto* fallback_request_state =
      exec_ctx.request_ctx()
          ->GetDataIfExists<KernelFallbackCompatRequestState>();

  if (!fallback_request_state) {
    KernelFallbackEmitError(
        exec_ctx, /*fallback_request_state=*/nullptr, kTfKernelNameToFallback,
        /*op_chain=*/nullptr, results.values(),
        tensorflow::errors::NotFound(
            "KernelFallbackCompatRequestState not found in RequestContext."));
    return;
  }

  auto* runner_cache =
      exec_ctx.request_ctx()
          ->resource_context()
          ->GetOrCreateResource<tfrt_stub::OpKernelRunnerCache>(
              kOpKernelRunnerCacheResourceName);
  auto attr_builder = [op_attr_array,
                       f](tensorflow::AttrValueMap* attr_value_map) {
    TF_RETURN_IF_ERROR(SetUpAttrValueMap(
        op_attr_array, /*op_func_attr_array*/ {}, attr_value_map));
    // Pass in a BEF function pointer with a I64 attribute.
    int64_t ptr_value = absl::bit_cast<int64_t>(&f.get());
    (*attr_value_map)["opaque_function_handle"].set_i(ptr_value);
    return absl::OkStatus();
  };
  auto kernel_runner_or_status = runner_cache->GetOrCreate(
      exec_ctx.location(), kTfKernelNameToFallback,
      ToAbslStringView(device.GetValue()), args.size() + 1, attr_builder,
      fallback_request_state->device_manager(),
      fallback_request_state->process_function_library_runtime());
  if (!kernel_runner_or_status.ok()) {
    KernelFallbackEmitError(exec_ctx, fallback_request_state,
                            kTfKernelNameToFallback,
                            /*op_chain=*/nullptr, results.values(),
                            kernel_runner_or_status.status());
    return;
  }
  auto* kernel_runner = kernel_runner_or_status.value();
  DCHECK(kernel_runner->IsAsync());

  auto& run_state = GetThreadLocalOpKernelRunState();
  auto clean_up_inputs =
      gtl::MakeCleanup([&]() { run_state.input_tf_tensors.clear(); });

  // TODO(bangdazhou): refactor the common logic into a util function.
  // Prepare the input tensors.
  auto& input_tf_tensors = run_state.input_tf_tensors;
  auto& input_tf_tensor_values = run_state.input_tf_tensor_values;
  DCHECK(input_tf_tensors.empty());
  input_tf_tensor_values.resize(args.size() + 1);
  for (int i = 0; i < args.size(); ++i) {
    auto* arg = args.values()[i];
    auto& fallback_tensor = arg->get<tensorflow::tfrt_stub::FallbackTensor>();
    // If the argument is immutable or unique, we can just keep the reference
    // without copying that invovles expensive atomic reference counting. And if
    // the argument is unique but mutable, then tensorflow optimizations like
    // buffer forwarding can be utilized. Otherwise, we conservatively copy the
    // tensor.
    if (!fallback_tensor.is_immutable() && !arg->IsUnique()) {
      input_tf_tensors.push_back(fallback_tensor.tensor());
    }
    input_tf_tensor_values[i].tensor = &fallback_tensor.tensor();
  }
  // exec_ctx is passed in as the last input. exec_ctx is only valid during the
  // period of one bef execution. It should not be stored and accessed after bef
  // execution completes.
  int64_t exec_ctx_intptr = absl::bit_cast<int64_t>(&exec_ctx);
  input_tf_tensors.push_back(tensorflow::Tensor(exec_ctx_intptr));
  input_tf_tensor_values.back().tensor = &input_tf_tensors.back();

  auto* tf_device =
      GetDeviceFromFallbackState(*fallback_request_state, *kernel_runner);

  SetUpParams(*kernel_runner, *fallback_request_state, tf_device, run_state);
  KernelFallbackExecuteCompatAsyncInternal<
      tensorflow::tfrt_stub::FallbackTensor>(
      exec_ctx, &run_state, *kernel_runner, /*op_chain=*/nullptr,
      results.values());
}

class TestAllocator : public tensorflow::AllocatorWrapper {
 public:
  TestAllocator() : tensorflow::AllocatorWrapper(tensorflow::cpu_allocator()) {}

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    std::printf("Using TestAllocator\n");
    fflush(stdout);
    return wrapped()->AllocateRaw(alignment, num_bytes);
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override {
    std::printf("Using TestAllocator\n");
    fflush(stdout);
    return wrapped()->AllocateRaw(alignment, num_bytes, allocation_attr);
  }
};

tensorflow::Allocator* GetTestAllocator() {
  static auto* const test_allocator = new TestAllocator;
  return test_allocator;
}

void RegisterKernelFallbackCompatKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("tfrt_fallback_async.const_tensor_proto",
                      TFRT_KERNEL(ConstTensorProto));
  registry->AddKernel("tfrt_fallback_async.executeop", FallbackAsyncExecuteOp);
  registry->AddKernel("tfrt_fallback_async.executeop.seq",
                      FallbackAsyncExecuteOpSeq);
  registry->AddKernel("tfrt_fallback_async.executeop.allocator",
                      FallbackAsyncExecuteOpWithAllocator);
  registry->AddKernel("tfrt_fallback_async.executeop.seq.allocator",
                      FallbackAsyncExecuteOpSeqWithAllocator);
  registry->AddKernel("tfrt_fallback_async.copy_if_small",
                      TFRT_KERNEL(FallbackCopyTensorIfSmall));
  registry->AddKernel("tfrt_fallback_async.createop",
                      TFRT_KERNEL(KernelFallbackCreateOp));
  registry->AddKernel("tfrt_fallback_async.set_resource",
                      TFRT_KERNEL(FallbackSetResource));
  registry->AddKernel("tfrt_fallback_async.get_resource",
                      TFRT_KERNEL(FallbackGetResource));
  registry->AddKernel("tfrt_fallback_async.batch_function",
                      TFRT_KERNEL(BatchFunction));
  registry->AddKernel("tfrt_fallback_async.const_dense_tensor",
                      TFRT_KERNEL(ConstDenseTensor));
  registry->AddKernel("tfrt_fallback_async.const_string_tensor",
                      TFRT_KERNEL(ConstStringTensor));
  registry->AddKernel("tfrt_fallback_async.predicate", TFRT_KERNEL(Predicate));

  // TODO(chky): Move test kernels to test-only library.
  registry->AddKernel("tfrt_fallback_async.get_test_allocator",
                      TFRT_KERNEL(GetTestAllocator));
}

TFRT_STATIC_KERNEL_REGISTRATION(RegisterKernelFallbackCompatKernels);

}  // namespace
}  // namespace tfd
}  // namespace tensorflow
