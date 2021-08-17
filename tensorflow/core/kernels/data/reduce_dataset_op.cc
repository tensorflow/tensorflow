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

#include "tensorflow/core/kernels/data/reduce_dataset_op.h"

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

}  // namespace

ReduceDatasetOp::ReduceDatasetOp(OpKernelConstruction* ctx)
    : HybridAsyncOpKernel(ctx, "tf_data_reduce_dataset") {
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("use_inter_op_parallelism",
                                   &params.use_inter_op_parallelism));
  params.use_default_device = false;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, "f", params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

Status ReduceDatasetOp::DoCompute(OpKernelContext* ctx) {
  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode("ReduceDatasetOp::DoCompute",
                                       {{"id", ctx->step_id()}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list("initial_state", &inputs));
  std::vector<Tensor> state(inputs.begin(), inputs.end());

  std::unique_ptr<CapturedFunction> captured_func;
  TF_RETURN_IF_ERROR(CapturedFunction::Create(
      ctx, func_metadata_, "other_arguments", &captured_func));

  IteratorContext::Params params(ctx);
  auto function_handle_cache =
      absl::make_unique<FunctionHandleCache>(params.flr);
  params.function_handle_cache = function_handle_cache.get();
  ResourceMgr resource_mgr;
  params.resource_mgr = &resource_mgr;
  CancellationManager cancellation_manager(ctx->cancellation_manager());
  params.cancellation_manager = &cancellation_manager;

  IteratorContext iter_ctx(std::move(params));
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
  TF_RETURN_IF_ERROR(
      captured_func->Instantiate(&iter_ctx, &instantiated_captured_func));

  std::unique_ptr<IteratorBase> iterator;
  if (ctx->function_library()->device()->device_type() == DEVICE_CPU) {
    DatasetBase* finalized_dataset = nullptr;
    TF_RETURN_IF_ERROR(FinalizeDataset(ctx, dataset, &finalized_dataset));
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
        &iter_ctx, /*parent=*/nullptr, "ReduceIterator", &iterator));
    finalized_dataset->Unref();
  } else {
    TF_RETURN_IF_ERROR(dataset->MakeIterator(&iter_ctx, /*parent=*/nullptr,
                                             "ReduceIterator", &iterator));
  }

  // Iterate through the input dataset.
  while (true) {
    if (ctx->cancellation_manager()->IsCancelled()) {
      return errors::Cancelled("Operation was cancelled");
    }
    std::vector<Tensor> next_input_element;
    bool end_of_input;
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &next_input_element, &end_of_input));
    if (end_of_input) {
      break;
    }

    // Run the reduce function to update the current state.
    std::vector<Tensor> args;
    args.reserve(state.size() + next_input_element.size());
    std::copy(state.begin(), state.end(), std::back_inserter(args));
    std::copy(next_input_element.begin(), next_input_element.end(),
              std::back_inserter(args));

    std::vector<Tensor> reduce_func_output;
    TF_RETURN_IF_ERROR(instantiated_captured_func->Run(
        &iter_ctx, std::move(args), &reduce_func_output, /*node=*/nullptr));
    if (reduce_func_output.size() != state.size()) {
      return errors::InvalidArgument(
          "The number of components of the initial state and the "
          "reduce "
          "function output does not match. (initial_state=",
          state.size(), ", output=", reduce_func_output.size(), ").");
    }
    std::swap(reduce_func_output, state);
  }

  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, state));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, state));
  for (size_t i = 0; i < state.size(); ++i) {
    ctx->set_output(i, state[i]);
  }
  return Status::OK();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("ReduceDataset").Device(DEVICE_CPU),
                        ReduceDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ReduceDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
