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
#include "tensorflow/core/tfrt/eager/function_cache.h"

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/tfrt/eager/transform_graph_function.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

void FunctionCache::RemoveFunction(string_view op_name) {
  mutex_lock l(cache_mu_);
  auto iter = cache_.begin();
  while (iter != cache_.end()) {
    if (iter->first.op_name == op_name) {
      iter = cache_.erase(iter);
    } else {
      ++iter;
    }
  }
}

tensorflow::Status FunctionCache::GetOrAddFunction(
    const std::string& op_name, const std::string& device_name,
    const tensorflow::DeviceSet& device_set,
    tensorflow::EagerContext* eager_ctx, tfrt::CoreRuntime* corert,
    RequestCtxBuilder request_ctx_fn, Location loc,
    tensorflow::TfrtFunctionCompileOptions compile_options,
    tfrt::ArrayRef<const Device*> input_devices,
    FunctionCache::FunctionCacheResult* result) {
  const CacheKey cache_key{op_name, device_name};
  {
    mutex_lock l(cache_mu_);
    auto& function_state = cache_[cache_key];
    if (function_state) {
      *result = FunctionCache::FunctionCacheResult{function_state, false};
      return tensorflow::Status::OK();
    }
  }

  tensorflow::FunctionLibraryDefinition* func_lib_def = eager_ctx->FuncLibDef();
  const tensorflow::FunctionDef* fdef = func_lib_def->Find(op_name);
  if (fdef == nullptr)
    return tensorflow::errors::NotFound(
        "Cannot find function from FunctionLibraryDefinition ", op_name);

  // Run graph optimizations using current runtime components before converting
  // the graph to MLIR module.
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  TF_RETURN_IF_ERROR(tensorflow::FunctionDefToBodyHelper(
      *fdef, tensorflow::AttrSlice(), func_lib_def, &fbody));

  // Transferring out the graph ownership from fbody.
  auto graph = std::unique_ptr<tensorflow::Graph>(fbody->graph);
  fbody->graph = nullptr;

  tensorflow::GraphDef graph_def;
  graph->ToGraphDef(&graph_def);
  tensorflow::FunctionLibraryDefinition reachable_lib_def =
      func_lib_def->ReachableDefinitions(graph_def);

  TF_RETURN_IF_ERROR(tensorflow::TransformGraphFunction(
      op_name, *fdef, device_name, device_set, eager_ctx,
      compile_options.enable_grappler, &fbody, std::move(graph), input_devices,
      &reachable_lib_def));

  BefBuffer bef_buffer;

  llvm::SmallVector<tfrt::string_view, 4> device_names;
  device_names.reserve(device_set.devices().size());
  for (auto& d : device_set.devices()) {
    device_names.push_back(d->name());
  }

  // Lower FunctionDef to BEF.
  TF_RETURN_IF_ERROR(tensorflow::ConvertFunctionToBef(
      op_name, fbody.get(), reachable_lib_def, device_names, compile_options,
      &bef_buffer));

  HostContext* host_ctx = corert->GetHostContext();
  auto bef_file =
      tfrt::BEFFile::Open(bef_buffer, host_ctx->GetKernelRegistry(),
                          host_ctx->diag_handler(), host_ctx->allocator());
  if (!bef_file)
    return tensorflow::errors::Internal(
        "Failed to open lowered BEF for function ", op_name, ".");

  const tfrt::Function* function = bef_file->GetFunction(op_name);
  if (!function)
    return tensorflow::errors::Internal(
        "Failed to get function from BEF for function ", op_name, ".");

  auto expected_fn = corert->MakeCompositeOp(function);
  if (!expected_fn)
    return tensorflow::errors::Internal(StrCat("Construct CoreRuntimeOp for ",
                                               op_name.c_str(), " failed. ",
                                               expected_fn.takeError()));

  TfrtDataTypeVector tfrt_arg_types;
  tensorflow::DataTypeVector tf_ret_types;

  for (const auto& arg_type : fbody->arg_types) {
    tfrt_arg_types.push_back(ConvertTfDTypeToTfrtDType(arg_type));
  }

  for (const auto& ret_type : fbody->ret_types) {
    tf_ret_types.push_back(ret_type);
  }

  auto runner_table =
      absl::make_unique<tensorflow::tfrt_stub::OpKernelRunnerTable>();
  RCReference<RequestContext> request_ctx;
  TF_RETURN_IF_ERROR(request_ctx_fn(runner_table.get(), &request_ctx));

  ExecutionContext exec_ctx{std::move(request_ctx), loc};
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file.get(), "_tfrt_fallback_init"));

  RCReference<FunctionState> entry = FunctionState::CreateFunctionState(
      tfrt_arg_types, tf_ret_types, std::move(bef_buffer), std::move(bef_file),
      std::move(expected_fn.get()), std::move(runner_table));

  mutex_lock l(cache_mu_);
  // Insert the new entry to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_[cache_key] = entry;
  *result = FunctionCache::FunctionCacheResult{std::move(entry), true};
  return tensorflow::Status::OK();
}

}  // namespace tf
}  // namespace tfrt
