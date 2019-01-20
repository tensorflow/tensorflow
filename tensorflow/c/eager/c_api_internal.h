/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/eager_profiler.h"
#include "tensorflow/core/public/version.h"

struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy policy{TFE_DEVICE_PLACEMENT_SILENT};
};

struct TFE_Context {
  TFE_Context(const tensorflow::SessionOptions& opts,
              TFE_ContextDevicePlacementPolicy default_policy, bool async,
              const tensorflow::DeviceMgr* device_mgr, bool device_mgr_owned,
              tensorflow::Rendezvous* rendezvous)
      : context(opts,
                static_cast<tensorflow::ContextDevicePlacementPolicy>(
                    default_policy),
                async, device_mgr, device_mgr_owned, rendezvous) {}

  tensorflow::EagerContext context;
};

struct TFE_TensorHandle {
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d,
                   tensorflow::Device* op_device)
      : handle(new tensorflow::TensorHandle(t, d, op_device, nullptr)) {}

  TFE_TensorHandle(tensorflow::TensorHandle* handle) : handle(handle) {}

  tensorflow::TensorHandle* handle;

  // Create a symbolic tensor.
  TFE_TensorHandle(TF_Output t, TF_DataType dtype)
      : handle(new tensorflow::TensorHandle(
            tensorflow::OutputGraphNode{t.oper, t.index},
            static_cast<tensorflow::DataType>(dtype))) {}
};

struct TFE_TensorDebugInfo {
  TFE_TensorDebugInfo(const std::vector<tensorflow::int64>& dims)
      : dev_dims(dims) {}

  // Fully-padded, minor-to-major.
  std::vector<tensorflow::int64> dev_dims;
};

struct TFE_Op {
  TFE_Op(TFE_Context* ctx, const char* op, bool is_function,
         const tensorflow::AttrTypeMap* t)
      : operation(&ctx->context, op, is_function, t) {}

  tensorflow::EagerOperation operation;
};

struct TFE_Profiler {
  TFE_Profiler(TFE_Context* ctx)
      : profiler(tensorflow::EagerProfiler::Create(&ctx->context)) {}

  std::unique_ptr<tensorflow::EagerProfiler> profiler;
};

namespace tensorflow {
// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace tensorflow

struct TFE_TraceContext {
  TF_Graph* const graph;

  unsigned int node_counter = 0;
  // Each tensor handle will have its ref count incremented when it's added as a
  // map key, and decremented when this object is destroyed.
  std::map<tensorflow::TensorHandle*, TF_Output> input_tensor_map;
  std::vector<std::pair<tensorflow::TensorHandle*, TF_Output>>* input_tensors =
      nullptr;

  TFE_TraceContext(TF_Graph* graph) : graph(graph) {}

  ~TFE_TraceContext() {
    delete input_tensors;
    for (auto input : input_tensor_map) {
      input.first->Unref();
    }
  }
};

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
