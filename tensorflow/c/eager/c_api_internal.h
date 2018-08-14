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
#include "tensorflow/core/public/version.h"

struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy policy{TFE_DEVICE_PLACEMENT_SILENT};
};

struct TFE_Context {
  explicit TFE_Context(const tensorflow::SessionOptions& opts,
                       TFE_ContextDevicePlacementPolicy default_policy,
                       bool async,
                       std::unique_ptr<tensorflow::DeviceMgr> device_mgr,
                       tensorflow::Rendezvous* rendezvous)
      : context(opts,
                static_cast<tensorflow::ContextDevicePlacementPolicy>(
                    default_policy),
                async, std::move(device_mgr), rendezvous) {}

  tensorflow::EagerContext context;
};

struct TFE_TensorHandle {
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d,
                   tensorflow::Device* op_device)
      : handle(new tensorflow::TensorHandle(t, d, op_device, nullptr)) {}

  TFE_TensorHandle(tensorflow::uint64 node_id, tensorflow::DataType dtype,
                   tensorflow::EagerContext* ctx)
      : handle(new tensorflow::TensorHandle(node_id, dtype, ctx)) {}

  TFE_TensorHandle(tensorflow::TensorHandle* handle) : handle(handle) {}

  tensorflow::TensorHandle* handle;
};

struct TFE_TensorDebugInfo {
  TFE_TensorDebugInfo(const std::vector<tensorflow::int64>& dims)
      : dev_dims(dims) {}

  // Fully-padded, minor-to-major.
  std::vector<tensorflow::int64> dev_dims;
};

struct TFE_Op {
  // t is NULL iff the TFE_Op corresponds to a TensorFlow function instead of a
  // primitive operation.
  TFE_Op(TFE_Context* ctx, const char* op, const tensorflow::AttrTypeMap* t)
      : operation(&ctx->context, op, t) {}

  tensorflow::EagerOperation operation;
};

namespace tensorflow {
// Set an AttrValue on the op. Doesn't handle the list types.
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status);
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
