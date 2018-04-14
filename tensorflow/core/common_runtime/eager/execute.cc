/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/execute.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

Status EagerExecute(EagerContext* ctx, Device* device,
                    const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
                    KernelAndDevice* kernel, NodeExecStats* maybe_stats,
                    TensorHandle** retvals, int num_retvals) {
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }

  std::vector<Tensor> outputs(1);
  const MemoryTypeVector* output_memory_types = nullptr;
  output_memory_types = &kernel->kernel()->output_memory_types();
  std::vector<Tensor> inputs(op_inputs.size());
  for (int i = 0; i < op_inputs.size(); ++i) {
    const Tensor* input_tensor = nullptr;
    TF_RETURN_IF_ERROR(op_inputs[i]->Tensor(&input_tensor));
    inputs[i] = *input_tensor;
  }
  // WARNING: kernel->Run utilizes the FunctionLibraryRuntime
  // (ctx->func_lib(device)), which in turn holds a pointer to func_lib_def.
  // But knowledge of the implementation
  // of FunctionLibraryRuntime tells us that func_lib_def is not accessed by
  // FunctionLibraryRuntime::Run(), so there is no thread-safety concern here.
  // This is quite subtle. Re-work things to make this better?  (Would it make
  // sense for FunctionLibraryRuntime to ensure thread-safe access to
  // FunctionLibraryDefinition?).  TODO(apassos) figure out how to record stats
  // for ops which are a part of functions.
  // TODO(agarwal): change Run to take vector of handles ?
  TF_RETURN_IF_ERROR(kernel->Run(&inputs, &outputs, maybe_stats));
  if (maybe_stats != nullptr) {
    maybe_stats->set_op_end_rel_micros(Env::Default()->NowMicros() -
                                       maybe_stats->all_start_micros());
    mutex_lock ml(*ctx->MetadataMu());
    if (ctx->ShouldStoreMetadata()) {
      auto* step_stats = ctx->RunMetadataProto()->mutable_step_stats();
      // Lazily initialize the RunMetadata with information about all devices if
      // this is the first call.
      while (step_stats->dev_stats_size() < ctx->devices()->size()) {
        step_stats->add_dev_stats();
      }
      // Find the current device's index.
      int device_idx = 0;
      for (int i = 0; i < ctx->devices()->size(); ++i) {
        if (ctx->devices()->at(i) == device) {
          device_idx = i;
          break;
        }
      }
      // Populate the device stats for this device.
      auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
      dev_stats->set_device(device->name());
      *dev_stats->add_node_stats() = *maybe_stats;
    }
  }
  DCHECK_EQ(num_retvals, outputs.size());
  Device* op_device = device;
  for (int i = 0; i < num_retvals; ++i) {
    Device* d = op_device;
    if (d != nullptr && output_memory_types != nullptr &&
        (*output_memory_types)[i] == HOST_MEMORY) {
      d = nullptr;
    }
    if (retvals[i] == nullptr) {
      retvals[i] = new TensorHandle(outputs[i], d, op_device, ctx);
    } else {
      retvals[i]->SetTensorAndDevice(outputs[i], d, op_device);
    }
  }
  return Status::OK();
}

Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         const char* device_name, TensorHandle** result) {
  TF_RETURN_IF_ERROR(ctx->GetStatus());
  Device* dstd = ctx->HostCPU();
  if (device_name != nullptr && strlen(device_name) > 0) {
    TF_RETURN_IF_ERROR(ctx->device_mgr()->LookupDevice(device_name, &dstd));
  }
  if (ctx->Async()) {
    // Note that `h` may not be currently ready. However execution order will
    // make sure that `h` is ready before the copy is actually done.
    CopyToDeviceNode* node = new CopyToDeviceNode(h, dstd, ctx);
    TensorHandle* output = node->dst();
    // Note that calling Add makes `node` accessible by the EagerExecutor
    // thread. So further accesses need to be thread-safe.
    ctx->ExecutorAdd(node);
    *result = output;
    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(h->CopyToDevice(ctx, dstd, result));
    return Status::OK();
  }
}

}  // namespace tensorflow
