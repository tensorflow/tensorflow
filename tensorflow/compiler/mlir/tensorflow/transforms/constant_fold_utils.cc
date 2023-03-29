/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"

#include <memory>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tsl/platform/mem.h"

namespace mlir {
namespace TF {

TFE_Context* GetContextForConstantFold() {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  // Only initialize single CPU.
  tensorflow::ConfigProto config_proto;
  // This is conceptually equal to what we do in python/eager/context.py but
  // with all GPU/TPU devices ignored and CPU only set to 1.
  (*config_proto.mutable_device_count())["CPU"] = 1;
#if TENSORFLOW_USE_ROCM
  (*config_proto.mutable_device_count())["GPU"] = 0;
#endif
  config_proto.add_device_filters("/device:CPU:*");
  // Limit the thread pool size. Without this, TF by default creates as many
  // threads as the number of CPUs (`port::MaxParallelism()`). This can be
  // expensive since this TFE context persists the entire program execution.
  config_proto.set_inter_op_parallelism_threads(2);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_NewBuffer(), TF_DeleteBuffer);
  DCHECK(config->data == nullptr);

  // Copy config_proto into config.
  {
    const size_t proto_size = config_proto.ByteSizeLong();
    void* buf = tsl::port::Malloc(proto_size);
    if (buf == nullptr) {
      LOG(ERROR) << "Failed to allocate memory to serialize ConfigProto "
                    "while creating context options for constant folding";
      return nullptr;
    }
    if (!config_proto.SerializeWithCachedSizesToArray(
            static_cast<uint8_t*>(buf))) {
      tsl::port::Free(buf);
      LOG(ERROR) << "Unable to serialize ConfigProto while creating context "
                    "options for constant folding";
      return nullptr;
    }
    config->data = buf;
    config->length = proto_size;
    config->data_deallocator = [](void* data, size_t length) {
      tsl::port::Free(data);
    };
  }

  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to set context options for constant folding: "
               << status.get();
    return nullptr;
  }

  // Input tensors are placed on the host CPU so use the explicit device
  // policy to fail if no CPU kernels are available for the op.
  TFE_ContextOptionsSetDevicePlacementPolicy(opts.get(),
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  auto ctx = TFE_NewContext(opts.get(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to create context for constant folding: "
               << status.get();
    return nullptr;
  }
  return ctx;
}

}  // namespace TF
}  // namespace mlir
