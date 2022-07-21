/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_

#include <string>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {

class TpuEmbeddingEngineState;

namespace tpu {

const char kTpuEmbeddingEngineStateInterfaceResourceName[] =
    "tpu_embedding_engine_state";

class TpuEmbeddingEngineStateInterface : public ResourceBase {
 public:
  explicit TpuEmbeddingEngineStateInterface(XLA_TpuEmbeddingEngineState* handle)
      : engine_state_(handle) {}

  ~TpuEmbeddingEngineStateInterface() override {
    if (engine_state_ != nullptr) {
      OpsApiFn()->TpuEmbeddingEngineState_FreeFn(engine_state_);
    }
  }

  tensorflow::TpuEmbeddingEngineState* GetState() const {
    if (engine_state_ == nullptr) {
      return nullptr;
    }
    return static_cast<tensorflow::TpuEmbeddingEngineState*>(
        OpsApiFn()->TpuEmbeddingEngineState_GetStateFn(engine_state_));
  }

  static TpuEmbeddingEngineStateInterface* Create() {
    XLA_TpuEmbeddingEngineState* state = nullptr;
    if (OpsApiFn()->TpuEmbeddingEngineState_CreateFn != nullptr) {
      state = OpsApiFn()->TpuEmbeddingEngineState_CreateFn();
    }
    return new TpuEmbeddingEngineStateInterface(state);
  }

  string DebugString() const override {
    return "TpuEmbeddingEngineStateInterface";
  }

 private:
  XLA_TpuEmbeddingEngineState* engine_state_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_
