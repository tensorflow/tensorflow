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
#ifndef EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_MESH_STATE_INTERFACE_H_
#define EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_MESH_STATE_INTERFACE_H_

#include <string>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"

namespace tensorflow {

class TpuMeshCommonState;

namespace tpu {

const char kTpuMeshCommonStateResourceName[] = "tpu_mesh_common_state";

class TpuMeshStateInterface : public tensorflow::ResourceBase {
 public:
  explicit TpuMeshStateInterface(XLA_TpuMeshState* handle)
      : mesh_state_(handle) {
  }

  ~TpuMeshStateInterface() override {
    if (mesh_state_ != nullptr) {
      TpuMeshState_Free(mesh_state_);
    }
  }

  static TpuMeshStateInterface* Create() {
    return new TpuMeshStateInterface(TpuMeshState_Create());
  }

  const XLA_TpuMeshState* data() const { return mesh_state_; }

  tensorflow::TpuMeshCommonState* mesh_common_state() const {
    return static_cast<tensorflow::TpuMeshCommonState*>(
        TpuMeshState_MeshCommonState(mesh_state_));
  }

  // Returns whether we should include the device assignment as a static field
  // to the TPU program. This also determines whether we should include the
  // device assignment as part of the compilation cache key.
  bool NeedsStaticDeviceAssignment(
      const TPUCompileMetadataProto& metadata,
      TpuCoreTypeEnum tpu_core_type) const {
    // Static device assignment enables XLA to perform certain optimization when
    // all cores are used in the replicated computation.
    return metadata.num_cores_per_replica() * metadata.num_replicas() ==
           TpuTopology_AvailableCoreCount(mesh_state_,
                                          tpu_core_type);
  }

  string DebugString() const override { return "TpuMeshStateInterface"; }

 private:
  XLA_TpuMeshState* mesh_state_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_MESH_STATE_INTERFACE_H_
