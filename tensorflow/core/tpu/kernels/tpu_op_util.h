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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_UTIL_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_UTIL_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"

namespace tensorflow {
namespace tpu {
// Creates a fingerprint given the name and the vector of shapes.
uint64 CreateFingerprintWithNameAndShapes(
    uint64 name, const std::vector<tensorflow::TensorShape>& shapes);

// Creates a unique compilation cache `key`.
TpuCompilationCacheKey CreateCompilationCacheKey(
    absl::string_view function_name, uint64 function_library_fingerprint,
    uint64 mlir_module_fingerprint, const OpInputList& guaranteed_constants,
    const std::vector<TensorShape>& dynamic_shapes,
    const TPUCompileMetadataProto& metadata,
    const TpuMeshStateInterface& mesh_state, uint64_t session_id = 0,
    ResourceMgr* resource_mgr = nullptr);
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_UTIL_H_
