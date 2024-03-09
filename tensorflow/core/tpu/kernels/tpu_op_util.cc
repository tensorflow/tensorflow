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
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"

namespace tensorflow {
namespace tpu {
namespace {
std::string CreateShapePrefix(
    const std::vector<tensorflow::TensorShape>& dynamic_shapes) {
  std::string shapes_prefix;
  for (const TensorShape& shape : dynamic_shapes) {
    for (int64_t size : shape.dim_sizes()) {
      absl::StrAppend(&shapes_prefix, size, ",");
    }
    absl::StrAppend(&shapes_prefix, ";");
  }
  return shapes_prefix;
}

// Include compilation configurations of the arguments that are not captured
// by the called graph.
std::string CreateConfigPrefix(const TPUCompileMetadataProto& metadata) {
  std::string config_prefix;
  for (const auto& arg : metadata.args()) {
    if (arg.is_same_data_across_replicas()) {
      absl::StrAppend(&config_prefix, ":s");
      // Same.
    } else {
      // Different.
      absl::StrAppend(&config_prefix, ":");
    }
    if (arg.enable_xla_sharding() ==
        tpu::TPUCompileMetadataProto::Arg::ALLOWED) {
      // Enabled.
      absl::StrAppend(&config_prefix, "e");
    }
    if (arg.unrestricted_layout()) {
      // Unrestricted.
      absl::StrAppend(&config_prefix, ":u");
    }
    absl::StrAppend(&config_prefix, ",type(", arg.dtype(), ")");
    if (arg.has_shape()) {
      absl::StrAppend(&config_prefix, ",shape(");
      for (const auto& dim : arg.shape().dim()) {
        absl::StrAppend(&config_prefix, dim.size(), ",");
      }
      absl::StrAppend(&config_prefix, ")");
    }
  }
  return config_prefix;
}
}  // namespace

uint64 CreateFingerprintWithNameAndShapes(
    uint64 name, const std::vector<tensorflow::TensorShape>& shapes) {
  std::string shape_prefix = CreateShapePrefix(shapes);
  VLOG(2) << "CreateFingerprintWithNameAndShapes, name: " << name
          << ", shape_prefix: " << shape_prefix;
  return TpuCompileInterface::Get()->FingerprintString(
      absl::StrCat(name, "_", shape_prefix));
}

// Return fingerprint_in_metadata if it's not empty; otherwise read input tensor
// data to compute the fingerprint.
std::string GuaranteedConstFingerprint(
    const string& fingerprint_in_metadata,
    const OpInputList& guaranteed_constants) {
  if (fingerprint_in_metadata.empty()) {
    uint64_t fingerprint = 0;
    for (const Tensor& constant : guaranteed_constants) {
      fingerprint = stream_executor::tpu::OpsApiFn()
                        ->TpuCompile_CreateGuaranteedConstFingerprintFn(
                            fingerprint, constant.tensor_data().data(),
                            constant.tensor_data().size());
    }
    return std::to_string(fingerprint);
  } else {
    return fingerprint_in_metadata;
  }
}

// The `guaranteed_constants` must be passed as reference due to the lazy
// evaluation of `guaranteed_const_fingerprint()` callback.
TpuCompilationCacheKey CreateCompilationCacheKey(
    absl::string_view function_name, uint64 function_library_fingerprint,
    uint64 mlir_module_fingerprint, const OpInputList& guaranteed_constants,
    const std::vector<TensorShape>& dynamic_shapes,
    const TPUCompileMetadataProto& metadata,
    const TpuMeshStateInterface& mesh_state, uint64_t session_id,
    ResourceMgr* resource_mgr) {
  VLOG(1) << "FunctionLibraryFingerprint:" << function_library_fingerprint;
  std::string shapes_prefix = CreateShapePrefix(dynamic_shapes);
  VLOG(1) << "shapes_prefix = " << shapes_prefix;
  std::string config_prefix = CreateConfigPrefix(metadata);
  VLOG(1) << "config_prefix = " << config_prefix;
  std::vector<int32_t> flattened_device_ids;
  if (metadata.has_device_assignment()) {
    for (const auto& device :
         metadata.device_assignment().computation_devices()) {
      flattened_device_ids.insert(flattened_device_ids.end(),
                                  device.replica_device_ids().begin(),
                                  device.replica_device_ids().end());
    }
  }
  CompilationCacheKeyResult result =
      stream_executor::tpu::OpsApiFn()->TpuCompile_CreateCompilationCacheKeyFn(
          CompilationCacheKeyProperty{
              config_prefix.data(), shapes_prefix.data(), function_name.data(),
              mlir_module_fingerprint, flattened_device_ids.data(),
              flattened_device_ids.size(), guaranteed_constants.size(),
              function_library_fingerprint, metadata.num_cores_per_replica(),
              metadata.num_replicas(), mesh_state.data(), session_id,
              resource_mgr});
  auto buffer_cleanup = gtl::MakeCleanup([result]() {
    stream_executor::tpu::OpsApiFn()->TpuCompile_DestroyCompilationCacheKeyFn(
        result);
  });
  TpuCompilationCacheKey key;
  key.prefix = result.key;
  key.debug_string = result.debug_string;
  key.session_id = session_id;

  // Guaranteed constants can be different across sessions. Use session_handle
  // and guaranteed_const fingerprint to guarantee no collision.
  if (guaranteed_constants.size() > 0) {
    key.has_guaranteed_const = true;
    key.session_handle = metadata.session_handle();
    // Both `metadata` and `guaranteed_constants` lifetime are captured by
    // reference based on the assumption that these variables lifetime is
    // managed through the `TPUCompileOpKernelImpl` that outlives the
    // lifetime of the compilation cache lookups.
    string fingerprint;
    key.guaranteed_const_fingerprint = [&metadata, &guaranteed_constants,
                                        fingerprint]() mutable {
      if (fingerprint.empty()) {
        fingerprint = GuaranteedConstFingerprint(
            metadata.guaranteed_const_fingerprint(), guaranteed_constants);
      }
      return fingerprint;
    };
  }
  return key;
}
}  // namespace tpu
}  // namespace tensorflow
