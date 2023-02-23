/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_H_
#define TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace dtensor {

// Configure a custom device which runs dtensor while executing
// operations on `underlying_devices`. Allocates `device_info` and fills
// `device`, which should then be passed to
// TFE_RegisterCustomDevice. This only affects eager execution.
//
// `device_name` arg should match the `device_name` argument to
// TFE_RegisterCustomDevice, and is the name of the custom device itself
// (e.g. pass it to `tf.device` to place operations on it from Python).
// TODO(b/268241383): Remove the `status = nullptr` overload.
void AllocateDTensorDevice(absl::string_view device_name,
                           TFE_CustomDevice* device, void** device_info,
                           TF_Status* status = nullptr);

// Add a mesh to the `DTensorDevice` indicated by `device_info`.
//
// `serialized_mesh` is a serialized Mesh proto.
//
// If `is_async` is true, it indicates the DTensor operations on this mesh will
// return immediately (with "non-ready" handles), otherwise block until
// executed. This is exposed as an option for ease of debugging, and will
// typically be on.
//
// `is_host_mesh` indicates this is a CPU mesh used only for sea-of-donuts-style
// host collectives.
//
// in_flight_nodes_limit throttles the number of inflight nodes in the eager
// async executors used by DTensor. The throttling bounds the memory usage
// of an eager training loop. Python API sets this value to 8 by default.
void AddMesh(const std::string& serialized_mesh, void* device_info,
             bool is_async, bool is_host_mesh, int in_flight_nodes_limit,
             TF_Status* status);

// Sets a requested layout for outputs of all operations.
void ExperimentalSetDefaultLayout(const std::string& serialized_layout,
                                  void* device_info, TF_Status* status);
void ExperimentalClearDefaultLayout(void* device_info, TF_Status* status);

// TODO(b/175928457): remove once the bug is fixed.
// Sets a requested default mesh.
void ExperimentalSetDefaultMesh(const std::string& serialized_mesh,
                                void* device_info, TF_Status* status);
void ExperimentalClearDefaultMesh(void* device_info, TF_Status* status);

// Determines whether tensors with a shape previously associated with only one
// layout use that layout if nothing else can be inferred.
void SetSameShapePolicy(void* device_info, bool enabled);

// Sets the global device ID-to-core ID mapping for a mesh. Global device IDs
// are equal to XLA replica IDs for the single XLA computation used by DTensor.
//
// See the comment above Mesh::tpu_core_ids() for some nuances.
void SetTPUCoreIDs(const std::string& mesh_name,
                   const std::vector<int>& tpu_core_ids, void* device_info,
                   TF_Status* status);

// TODO(b/187112276): Delete once we have the TPUCoreIDs live with Device.
void ClearTPUCoreIDs(void* device_info);

// Returns TPU core locations when given a list of TPU core IDs.
std::vector<std::vector<int>> TPUCoreIDsToLocations(
    TFE_Context* context, const std::vector<int>& tpu_core_ids,
    void* device_info);

// Returns TPU core IDs when given a list of TPU core locations.
std::vector<int> TPUCoreLocationsToIDs(
    TFE_Context* context,
    const std::vector<std::vector<int>>& tpu_core_locations, void* device_info);

// Pack `inputs` tensors into a single parallel tensor handle.
TFE_TensorHandle* Pack(TFE_Context* context, int num_inputs,
                       TFE_TensorHandle** inputs,
                       const std::string& string_layout, void* device_info,
                       TF_Status* status);

// Returns the raw components placed on each device of `inputs`'s mesh.
std::vector<TFE_TensorHandle*> Unpack(TFE_Context* context,
                                      TFE_TensorHandle* input,
                                      void* device_info, TF_Status* status);

// Returns the layout of the dtensor 'input'.
std::string FetchLayout(TFE_Context* context, TFE_TensorHandle* input,
                        void* device_info, TF_Status* status);

// Returns whether `input` is a dtensor.
bool IsDTensor(TFE_Context* context, TFE_TensorHandle* input, void* device_info,
               TF_Status* status);

// Pack `indices`, `values`, `shapes` tensors into a SparseTensorWithLayout.
TFE_TensorHandle* SparsePack(TFE_Context* context, int num_inputs,
                             TFE_TensorHandle** indices,
                             TFE_TensorHandle** values,
                             TFE_TensorHandle** shapes,
                             const std::string& string_layout,
                             void* device_info, TF_Status* status);

// Returns whether `input` is a sparse dtensor. Used in `Unpack` at the python
// level to determine whether we should wrap component tensors back into a
// SparseTensor.
bool IsSparseDTensor(TFE_Context* context, TFE_TensorHandle* input,
                     void* device_info, TF_Status* status);

// Returns a dictionary with cache hits and cache miss information.
// Cache hit count is mapped under 'hit', and cache miss count is mapped under
// 'miss'.
std::unordered_map<std::string, int> GetFunctionCacheHitAndMissCount(
    TFE_Context* context, void* device_info, TF_Status* status);

// Sets the layouts for the elements emitted by an iterator resource tensor.
void SetIteratorElementLayouts(TFE_Context* context, TFE_TensorHandle* input,
                               const std::vector<std::string>& string_layouts,
                               void* device_info, TF_Status* status);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_H_
