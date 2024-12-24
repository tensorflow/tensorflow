// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_SOUTHBOUND_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_SOUTHBOUND_H_

#include <memory>
#include <optional>
#include <string>

#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace google_tensor {

class Southbound {
 public:
  using Ptr = std::unique_ptr<Southbound>;
  struct ThrFunctions;

  Southbound(Southbound&) = delete;
  Southbound(Southbound&&) = delete;
  Southbound& operator=(const Southbound&) = delete;
  Southbound& operator=(Southbound&&) = delete;

  ~Southbound();

  static Expected<Ptr> Create(std::optional<std::string> shared_library_dir);

  const ThrFunctions& api() const { return *api_; }

 private:
  Southbound();
  Expected<void> LoadSymbols(std::optional<std::string> shared_library_dir);

  void* dlib_handle_ = nullptr;
  std::unique_ptr<ThrFunctions> api_;
};

// A convenient struct for holding function pointers to SouthBound symbols.
// These function pointers will be loaded to the shared library on device during
// runtime.
struct Southbound::ThrFunctions {
  decltype(&thrInitialize) thr_initialize = nullptr;

  decltype(&thrGetVendorApiVersion) thr_get_vendor_api_version = nullptr;
  decltype(&thrGetVendorId) thr_get_vendor_id = nullptr;

  decltype(&thrContextCreate) thr_context_create = nullptr;
  decltype(&thrContextDelete) thr_context_delete = nullptr;

  decltype(&thrGraphCreate) thr_graph_create = nullptr;
  decltype(&thrGraphDelete) thr_graph_delete = nullptr;

  decltype(&thrGraphAddEdge) thr_graph_add_edge = nullptr;
  decltype(&thrGraphAddSqNode) thr_graph_add_sq_node = nullptr;

  decltype(&thrGraphConnectNodeInput) thr_graph_connect_node_input = nullptr;
  decltype(&thrGraphConnectNodeOutput) thr_graph_connect_node_output = nullptr;

  decltype(&thrGraphSetInputEdge) thr_graph_set_input_edge = nullptr;
  decltype(&thrGraphSetOutputEdge) thr_graph_set_output_edge = nullptr;

  decltype(&thrGraphAnnotateGraph) thr_graph_annotate_graph = nullptr;
  decltype(&thrGraphAnnotateEdge) thr_graph_annotate_edge = nullptr;
  decltype(&thrGraphAnnotateNode) thr_graph_annotate_node = nullptr;

  decltype(&thrLoadSqContainer) thr_load_sq_container = nullptr;
  decltype(&thrLoadSqContainerFd) thr_load_sq_container_fd = nullptr;
  decltype(&thrLoadSqContainerFile) thr_load_sq_container_file = nullptr;
  decltype(&thrUnloadSqContainer) thr_unload_sq_container = nullptr;

  decltype(&thrGraphAssignSq) thr_graph_assign_sq = nullptr;
  decltype(&thrSqQueryScratchPad) thr_sq_query_scratch_pad = nullptr;
  decltype(&thrSqAttachScratchPadBuffer) thr_sq_attach_scratch_pad_buffer =
      nullptr;

  decltype(&thrRegisterBuffer) thr_register_buffer = nullptr;
  decltype(&thrRegisterBufferWithOffset) thr_register_buffer_with_offset =
      nullptr;
  decltype(&thrUnregisterBuffer) thr_unregister_buffer = nullptr;

  decltype(&thrInvocationContextGet) thr_invocation_context_get = nullptr;
  decltype(&thrInvocationContextDelete) thr_invocation_context_delete = nullptr;

  decltype(&thrInvocationContextAttachBuffer)
      thr_invocation_context_attach_buffer = nullptr;
  decltype(&thrInvocationContextDetachBuffer)
      thr_invocation_context_detach_buffer = nullptr;

  decltype(&thrInvocationContextPrepareForInvoke)
      thr_invocation_context_prepare_for_invoke = nullptr;
  decltype(&thrInvocationContextInvokeOnce) thr_invocation_context_invoke_once =
      nullptr;
  decltype(&thrInvocationContextWait) thr_invocation_context_wait = nullptr;

  decltype(&thrInvocationContextAttachInputBufferSyncFence)
      thr_invocation_context_attach_input_buffer_sync_fence = nullptr;
  decltype(&thrInvocationContextGetOutputBufferSyncFence)
      thr_invocation_context_get_output_buffer_sync_fence = nullptr;

  decltype(&thrInvocationContextQueryNodeScratchPad)
      thr_invocation_context_query_node_scratch_pad = nullptr;
  decltype(&thrInvocationContextAttachScratchPadBuffer)
      thr_invocation_context_attach_scratch_pad_buffer = nullptr;

  decltype(&thrVendorSetSystemAttributeStr)
      thr_vendor_set_system_attribute_str = nullptr;
  decltype(&thrVendorSetSystemAttributeInt64)
      thr_vendor_set_system_attribute_int64 = nullptr;
};

}  // namespace google_tensor
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_SOUTHBOUND_H_
