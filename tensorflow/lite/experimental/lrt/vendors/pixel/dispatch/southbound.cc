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

#include "tensorflow/lite/experimental/lrt/vendors/pixel/dispatch/southbound.h"

#include <dlfcn.h>

#include "absl/log/absl_log.h"

#define Load(H, S)                                               \
  H = reinterpret_cast<decltype(&S)>(::dlsym(dlib_handle_, #S)); \
  if (!H) {                                                      \
    ABSL_LOG(WARNING) << "Failed to load symbol " << #S << ": "  \
                      << ::dlerror();                            \
  }

namespace lrt {
namespace pixel {

Southbound::Southbound() : thr_functions_(new ThrFunctions) {}

Southbound::~Southbound() {
  if (dlib_handle_) {
    ::dlclose(dlib_handle_);
  }
}

absl::StatusOr<std::unique_ptr<Southbound>> Southbound::Create() {
  std::unique_ptr<Southbound> southbound(new Southbound);
  if (auto status = southbound->LoadSymbols(); !status.ok()) {
    return status;
  }

  return southbound;
}

absl::Status Southbound::LoadSymbols() {
  dlib_handle_ = ::dlopen(kSouthBoundLibPath, RTLD_NOW | RTLD_LOCAL);
  if (!dlib_handle_) {
    return absl::InternalError("Failed to load Southbound shared library");
  }

  // Binds all supported symbols from the shared library to the function
  // pointers.
  Load(thr_functions_->thr_initialize, thrInitialize);

  Load(thr_functions_->thr_get_vendor_api_version, thrGetVendorApiVersion);
  Load(thr_functions_->thr_get_vendor_id, thrGetVendorId);

  Load(thr_functions_->thr_context_create, thrContextCreate);
  Load(thr_functions_->thr_context_delete, thrContextDelete);

  Load(thr_functions_->thr_graph_create, thrGraphCreate);
  Load(thr_functions_->thr_graph_delete, thrGraphDelete);

  Load(thr_functions_->thr_graph_add_edge, thrGraphAddEdge);
  Load(thr_functions_->thr_graph_add_sq_node, thrGraphAddSqNode);

  Load(thr_functions_->thr_graph_connect_node_input, thrGraphConnectNodeInput);
  Load(thr_functions_->thr_graph_connect_node_output,
       thrGraphConnectNodeOutput);

  Load(thr_functions_->thr_graph_set_input_edge, thrGraphSetInputEdge);
  Load(thr_functions_->thr_graph_set_output_edge, thrGraphSetOutputEdge);

  Load(thr_functions_->thr_graph_annotate_graph, thrGraphAnnotateGraph);
  Load(thr_functions_->thr_graph_annotate_edge, thrGraphAnnotateEdge);
  Load(thr_functions_->thr_graph_annotate_node, thrGraphAnnotateNode);

  Load(thr_functions_->thr_load_sq_container, thrLoadSqContainer);
  Load(thr_functions_->thr_load_sq_container_fd, thrLoadSqContainerFd);
  Load(thr_functions_->thr_load_sq_container_file, thrLoadSqContainerFile);
  Load(thr_functions_->thr_unload_sq_container, thrUnloadSqContainer);

  Load(thr_functions_->thr_graph_assign_sq, thrGraphAssignSq);
  Load(thr_functions_->thr_sq_query_scratch_pad, thrSqQueryScratchPad);
  Load(thr_functions_->thr_sq_attach_scratch_pad_buffer,
       thrSqAttachScratchPadBuffer);

  Load(thr_functions_->thr_register_buffer, thrRegisterBuffer);
  Load(thr_functions_->thr_register_buffer_with_offset,
       thrRegisterBufferWithOffset);
  Load(thr_functions_->thr_unregister_buffer, thrUnregisterBuffer);

  Load(thr_functions_->thr_invocation_context_get, thrInvocationContextGet);
  Load(thr_functions_->thr_invocation_context_delete,
       thrInvocationContextDelete);

  Load(thr_functions_->thr_invocation_context_attach_buffer,
       thrInvocationContextAttachBuffer);
  Load(thr_functions_->thr_invocation_context_detach_buffer,
       thrInvocationContextDetachBuffer);

  Load(thr_functions_->thr_invocation_context_prepare_for_invoke,
       thrInvocationContextPrepareForInvoke);
  Load(thr_functions_->thr_invocation_context_invoke_once,
       thrInvocationContextInvokeOnce);
  Load(thr_functions_->thr_invocation_context_wait, thrInvocationContextWait);

  Load(thr_functions_->thr_invocation_context_attach_input_buffer_sync_fence,
       thrInvocationContextAttachInputBufferSyncFence);
  Load(thr_functions_->thr_invocation_context_get_output_buffer_sync_fence,
       thrInvocationContextGetOutputBufferSyncFence);

  Load(thr_functions_->thr_invocation_context_query_node_scratch_pad,
       thrInvocationContextQueryNodeScratchPad);
  Load(thr_functions_->thr_invocation_context_attach_scratch_pad_buffer,
       thrInvocationContextAttachScratchPadBuffer);

  Load(thr_functions_->thr_vendor_set_system_attribute_str,
       thrVendorSetSystemAttributeStr);
  Load(thr_functions_->thr_vendor_set_system_attribute_int64,
       thrVendorSetSystemAttributeInt64);

  ABSL_LOG(INFO) << "SouthBound symbols loaded.";
  return {};
}

}  // namespace pixel
}  // namespace lrt
