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

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/southbound.h"

#include <dlfcn.h>

#include <memory>
#include <optional>
#include <string>

#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#define Load(H, S)                                                 \
  H = reinterpret_cast<decltype(&S)>(::dlsym(dlib_handle_, #S));   \
  if (!H) {                                                        \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #S, \
               ::dlerror());                                       \
  }

namespace litert {
namespace google_tensor {

namespace {

// The SouthBound APIs are implemented in the EdgeTPU libraries.
// It used to be implemented in the libedgetpu_util.so and has been moved to
// libedgetpu_litert.so in newer Android builds.
constexpr const char* kLiteRtLibPath = "/vendor/lib64/libedgetpu_litert.so";
constexpr const char* kEdgeTpuUtilLibPath = "/vendor/lib64/libedgetpu_util.so";

}  // namespace

Southbound::Southbound() : api_(new ThrFunctions) {}

Southbound::~Southbound() {
  if (dlib_handle_) {
    ::dlclose(dlib_handle_);
  }
}

Expected<Southbound::Ptr> Southbound::Create(
    std::optional<std::string> shared_library_dir) {
  Ptr southbound(new Southbound);
  if (auto status = southbound->LoadSymbols(shared_library_dir); !status) {
    return Unexpected(status.Error());
  }

  return southbound;
}

Expected<void> Southbound::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  // Always load the Southbound API library from the vendor partition.
  (void)shared_library_dir;

  // Try loading the new EdgeTPU LiteRT library first. If it fails, it might be
  // because the Android build is too old. In that case, load the old EdgeTPU
  // utility library.
  dlib_handle_ = ::dlopen(kLiteRtLibPath, RTLD_NOW | RTLD_LOCAL);
  if (!dlib_handle_) {
    dlib_handle_ = ::dlopen(kEdgeTpuUtilLibPath, RTLD_NOW | RTLD_LOCAL);
    if (!dlib_handle_) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to load Southbound shared library");
    }
  }

  // Binds all supported symbols from the shared library to the function
  // pointers.
  Load(api_->thr_initialize, thrInitialize);

  Load(api_->thr_get_vendor_api_version, thrGetVendorApiVersion);
  Load(api_->thr_get_vendor_id, thrGetVendorId);

  Load(api_->thr_context_create, thrContextCreate);
  Load(api_->thr_context_delete, thrContextDelete);

  Load(api_->thr_graph_create, thrGraphCreate);
  Load(api_->thr_graph_delete, thrGraphDelete);

  Load(api_->thr_graph_add_edge, thrGraphAddEdge);
  Load(api_->thr_graph_add_sq_node, thrGraphAddSqNode);

  Load(api_->thr_graph_connect_node_input, thrGraphConnectNodeInput);
  Load(api_->thr_graph_connect_node_output, thrGraphConnectNodeOutput);

  Load(api_->thr_graph_set_input_edge, thrGraphSetInputEdge);
  Load(api_->thr_graph_set_output_edge, thrGraphSetOutputEdge);

  Load(api_->thr_graph_annotate_graph, thrGraphAnnotateGraph);
  Load(api_->thr_graph_annotate_edge, thrGraphAnnotateEdge);
  Load(api_->thr_graph_annotate_node, thrGraphAnnotateNode);

  Load(api_->thr_load_sq_container, thrLoadSqContainer);
  Load(api_->thr_load_sq_container_fd, thrLoadSqContainerFd);
  Load(api_->thr_load_sq_container_file, thrLoadSqContainerFile);
  Load(api_->thr_unload_sq_container, thrUnloadSqContainer);

  Load(api_->thr_graph_assign_sq, thrGraphAssignSq);
  Load(api_->thr_sq_query_scratch_pad, thrSqQueryScratchPad);
  Load(api_->thr_sq_attach_scratch_pad_buffer, thrSqAttachScratchPadBuffer);

  Load(api_->thr_register_buffer, thrRegisterBuffer);
  Load(api_->thr_register_buffer_with_offset, thrRegisterBufferWithOffset);
  Load(api_->thr_unregister_buffer, thrUnregisterBuffer);

  Load(api_->thr_invocation_context_get, thrInvocationContextGet);
  Load(api_->thr_invocation_context_delete, thrInvocationContextDelete);

  Load(api_->thr_invocation_context_attach_buffer,
       thrInvocationContextAttachBuffer);
  Load(api_->thr_invocation_context_detach_buffer,
       thrInvocationContextDetachBuffer);

  Load(api_->thr_invocation_context_prepare_for_invoke,
       thrInvocationContextPrepareForInvoke);
  Load(api_->thr_invocation_context_invoke_once,
       thrInvocationContextInvokeOnce);
  Load(api_->thr_invocation_context_wait, thrInvocationContextWait);

  Load(api_->thr_invocation_context_attach_input_buffer_sync_fence,
       thrInvocationContextAttachInputBufferSyncFence);
  Load(api_->thr_invocation_context_get_output_buffer_sync_fence,
       thrInvocationContextGetOutputBufferSyncFence);
  Load(api_->thr_invocation_context_detach_input_buffer_sync_fence,
       thrInvocationContextDetachInputBufferSyncFence);

  Load(api_->thr_invocation_context_query_node_scratch_pad,
       thrInvocationContextQueryNodeScratchPad);
  Load(api_->thr_invocation_context_attach_scratch_pad_buffer,
       thrInvocationContextAttachScratchPadBuffer);

  Load(api_->thr_invocation_context_start_metrics_collection,
       thrInvocationContextStartMetricsCollection);
  Load(api_->thr_invocation_context_stop_metrics_collection,
       thrInvocationContextStopMetricsCollection);

  Load(api_->thr_vendor_set_system_attribute_str,
       thrVendorSetSystemAttributeStr);
  Load(api_->thr_vendor_set_system_attribute_int64,
       thrVendorSetSystemAttributeInt64);

  LITERT_LOG(LITERT_INFO, "SouthBound symbols loaded");
  return {};
}

}  // namespace google_tensor
}  // namespace litert
