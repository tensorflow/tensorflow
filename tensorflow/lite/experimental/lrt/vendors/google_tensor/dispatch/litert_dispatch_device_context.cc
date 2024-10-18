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

#include "tensorflow/lite/experimental/lrt/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"

#include <dlfcn.h>

#include <memory>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  if (!thr_graphs_.empty()) {
    auto thr_graph_delete = southbound_.thr_functions().thr_graph_delete;
    if (!thr_graph_delete) {
      ABSL_LOG(ERROR) << "thr_graph_delete not found";
    } else {
      for (auto* thr_graph : thr_graphs_) {
        thr_graph_delete(thr_graph);
      }
    }
  }

  if (thr_context_) {
    auto thr_context_delete = southbound_.thr_functions().thr_context_delete;
    if (!thr_context_delete) {
      ABSL_LOG(ERROR) << "thr_context_delete not found";
    } else {
      thr_context_delete(thr_context_);
    }
  }
}

absl::StatusOr<std::unique_ptr<LiteRtDispatchDeviceContextT>>
LiteRtDispatchDeviceContextT::Create(
    const litert::google_tensor::Southbound& southbound) {
  std::unique_ptr<LiteRtDispatchDeviceContextT> device_context(
      new LiteRtDispatchDeviceContextT(southbound));

  auto thr_context_create = southbound.thr_functions().thr_context_create;
  if (!thr_context_create) {
    return absl::InternalError("thr_context_create not found");
  }

  device_context->thr_context_ = thr_context_create();
  return device_context;
}
