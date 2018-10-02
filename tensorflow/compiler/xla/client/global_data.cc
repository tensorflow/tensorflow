/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/global_data.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

GlobalData::GlobalData(ServiceInterface* parent, GlobalDataHandle handle)
    : handle_(std::move(handle)), parent_(parent) {}

GlobalData::~GlobalData() {
  UnregisterRequest request;
  *request.mutable_data() = handle_;
  UnregisterResponse response;
  VLOG(1) << "requesting to unregister " << handle_.ShortDebugString();
  Status s = parent_->Unregister(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    LOG(WARNING) << "failed to unregister " << handle_.ShortDebugString()
                 << "; continuing anyway...";
  }
}

}  // namespace xla
