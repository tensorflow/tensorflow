/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_INIT_H_
#define TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_INIT_H_

#include "tensorflow/core/common_runtime/local_session_selection.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Use TfrtSession as the Session implementation for local session.
//
// TODO(jingdong): Merge this function with the InitializeTfrtSession() in
// tfrt_session.h after we decouple TPU logic from TfrtSession.
inline absl::Status InitializeTfrtSession() {
  SetDefaultLocalSessionImpl(LocalSessionImpl::kTfrtSession);
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_INIT_H_
