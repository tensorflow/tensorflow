/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/local_session_selection.h"

namespace tensorflow {

static LocalSessionImpl default_local_session =
    LocalSessionImpl::kDirectSession;

void SetDefaultLocalSessionImpl(LocalSessionImpl impl) {
  default_local_session = impl;
}

LocalSessionImpl GetDefaultLocalSessionImpl() { return default_local_session; }

}  // namespace tensorflow
