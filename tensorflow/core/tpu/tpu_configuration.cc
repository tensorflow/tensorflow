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

#include "tensorflow/core/tpu/tpu_configuration.h"

namespace tensorflow {

namespace {

ResourceMgr* GetGlobalResourceMgr() {
  static ResourceMgr* const rmgr = new ResourceMgr();
  return rmgr;
}

}  // namespace

#if !defined(PLATFORM_GOOGLE)
// Used only by Google-internal tests, so deliberately left empty.
void MaybeInitializeTPUSystemForTests() {}
#endif

ResourceMgr* GetTPUConfigResourceMgr() {
  MaybeInitializeTPUSystemForTests();

  // Put all TPU-related state in the global ResourceMgr. This includes the
  // TpuPodState, compilation cache, etc. We don't use the TPU_SYSTEM
  // ResourceMgr because there may be more than one TPU_SYSTEM ResourceMgr when
  // DirectSession or isolate_session_state are used.
  return GetGlobalResourceMgr();
}

}  // namespace tensorflow
