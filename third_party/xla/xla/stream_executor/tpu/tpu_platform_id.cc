/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_platform_id.h"

#include "xla/stream_executor/platform.h"

namespace tensorflow {
namespace tpu {

::stream_executor::Platform::Id GetTpuPlatformId() {
  // We can't use the PLATFORM_DEFINE_ID macro because of potential
  // initialization-order-fiasco errors.
  static int plugin_id_value = 42;
  const ::stream_executor::Platform::Id platform_id = &plugin_id_value;
  return platform_id;
}

}  // namespace tpu
}  // namespace tensorflow
