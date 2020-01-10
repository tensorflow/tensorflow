/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/cl_memory.h"

namespace tflite {
namespace gpu {
namespace cl {

cl_mem_flags ToClMemFlags(AccessType access_type) {
  switch (access_type) {
    case AccessType::READ:
      return CL_MEM_READ_ONLY;
    case AccessType::WRITE:
      return CL_MEM_WRITE_ONLY;
    case AccessType::READ_WRITE:
      return CL_MEM_READ_WRITE;
  }

  return CL_MEM_READ_ONLY;  // unreachable
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
