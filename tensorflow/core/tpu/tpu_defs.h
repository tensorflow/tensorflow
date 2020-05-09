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

// Common definitions related to TPUs.

#ifndef TENSORFLOW_CORE_TPU_TPU_DEFS_H_
#define TENSORFLOW_CORE_TPU_TPU_DEFS_H_

namespace tensorflow {

// Name of the TPU device, which corresponds to a single core.
extern const char* const DEVICE_TPU_NODE;  // "TPU";

// The TPU_REPLICATED_CORE device is a virtual device corresponding to one core
// of a replicated TPU computation. Only valid within the body of a
// TPUReplicate computation.
extern const char* const DEVICE_TPU_REPLICATED_CORE;

extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM";

// Name of the XLA_TPU_JIT compilation device, which is an internal device to
// compile graphs for TPU. Not registered as a device; no operators can be
// assigned to this device by a user.
extern const char* const DEVICE_TPU_XLA_JIT;  // "XLA_TPU_JIT";

// Attribute used internally to pass "is_mirrored_variable" attribute on
// TPUReplicatedInput nodes to _TPUReplicate.
extern const char* const TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR;

// Attribute used internally to annoate ops which might consume TPU FastMem
// variable.
extern const char* const TPU_FAST_MEM_ATTR;  // "_TPU_FAST_MEM"

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_DEFS_H_
