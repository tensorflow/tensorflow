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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_OPTIONS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_OPTIONS_H_

#include <string>

namespace tensorflow {
namespace internal {

// Setter and getter that determine how TPUCompile responds to cancelled
// compilation.  By default this is true, meaning cancelled compilation will
// abort the process, since that's the only mechanism we have available.
//
// Setting this to false allows the process to remain alive, and should only be
// used in tests.
void SetTpuCompilationCancellationTerminatesProcess(bool b);
bool TpuCompilationCancellationTerminatesProcess();

// Setter and getter that determine whether TPU compilation failure will cause
// chips to close. By default this is true, it is suitable for training. For
// inference, we never want servers to die and thus chips will keep alive.
// See b/109873767.
void SetTpuCompilationFailureClosesChips(bool value);
bool TpuCompilationFailureClosesChips();

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_OPTIONS_H_
