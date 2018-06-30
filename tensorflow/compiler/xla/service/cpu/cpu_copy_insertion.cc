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

#include "tensorflow/compiler/xla/service/cpu/cpu_copy_insertion.h"

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> CpuCopyInsertion::Run(HloModule* module) {
  CopyInsertion generic_copy_insertion;

  TF_ASSIGN_OR_RETURN(bool generic_changed, generic_copy_insertion.Run(module));

  // The CPU backend needs additional copies added due to deficiencies in
  // buffer assignment.
  TF_ASSIGN_OR_RETURN(bool buffer_assignment_changed,
                      CopyInsertion::AddCopiesForBufferAssignment(module));

  return generic_changed || buffer_assignment_changed;
}

}  // namespace xla
