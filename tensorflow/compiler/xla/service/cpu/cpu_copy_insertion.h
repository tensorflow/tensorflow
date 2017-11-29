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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COPY_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COPY_INSERTION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Besides the modifications made by the generic xla::CopyInsertion, this
// CPU-specific copy insertion pass also adds copies to values live out of
// computations satisfying certain conditions (defined by constant or parameter,
// etc). This is necessary because of deficiencies of buffer
// assignment. Specifically, buffer assignment is computation-scoped and does
// not recognized aliasing between arguments and outputs of computations.
//
// TODO(b/62548313): Remove this when buffer assignment is smarter
// (module-scoped).
class CpuCopyInsertion : public HloPassInterface {
 public:
  tensorflow::StringPiece name() const override { return "copy-insertion"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COPY_INSERTION_H_
