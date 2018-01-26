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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COPY_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COPY_INSERTION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Besides the modifications made by the generic xla::CopyInsertion, this
// GPU-specific copy insertion also materializes operands of library calls by
// inserting kCopy instructions.
class GpuCopyInsertion : public HloPassInterface {
 public:
  tensorflow::StringPiece name() const override { return "copy-insertion"; }

  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Returns a copy of `hlo`. Looks in inserted_copies_ first to avoid making
  // duplicate copies.
  StatusOr<HloInstruction*> FindOrInsertCopy(HloInstruction* hlo);

  // A map containing all copies inserted to materialize operands of library
  // calls. The key is the copied instruction and the value is the copy.
  tensorflow::gtl::FlatMap<HloInstruction*, HloInstruction*> inserted_copies_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COPY_INSERTION_H_
