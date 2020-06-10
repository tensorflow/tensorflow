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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"

#include "tensorflow/core/platform/casts.h"

namespace tensorflow {
namespace tpu {

TpuCompilationCacheEntry::TpuCompilationCacheEntry(
    const TpuProgramGroupInterface* tpu_program_group, int core_index)
    : tpu_program_group_(
          tensorflow::down_cast<const TpuProgramGroup*>(tpu_program_group)),
      core_index_(core_index) {}

// Constructor for an empty entry.
TpuCompilationCacheEntry::TpuCompilationCacheEntry()
    : tpu_program_group_(nullptr) {}

const TPUExecutableInfoProto* TpuCompilationCacheEntry::get_executable_info()
    const {
  return &(tpu_program_group_->executable_info());
}

const TPUHostTransferInfoProto*
TpuCompilationCacheEntry::get_host_transfer_info() const {
  return &(tpu_program_group_->host_transfer_info());
}

const xla::HloProto* TpuCompilationCacheEntry::get_hlo_metadata() const {
  return tpu_program_group_->hlo_metadatas()[core_index_].get();
}

// TODO(henrytan,jiawenhao): When should we expect more than one
// XLA_TpuProgram* per TpuProgram? Remove the program_count CHECK below then.
const XLA_TpuProgram* TpuCompilationCacheEntry::get_tpu_program() const {
  CHECK_EQ(tpu_program_group_->program_count(), 1);
  return tpu_program_group_->tpu_programs()[core_index_];
}

}  // namespace tpu
}  // namespace tensorflow
