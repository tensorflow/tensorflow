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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"

namespace tensorflow {
namespace tpu {

// A version of `CompilationCacheEntry` to access Tpu binary program
// `XLA_TpuProgram`.
class TpuCompilationCacheEntry {
 public:
  explicit TpuCompilationCacheEntry(
      const TpuProgramGroupInterface* tpu_program_group, int core_index);
  // Constructor for an empty entry.
  TpuCompilationCacheEntry();
  const TPUExecutableInfoProto* get_executable_info() const;
  const TPUHostTransferInfoProto* get_host_transfer_info() const;
  const xla::HloProto* get_hlo_metadata() const;
  // TODO(henrytan): maybe nicer to return C++ wrapper of `XLA_TpuProgram`
  const XLA_TpuProgram* get_tpu_program() const;

 private:
  const TpuProgramGroup* tpu_program_group_;
  int core_index_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_H_
