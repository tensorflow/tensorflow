/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_UTIL_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_UTIL_H_

#include <memory>

#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"

namespace tensorflow {
namespace tpu {
namespace reshard_variables {

Status FlushProgramMemory(se::Platform* platform, int device_ordinal);

Status CheckIsValidKey(const Tensor& key);

bool IsDefaultKey(const Tensor& key);

Status GetComputationCacheEntry(
    const Tensor& key, string* rendezvous_key_base,
    std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target);

xla::StatusOr<xla::ShapeTree<xla::MaybeOwningDeviceMemory>> BuildInputBuffers(
    OpKernelContext* context, const std::vector<VariableInfo>& variables,
    const xla::Shape& input_host_shape, xla::Backend* backend,
    int device_ordinal, se::Stream* stream);

Status PerformCompaction(stream_executor::Stream* stream);

Status UpdateOutputVariables(
    OpKernelContext* context, xla::ScopedShapedBuffer result_buffers,
    absl::Span<const TensorShapeProto* const> output_tensor_shape_protos,
    xla::Backend* backend, se::Stream* stream, int device_ordinal,
    const std::vector<VariableInfo>& variables,
    const std::shared_ptr<se::Event>& definition_event);

}  // namespace reshard_variables
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_UTIL_H_
