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

#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

CholeskyThunk::CholeskyThunk(const CholeskyOptions& options,
                             BufferAllocation::Slice a_buffer,
                             BufferAllocation::Slice workspace_buffer,
                             BufferAllocation::Slice info_buffer,
                             PrimitiveType type, int64 batch_size, int64 n,
                             const HloInstruction* hlo)
    : Thunk(Kind::kCholesky, hlo),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      info_buffer_(info_buffer),
      type_(type),
      batch_size_(batch_size),
      a_batch_stride_(n * n *
                      ShapeUtil::ByteSizeOfPrimitiveType(
                          hlo->operand(0)->shape().element_type())),
      n_(n) {}

Status CholeskyThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  VLOG(3) << "type=" << PrimitiveType_Name(type_)
          << " uplo=" << se::blas::UpperLowerString(uplo_)
          << " batch_size=" << batch_size_ << " n=" << n_
       << " a=" << a_buffer_.ToString()
       << " workspace=" << workspace_buffer_.ToString()
       << " info=" << info_buffer_.ToString();

  CusolverContext* context;
  {
    tensorflow::mutex_lock lock(mu_);
    auto result = contexts_.emplace(stream, CusolverContext());
    if (result.second) {
      TF_ASSIGN_OR_RETURN(result.first->second,
                          CusolverContext::Create(stream));
    }
    context = &result.first->second;
  }

  char* a_base = static_cast<char*>(
      buffer_allocations.GetDeviceAddress(a_buffer_).opaque());
  int* info_base = static_cast<int*>(
      buffer_allocations.GetDeviceAddress(info_buffer_).opaque());
  se::DeviceMemoryBase workspace_data =
      buffer_allocations.GetDeviceAddress(workspace_buffer_);
  for (int64 i = 0; i < batch_size_; ++i) {
    se::DeviceMemoryBase a_data =
        se::DeviceMemoryBase(a_base + i * a_batch_stride_, a_batch_stride_);
    se::DeviceMemory<int> info_data(
        se::DeviceMemoryBase(info_base + i, sizeof(int)));
    switch (type_) {
      case F32: {
        TF_RETURN_IF_ERROR(
            context->Potrf(uplo_, n_, se::DeviceMemory<float>(a_data), n_,
                           info_data, se::DeviceMemory<float>(workspace_data)));
        break;
      }
      case F64: {
        TF_RETURN_IF_ERROR(context->Potrf(
            uplo_, n_, se::DeviceMemory<double>(a_data), n_, info_data,
            se::DeviceMemory<double>(workspace_data)));
        break;
      }
      case C64: {
        TF_RETURN_IF_ERROR(context->Potrf(
            uplo_, n_, se::DeviceMemory<std::complex<float>>(a_data), n_,
            info_data, se::DeviceMemory<std::complex<float>>(workspace_data)));
        break;
      }
      case C128: {
        TF_RETURN_IF_ERROR(context->Potrf(
            uplo_, n_, se::DeviceMemory<std::complex<double>>(a_data), n_,
            info_data, se::DeviceMemory<std::complex<double>>(workspace_data)));
        break;
      }
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type_));
    }
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
