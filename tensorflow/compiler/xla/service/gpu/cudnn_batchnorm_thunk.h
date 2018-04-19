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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

// This file contains thunks which call into cudnn to run the various flavors of
// batch normalization: BatchNormInference, BatchNormTraining, and
// BatchNormGrad, known to cudnn as BatchNormForwardInference,
// BatchNormForwardTraining, and BatchNormBackward.
//
// As an alternative to using these thunks, XLA can decompose batchnorm HLOs
// into smaller components using the BatchNormRewriter pass.  This can result in
// faster code because those individual components can fuse into their
// inputs/outputs, but it may also be slower if cudnn's batchnorm implementation
// outperforms the code XLA generates for these components.
//
// Currently these thunks require that their inputs are F32s.
//
// Note that these thunks do not take full advantage of the cudnn batchnorm
// functions.  For example, cudnn lets you bias and/or scale the input/output,
// but these thunks don't currently support that.

class CudnnBatchNormForwardInferenceThunk : public Thunk {
 public:
  CudnnBatchNormForwardInferenceThunk(const BufferAllocation::Slice& operand,
                                      const BufferAllocation::Slice& scale,
                                      const BufferAllocation::Slice& offset,
                                      const BufferAllocation::Slice& mean,
                                      const BufferAllocation::Slice& variance,
                                      float epsilon, int64 feature_index,
                                      const BufferAllocation::Slice& output,
                                      const HloInstruction* hlo);

  CudnnBatchNormForwardInferenceThunk(
      const CudnnBatchNormForwardInferenceThunk&) = delete;
  CudnnBatchNormForwardInferenceThunk& operator=(
      const CudnnBatchNormForwardInferenceThunk&) = delete;

  Status ExecuteOnStream(const BufferAllocations& buffer_allocations,
                         perftools::gputools::Stream* stream) override;

 private:
  BufferAllocation::Slice operand_;
  BufferAllocation::Slice scale_;
  BufferAllocation::Slice offset_;
  BufferAllocation::Slice mean_;
  BufferAllocation::Slice variance_;
  float epsilon_;
  int64 feature_index_;
  BufferAllocation::Slice output_;
};

class CudnnBatchNormForwardTrainingThunk : public Thunk {
 public:
  CudnnBatchNormForwardTrainingThunk(
      const BufferAllocation::Slice& operand,
      const BufferAllocation::Slice& scale,
      const BufferAllocation::Slice& offset, float epsilon, int64 feature_index,
      const BufferAllocation::Slice& output_data,
      const BufferAllocation::Slice& output_mean,
      const BufferAllocation::Slice& output_inv_stddev,
      const BufferAllocation::Slice& output_tuple, const HloInstruction* hlo);

  CudnnBatchNormForwardTrainingThunk(
      const CudnnBatchNormForwardTrainingThunk&) = delete;
  CudnnBatchNormForwardTrainingThunk& operator=(
      const CudnnBatchNormForwardTrainingThunk&) = delete;

  Status ExecuteOnStream(const BufferAllocations& buffer_allocations,
                         perftools::gputools::Stream* stream) override;

 private:
  BufferAllocation::Slice operand_;
  BufferAllocation::Slice scale_;
  BufferAllocation::Slice offset_;
  float epsilon_;
  int64 feature_index_;
  BufferAllocation::Slice output_data_;
  BufferAllocation::Slice output_mean_;
  BufferAllocation::Slice output_inv_stddev_;
  BufferAllocation::Slice output_tuple_;
};

class CudnnBatchNormBackwardThunk : public Thunk {
 public:
  CudnnBatchNormBackwardThunk(const BufferAllocation::Slice& operand,
                              const BufferAllocation::Slice& scale,
                              const BufferAllocation::Slice& mean,
                              const BufferAllocation::Slice& inv_stddev,
                              const BufferAllocation::Slice& grad_output,
                              float epsilon, int64 feature_index,
                              const BufferAllocation::Slice& output_grad_data,
                              const BufferAllocation::Slice& output_grad_scale,
                              const BufferAllocation::Slice& output_grad_offset,
                              const BufferAllocation::Slice& output_tuple,
                              const HloInstruction* hlo);

  CudnnBatchNormBackwardThunk(const CudnnBatchNormBackwardThunk&) = delete;
  CudnnBatchNormBackwardThunk& operator=(const CudnnBatchNormBackwardThunk&) =
      delete;

  Status ExecuteOnStream(const BufferAllocations& buffer_allocations,
                         perftools::gputools::Stream* stream) override;

 private:
  BufferAllocation::Slice operand_;
  BufferAllocation::Slice scale_;
  BufferAllocation::Slice mean_;
  BufferAllocation::Slice inv_stddev_;
  BufferAllocation::Slice grad_output_;
  float epsilon_;
  int64 feature_index_;
  BufferAllocation::Slice output_grad_data_;
  BufferAllocation::Slice output_grad_scale_;
  BufferAllocation::Slice output_grad_offset_;
  BufferAllocation::Slice output_tuple_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_THUNK_H_
