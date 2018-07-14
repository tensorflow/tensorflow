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
#ifdef TENSORFLOW_USE_ROCM
#ifndef TENSORFLOW_RTG_LAUNCH_OP_
#define TENSORFLOW_RTG_LAUNCH_OP_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/core/graph/rocm/attr_to_rtg.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace tensorflow {

// RTGLaunchOp is used to replace a region of the TensorFlow graph
// which will be executed using RTG lib.  The RTGLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// 
class RTGLaunchOp : public OpKernel {
 public:
  explicit RTGLaunchOp(OpKernelConstruction* ctx);
  ~RTGLaunchOp() override;

  void Compute(OpKernelContext* ctx) override;
 private:
  void * program;
  int required_bytes;
  TF_DISALLOW_COPY_AND_ASSIGN(RTGLaunchOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_RTG_LAUNCH_OP_
#endif // TENSORFLOW_USE_ROCM
