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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_MAP_DEFUN_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_MAP_DEFUN_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// This op runs a given defun on slices of the input arguments. The function
// given by "f" is assumed to be stateless, and is executed concurrently
// on all the slices; up to batch_size (i.e. the 0th dimension of each argument)
// functions will be scheduled at once.
//
// The "max_intra_op_parallelism" attr, which defaults to 1, can be used to
// limit the intra op parallelism. To limit inter-op parallelism, a user
// can set a private threadpool on the dataset using `tf.data.Options`'s
// `ThreadingOptions`.
//
// Note that this op is not exposed to users directly, but is invoked in
// tf.data rewrites.
class MapDefunOp : public AsyncOpKernel {
 public:
  static constexpr const char* const kArguments = "arguments";
  static constexpr const char* const kCapturedInputs = "captured_inputs";
  static constexpr const char* const kTarguments = "Targuments";
  static constexpr const char* const kTcaptured = "Tcaptured";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kFunc = "f";
  static constexpr const char* const kMaxIntraOpParallelism =
      "max_intra_op_parallelism";

  explicit MapDefunOp(OpKernelConstruction* ctx);

  ~MapDefunOp() override = default;

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  struct ComputeOptions;
  class MapFunctionCallFrame;

  void SetRunOptions(OpKernelContext* ctx,
                     FunctionLibraryRuntime::Options* opts,
                     ComputeOptions* compute_opts, bool always_collect_stats);

  // Get inputs to Compute and check that they are valid.
  Status SetupArgs(OpKernelContext* ctx, ComputeOptions** compute_opts);

  Status SetupOutputs(OpKernelContext* ctx, ComputeOptions* opts);

  FunctionLibraryRuntime::Handle func_handle_;
  std::vector<PartialTensorShape> output_shapes_;
  // If this value is positive, limit the max intra op parallelism when the
  // function is run on slices of the input.
  int max_intra_op_parallelism_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_MAP_DEFUN_DATASET_OP_H_
