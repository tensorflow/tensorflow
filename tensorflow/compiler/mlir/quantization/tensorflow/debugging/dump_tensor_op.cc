/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"

namespace tensorflow {

absl::Status SaveTensorProtoToFile(const Tensor& tensor, std::string file_path,
                                   tsl::Env* env) {
  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);

  std::unique_ptr<tsl::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(file_path, &file));
  absl::Status append_result = file->Append(tensor_proto.SerializeAsString());
  absl::Status close_result = file->Close();

  return append_result.ok() ? close_result : append_result;
}

// DumpTensor op saves entire value of input to as a tensor proto into a
// specified directory and filename. When enabled is set to false, op is
// disabled and won't save any value.
REGISTER_OP("DumpTensor")
    .Input("tensor_data: T")
    .Attr("log_dir_path: string")
    .Attr("file_name: string")
    .Attr("T: type")
    .Attr("enabled: bool")
    .SetIsStateful();

class DumpTensorOp : public OpKernel {
 public:
  explicit DumpTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string log_dir_path;
    string file_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_dir_path", &log_dir_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enabled", &enabled_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("file_name", &file_name));
    OP_REQUIRES_OK(ctx, ctx->env()->RecursivelyCreateDir(log_dir_path));

    tensor_data_path_ = io::JoinPath(log_dir_path, file_name);
  }

  void Compute(OpKernelContext* ctx) override {
    if (enabled_) {
      const Tensor& tensor_data = ctx->input(0);

      OP_REQUIRES_OK(ctx, SaveTensorProtoToFile(tensor_data, tensor_data_path_,
                                                ctx->env()));
    }
  }

 private:
  std::string tensor_data_path_;
  bool enabled_;
};

REGISTER_KERNEL_BUILDER(Name("DumpTensor").Device(DEVICE_CPU), DumpTensorOp);

}  // namespace tensorflow
