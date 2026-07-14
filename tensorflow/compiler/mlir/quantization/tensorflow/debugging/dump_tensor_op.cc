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
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

absl::Status SaveSerializedProtoToFile(const absl::string_view serialized_proto,
                                       const absl::string_view file_path,
                                       tsl::Env* env) {
  std::unique_ptr<tsl::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(std::string(file_path), &file));
  absl::Status append_result = file->Append(serialized_proto);
  absl::Status close_result = file->Close();

  return append_result.ok() ? close_result : append_result;
}

// `DumpTensor` op saves entire value of input to as a tensor proto into a
// specified directory and filename. When enabled is set to false, op is
// disabled and won't save any value. It also creates `QuantizationUnit` proto
// with `func_name` and `node_name` to identify the op.
REGISTER_OP("DumpTensor")
    .Input("tensor_data: T")
    .Attr("log_dir_path: string")
    .Attr("file_name: string")
    .Attr("T: type")
    .Attr("enabled: bool")
    .Attr("func_name: string")
    .Attr("node_name: string")
    .SetIsStateful();

class DumpTensorOp : public OpKernel {
 public:
  explicit DumpTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string log_dir_path;
    string file_name;
    string func_name;
    string node_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("log_dir_path", &log_dir_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("enabled", &enabled_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("file_name", &file_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("func_name", &func_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("node_name", &node_name));
    OP_REQUIRES_OK(ctx, ctx->env()->RecursivelyCreateDir(log_dir_path));

    std::string tensor_data_path = io::JoinPath(log_dir_path, file_name);
    OP_REQUIRES_OK(
        ctx, ctx->env()->NewWritableFile(tensor_data_path, &tensor_data_file_));

    // Turn on Zlib compression.
    io::RecordWriterOptions options =
        io::RecordWriterOptions::CreateRecordWriterOptions(
            io::compression::kZlib);
    tensor_data_writer_ =
        std::make_unique<io::RecordWriter>(tensor_data_file_.get(), options);
    OP_REQUIRES(ctx, tensor_data_writer_ != nullptr,
                absl::AbortedError("Could not create record writer"));

    // Fetch func_name and node_name from attributes and save as proto.
    quantization::UnitWiseQuantizationSpec::QuantizationUnit quant_unit_proto;
    quant_unit_proto.set_func_name(func_name);
    quant_unit_proto.set_node_name(node_name);

    string quant_unit_path = io::JoinPath(log_dir_path, "quant_unit.pb");
    OP_REQUIRES_OK(
        ctx, SaveSerializedProtoToFile(quant_unit_proto.SerializeAsString(),
                                       quant_unit_path, ctx->env()));
  }

  ~DumpTensorOp() override {
    (void)tensor_data_writer_->Flush();
    (void)tensor_data_writer_->Close();
    (void)tensor_data_file_->Close();
  }

  void Compute(OpKernelContext* ctx) override {
    if (!enabled_) return;

    const Tensor& tensor_data = ctx->input(0);

    TensorProto tensor_proto;
    tensor_data.AsProtoTensorContent(&tensor_proto);

    OP_REQUIRES_OK(ctx, tensor_data_writer_->WriteRecord(
                            tensor_proto.SerializeAsString()));
  }

 private:
  bool enabled_;
  std::unique_ptr<tsl::WritableFile> tensor_data_file_;
  std::unique_ptr<io::RecordWriter> tensor_data_writer_;
};

REGISTER_KERNEL_BUILDER(Name("DumpTensor").Device(DEVICE_CPU), DumpTensorOp);

}  // namespace tensorflow
