/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/logging_ops.h"

#include <iostream>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {

namespace {

// If the following string is found at the beginning of an output stream, it
// will be interpreted as a file path.
const char kOutputStreamEscapeStr[] = "file://";

// A mutex that guards appending strings to files.
static mutex* file_mutex = new mutex();

// Appends the given data to the specified file. It will create the file if it
// doesn't already exist.
Status AppendStringToFile(const std::string& fname, StringPiece data,
                          Env* env) {
  // TODO(ckluk): If opening and closing on every log causes performance issues,
  // we can reimplement using reference counters.
  mutex_lock l(*file_mutex);
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewAppendableFile(fname, &file));
  Status a = file->Append(data);
  Status c = file->Close();
  return a.ok() ? c : a;
}

}  // namespace

AssertOp::AssertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
}

void AssertOp::Compute(OpKernelContext* ctx) {
  const Tensor& cond = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(cond.shape()),
              errors::InvalidArgument("In[0] should be a scalar: ",
                                      cond.shape().DebugString()));

  if (cond.scalar<bool>()()) {
    return;
  }
  string msg = "assertion failed: ";
  for (int i = 1; i < ctx->num_inputs(); ++i) {
    strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                       "]");
    if (i < ctx->num_inputs() - 1) strings::StrAppend(&msg, " ");
  }
  ctx->SetStatus(errors::InvalidArgument(msg));
}

REGISTER_KERNEL_BUILDER(Name("Assert")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("condition")
                            .HostMemory("data"),
                        AssertOp);

class PrintOp : public OpKernel {
 public:
  explicit PrintOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), call_counter_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
    if (first_n_ >= 0) {
      mutex_lock l(mu_);
      if (call_counter_ >= first_n_) return;
      call_counter_++;
    }
    string msg;
    strings::StrAppend(&msg, message_);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                         "]");
    }
    std::cerr << msg << std::endl;
  }

 private:
  mutex mu_;
  int64_t call_counter_ TF_GUARDED_BY(mu_) = 0;
  int64_t first_n_ = 0;
  int32 summarize_ = 0;
  string message_;
};

REGISTER_KERNEL_BUILDER(Name("Print").Device(DEVICE_CPU), PrintOp);

class PrintV2Op : public OpKernel {
 public:
  explicit PrintV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_stream", &output_stream_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end", &end_));

    SetFilePathIfAny();
    if (!file_path_.empty()) return;

    auto output_stream_index =
        std::find(std::begin(valid_output_streams_),
                  std::end(valid_output_streams_), output_stream_);

    if (output_stream_index == std::end(valid_output_streams_)) {
      string error_msg = strings::StrCat(
          "Unknown output stream: ", output_stream_, ", Valid streams are:");
      for (auto valid_stream : valid_output_streams_) {
        strings::StrAppend(&error_msg, " ", valid_stream);
      }
      OP_REQUIRES(ctx, false, errors::InvalidArgument(error_msg));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(input_->shape()),
        errors::InvalidArgument("Input is expected to be scalar, but got ",
                                input_->shape()));
    const string& msg = input_->scalar<tstring>()();

    string ended_msg = strings::StrCat(msg, end_);

    if (!file_path_.empty()) {
      // Outputs to a file at the specified path.
      OP_REQUIRES_OK(ctx,
                     AppendStringToFile(file_path_, ended_msg, ctx->env()));
      return;
    }

    if (logging::LogToListeners(ended_msg, "")) {
      return;
    }

    if (output_stream_ == "stdout") {
      std::cout << ended_msg << std::flush;
    } else if (output_stream_ == "stderr") {
      std::cerr << ended_msg << std::flush;
    } else if (output_stream_ == "log(info)") {
      LOG(INFO) << ended_msg << std::flush;
    } else if (output_stream_ == "log(warning)") {
      LOG(WARNING) << ended_msg << std::flush;
    } else if (output_stream_ == "log(error)") {
      LOG(ERROR) << ended_msg << std::flush;
    } else {
      string error_msg = strings::StrCat(
          "Unknown output stream: ", output_stream_, ", Valid streams are:");
      for (auto valid_stream : valid_output_streams_) {
        strings::StrAppend(&error_msg, " ", valid_stream);
      }
      strings::StrAppend(&error_msg, ", or file://<filename>");
      OP_REQUIRES(ctx, false, errors::InvalidArgument(error_msg));
    }
  }

  const char* valid_output_streams_[5] = {"stdout", "stderr", "log(info)",
                                          "log(warning)", "log(error)"};

 private:
  string end_;
  // Either output_stream_ or file_path_ (but not both) will be non-empty.
  string output_stream_;
  string file_path_;

  // If output_stream_ is a file path, extracts it to file_path_ and clears
  // output_stream_; otherwise sets file_paths_ to "".
  void SetFilePathIfAny() {
    if (absl::StartsWith(output_stream_, kOutputStreamEscapeStr)) {
      file_path_ = output_stream_.substr(strlen(kOutputStreamEscapeStr));
      output_stream_ = "";
    } else {
      file_path_ = "";
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PrintV2").Device(DEVICE_CPU), PrintV2Op);

class TimestampOp : public OpKernel {
 public:
  explicit TimestampOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, !OpDeterminismRequired(),
                errors::FailedPrecondition(
                    "Timestamp cannot be called when determinism is enabled"));
    TensorShape output_shape;  // Default shape is 0 dim, 1 element
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    auto output_scalar = output_tensor->scalar<double>();
    double now_us = static_cast<double>(Env::Default()->NowMicros());
    double now_s = now_us / 1000000;
    output_scalar() = now_s;
  }
};

REGISTER_KERNEL_BUILDER(Name("Timestamp").Device(DEVICE_CPU), TimestampOp);

}  // end namespace tensorflow
