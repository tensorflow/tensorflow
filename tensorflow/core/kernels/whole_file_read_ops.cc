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

// See docs in ../ops/io_ops.cc.

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_base.pb.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

static Status ReadEntireFile(Env* env, const string& filename,
                             string* contents) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
  io::RandomAccessInputStream input_stream(file.get());
  io::BufferedInputStream in(&input_stream, 1 << 20);
  TF_RETURN_IF_ERROR(in.ReadAll(contents));
  return Status::OK();
}

class WholeFileReader : public ReaderBase {
 public:
  WholeFileReader(Env* env, const string& node_name)
      : ReaderBase(strings::StrCat("WholeFileReader '", node_name, "'")),
        env_(env) {}

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    *key = current_work();
    TF_RETURN_IF_ERROR(ReadEntireFile(env_, *key, value));
    *produced = true;
    *at_end = true;
    return Status::OK();
  }

  // Stores state in a ReaderBaseState proto, since WholeFileReader has
  // no additional state beyond ReaderBase.
  Status SerializeStateLocked(string* state) override {
    ReaderBaseState base_state;
    SaveBaseState(&base_state);
    base_state.SerializeToString(state);
    return Status::OK();
  }

  Status RestoreStateLocked(const string& state) override {
    ReaderBaseState base_state;
    if (!ParseProtoUnlimited(&base_state, state)) {
      return errors::InvalidArgument("Could not parse state for ", name(), ": ",
                                     str_util::CEscape(state));
    }
    TF_RETURN_IF_ERROR(RestoreBaseState(base_state));
    return Status::OK();
  }

 private:
  Env* env_;
};

class WholeFileReaderOp : public ReaderOpKernel {
 public:
  explicit WholeFileReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory(
        [this, env]() { return new WholeFileReader(env, name()); });
  }
};

REGISTER_KERNEL_BUILDER(Name("WholeFileReader").Device(DEVICE_CPU),
                        WholeFileReaderOp);
REGISTER_KERNEL_BUILDER(Name("WholeFileReaderV2").Device(DEVICE_CPU),
                        WholeFileReaderOp);

class ReadFileOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* input;
    OP_REQUIRES_OK(context, context->input("filename", &input));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                errors::InvalidArgument(
                    "Input filename tensor must be scalar, but had shape: ",
                    input->shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("contents",
                                                     TensorShape({}), &output));
    OP_REQUIRES_OK(context,
                   ReadEntireFile(context->env(), input->scalar<string>()(),
                                  &output->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("ReadFile").Device(DEVICE_CPU), ReadFileOp);

class WriteFileOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* filename_input;
    const Tensor* contents_input;
    OP_REQUIRES_OK(context, context->input("filename", &filename_input));
    OP_REQUIRES_OK(context, context->input("contents", &contents_input));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(filename_input->shape()),
                errors::InvalidArgument(
                    "Input filename tensor must be scalar, but had shape: ",
                    filename_input->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_input->shape()),
                errors::InvalidArgument(
                    "Contents tensor must be scalar, but had shape: ",
                    contents_input->shape().DebugString()));
    const string& filename = filename_input->scalar<string>()();
    const string dir = io::Dirname(filename).ToString();
    if (!context->env()->FileExists(dir).ok()) {
      OP_REQUIRES_OK(context, context->env()->RecursivelyCreateDir(dir));
    }
    OP_REQUIRES_OK(context,
                   WriteStringToFile(context->env(), filename,
                                     contents_input->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("WriteFile").Device(DEVICE_CPU), WriteFileOp);
}  // namespace tensorflow
