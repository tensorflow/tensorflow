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

#include "tensorflow/core/framework/reader_op_kernel.h"

namespace tensorflow {

ReaderOpKernel::ReaderOpKernel(OpKernelConstruction* context)
    : OpKernel(context), have_handle_(false) {
  OP_REQUIRES_OK(context, context->allocate_persistent(
                              tensorflow::DT_STRING,
                              tensorflow::TensorShape({2}), &handle_, nullptr));
}

ReaderOpKernel::~ReaderOpKernel() {
  if (have_handle_ && cinfo_.resource_is_private_to_kernel()) {
    TF_CHECK_OK(cinfo_.resource_manager()->Delete<ReaderInterface>(
        cinfo_.container(), cinfo_.name()));
  }
}

void ReaderOpKernel::Compute(OpKernelContext* ctx) {
  mutex_lock l(mu_);
  if (!have_handle_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(), false));
    ReaderInterface* reader;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()->LookupOrCreate<ReaderInterface>(
                       cinfo_.container(), cinfo_.name(), &reader,
                       [this](ReaderInterface** ret) {
                         *ret = factory_();
                         return Status::OK();
                       }));
    reader->Unref();
    auto h = handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    have_handle_ = true;
  }
  ctx->set_output_ref(0, &mu_, handle_.AccessTensor(ctx));
}

}  // namespace tensorflow
