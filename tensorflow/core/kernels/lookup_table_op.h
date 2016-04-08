/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_
#define TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Lookup table op that supports different table implementations specified by
// the 'Container' template. Container must be derived from LookupInterface. The
// key and value are of the templated type "key_dtype" and "value_dtype"
// respectively.
template <class Container, class key_dtype, class value_dtype>
class LookupTableOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit LookupTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(tensorflow::DT_STRING,
                                                 tensorflow::TensorShape({2}),
                                                 &table_handle_, nullptr));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));
      auto creator = [this](lookup::LookupInterface** ret) {
        *ret = new Container();
        return Status::OK();
      };

      lookup::LookupInterface* table = nullptr;
      OP_REQUIRES_OK(
          ctx, cinfo_.resource_manager()
                   ->template LookupOrCreate<lookup::LookupInterface>(
                       cinfo_.container(), cinfo_.name(), &table, creator));
      core::ScopedUnref unref_me(table);

      OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                              *table, DataTypeToEnum<key_dtype>::v(),
                              DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

      auto h = table_handle_.AccessTensor(ctx)->template flat<string>();
      h(0) = cinfo_.container();
      h(1) = cinfo_.name();
      table_handle_set_ = true;
    }
    ctx->set_output_ref(0, &mu_, table_handle_.AccessTensor(ctx));
  }

  ~LookupTableOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(
          cinfo_.resource_manager()->template Delete<lookup::LookupInterface>(
              cinfo_.container(), cinfo_.name()));
    }
  }

 private:
  mutex mu_;
  PersistentTensor table_handle_ GUARDED_BY(mu_);
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;

  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_
