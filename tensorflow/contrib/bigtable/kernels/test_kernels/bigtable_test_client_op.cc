/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/bigtable/kernels/bigtable_lib.h"
#include "tensorflow/contrib/bigtable/kernels/test_kernels/bigtable_test_client.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

namespace {

class BigtableTestClientOp : public OpKernel {
 public:
  explicit BigtableTestClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~BigtableTestClientOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<BigtableClientResource>(cinfo_.container(),
                                                cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }
  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));
      BigtableClientResource* resource;
      OP_REQUIRES_OK(ctx,
                     mgr->LookupOrCreate<BigtableClientResource>(
                         cinfo_.container(), cinfo_.name(), &resource,
                         [this, ctx](BigtableClientResource** ret)
                             EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                               std::shared_ptr<bigtable::DataClient> client(
                                   new BigtableTestClient());
                               // Note: must make explicit copies to sequence
                               // them before the move of client.
                               string project_id = client->project_id();
                               string instance_id = client->instance_id();
                               *ret = new BigtableClientResource(
                                   std::move(project_id),
                                   std::move(instance_id), std::move(client));
                               return Status::OK();
                             }));
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<BigtableClientResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTestClient").Device(DEVICE_CPU),
                        BigtableTestClientOp);

}  // namespace
}  // namespace tensorflow
