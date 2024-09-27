/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class DeviceIndexOp : public XlaOpKernel {
 public:
  explicit DeviceIndexOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_names", &device_names_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // When compiling we are not executing on any physical device, so we return
    // a sentinel value (size of the list of devices).
    ctx->SetOutput(
        0, xla::ConstantR0<int32>(ctx->builder(), device_names_.size()));
  }

 private:
  std::vector<string> device_names_;
};

REGISTER_XLA_OP(Name("DeviceIndex"), DeviceIndexOp);

}  // namespace
}  // namespace tensorflow
