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
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel_test_base.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

class BaseKernel : public ::tensorflow::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(::tensorflow::OpKernelContext* context) override {}
  virtual int Which() const = 0;
};

template <int WHICH>
class LabeledKernel : public BaseKernel {
 public:
  using BaseKernel::BaseKernel;
  int Which() const override { return WHICH; }
};

class KernelTest : public OpKernelBuilderTest {
  void SetUp() override { setenv(kDisableJitKernelsEnvVar, "1", 1); }
};

REGISTER_OP("JitKernel");
REGISTER_KERNEL_BUILDER(
    Name("JitKernel").Device(DEVICE_CPU).Label(kJitKernelLabel),
    LabeledKernel<4>);

TEST_F(KernelTest, Filter) {
  ExpectFailure("JitKernel", DEVICE_CPU, {absl::StrCat("_kernel|string|''")},
                error::NOT_FOUND);
  ExpectFailure("JitKernel", DEVICE_CPU,
                {absl::StrCat("_kernel|string|'", kJitKernelLabel, "'")},
                error::NOT_FOUND);
}

}  // namespace
}  // namespace tensorflow
