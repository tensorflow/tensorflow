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

#include "tensorflow/compiler/xla/service/gpu/hlo_algorithm_denylist.h"

#include <cstdlib>
#include <string>

#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/resource_loader.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class DenylistTest : public testing::Test {
 protected:
  DenylistTest() {
    std::string existing_xla_flags;
    const char* env = std::getenv("XLA_FLAGS");
    if (env != nullptr) {
      existing_xla_flags = absl::StrCat(env, " ");
    }

    tsl::setenv(
        "XLA_FLAGS",
        absl::StrCat(existing_xla_flags, "--xla_gpu_algorithm_denylist_path=",
                     tsl::GetDataDependencyFilepath(tsl::io::JoinPath(
                         "tensorflow", "compiler", "xla", "service", "gpu",
                         "data", "hlo_algorithm_denylist.pbtxt")))
            .data(),
        /*overwrite=*/true);
  }
};

TEST_F(DenylistTest, DefaultTest) {
  tensorflow::ComputeCapability cc;
  cc.set_major(7);
  cc.set_minor(0);
  tensorflow::CudnnVersion cudnn_version;
  cudnn_version.set_major(7);
  cudnn_version.set_minor(6);
  cudnn_version.set_patch(2);
  auto list = GetDisabledConvAlgorithms(
      cc, cudnn_version, /*blas_version=*/"9000",
      R"((f16[256,112,112,64]{3,2,1,0}, u8[0]{0}) custom-call(f16[256,224,224,4]{3,2,1,0}, f16[7,7,4,64]{2,1,0,3}), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}")");
  ASSERT_EQ(4, list.size());
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(0, false), list[0]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(0, true), list[1]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(1, false), list[2]);
  EXPECT_EQ(stream_executor::dnn::AlgorithmDesc(1, true), list[3]);
}

TEST_F(DenylistTest, NegativeTest) {
  tensorflow::ComputeCapability cc;
  cc.set_major(7);
  cc.set_minor(0);
  tensorflow::CudnnVersion cudnn_version;
  cudnn_version.set_major(7);
  cudnn_version.set_minor(6);
  cudnn_version.set_minor(2);
  auto list =
      GetDisabledConvAlgorithms(cc, cudnn_version, "9000", R"(invalid hlo)");
  ASSERT_EQ(0, list.size());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
