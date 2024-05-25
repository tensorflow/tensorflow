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

#include <gtest/gtest.h>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_options.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tsl/framework/device_type.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

TEST(PjrtGpuClientCreateTest, TestGpuCreateOption) {
  PjrtClientFactoryOptions options = PjrtClientFactoryOptions();
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, xla::PjrtClientFactoryRegistry::Get().GetPjrtClient(
                       tsl::DeviceType(tensorflow::DEVICE_GPU), options));
}

}  // namespace

}  // namespace xla
