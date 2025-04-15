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
#include "tensorflow/core/data/test_utils.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace data {

absl::StatusOr<std::unique_ptr<TestContext>> TestContext::Create() {
  auto ctx = std::unique_ptr<TestContext>(new TestContext());
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", 1});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  ctx->device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
  FunctionDefLibrary proto;
  ctx->lib_def_ =
      std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  ctx->pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
      ctx->device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, ctx->lib_def_.get(), opts);
  ctx->runner_ = [](const std::function<void()>& fn) { fn(); };
  ctx->params_.function_library = ctx->pflr_->GetFLR("/device:CPU:0");
  ctx->params_.device = ctx->device_mgr_->ListDevices()[0];
  ctx->params_.runner = &ctx->runner_;
  ctx->op_ctx_ = std::make_unique<OpKernelContext>(&ctx->params_, 0);
  ctx->iter_ctx_ = std::make_unique<IteratorContext>(ctx->op_ctx_.get());
  return ctx;
}

}  // namespace data
}  // namespace tensorflow
