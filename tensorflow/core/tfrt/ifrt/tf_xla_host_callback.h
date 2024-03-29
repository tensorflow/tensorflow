/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_TF_XLA_HOST_CALLBACK_H_
#define TENSORFLOW_CORE_TFRT_IFRT_TF_XLA_HOST_CALLBACK_H_

#include <any>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/runtime/step_id.h"

namespace tensorflow {
namespace ifrt_serving {

// A host callback implementation to run a TF graph within XLA computation.
class TfXlaHostCallback {
 public:
  static absl::StatusOr<std::unique_ptr<TfXlaHostCallback>> CreateCallback(
      const tensorflow::ConfigProto& session_config,
      absl::Span<const tensorflow::FunctionDef> functions,
      absl::string_view entry_function_name,
      std::shared_ptr<tensorflow::StaticDeviceMgr> device_mgr);

  absl::Status Call(
      tensorflow::tfrt_stub::StepId client_step_id,
      absl::Span<const std::unique_ptr<xla::LiteralBase>> operands,
      absl::Span<const std::unique_ptr<xla::MutableLiteralBase>> results);

 private:
  TfXlaHostCallback(absl::string_view entry_function_name,
                    std::shared_ptr<tensorflow::StaticDeviceMgr> device_mgr,
                    tensorflow::EagerContextPtr ctx)
      : device_mgr_(device_mgr),
        ctx_(std::move(ctx)),
        entry_function_name_(entry_function_name) {}

  // DeviceMgr shared across one or more host callbacks. Stored here to keep it
  // alive until the host callback is deallocated.
  std::shared_ptr<tensorflow::StaticDeviceMgr> device_mgr_;

  // Per-callback TF Eager context.
  tensorflow::EagerContextPtr ctx_;

  // Entry function name to be called on invocation. Specified by
  // `XlaHostCallbackProto.TfCallback`.
  std::string entry_function_name_;
};

absl::StatusOr<std::shared_ptr<tensorflow::StaticDeviceMgr>>
CreateTfStaticDeviceMgr();

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_TF_XLA_HOST_CALLBACK_H_
