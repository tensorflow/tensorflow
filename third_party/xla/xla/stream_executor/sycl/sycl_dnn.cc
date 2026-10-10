/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_dnn.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dnnl.hpp"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {

OnednnSupport::OnednnSupport(StreamExecutor* parent) : parent_(parent) {}

absl::Status OnednnSupport::Init() { return absl::OkStatus(); }

absl::StatusOr<dnn::VersionInfo> OnednnSupport::GetOnednnVersion() {
  const dnnl_version_t* v = dnnl::version();
  if (v == nullptr) {
    return absl::InternalError("Failed to query oneDNN version.");
  }
  return dnn::VersionInfo(v->major, v->minor, v->patch);
}

absl::StatusOr<dnn::VersionInfo> OnednnSupport::GetVersion() {
  return GetOnednnVersion();
}

namespace {
class OnednnConvRunner : public dnn::ConvRunner {
 public:
  OnednnConvRunner(OneDnnConvPrimitiveDesc onednn_conv_primitive_desc,
                   size_t workspace_size)
      : onednn_conv_primitive_desc_(std::move(onednn_conv_primitive_desc)),
        workspace_size_(workspace_size) {}
  std::string ToString() const override { return "OnednnConvRunner"; }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return dnn::AlgorithmDesc(-1, false, workspace_size_);
  }

  absl::Status operator()(Stream* stream,
                          dnn::ProfileResult* output_profile_result,
                          DeviceAddressBase scratch_memory,
                          DeviceAddressBase input_data,
                          DeviceAddressBase filter_data,
                          DeviceAddressBase output_data) const override {
    // Implemented as part of a follow-up PR.
    return absl::UnimplementedError(
        "OnednnConvRunner operator() is not implemented for SYCL");
  }

 private:
  OneDnnConvPrimitiveDesc onednn_conv_primitive_desc_;
  size_t workspace_size_ = 0;
};

class OnednnFusedConvRunner : public dnn::FusedConvRunner {
 public:
  OnednnFusedConvRunner(OneDnnConvPrimitiveDesc onednn_conv_primitive_desc,
                        size_t workspace_size)
      : onednn_conv_primitive_desc_(std::move(onednn_conv_primitive_desc)),
        workspace_size_(workspace_size) {}
  std::string ToString() const override { return "OnednnFusedConvRunner"; }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return dnn::AlgorithmDesc(-1, false, workspace_size_);
  }

  absl::Status operator()(Stream* stream,
                          dnn::ProfileResult* output_profile_result,
                          DeviceAddressBase scratch_memory,
                          DeviceAddressBase input_data,
                          DeviceAddressBase filter_data,
                          DeviceAddressBase side_input_data,
                          DeviceAddressBase bias_data,
                          DeviceAddressBase output_data) const override {
    // Implemented as part of a follow-up PR.
    return absl::UnimplementedError(
        "OnednnFusedConvRunner operator() is not implemented for SYCL");
  }

 private:
  OneDnnConvPrimitiveDesc onednn_conv_primitive_desc_;
  size_t workspace_size_ = 0;
};
}  // namespace

absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
OnednnSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  ABSL_ASSIGN_OR_RETURN(
      OneDnnConvPrimitiveDesc primitive_desc,
      CreateOneDnnConvPrimitiveDesc(
          OneDnnConvConfig{kind, input_type, output_type, input_descriptor,
                           filter_descriptor, output_descriptor,
                           convolution_descriptor},
          stream));
  size_t workspace_size = 0;
  return std::make_unique<OnednnConvRunner>(std::move(primitive_desc),
                                            workspace_size);
}

absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
OnednnSupport::FusedConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::ActivationMode activation_mode) {
  ABSL_ASSIGN_OR_RETURN(
      OneDnnConvPrimitiveDesc primitive_desc,
      CreateOneDnnConvPrimitiveDesc(
          OneDnnConvConfig{
              kind, element_type, output_type, input_descriptor,
              filter_descriptor, output_descriptor, convolution_descriptor,
              conv_scale,
              OneDnnConvConfig::Fusion{activation_mode, side_input_scale,
                                       leakyrelu_alpha}},
          stream));
  size_t workspace_size = 0;
  return std::make_unique<OnednnFusedConvRunner>(std::move(primitive_desc),
                                                 workspace_size);
}

void initialize_onednn() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          sycl::kSyclPlatformId, "oneDNN",
          [](StreamExecutor* parent) -> dnn::DnnSupport* {
            sycl::OnednnSupport* dnn = new sycl::OnednnSupport(parent);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register oneDNN factory: " << status.message();
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_onednn, {
  stream_executor::sycl::initialize_onednn();
});
