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

#include "tensorflow/core/kernels/gpu_utils.h"

#if GOOGLE_CUDA

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/protobuf/conv_autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {
namespace {

tensorflow::CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
  tensorflow::CudnnVersion cudnn_version;
  if (auto* dnn = stream_executor->AsDnn()) {
    se::port::StatusOr<se::dnn::VersionInfo> version_or = dnn->GetVersion();
    if (version_or.ok()) {
      const auto& version = version_or.ValueOrDie();
      cudnn_version.set_major(version.major_version());
      cudnn_version.set_minor(version.minor_version());
      cudnn_version.set_patch(version.patch());
    }
  }
  return cudnn_version;
}

tensorflow::ComputeCapability GetComputeCapability(
    se::StreamExecutor* stream_executor) {
  tensorflow::ComputeCapability cc;
  int cc_major, cc_minor;
  stream_executor->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                  &cc_minor);
  cc.set_major(cc_major);
  cc.set_minor(cc_minor);
  return cc;
}

}  // namespace

void LogConvAutotuneResults(const NodeDef& node, const Tensor& input,
                            const Tensor& filter, const Tensor& output,
                            se::StreamExecutor* stream_exec,
                            absl::Span<const AutotuneResult> results) {
  AutotuningLog log;
  ConvNodeDef instr;
  *instr.mutable_conv() = node;
  input.shape().AsProto(instr.mutable_input()->mutable_tensor_shape());
  instr.mutable_input()->set_dtype(input.dtype());
  filter.shape().AsProto(instr.mutable_filter()->mutable_tensor_shape());
  instr.mutable_filter()->set_dtype(filter.dtype());
  output.shape().AsProto(instr.mutable_output()->mutable_tensor_shape());
  instr.mutable_output()->set_dtype(output.dtype());
  log.mutable_instr()->PackFrom(std::move(instr));
  *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
  *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
  log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
  for (const auto& result : results) {
    *log.add_results() = result;
  }
  Logger::Singleton()->LogProto(log);
}

void LogFusedConvAutotuneResults(const NodeDef& node, const Tensor& input,
                                 const Tensor& filter, const Tensor& output,
                                 const Tensor& bias, const Tensor* side_input,
                                 se::StreamExecutor* stream_exec,
                                 absl::Span<const AutotuneResult> results) {
  AutotuningLog log;
  ConvNodeDef instr;
  *instr.mutable_conv() = node;
  input.shape().AsProto(instr.mutable_input()->mutable_tensor_shape());
  instr.mutable_input()->set_dtype(input.dtype());
  filter.shape().AsProto(instr.mutable_filter()->mutable_tensor_shape());
  instr.mutable_filter()->set_dtype(filter.dtype());
  output.shape().AsProto(instr.mutable_output()->mutable_tensor_shape());
  instr.mutable_output()->set_dtype(output.dtype());
  bias.shape().AsProto(instr.mutable_bias()->mutable_tensor_shape());
  instr.mutable_bias()->set_dtype(bias.dtype());
  if (side_input) {
    side_input->shape().AsProto(
        instr.mutable_side_input()->mutable_tensor_shape());
    instr.mutable_side_input()->set_dtype(side_input->dtype());
  }
  log.mutable_instr()->PackFrom(std::move(instr));
  *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
  *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
  log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
  for (const auto& result : results) {
    *log.add_results() = result;
  }
  Logger::Singleton()->LogProto(log);
}

Status BestCudnnConvAlgorithm(absl::Span<const AutotuneResult> results,
                              se::dnn::AlgorithmConfig* algo) {
  // TODO(jlebar): Exclude conv ops with failures, once we have failure checking
  // and have confidence that it's correct.

  const AutotuneResult* best_result = std::min_element(
      results.begin(), results.end(),
      [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return proto_utils::FromDurationProto(lhs.run_time()) <
               proto_utils::FromDurationProto(rhs.run_time());
      });

  const AutotuneResult* best_result_no_scratch = std::min_element(
      results.begin(), results.end(),
      [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return std::make_tuple(lhs.scratch_bytes(),
                               proto_utils::FromDurationProto(lhs.run_time())) <
               std::make_tuple(rhs.scratch_bytes(),
                               proto_utils::FromDurationProto(rhs.run_time()));
      });

  if (best_result == results.end()) {
    return errors::NotFound("No algorithm worked!");
  }
  algo->set_algorithm({best_result->conv().algorithm(),
                       best_result->conv().tensor_ops_enabled()});
  if (best_result_no_scratch != results.end() &&
      best_result_no_scratch->scratch_bytes() == 0) {
    algo->set_algorithm_no_scratch(
        {best_result_no_scratch->conv().algorithm(),
         best_result_no_scratch->conv().tensor_ops_enabled()});
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
