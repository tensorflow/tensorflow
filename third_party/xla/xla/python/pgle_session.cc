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

#include "xla/python/pgle_session.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/aggregate_profile.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/profiler_session.h"

namespace xla {
std::unique_ptr<tsl::ProfilerSession> PGLESession::Trace() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return tsl::ProfilerSession::Create(options);
}

void PGLESession::StopTrace(
    std::unique_ptr<tsl::ProfilerSession> profiler_session) {
  tensorflow::profiler::XSpace xspace;
  // Disables the ProfilerSession
  xla::ThrowIfError(profiler_session->CollectData(&xspace));
  tensorflow::profiler::ProfiledInstructionsProto fdo_profile;
  xla::ThrowIfError(
      xla::ConvertXplaneToProfiledInstructionsProto({xspace}, &fdo_profile));
  fdo_profiles_.push_back(std::move(fdo_profile));
}

absl::StatusOr<std::string> PGLESession::GetFdoProfile(
    xla::PjRtClient* pjrt_client, const CompileOptions& compile_options) const {
  tensorflow::profiler::ProfiledInstructionsProto result_proto;
  xla::AggregateProfiledInstructionsProto(fdo_profiles_, &result_proto);
  TF_ASSIGN_OR_RETURN(
      auto aggregated_between_hosts_cost,
      AggregateFdoProfileCrossHost(result_proto, pjrt_client, compile_options));
  return aggregated_between_hosts_cost.SerializeAsString();
}

absl::StatusOr<tensorflow::profiler::ProfiledInstructionsProto>
PGLESession::AggregateFdoProfileCrossHost(
    tensorflow::profiler::ProfiledInstructionsProto fdo_profile,
    xla::PjRtClient* pjrt_client, const CompileOptions& compile_options) const {
  // Store the fdo_profile name and cost.
  std::vector<std::string> fdo_profile_names;
  std::vector<float> fdo_profile_costs;
  for (const auto& cost : fdo_profile.costs()) {
    fdo_profile_names.push_back(cost.name());
    fdo_profile_costs.push_back(cost.cost_us());
  }
  int64_t fdo_profile_count = fdo_profile_costs.size();

  // Build AllReduce across all the replicas.
  XlaBuilder reduction_builder("reduction");
  XlaComputation add_func = CreateScalarAddComputation(F32, &reduction_builder);
  XlaBuilder builder("reduce");
  Shape shape = ShapeUtil::MakeShape(F32, {fdo_profile_count});
  auto inp = Parameter(&builder, 0, shape, "input");
  auto all_reduce = AllReduce(inp, add_func);

  // Compile the AllReduce computation.
  TF_ASSIGN_OR_RETURN(auto computation, builder.Build(all_reduce));
  TF_ASSIGN_OR_RETURN(auto executable,
                      pjrt_client->Compile(computation, compile_options));

  // Executable the AllReduce computation.
  absl::Span<xla::PjRtDevice* const> devices =
      executable->addressable_devices();

  std::vector<std::unique_ptr<PjRtBuffer>> devices_buffers;
  int num_replicas = devices.size();
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.resize(num_replicas);
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results;
  results.resize(num_replicas);
  ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  for (int replica = 0; replica < num_replicas; ++replica) {
    TF_ASSIGN_OR_RETURN(
        buffers[replica],
        pjrt_client->BufferFromHostBuffer(
            fdo_profile_costs.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kZeroCopy, nullptr,
            devices[replica]));

    TF_ASSIGN_OR_RETURN(
        results[replica],
        executable->ExecuteSharded({buffers[replica].get()}, devices[replica],
                                   execute_options));
  }

  // Get the result.
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> cost_literal,
                      results[0][0]->ToLiteralSync());

  tensorflow::profiler::ProfiledInstructionsProto result_profile;
  for (int i = 0; i < fdo_profile_count; ++i) {
    auto* cost = result_profile.add_costs();
    cost->set_name(fdo_profile_names[i]);
    cost->set_cost_us(cost_literal->data<float>()[i]);
  }

  return result_profile;
}

PGLESessionRunner::PGLESessionHandler::PGLESessionHandler(
    PGLESession& pgle_session)
    : pgle_session_(pgle_session) {
  profiler_session_ = pgle_session_.Trace();
}

PGLESessionRunner::PGLESessionHandler::~PGLESessionHandler() {
  pgle_session_.StopTrace(std::move(profiler_session_));
}

std::optional<PGLESessionRunner::PGLESessionHandler> PGLESessionRunner::Run(
    int process_index) {
  if (pgle_data_collecting_retries_ <= 0 ||
      call_times_ > pgle_data_collecting_retries_) {
    return std::nullopt;
  }

  call_times_++;
  // On multi-host environment collect only on the first host.
  if (distributed_client_ != nullptr && process_index != 0) {
    return std::nullopt;
  }

  return std::optional<PGLESessionRunner::PGLESessionHandler>(pgle_session_);
}

absl::StatusOr<std::optional<std::string>> PGLESessionRunner::GetFdoProfile(
    xla::PjRtClient* pjrt_client, const CompileOptions& options) {
  if (collected_fdo_.has_value()) {
    return collected_fdo_;
  }

  if (pgle_data_collecting_retries_ <= 0 ||
      call_times_ != pgle_data_collecting_retries_) {
    return std::nullopt;
  }

  if (distributed_client_ == nullptr) {
    return pgle_session_.GetFdoProfile(pjrt_client, options);
  }
  std::string fdo_profile;
  if (pjrt_client->process_index() == 0) {
    LOG(INFO) << "Collecting fdo profile: " << distributed_data_key_;
    TF_ASSIGN_OR_RETURN(fdo_profile,
                        pgle_session_.GetFdoProfile(pjrt_client, options));

    TF_RETURN_IF_ERROR(
        distributed_client_->KeyValueSet(distributed_data_key_, fdo_profile));
  } else {
    LOG(INFO) << "Waiting for fdo profile: " << distributed_data_key_;
    TF_ASSIGN_OR_RETURN(fdo_profile,
                        distributed_client_->BlockingKeyValueGet(
                            distributed_data_key_, absl::Minutes(20)));
  }

  collected_fdo_ = fdo_profile;
  return fdo_profile;
}

std::shared_ptr<PGLESessionRunner> PGLESessionRunnerFactory::Create(
    std::optional<int64_t> pgle_data_collecting_retries,
    std::optional<std::string> module_fingerprint,
    std::shared_ptr<xla::DistributedRuntimeClient> distributed_client) {
  if (!pgle_data_collecting_retries.has_value() ||
      !module_fingerprint.has_value()) {
    return empty_runner_;
  }

  // Amount of retries is a part of key to support chenging the value during the
  // task.
  std::string session_key =
      module_fingerprint.value() +
      std::to_string(pgle_data_collecting_retries.value());
  if (!module_to_session_runner_.contains(session_key)) {
    std::string distributed_data_key = session_key + "_fdo_profile";
    module_to_session_runner_.insert(
        {session_key, std::make_shared<PGLESessionRunner>(
                          pgle_data_collecting_retries.value(),
                          distributed_data_key, distributed_client)});
  }

  return module_to_session_runner_[session_key];
}
}  // namespace xla
