/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/conv_algorithm_picker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_autotuning.pb.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/hlo_algorithm_denylist.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/env_var.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "third_party/gpus/cudnn/cudnn.h"  // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn_version.h"
#if CUDNN_VERSION >= 90000
#include "third_party/gpus/cudnn/cudnn_ops.h"
#else
#include "third_party/gpus/cudnn/cudnn_ops_infer.h"
#endif  // CUDNN_VERSION >= 90000
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {
namespace {

using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
using std::optional;

class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal,
                   se::DeviceMemoryAllocator* memory_allocator)
      : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

  int64_t GetMemoryLimitInBytes() override {
    return ScratchAllocator::GetDefaultMemoryLimitInBytes();
  }
  int64_t TotalAllocatedBytes() { return total_allocated_bytes_; }

  static int64_t GetDefaultMemoryLimitInBytes() {
    int64_t value;
    TF_CHECK_OK(tsl::ReadInt64FromEnvVar("TF_CUDNN_WORKSPACE_LIMIT_IN_MB",
                                         1LL << 12, &value));
    return value * (1LL << 20);
  }

  absl::StatusOr<se::DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size) override;

  template <typename T>
  absl::StatusOr<se::DeviceMemory<T>> Allocate(int64_t num_elements) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemory<uint8_t> bytes,
                        AllocateBytes(num_elements * sizeof(T)));
    return se::DeviceMemory<T>(bytes);
  }

 private:
  const int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;
  std::vector<se::OwningDeviceMemory> allocated_buffers_;
  int64_t total_allocated_bytes_ = 0;
};

absl::StatusOr<se::DeviceMemory<uint8_t>> ScratchAllocator::AllocateBytes(
    int64_t byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Allocating %d bytes exceeds the memory limit of %d bytes.", byte_size,
        GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8_t>(buffer_addr);
}

absl::StatusOr<std::vector<GenericConvRunner>> GetAlgorithms(
    const GpuConvConfig& config, se::Stream* stream, bool use_cudnn_frontend,
    bool use_fallback, const se::NumericOptions& numeric_options) {
  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDNNConvKindFromCudnnConvKind(config.kind));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                      GetDNNDataTypeFromPrimitiveType(config.input_type));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType output_type,
                      GetDNNDataTypeFromPrimitiveType(config.output_type));

  se::StreamExecutor* stream_exec = stream->parent();
  std::vector<GenericConvRunner> result;

  auto dnn = stream_exec->AsDnn();
  if (dnn == nullptr) {
    return absl::InvalidArgumentError("No DNN in stream executor.");
  }
  switch (kind) {
    default:
      return Internal("Unknown ConvolutionKind %d", kind);
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      if (!config.fusion) {
        return Internal(
            "GpuConvConfig had fusion ConvolutionKind but no FusionConfig.");
      }
      std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
      TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          use_cudnn_frontend,
          // This refers to the kind of convolution op inside the fusion, not
          // the whole fused graph.
          se::dnn::ConvolutionKind::FORWARD, input_type,
          BiasTypeForInputType(input_type), output_type,
          /* conv_input_scale = */ config.conv_result_scale,
          /* side_input_scale = */ config.fusion->side_input_scale,
          /* leakyrelu_alpha = */ config.fusion->leakyrelu_alpha, stream,
          config.input_descriptor, config.filter_descriptor,
          config.bias_descriptor, config.output_descriptor, config.conv_desc,
          use_fallback, config.fusion->mode, numeric_options, &runners));
      for (auto& runner : runners) {
        TF_ASSIGN_OR_RETURN(
            auto runner_cache,
            se::dnn::LazyOpRunner<se::dnn::FusedConvOp>::FromOpRunner(
                std::move(runner)));
        result.emplace_back(std::move(runner_cache));
      }
      break;
    }

    case se::dnn::ConvolutionKind::FORWARD_GRAPH: {
      std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>> runners;
      // This path is cuDNN-only, where the DeviceMemoryBase arguments and the
      // allocator are unused; so, they're all provided as nullptr.
      TF_RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
          kind, input_type, output_type, stream, config.input_descriptor,
          config.filter_descriptor, config.output_descriptor, config.conv_desc,
          use_fallback, numeric_options, &runners, config.serialized_graph));
      for (auto& runner : runners) {
        TF_ASSIGN_OR_RETURN(
            auto runner_cache,
            se::dnn::LazyOpRunner<se::dnn::GraphConvOp>::FromOpRunner(
                std::move(runner)));
        result.emplace_back(std::move(runner_cache));
      }
      break;
    }

    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER: {
      std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
      // This path is cuDNN-only, where the DeviceMemoryBase arguments and the
      // allocator are unused; so, they're all provided as nullptr.
      TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
          use_cudnn_frontend, kind, input_type, output_type, stream,
          config.input_descriptor,
          /* input_data = */ DeviceMemoryBase(nullptr),
          config.filter_descriptor,
          /* filter_data = */ DeviceMemoryBase(nullptr),
          config.output_descriptor,
          /* output_data = */ DeviceMemoryBase(nullptr), config.conv_desc,
          use_fallback, nullptr, numeric_options, &runners));

      for (auto& runner : runners) {
        TF_ASSIGN_OR_RETURN(
            auto runner_cache,
            se::dnn::LazyOpRunner<se::dnn::ConvOp>::FromOpRunner(
                std::move(runner)));
        result.emplace_back(std::move(runner_cache));
      }
      break;
    }
  }

  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<const se::dnn::ConvRunner>>>
GetMIOpenAlgorithms(const HloCustomCallInstruction* instr,
                    absl::Span<se::DeviceMemoryBase> operand_buffers,
                    absl::Span<se::DeviceMemoryBase> result_buffers,
                    se::StreamExecutor* stream_exec,
                    ScratchAllocator* scratch_allocator, se::Stream* stream,
                    const se::NumericOptions& numeric_options) {
  TF_ASSIGN_OR_RETURN(GpuConvConfig config, GetGpuConvConfig(instr));

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDNNConvKindFromCudnnConvKind(config.kind));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType dtype,
                      GetDNNDataTypeFromPrimitiveType(config.output_type));

  TF_ASSIGN_OR_RETURN(
      GpuConvParams params,
      GetGpuConvParams(config, operand_buffers, result_buffers));

  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
  auto dnn = stream_exec->AsDnn();
  if (dnn == nullptr) {
    return absl::InvalidArgumentError("No DNN in stream executor.");
  }
  TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
      /* use_cudnn_frontend = */ false, kind, dtype, dtype, stream,
      params.config->input_descriptor, params.input_buf,
      params.config->filter_descriptor, params.filter_buf,
      params.config->output_descriptor, params.output_buf,
      params.config->conv_desc,
      /* use_fallback = */ false, scratch_allocator, numeric_options,
      &runners));

  return runners;
}

std::string NumBytesToString(int64_t bytes) {
  return absl::StrCat(tsl::strings::HumanReadableNumBytes(bytes), " (", bytes,
                      "B)");
}

CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
  se::dnn::VersionInfo version = GetDnnVersionInfoOrDefault(stream_executor);
  CudnnVersion cudnn_version;
  cudnn_version.set_major(version.major_version());
  cudnn_version.set_minor(version.minor_version());
  cudnn_version.set_patch(version.patch());

  return cudnn_version;
}

ComputeCapability GetComputeCapability(se::StreamExecutor* stream_executor) {
  ComputeCapability cc;
  se::CudaComputeCapability se_cc =
      stream_executor->GetDeviceDescription().cuda_compute_capability();
  cc.set_major(se_cc.major);
  cc.set_minor(se_cc.minor);
  return cc;
}

void PrintPlatformInfo(const se::Stream* stream) {
  auto* se = stream->parent();
  const auto& desc = se->GetDeviceDescription();
  LOG(ERROR) << "Device: " << desc.name();
  LOG(ERROR) << "Platform: " << desc.platform_version();
  LOG(ERROR) << "Driver: " << desc.driver_version();
  LOG(ERROR) << "Runtime: " << desc.runtime_version();

  auto dnn_version = GetDnnVersionInfo(se);
  if (dnn_version.ok()) {
    auto v = dnn_version.value();
    LOG(ERROR) << "cudnn version: " << v.major_version() << "."
               << v.minor_version() << "." << v.patch();
  }
}

// Returns true if the redzones in `allocator`'s allocations are unmodified.
//
// If the redzones are modified, logs an error, sets the appropriate failure
// bits on `result`, and returns false.
//
// Returns a absl::Status if an unexpected error has occurred, and the stream
// has been poisoned.
//
// `name` is a user-friendly name for the set of redzones being checked, e.g.
// "input/output" or "scratch".
absl::StatusOr<bool> CheckRedzones(const se::RedzoneAllocator& allocator,
                                   se::Stream* stream, absl::string_view name,
                                   std::string_view instr_str,
                                   AutotuneResult* result) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CudnnConvAlgorithmPicker checking redzones",
                                 2);
  using RedzoneCheckStatus = se::RedzoneAllocator::RedzoneCheckStatus;
  TF_ASSIGN_OR_RETURN(RedzoneCheckStatus redzone_check,
                      allocator.CheckRedzones());
  if (redzone_check.ok()) {
    return true;
  }

  auto* fail = result->mutable_failure();
  fail->set_kind(AutotuneResult::REDZONE_MODIFIED);
  *fail->mutable_msg() = redzone_check.RedzoneFailureMsg();
  fail->set_buffer_address(
      reinterpret_cast<uint64_t>(redzone_check.user_buffer_address));

  LOG(ERROR) << absl::StreamFormat(
      "Detected cudnn out-of-bounds write in conv %s buffer! This is likely a "
      "cudnn bug. We will skip this algorithm in the future, but your GPU "
      "state may already be corrupted, leading to incorrect results. Within "
      "Google, no action is needed on your part. Outside of Google, please "
      "ensure you're running the latest version of cudnn. If that doesn't fix "
      "the problem, please file a bug with this full error message and we'll "
      "contact nvidia.",
      name);
  LOG(ERROR) << redzone_check.RedzoneFailureMsg();
  LOG(ERROR) << "HloInstruction " << instr_str;
  PrintPlatformInfo(stream);
  return false;
}

}  // anonymous namespace

bool ShouldInitConvData(const HloModuleConfig& hlo_module_config) {
  const int32_t conv_autotune_level =
      hlo_module_config.debug_options().xla_gpu_autotune_level();
  return conv_autotune_level >= 2;
}

bool ShouldCheckConv(const HloModuleConfig& hlo_module_config) {
  const int32_t conv_autotune_level =
      hlo_module_config.debug_options().xla_gpu_autotune_level();
  return conv_autotune_level >= 4;
}

absl::StatusOr<AutotuneResult> GpuConvAlgorithmPicker::PickBestAlgorithm(
    const HloCustomCallInstruction* instr) {
  return AutotunerUtil::Autotune(
      instr, config_, [&] { return PickBestAlgorithmNoCache(instr); });
}

absl::StatusOr<AutotuneResult> GpuConvAlgorithmPicker::PickBestAlgorithmNoCache(
    const HloCustomCallInstruction* instr) {
  if (config_.IsDeviceless()) {
    // Return an autotune result with algo id -1, which means that we autotune
    // at runtime.
    AutotuneResult result;
    result.mutable_algorithm()->set_algo_id(-1);
    return result;
  }

  se::StreamExecutor* stream_exec = config_.GetExecutor();
  // Don't run this function concurrently on the same GPU.
  //
  // This is a bit of a hack and doesn't protect us against arbitrary concurrent
  // use of a GPU, but it's sufficient to let us compile two HLO modules
  // concurrently and then run them sequentially.
  //
  // Putting the lock in here rather than in PickBestAlgorithmNoCache lets us
  // avoid ever doing duplicate work.  If we have a cache miss, only one thread
  // will run PickBestAlgorithmImpl for a particular device.
  absl::MutexLock lock(&GetGpuMutex(stream_exec));

  // Make sure any previous activity on this executor is done. We don't want
  // other work still running on the GPU to interfere with autotuning.
  if (!stream_exec->SynchronizeAllActivity()) {
    return Internal(
        "Failed to synchronize GPU for autotuning conv instruction");
  }

  absl::StatusOr<AutotuneResult> result_or(Internal("Unknown platform."));
  // Check StreamExecutor on which platform it is. ROCm and Cuda implementation
  // have diverged. Specifically, we need to make sure redzone allocator related
  // utilities are not used in ROCm routine
  se::Platform::Id platform_id = stream_exec->GetPlatform()->id();
  if (platform_id == se::rocm::kROCmPlatformId) {
    result_or = PickBestAlgorithmNoCacheRocm(instr);
  } else if (platform_id == se::cuda::kCudaPlatformId) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
    result_or = PickBestAlgorithmNoCacheCuda(instr);
#endif
  }

  return result_or;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)

absl::StatusOr<GpuConvAlgorithmPicker::AutotuneRuntimeArguments>
GpuConvAlgorithmPicker::AutotuneRuntimeArguments::FromInstruction(
    const HloCustomCallInstruction* instr, const AutotuneConfig& config,
    const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(auto rz_buffers,
                      RedzoneBuffers::FromInstruction(
                          *instr, config, debug_options,
                          RedzoneBuffers::kAllInputsOutputsNoScratch));

  // Get canonical HLO.
  std::string canonical_hlo(
      AutotuneCacheKey(config.GetExecutor()->GetDeviceDescription().model_str(),
                       *instr)
          .GetHlo());

  TF_ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));

  GpuConvAlgorithmPicker::AutotuneRuntimeArguments runtime_arguments = {
      instr->GetModule()->config(),
      std::move(rz_buffers),
      std::move(gpu_conv_config),
      {canonical_hlo}};

  return runtime_arguments;
}

struct CudnnVersionRange {
  using TupleVersion = std::tuple<int, int, int>;
  TupleVersion begin;
  TupleVersion end;

  bool IsInRange(const CudnnVersion& other) const {
    TupleVersion other_version{other.major(), other.minor(), other.patch()};
    return begin <= other_version && other_version < end;
  }

  CudnnVersionRange(const CudnnVersion& begin, const CudnnVersion& end)
      : begin(begin.major(), begin.minor(), begin.patch()),
        end(end.major(), end.minor(), end.patch()) {}

  CudnnVersionRange(const TupleVersion& begin, const TupleVersion& end)
      : begin(begin), end(end) {}
};

struct ComputeCapabilityRange {
  using TupleComputeCapability = std::tuple<int, int>;
  TupleComputeCapability begin;
  TupleComputeCapability end;

  bool IsInRange(const ComputeCapability& other) const {
    TupleComputeCapability other_cc{other.major(), other.minor()};
    return begin <= other_cc && other_cc < end;
  }
};

struct DisabledAlgorithm {
  CudnnVersionRange cudnn_version_range;
  ComputeCapabilityRange compute_capability_range;
  int algo_id;
};

// TODO(b/343101418): Remove this once the bug is fixed in upstream cudnn and
// once we updated to that cudnn version.
static const DisabledAlgorithm kDisabledAlgorithms[] = {
    {/*.cudnn_version_range=*/{/*.begin=*/{9, 0, 0}, /*.end=*/{10, 0, 0}},
     /*.compute_capability_range=*/{/*.begin=*/{6, 0}, /*.end=*/{8, 0}},
     /*.algo_id=*/14}};

// There are three tiers of errors possible here: returning a failed
// absl::StatusOr means autotuning fails immediately; returning an
// AutotuneResult with a failure code other than DISQUALIFIED means autotuning
// fails if crash_on_checking_failure is set; and returning a DISQUALIFIED
// AutotuneResult simply skips the engine/algorithm while recording a reason for
// skipping it.
absl::StatusOr<AutotuneResult> GpuConvAlgorithmPicker::AutotuneOneConvRunner(
    GenericConvRunner* const runner,
    std::optional<ReferenceResult>* reference_result,
    absl::Span<const AlgorithmDesc> disabled_algos,
    std::optional<AutotuneCacheKey> instruction_info,
    const AutotuneRuntimeArguments& runtime_arguments) {
  auto alg = runner->ToAlgorithmDesc();

  se::StreamExecutor* stream_exec = config_.GetExecutor();
  XLA_SCOPED_LOGGING_TIMER_LEVEL(
      absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                   alg.ToString()),
      2);

  auto make_failure = [&alg](AutotuneResult::FailureKind kind,
                             absl::string_view msg) {
    AutotuneResult result;
    *result.mutable_algorithm() = alg.ToProto();
    result.mutable_failure()->set_kind(kind);
    result.mutable_failure()->set_msg(/* *sigh* */ msg.data(), msg.size());
    return result;
  };

  AlgorithmDesc alg_key(alg.algo_id(), alg.tensor_ops_enabled(), std::nullopt);

  std::string instr_str = instruction_info.has_value()
                              ? std::string(instruction_info->GetHlo())
                              : "<unknown>";

  for (const auto& disabled_algo : kDisabledAlgorithms) {
    if (disabled_algo.cudnn_version_range.IsInRange(
            GetCudnnVersion(stream_exec)) &&
        disabled_algo.compute_capability_range.IsInRange(
            GetComputeCapability(stream_exec)) &&
        disabled_algo.algo_id == alg.algo_id()) {
      LOG(INFO) << "Omitted potentially buggy algorithm " << alg.ToString()
                << " for conv " << instr_str;
      return make_failure(AutotuneResult::DISQUALIFIED,
                          "Disqualified for being known-buggy.");
    }
  }

  if (absl::c_linear_search(disabled_algos, alg_key)) {
    LOG(INFO) << "Omitted potentially buggy algorithm " << alg.ToString()
              << " for conv " << instr_str;
    return make_failure(AutotuneResult::DISQUALIFIED,
                        "Disqualified for being known-buggy.");
  }

  GpuConvConfig config = runtime_arguments.gpu_conv_config;
  auto activation_mode =
      config.fusion ? config.fusion->mode : se::dnn::ActivationMode::kNone;

  // For fused convolutions with the identity function as the activation, only
  // ALGO_IMPLICIT_PRECOMP_GEMM does the right thing. Other algorithms
  // silently do Relu. See
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
  //
  // For cuDNN Frontend, there is no way to check whether we're using a broken
  // algorithm, so on versions where some algorithms are broken, we don't use
  // the cuDNN Frontend for these convs at all.  As such, if we get a
  // frontend-based runner, we can be sure it's not one of the broken
  // algorithms we're checking for.
  if (!alg.is_cudnn_frontend() &&
      config.kind == CudnnConvKind::kForwardActivation &&
      activation_mode == se::dnn::ActivationMode::kNone &&
      alg.algo_id() != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
    return make_failure(AutotuneResult::DISQUALIFIED,
                        "Disqualified for implicit RELU.");
  }

  TF_ASSIGN_OR_RETURN(
      se::RedzoneAllocator scratch_allocator,
      AutotunerUtil::CreateRedzoneAllocator(
          config_, runtime_arguments.hlo_module_config.debug_options()));

  se::dnn::ProfileResult profile_result;
  VLOG(4) << "Trying algorithm " << alg.ToString() << " for " << instr_str;

  SlowOperationAlarm alarm(absl::Seconds(1), [&] {
    return absl::StrFormat(
        "Trying algorithm %s for conv %s is taking a while...", alg.ToString(),
        instr_str);
  });

  std::optional<size_t> workspace_size =
      runner->ToAlgorithmDesc().workspace_size();
  if (!workspace_size) {
    return make_failure(AutotuneResult::UNKNOWN,
                        "Internal error: missing workspace size from "
                        "OpRunner::ToAlgorithmDesc()");
  }

  auto scratch_or = scratch_allocator.AllocateBytes(*workspace_size);
  if (!scratch_or.ok()) {
    return make_failure(AutotuneResult::DISQUALIFIED,
                        absl::StrCat("Scratch allocation failed: ",
                                     scratch_or.status().ToString()));
  }
  se::DeviceMemoryBase scratch_memory = scratch_or.value();

  // Use assignment instead of brace-list to make GCC 4.9 happy.
  RunConvOptions options;
  options.runner_cache = runner;
  // The following plan timing code is based on
  // https://github.com/NVIDIA/cudnn-frontend/blob/60496f42fdc7a4ccc059f5934e306e728a756755/include/cudnn_frontend_find_plan.h
  float max_time = 0;
  float min_time = std::numeric_limits<float>::max();
  absl::Status launch_status;
  std::vector<se::DeviceMemoryBase> operand_buffers =
      runtime_arguments.rz_buffers.input_buffers();
  std::vector<se::DeviceMemoryBase> result_buffers =
      runtime_arguments.rz_buffers.output_buffers();

  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());

  // Dry-run to warmup the plan.
  launch_status = RunGpuConv(config, operand_buffers, result_buffers,
                             scratch_memory, stream, options);
  // Flag that a warm-up run has been executed; this allows the GpuTimer for
  // the main measurement to safely use the delay kernel pattern, even if lazy
  // module loading is enabled.
  options.profile_result = &profile_result;
  profile_result.set_warmup_run_executed(true);
  constexpr int kMaxIter = 10;
  // Iterate until the new measurement is within kThreshold of the current
  // minimum.
  int num_iters = 0;
  for (; num_iters < kMaxIter && launch_status.ok(); ++num_iters) {
    launch_status = RunGpuConv(config, operand_buffers, result_buffers,
                               scratch_memory, stream, options);
    if (!profile_result.is_valid()) {
      break;
    }
    float old_min_time = min_time;
    min_time = std::min(min_time, profile_result.elapsed_time_in_ms());
    max_time = std::max(max_time, profile_result.elapsed_time_in_ms());

    constexpr float kThreshold = 0.05f;
    if (std::abs(profile_result.elapsed_time_in_ms() - old_min_time) /
            old_min_time <
        kThreshold) {
      break;
    }
  }
  if (!launch_status.ok()) {
    VLOG(5) << "Launch failed: " << launch_status;
    return make_failure(
        AutotuneResult::DISQUALIFIED,
        absl::StrCat("Profiling failure on cuDNN engine ", alg.ToString(), ": ",
                     launch_status.ToString()));
  }
  if (!profile_result.is_valid()) {
    VLOG(5) << "Launch succeeded but profile result is invalid.";
    // Not DISQUALIFIED: this means something went wrong internally.
    return make_failure(
        AutotuneResult::UNKNOWN,
        absl::StrCat("Launch succeeded but profile result is invalid, "
                     "with cuDNN engine ",
                     alg.ToString(), ": ", launch_status.ToString()));
  }
  VLOG(4) << "Best time: " << min_time << " ms. Worst time: " << max_time
          << " ms. Total iterations: " << num_iters;
  int64_t scratch_bytes_used =
      scratch_allocator.TotalAllocatedBytesExcludingRedzones();

  AutotuneResult result;
  *result.mutable_algorithm() = alg.ToProto();
  result.set_scratch_bytes(scratch_bytes_used);
  *result.mutable_run_time() =
      tsl::proto_utils::ToDurationProto(absl::Milliseconds(min_time));

  if (!ShouldCheckConv(runtime_arguments.hlo_module_config)) {
    if (!reference_result->has_value()) {
      (*reference_result) = {
          alg, std::vector<DeviceMemoryBase>(result_buffers.size())};
    }
    return result;
  }

  // Check for writes to redzones.
  TF_ASSIGN_OR_RETURN(
      bool input_output_allocator_redzone_clear,
      CheckRedzones(runtime_arguments.rz_buffers.RedzoneAllocator(), stream,
                    "input/output", instr_str, &result));

  TF_ASSIGN_OR_RETURN(
      bool scratch_allocator_redzone_clear,
      CheckRedzones(scratch_allocator, stream, "scratch", instr_str, &result));

  if (!input_output_allocator_redzone_clear ||
      !scratch_allocator_redzone_clear) {
    if (runtime_arguments.canonical_hlo.has_value()) {
      std::string canonical_hlo = runtime_arguments.canonical_hlo.value();
      std::string blas_version;
      if (auto* blas = stream_exec->AsBlas()) {
        (void)blas->GetVersion(&blas_version);
      }

      AlgorithmDenylist proto;
      auto entry = proto.add_entries();
      entry->set_hlo(canonical_hlo);
      *entry->mutable_cc() = GetComputeCapability(stream_exec);
      *entry->mutable_cudnn_version() = GetCudnnVersion(stream_exec);
      entry->set_blas_version(blas_version);
      auto algo = entry->add_algos();
      algo->set_id(alg.algo_id());
      algo->set_tensor_ops(alg.tensor_ops_enabled());

      LOG(ERROR) << "To denylist this algorithm for this convolution, "
                    "copy-paste the following "
                    "proto to the denylist file pointed by XLA_FLAGS "
                    "--xla_gpu_algorithm_denylist_path="
                 << GetDebugOptionsFromFlags().xla_gpu_algorithm_denylist_path()
                 << " : " << proto.ShortDebugString();
    }

    // CheckRedzones has modified the result in-place to include a failure.
    return result;
  }

  if (reference_result->has_value()) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::CompareEqual", 2);
    BufferComparator comparator(runtime_arguments.rz_buffers.output_shape(),
                                runtime_arguments.hlo_module_config);
    for (int i = 0; i < result_buffers.size(); ++i) {
      absl::StatusOr<bool> compare_result = comparator.CompareEqual(
          stream, (*reference_result)->buffers[i], result_buffers[i]);
      if (!compare_result.ok()) {
        LOG(ERROR) << "Unable to compare "
                   << (*reference_result)->algorithm.ToString() << " against "
                   << alg.ToString() << " for " << instr_str << ": "
                   << compare_result.status();
        if (compare_result.status().code() ==
            absl::StatusCode::kResourceExhausted) {
          // Possibly OOM. Propagate the error.
          return compare_result.status();
        }
        const DebugOptions& debug_options =
            runtime_arguments.hlo_module_config.debug_options();
        CHECK(!debug_options.xla_gpu_crash_on_verification_failures());
      } else if (!compare_result.value()) {
        LOG(ERROR)
            << "Results mismatch between different convolution algorithms. "
               "This is likely a bug/unexpected loss of precision in cudnn.\n"
            << instr_str << " for " << (*reference_result)->algorithm.ToString()
            << " vs " << alg.ToString();
        PrintPlatformInfo(stream);
        if (instruction_info.has_value()) {
          VLOG(2) << "Full module on failure: \n"
                  << instruction_info->GetModelStr();
        }
        auto* fail = result.mutable_failure();
        fail->set_kind(AutotuneResult::WRONG_RESULT);
        fail->set_buffer_address(
            reinterpret_cast<uint64_t>(result_buffers[i].opaque()));
        *fail->mutable_reference_algorithm() =
            (*reference_result)->algorithm.ToProto();
      }
    }
  } else {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("Memcpy Reference Result", 2);
    std::vector<DeviceMemoryBase> reference_result_buffers(
        result_buffers.size());
    for (int i = 0; i < result_buffers.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          reference_result_buffers[i],
          runtime_arguments.rz_buffers.RedzoneAllocator().AllocateBytes(
              result_buffers[i].size()));
      TF_RETURN_IF_ERROR(stream->Memcpy(&reference_result_buffers[i],
                                        result_buffers[i],
                                        result_buffers[i].size()));
    }
    (*reference_result) = {alg, reference_result_buffers};
  }

  return result;
}

absl::StatusOr<AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheCuda(
    const HloCustomCallInstruction* instr) {
  AutotuneCacheKey instruction_info{config_.GetModelStr(), *instr};
  std::string instr_str(instruction_info.GetHlo());
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr_str));

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();
  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  std::string blas_version;
  se::StreamExecutor* stream_exec = config_.GetExecutor();
  if (auto* blas = stream_exec->AsBlas()) {
    (void)blas->GetVersion(&blas_version);
  }

  std::vector<AlgorithmDesc> disabled_algos;
  TF_ASSIGN_OR_RETURN(
      AutotuneRuntimeArguments runtime_arguments,
      AutotuneRuntimeArguments::FromInstruction(instr, config_, debug_options));
  if (runtime_arguments.canonical_hlo.has_value()) {
    disabled_algos = GetDisabledConvAlgorithms(
        GetComputeCapability(stream_exec), GetCudnnVersion(stream_exec),
        blas_version, runtime_arguments.canonical_hlo.value());
  }

  const bool cudnn_frontend_enabled =
      debug_options.xla_gpu_enable_cudnn_frontend();
  const bool deterministic_ops =
      debug_options.xla_gpu_deterministic_ops() ||
      debug_options.xla_gpu_exclude_nondeterministic_ops();
  bool allow_tf32 = true;
  // TODO(b/284371623): Properly set allow_tf32 even if instr==nullptr, which is
  // the case when running an AOT compiled executable with runtime autotuning.
  if (instr) {
    allow_tf32 = absl::c_all_of(
        instr->precision_config().operand_precision(),
        [](int precision) { return precision <= PrecisionConfig::HIGH; });
  }
  const se::NumericOptions numeric_options{deterministic_ops, allow_tf32};

  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm suffices. It doesn't make
  // this algorithm considered correct, though.
  std::optional<ReferenceResult> reference_result;

  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());
  TF_ASSIGN_OR_RETURN(
      std::vector<GenericConvRunner> runners,
      GetAlgorithms(runtime_arguments.gpu_conv_config, stream,
                    cudnn_frontend_enabled,
                    /* use_fallback = */ false, numeric_options));

  std::vector<AutotuneResult> profile_results;
  for (auto& runner_cache : runners) {
    TF_ASSIGN_OR_RETURN(
        auto result,
        AutotuneOneConvRunner(&runner_cache, &reference_result, disabled_algos,
                              instruction_info, runtime_arguments));
    profile_results.emplace_back(std::move(result));
  }

  // If any algorithm has worked, we'll skip the fallback algorithms, since
  // they include some very slow algorithms.
  if (!reference_result) {
    LOG(WARNING) << "None of the algorithms provided by cuDNN heuristics "
                    "worked; trying fallback algorithms.";
    if (runtime_arguments.canonical_hlo.has_value()) {
      LOG(WARNING) << "Conv: " << runtime_arguments.canonical_hlo.value();
    }

    TF_ASSIGN_OR_RETURN(
        std::vector<GenericConvRunner> fallback_runners,
        GetAlgorithms(runtime_arguments.gpu_conv_config, stream,
                      cudnn_frontend_enabled,
                      /* use_fallback = */ true, numeric_options));

    for (auto& runner_cache : fallback_runners) {
      TF_ASSIGN_OR_RETURN(
          auto result, AutotuneOneConvRunner(&runner_cache, &reference_result,
                                             disabled_algos, instruction_info,
                                             runtime_arguments));
      profile_results.emplace_back(std::move(result));
    }
  }

  // Log the autotuning result.
  if (instr) {
    AutotuningLog log;
    {
      ConvInstructionLog instr_log;
      *instr_log.mutable_instruction() = instr->ToProto();
      for (int i = 0; i < instr->operand_count(); i++) {
        *instr_log.add_operand_shapes() = instr->operand(i)->shape().ToProto();
        instr_log.add_operand_addresses(reinterpret_cast<uint64_t>(
            runtime_arguments.rz_buffers.input_buffers()[i].opaque()));
      }
      for (se::DeviceMemoryBase result_buffer :
           runtime_arguments.rz_buffers.output_buffers()) {
        instr_log.add_result_addresses(
            reinterpret_cast<uint64_t>(result_buffer.opaque()));
      }
      log.mutable_instr()->PackFrom(instr_log);
    }
    for (const auto& profile : profile_results) {
      *log.add_results() = profile;
    }
    *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
    *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
    log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
    log.set_blas_version(blas_version);
    VLOG(2) << "Autotuning result: " << log.ShortDebugString();
    // If we crash on checking failure, we are in a testing/benchmark mode.
    if (crash_on_checking_failure) {
      // Crash on miscompares and redzone violations if desired.
      for (const auto& profile : profile_results) {
        if (profile.has_failure() &&
            profile.failure().kind() != AutotuneResult::DISQUALIFIED) {
          LOG(FATAL) << "crash_on_checking_failure encountered errors:\n\n"
                     << log.DebugString();  // NOLINT
        }
      }
    }
  }

  TF_ASSIGN_OR_RETURN(AutotuneResult selected_algorithm,
                      PickBestResult(profile_results, instr_str,
                                     runtime_arguments.hlo_module_config));
  return selected_algorithm;
}
#endif

absl::StatusOr<AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheRocm(
    const HloCustomCallInstruction* instr) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr->ToString()));

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();
  const bool deterministic_ops =
      debug_options.xla_gpu_deterministic_ops() ||
      debug_options.xla_gpu_exclude_nondeterministic_ops();
  const bool allow_tf32 = absl::c_all_of(
      instr->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::NumericOptions numeric_options{deterministic_ops, allow_tf32};

  se::StreamExecutor* stream_exec = config_.GetExecutor();
  const auto device_ordinal = stream_exec->device_ordinal();
  std::vector<se::DeviceMemoryBase> operand_buffers;

  // allocator either points to this->allocator_ or, if that's null, to a
  // se::StreamExecutorMemoryAllocator for stream_exec.
  se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());
  const auto initialize_buffer = [stream](DeviceMemoryBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    return stream->MemZero(&buffer, buffer.size());
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    TF_RETURN_IF_ERROR(initialize_buffer(buffer));
    operand_buffers.push_back(buffer);
  }

  std::vector<se::DeviceMemoryBase> result_buffers(
      instr->shape().tuple_shapes_size());
  if (instr->shape().IsTuple()) {
    for (int i = 0; i < instr->shape().tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          result_buffers[i],
          input_output_allocator.AllocateBytes(
              ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(i))));
      TF_RETURN_IF_ERROR(initialize_buffer(result_buffers[i]));
    }
  } else {
    TF_ASSIGN_OR_RETURN(
        result_buffers[0],
        input_output_allocator.AllocateBytes(
            ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(0))));
    TF_RETURN_IF_ERROR(initialize_buffer(result_buffers[0]));
  }

  ScratchAllocator scratch_allocator(device_ordinal, allocator);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners,
      GetMIOpenAlgorithms(instr, absl::MakeSpan(operand_buffers),
                          absl::MakeSpan(result_buffers), stream_exec,
                          &scratch_allocator, stream, numeric_options));

  std::vector<AutotuneResult> profile_results;

  if (runners.size() == 1) {
    TF_ASSIGN_OR_RETURN(auto alg, runners[0]->ToAlgorithmDesc());
    auto algorithm_proto = alg.ToProto();
    profile_results.emplace_back();
    auto& result = profile_results.back();
    *result.mutable_algorithm() = algorithm_proto;

    result.set_scratch_bytes(runners[0]->GetWorkspaceSize());

    // TODO(awpr): if the profile result time for a singleton algorithm is
    // needed, plumb it via OpRunner; we'll need to do this to let TF ops avoid
    // re-profiling ROCm algorithms anyway.
    *result.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(absl::Milliseconds(-1));
  } else {
    TF_ASSIGN_OR_RETURN(GpuConvConfig config, GetGpuConvConfig(instr));
    for (auto& runner : runners) {
      TF_ASSIGN_OR_RETURN(auto alg, runner->ToAlgorithmDesc());
      XLA_SCOPED_LOGGING_TIMER_LEVEL(
          absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                       alg.ToString()),
          2);

      se::dnn::ProfileResult profile_result;
      VLOG(4) << "Trying algorithm " << alg.ToString() << " for "
              << instr->ToString();

      TF_ASSIGN_OR_RETURN(
          DeviceMemoryBase scratch_memory,
          scratch_allocator.AllocateBytes(runner->GetWorkspaceSize()));

      TF_ASSIGN_OR_RETURN(auto lazy_runner,
                          se::dnn::LazyOpRunner<se::dnn::ConvOp>::FromOpRunner(
                              std::move(runner)));

      GenericConvRunner runner_cache(std::move(lazy_runner));

      // Use assignment instead of brace-list to make GCC 4.9 happy.
      RunConvOptions options;
      options.profile_result = &profile_result;
      options.runner_cache = &runner_cache;
      absl::Status launch_status =
          RunGpuConv(config, absl::MakeSpan(operand_buffers), result_buffers,
                     scratch_memory, stream, options);

      if (!launch_status.ok()) {
        continue;
      }

      if (!profile_result.is_valid()) {
        continue;
      }

      profile_results.emplace_back();
      AutotuneResult& result = profile_results.back();
      *result.mutable_algorithm() = alg.ToProto();

      int64_t scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
      result.set_scratch_bytes(scratch_bytes_used);
      *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    }
  }

  TF_ASSIGN_OR_RETURN(AutotuneResult selected_algorithm,
                      PickBestResult(profile_results, instr->ToString(),
                                     instr->GetModule()->config()));
  return selected_algorithm;
}

absl::StatusOr<bool> GpuConvAlgorithmPicker::RunOnInstruction(
    HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  const bool strict = instr->parent()
                          ->parent()
                          ->config()
                          .debug_options()
                          .xla_gpu_strict_conv_algorithm_picker();

  absl::StatusOr<AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    auto msg = absl::StrFormat(
        "Failed to determine best cudnn convolution algorithm for:\n%s\n\n"
        "Original error: %s",
        instr->ToString(), best_algo_or.status().ToString());

    if (strict) {
      return Unknown(
          "%s\n\nTo ignore this failure and try to use a fallback algorithm "
          "(which may have suboptimal performance), use "
          "XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false.  Please "
          "also file a bug for the root cause of failing autotuning.",
          msg);
    }
    LOG(WARNING)
        << msg << "\n\nAs a result, convolution performance may be suboptimal.";
    return false;
  }

  auto best_algo = std::move(best_algo_or).value();
  VLOG(3) << "Setting cudnn conv to use algorithm "
          << best_algo.conv().algorithm() << " and "
          << NumBytesToString(best_algo.scratch_bytes())
          << " of scratch memory: " << instr->ToString()
          << " tensor_ops_enabled: " << best_algo.conv().tensor_ops_enabled();

  // Replace instr with a new CustomCall which has the correct algorithm, and
  // whose output shape has the appropriate amount of scratch memory.
  HloComputation* computation = instr->parent();
  std::vector<Shape> new_call_element_shapes;
  // Add the shapes of the outputs of the convolution.
  new_call_element_shapes.reserve(instr->shape().tuple_shapes_size() - 1);
  for (int i = 0; i < instr->shape().tuple_shapes_size() - 1; ++i) {
    new_call_element_shapes.emplace_back(instr->shape().tuple_shapes(i));
  }
  // The final element is the size of the workspace.
  new_call_element_shapes.emplace_back(
      ShapeUtil::MakeShape(U8, {best_algo.scratch_bytes()}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr->backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig& backend_config =
      *gpu_backend_config.mutable_cudnn_conv_backend_config();
  *backend_config.mutable_algorithm() = best_algo.algorithm();
  backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
      best_algo.scratch_bytes());

  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, instr->operands()));

  // Preserve the name of the old instruction.  This is safe because we're going
  // to remove the old one anyway, and it makes it easier to trace how our conv
  // is transformed through all our passes.
  new_call->SetAndSanitizeName(instr->name());

  VLOG(3) << "Replacing convolution " << instr->ToString() << " with "
          << new_call->ToString();

  TF_RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  std::vector<HloInstruction*> new_tuple_elements;
  new_tuple_elements.reserve(new_call->shape().tuple_shapes_size() - 1);
  for (int i = 0; i < new_call->shape().tuple_shapes_size() - 1; ++i) {
    new_tuple_elements.emplace_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.emplace_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));

  TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, new_tuple));
  return true;
}

absl::StatusOr<bool> GpuConvAlgorithmPicker::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCandidate(instr)) {
      convs.push_back(instr);
    }
  }

  bool changed = false;
  for (HloInstruction* instr : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr));
    changed |= result;
  }
  return changed;
}

absl::StatusOr<bool> GpuConvAlgorithmPicker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GpuConvAlgorithmPicker for ", module->name()));

  if (!IsEnabled(module)) {
    VLOG(3) << "Convolution auto-tuning disabled, GpuConvAlgorithmPicker "
               "returning early.";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
