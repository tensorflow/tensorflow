/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/core/autotuner/gpu_profiler.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/core/autotuner/profiler.h"
#include "xla/backends/gpu/core/autotuner/redzone_buffers.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/backends/gpu/runtime/gpu_executable.h"
#include "xla/backends/gpu/runtime/gpu_executable_run_options.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/util/buffer_slice_merge.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace gpu {

namespace {

// Minimum size of the buffer used to flush the caches, in case the device does
// not report its L2 cache size.
constexpr int64_t kMinCacheFlushBytes = int64_t{64} << 20;  // 64 MiB

// First-principles hardware noise model and competitive margin-aware early
// stopping for GPU autotuning (`GpuProfiler::Profile`).
//
// Let `mu_k` be the true steady-state execution time of candidate `k`. In
// `GpuProfiler::Profile`, timed run `i` in `{0, ..., N - 1}` measures:
//
//   T_i = mu_k + delta_0 * I[i == 0] + eps_quant,i + eps_boundary,i
//         + eta_i * mu_k
//
// with four physical error sources on modern GPUs:
//   1. CUDA event timer quantization (`eps_quant`, sigma_q ~ 9 ns):
//      `cuEventElapsedTime` samples the 32 ns GPU `GLOBALTIMER` tick at start
//      and stop (`sqrt(2 / 12) * 32 ns ~= 13 ns`), which is negligible above
//      1 us.
//   2. Additive stream-boundary and cache-flush drain jitter (`eps_boundary`,
//      `kBoundaryJitterStdDevNs = 1,000 ns`): each timed pass is preceded by a
//      `2 * L2` `MemZero` (`FlushCaches()`, ~30-36 us on H100/B200/GB300).
//      Residual L2/DRAM writeback drain, GPU front-end CTA launch latency, and
//      last-wave tail drain contribute ~1 us of run-to-run standard deviation
//      independent of kernel duration.
//   3. First-run post-host-sync idle/clock transient (`delta_0`,
//      `kFirstRunIdleBiasNs = 2,000 ns`): between the warm-up pass and
//      `run == 0`, `Profile()` calls `stream_->BlockHostUntilDone()` and runs
//      host-side `CheckRedzones()`, leaving the GPU idle while GPC boost clocks
//      droop and power gates engage. A single ~33 us `MemZero` before
//      `run == 0` is shorter than the ~100-200 us GPU DVFS ramp-up window,
//      biasing `T_0` upward by up to ~2 us relative to runs `1..N-1` (which
//      execute back-to-back on stream without host synchronization).
//   4. Multiplicative memory-subsystem variance (`eta_i * mu_k`,
//      `kRelativeNoiseStdDev = 0.01`): periodic HBM refresh (`tREFI ~ 3.9 us`)
//      and L2/NoC partition arbitration scale proportionally (~1%) with kernel
//      duration.
//
// From this model, additional timed runs after `run == 0` cannot improve
// selection accuracy in two regimes:
//   (a) Saturation / cost-regret cap (`T_0 >= kMaxMultiRunDurationNs`, 500 us):
//       once `T_0 >= 500 us`, the combined relative error from the transient
//       and boundary terms is `hypot(1 us, 2 us) / 500 us < 0.45%`, which is
//       strictly dominated by the stationary 1% HBM variance floor while each
//       extra pass costs `T_flush + T_0 > 530 us` (>15x the L2 flush cost).
//   (b) Competitive 3-sigma contender test against `T_best`: let `T_best` be
//       the lowest mean duration on the same `GpuInputBuffers` among
//       candidates that passed `ConfigRunner` correctness checks and are
//       eligible for selection (reported via `RecordValidatedCandidate`, with
//       `ConfigRunner` profiling trusted reference backends first). Under
//       the null hypothesis that candidate `k` is at least as fast as the
//       leader (`mu_k <= mu_best`), `T_0` has upper idle bias `<= delta_0` and
//       the difference `T_0 - T_best` (1 run vs. an N-run mean) has combined
//       standard error:
//
//         sigma_diff(T_best, N) =
//             sqrt(1 + 1 / N) * hypot(sigma_0, sigma_m * T_best)
//
//       At `z = 3` (99.87% one-sided normal confidence), any candidate with
//
//         T_0 > T_best + delta_0 + 3 * sigma_diff(T_best, N)
//
//       cannot plausibly beat the current leader and stops after `run == 0`.
//       Contenders within this margin execute all `num_timed_runs` passes so
//       first-run transients are averaged out before ranking close tiles.
//       This test requires active L2 flushing before every timed run: without
//       it, `run == 0` follows the L2-perturbing warm-up redzone check while
//       runs `1..N-1` are L2-hot, so `T_0` carries an unbounded cold-cache
//       bias relative to a multi-run `T_best` and only regime (a) applies.
constexpr int64_t kMaxMultiRunDurationNs = 500'000;  // 500 us
constexpr double kFirstRunIdleBiasNs = 2'000.0;      // 2.0 us
constexpr double kBoundaryJitterStdDevNs = 1'000.0;  // 1.0 us
constexpr double kRelativeNoiseStdDev = 0.01;        // 1.0%
constexpr double kContenderConfidenceSigma = 3.0;    // 99.87% one-sided

bool ShouldStopAfterFirstRun(int64_t first_run_ns, int64_t best_duration_ns,
                             int num_timed_runs, bool l2_flush_active) {
  if (num_timed_runs <= 1) {
    return true;
  }
  // Mock or unprofiled executables may report <= 0 (`GpuExecutable` clamps real
  // timings to >= 1 ns in `MaybeSyncAndProfile`), and kernels at or above
  // `kMaxMultiRunDurationNs` have <0.45% transient/boundary error.
  if (first_run_ns <= 0 || first_run_ns >= kMaxMultiRunDurationNs) {
    return true;
  }
  if (!l2_flush_active || best_duration_ns <= 0 ||
      best_duration_ns == std::numeric_limits<int64_t>::max() ||
      first_run_ns <= best_duration_ns) {
    return false;
  }
  const double best_ns = static_cast<double>(best_duration_ns);
  const double per_run_sigma_ns =
      std::hypot(kBoundaryJitterStdDevNs, kRelativeNoiseStdDev * best_ns);
  const double diff_sigma_ns =
      std::sqrt(1.0 + 1.0 / static_cast<double>(num_timed_runs)) *
      per_run_sigma_ns;
  const double contender_upper_bound_ns =
      best_ns + kFirstRunIdleBiasNs + kContenderConfidenceSigma * diff_sigma_ns;
  return static_cast<double>(first_run_ns) > contender_upper_bound_ns;
}

std::vector<ExecutionInput> CreateExecutionInputsFromBuffers(
    absl::Span<se::DeviceAddressBase const> buffers,
    absl::Span<Shape const> shapes) {
  CHECK_EQ(buffers.size(), shapes.size());
  std::vector<ExecutionInput> inputs;
  for (int i = 0; i < buffers.size(); ++i) {
    inputs.emplace_back(shapes.at(i));
    // Our executable doesn't have input-output aliasing, so we can pass
    // unowned input buffers.
    inputs.back().SetUnownedBuffer(
        /*index=*/{}, MaybeOwningDeviceAddress(/*unowned=*/buffers.at(i)));
  }
  return inputs;
}

int GetScratchBytes(const GpuExecutable& executable) {
  // Group scratch slices by allocation.
  absl::flat_hash_map<const BufferAllocation*,
                      std::vector<BufferAllocation::Slice>>
      alloc_to_slices;

  CHECK_OK(executable.thunk_executor().thunks().WalkNested(
      [&alloc_to_slices](const Thunk* thunk) {
        for (const auto& buffer_use : thunk->buffer_uses()) {
          // ContentValidity::kUndefined means the buffer is a scratch buffer.
          if (buffer_use.content_validity() ==
              BufferUse::ContentValidity::kUndefined) {
            const auto& slice = buffer_use.slice();
            alloc_to_slices[slice.allocation()].push_back(slice);
          }
        }

        return absl::OkStatus();
      }));

  int64_t scratch_bytes = 0;
  for (auto& [alloc, slices] : alloc_to_slices) {
    // We might be re-using the same physical buffer for different slices (e.g.
    // we need to use the same scratch buffer for two sequential operations), so
    // we need to merge overlapping slices.
    for (const auto& merged_slice : MergeOverlappingSlices(slices)) {
      scratch_bytes += merged_slice.size();
    }
  }

  return static_cast<int>(scratch_bytes);
}

// Initialize a specific input buffer with custom values.
// This is useful for initializing constant parameters like group sizes
// in group-gemm operations after buffers are created.
static absl::Status InitializeInputBuffer(GpuInputBuffers& gpu_buffers,
                                          se::Stream* stream, int buffer_index,
                                          const void* values,
                                          size_t size_bytes) {
  RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;

  if (buffer_index < 0 || buffer_index >= rz_buffers.input_buffers().size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid buffer_index %d, must be in range [0, %d)",
                        buffer_index, rz_buffers.input_buffers().size()));
  }

  se::DeviceAddressBase buffer = rz_buffers.input_buffers()[buffer_index];
  ABSL_RETURN_IF_ERROR(stream->Memcpy(const_cast<se::DeviceAddressBase*>(&buffer),
                                 values, size_bytes));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return absl::OkStatus();
}

// Initialize input buffers based on the operation type.
// This function examines the HLO instruction and initializes
// specific input buffers based on the operation's requirements.
static absl::Status InitializeBuffersIfRequiredByOpcode(
    const HloInstruction* instr, GpuInputBuffers& gpu_buffers,
    se::Stream* stream) {
  if (instr == nullptr) {
    return absl::OkStatus();
  }

  // Handle kFusion with kRaggedDot (kRaggedContracting autotuning).
  // The kernel reads GS via the function argument that the fusion's
  // group_sizes operand maps to (= that ENTRY parameter = that input buffer).
  // RunHloPasses replaces the FUSION's group_sizes operand with a balanced
  // constant in the ENTRY computation, but the KERNEL still reads from its
  // argument. Initialize that buffer with balanced GS values so the kernel
  // computes correct start_m and m_loop_count instead of using random data.
  if (instr->opcode() == HloOpcode::kFusion && instr->operand_count() >= 3) {
    const HloComputation* fc = instr->fused_instructions_computation();
    const HloInstruction* ragged_dot_instr =
        hlo_query::GetFirstInstructionWithOpcode(*fc, HloOpcode::kRaggedDot);
    if (ragged_dot_instr != nullptr) {
      const auto* ragged_dot = Cast<HloRaggedDotInstruction>(ragged_dot_instr);
      // The buffer to fill is the fusion operand feeding group_sizes. If
      // group_sizes is materialized inside the fusion instead, there is no
      // input buffer to initialize and the kernel reads well-defined values
      // anyway, so leave the buffers alone.
      const HloInstruction* gs_param = ragged_dot->operand(2);
      if (gs_param->opcode() != HloOpcode::kParameter) {
        return absl::OkStatus();
      }
      const int64_t gs_index = gs_param->parameter_number();
      const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
      const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
      const int64_t ragged_dim_size =
          ragged_dot->operand(0)->shape().dimensions(lhs_ragged_dim);
      // group_sizes is [G], or [batch..., G] for a batched
      // kRaggedNonContracting dot, in which case the same balanced pattern
      // repeats once per batch element.
      const Shape& gs_shape = instr->operand(gs_index)->shape();
      const int64_t groups_per_batch = gs_shape.dimensions().back();
      if (groups_per_batch == 0) {
        return absl::OkStatus();
      }
      const int64_t num_elements = ShapeUtil::ElementsIn(gs_shape);
      const int64_t base_group_size = ragged_dim_size / groups_per_batch;
      const int64_t remainder_rows = ragged_dim_size % groups_per_batch;

      // Hands the first remainder_rows groups of each batch element one extra
      // row, so that the group sizes sum to exactly ragged_dim_size.
      auto initialize_gs_buffer = [&](auto type_sample) -> absl::Status {
        using T = decltype(type_sample);
        std::vector<T> gs_vals(num_elements);
        for (int64_t i = 0; i < num_elements; ++i) {
          gs_vals[i] = static_cast<T>((i % groups_per_batch) < remainder_rows
                                          ? base_group_size + 1
                                          : base_group_size);
        }
        return InitializeInputBuffer(gpu_buffers, stream, gs_index,
                                     gs_vals.data(), num_elements * sizeof(T));
      };

      // The emitter sign-extends group_sizes from any of these index types
      // (see IsTritonSupportedRaggedDot), so the buffer must be filled with
      // elements of exactly that width.
      switch (gs_shape.element_type()) {
        case S8:
          ABSL_RETURN_IF_ERROR(initialize_gs_buffer(int8_t{}));
          break;
        case S16:
          ABSL_RETURN_IF_ERROR(initialize_gs_buffer(int16_t{}));
          break;
        case S32:
          ABSL_RETURN_IF_ERROR(initialize_gs_buffer(int32_t{}));
          break;
        case S64:
          ABSL_RETURN_IF_ERROR(initialize_gs_buffer(int64_t{}));
          break;
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Unexpected group_sizes type for ragged-dot autotuning: ",
              gs_shape.ToString()));
      }
      VLOG(3) << "GpuProfiler: initialized GS input buffer (arg[" << gs_index
              << "]) with balanced values [base_group_size=" << base_group_size
              << ",remainder_rows=" << remainder_rows
              << ",groups_per_batch=" << groups_per_batch
              << ",ragged_dim_size=" << ragged_dim_size << "]";
      return absl::OkStatus();
    }
  }

  // Handle group-gemm operations
  if (instr->opcode() == HloOpcode::kCustomCall &&
      instr->custom_call_target() == "__cublas$lt$groupedMatmul") {
    // Get the backend config to extract ragged dimension information
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr->backend_config<GpuBackendConfig>());
    const GroupedGemmBackendConfig& grouped_config =
        gpu_config.grouped_gemm_backend_config();
    const RaggedDotDimensionNumbers& ragged_dims =
        grouped_config.ragged_dot_dimension_numbers();

    // Get ragged dimension index from the config
    int64_t ragged_dim_index = ragged_dims.lhs_ragged_dimensions(0);

    // Get ragged dimension size from LHS operand
    const Shape& lhs_shape = instr->operand(0)->shape();
    int64_t ragged_dim_size = lhs_shape.dimensions(ragged_dim_index);

    // Get number of groups from the group sizes parameter shape
    // The shape can be 1D [num_groups] or 2D [batch_size, groups_per_batch]
    const Shape& group_sizes_shape =
        instr->operand(instr->operand_count() - 1)->shape();
    int64_t groups_per_batch =
        group_sizes_shape.dimensions(group_sizes_shape.dimensions().size() - 1);
    int64_t total_elements = ShapeUtil::ElementsIn(group_sizes_shape);

    // Calculate group sizes based on groups_per_batch (not total elements)
    int64_t base_group_size = ragged_dim_size / groups_per_batch;
    int64_t last_group_size =
        ragged_dim_size - ((groups_per_batch - 1) * base_group_size);

    VLOG(3) << "  Ragged dim index: " << ragged_dim_index;
    VLOG(3) << "  Ragged dim size: " << ragged_dim_size;
    VLOG(3) << "  Group sizes shape: " << group_sizes_shape.ToString();
    VLOG(3) << "  Groups per batch: " << groups_per_batch;
    VLOG(3) << "  Total elements in group sizes buffer: " << total_elements;
    VLOG(3) << "  Base group size (first n-1 groups): " << base_group_size;
    VLOG(3) << "  Last group size: " << last_group_size;

    // Determine the type of the group sizes parameter
    PrimitiveType group_sizes_type = group_sizes_shape.element_type();

    if (group_sizes_type == S32) {
      std::vector<int32_t> group_sizes(total_elements);
      // Fill with the pattern: [base_size, base_size, ..., last_size]
      // repeated for each batch
      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t group_idx_in_batch = i % groups_per_batch;
        if (group_idx_in_batch == groups_per_batch - 1) {
          group_sizes[i] = static_cast<int32_t>(last_group_size);
        } else {
          group_sizes[i] = static_cast<int32_t>(base_group_size);
        }
      }
      ABSL_RETURN_IF_ERROR(InitializeInputBuffer(
          gpu_buffers, stream,
          instr->operand_count() - 1,  // Last parameter is group sizes
          group_sizes.data(), total_elements * sizeof(int32_t)));
    } else if (group_sizes_type == S64) {
      std::vector<int64_t> group_sizes(total_elements);
      // Fill with the pattern: [base_size, base_size, ..., last_size]
      // repeated for each batch
      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t group_idx_in_batch = i % groups_per_batch;
        if (group_idx_in_batch == groups_per_batch - 1) {
          group_sizes[i] = last_group_size;
        } else {
          group_sizes[i] = base_group_size;
        }
      }
      ABSL_RETURN_IF_ERROR(InitializeInputBuffer(
          gpu_buffers, stream,
          instr->operand_count() - 1,  // Last parameter is group sizes
          group_sizes.data(), total_elements * sizeof(int64_t)));
    }
  }

  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<GpuProfiler> GpuProfiler::Create(
    se::StreamExecutor* stream_executor, ProfileOptions options,
    se::DeviceAddressAllocator* external_allocator) {
  std::unique_ptr<se::DeviceAddressAllocator> owned_allocator;
  se::DeviceAddressAllocator* active_allocator = external_allocator;

  if (active_allocator == nullptr) {
    VLOG(1) << "No external allocator provided, creating a new allocator.";
    owned_allocator =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_executor);
    active_allocator = owned_allocator.get();
  }

  // TODO(b/442997461): Create a new stream using
  // `stream_executor->CreateStream()` instead of reusing the allocator stream
  // once we can handle cuBLAS using multiple streams.
  auto stream = active_allocator->GetStream(stream_executor->device_ordinal());
  if (!stream.ok()) {
    LOG(ERROR) << "Failed to create stream: " << stream.status();
    return nullptr;
  }

  // Overwriting twice the L2 cache size evicts all previously cached lines.
  const int64_t cache_flush_bytes =
      options.flush_caches
          ? std::max(
                2 * stream_executor->GetDeviceDescription().l2_cache_size(),
                kMinCacheFlushBytes)
          : 0;
  return absl::WrapUnique(new GpuProfiler(
      stream_executor, active_allocator, std::move(owned_allocator),
      stream.value(), options, cache_flush_bytes));
}

GpuProfiler::~GpuProfiler() {
  if (!cache_flush_buffer_.is_null()) {
    absl::Status status = stream_->BlockHostUntilDone();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to synchronize stream before freeing cache flush "
                    "buffer: "
                 << status;
    }
  }
}

absl::StatusOr<std::unique_ptr<InputBuffers>> GpuProfiler::CreateInputBuffers(
    const Executable* executable, const HloInstruction* instr) {
  if (!cache_flush_buffer_.is_null()) {
    ABSL_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    ABSL_RETURN_IF_ERROR(cache_flush_buffer_.Free());
  }
  cache_flush_alloc_failed_ = false;
  ABSL_ASSIGN_OR_RETURN(
      RedzoneBuffers buffers,
      RedzoneBuffers::FromProgramShape(
          executable->compute_computation_layout().ComputeProgramShape(),
          RedzoneBuffers::BuffersToCreate::kAllInputs,
          options_.should_init_buffers,
          /*should_check_correctness=*/true, options_.redzone_padding_bytes,
          allocator_, stream_));
  auto gpu_buffers = std::make_unique<GpuInputBuffers>();
  gpu_buffers->redzone_buffers = std::move(buffers);

  // Initialize buffers based on operation type
  ABSL_RETURN_IF_ERROR(
      InitializeBuffersIfRequiredByOpcode(instr, *gpu_buffers, stream_));

  return gpu_buffers;
}

absl::StatusOr<ProfileResult> GpuProfiler::Profile(
    Executable* executable, const InputBuffers& buffers) {
  const GpuInputBuffers& gpu_buffers =
      absl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;

  ProfileResult result;
  if (auto* gpu_executable = dynamic_cast<const GpuExecutable*>(executable);
      gpu_executable != nullptr) {
    result.scratch_bytes = GetScratchBytes(*gpu_executable);
  }
  {
    // Warm-up run: route every buffer the executable allocates on-demand
    // (result/output buffers and workspace/scratch buffers alike) through a
    // RedzoneDeviceAddressAllocator, so that any out-of-bounds write lands in
    // a mapped post-redzone rather than causing a GPU VM fault.
    std::optional<se::RedzoneDeviceAddressAllocator> warmup_rz;
    se::DeviceAddressAllocator* warmup_alloc = allocator_;
    if (options_.redzone_padding_bytes > 0) {
      warmup_rz.emplace(stream_, allocator_, options_.redzone_padding_bytes);
      warmup_alloc = &warmup_rz.value();
    }
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());
    ABSL_RETURN_IF_ERROR(Execute(executable, std::move(execution_inputs),
                            /*profile=*/nullptr, warmup_alloc)
                        .status());
    ABSL_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    if (warmup_rz.has_value()) {
      absl::ReaderMutexLock lock(GetGpuMutex(stream_executor_));
      ABSL_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check,
                       warmup_rz->CheckRedzones());
      if (!rz_check.ok()) {
        std::string redzone_failure_msg = rz_check.RedzoneFailureMsg();
        VLOG(1) << "Autotuning candidate discarded: out-of-bounds write "
                   "detected past an allocated buffer. "
                << redzone_failure_msg;
        return absl::InternalError(absl::StrCat(
            "Autotuning candidate rejected: kernel wrote past its allocated "
            "buffer. ",
            redzone_failure_msg));
      }
    }
  }

  // Average over `num_timed_runs` for competitive candidates so run-to-run
  // jitter and first-run post-host-sync transients do not flip selection among
  // close tiles, while stopping after `run == 0` when `ShouldStopAfterFirstRun`
  // determines extra passes cannot plausibly change the winner.
  int num_timed_runs = std::max(1, options_.num_timed_runs);
  absl::Duration total_duration = absl::ZeroDuration();
  ExecutionProfile profile;
  for (int run = 0; run < num_timed_runs; ++run) {
    ABSL_RETURN_IF_ERROR(FlushCaches());
    profile.Clear();
    profile.set_warmup_run_executed(true);
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());

    ABSL_ASSIGN_OR_RETURN(
        ExecutionOutput execution_output,
        Execute(executable, std::move(execution_inputs), &profile, allocator_));

    const int64_t duration_ns = profile.compute_time_ns();
    total_duration += absl::Nanoseconds(duration_ns);
    // Evaluated after `FlushCaches()` so allocation failures on `run == 0`
    // are reflected.
    const bool l2_flush_active =
        cache_flush_bytes_ > 0 && !cache_flush_alloc_failed_;
    if (run == 0 && options_.adaptive_early_stopping &&
        ShouldStopAfterFirstRun(
            duration_ns,
            gpu_buffers.best_duration_ns.load(std::memory_order_relaxed),
            num_timed_runs, l2_flush_active)) {
      num_timed_runs = 1;
    }
    if (run == num_timed_runs - 1) {
      result.output_buffer = execution_output.Commit().ConsumeResult();
    }
  }
  result.duration = total_duration / num_timed_runs;
  return result;
}

void GpuProfiler::RecordValidatedCandidate(const InputBuffers& buffers,
                                           absl::Duration duration) {
  if (!options_.adaptive_early_stopping) {
    return;
  }
  absl::down_cast<const GpuInputBuffers&>(buffers).RecordCandidateDurationNs(
      absl::ToInt64Nanoseconds(duration));
}

absl::Status GpuProfiler::FlushCaches() {
  if (cache_flush_bytes_ <= 0 || cache_flush_alloc_failed_) {
    return absl::OkStatus();
  }
  auto allocate_flush_buffer = [&]() {
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocator_->Allocate(stream_executor_->device_ordinal(),
                             cache_flush_bytes_,
                             /*retry_on_failure=*/false);
    if (!buffer.ok()) {
      cache_flush_alloc_failed_ = true;
      if (!logged_cache_flush_alloc_failure_) {
        logged_cache_flush_alloc_failure_ = true;
        LOG(WARNING) << "Failed to allocate " << cache_flush_bytes_
                     << " bytes to flush caches between autotuning runs, "
                        "profiling with warm caches: "
                     << buffer.status();
      }
    }
    return buffer;
  };

  if (allocator_->AllowsAsynchronousDeallocation()) {
    // Pool and VMM allocators handle stream-ordered reuse in Deallocate(), so
    // release the scratch buffer immediately after scheduling MemZero() rather
    // than holding it across input-buffer creation, warm-up redzones, or
    // timed executions.
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocate_flush_buffer();
    if (!buffer.ok()) {
      return absl::OkStatus();
    }
    return stream_->MemZero(buffer->ptr(), (*buffer)->size());
  }

  if (cache_flush_buffer_.is_null()) {
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocate_flush_buffer();
    if (!buffer.ok()) {
      return absl::OkStatus();
    }
    cache_flush_buffer_ = *std::move(buffer);
  }
  return stream_->MemZero(cache_flush_buffer_.ptr(),
                          cache_flush_buffer_->size());
}

absl::StatusOr<ExecutionOutput> GpuProfiler::Execute(
    Executable* executable, std::vector<ExecutionInput> inputs,
    ExecutionProfile* profile, se::DeviceAddressAllocator* allocator) {
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_->device_ordinal());
  run_options.set_stream(stream_);
  run_options.set_allocator(allocator);
  run_options.set_gpu_executable_run_options(&gpu_opts);
  run_options.set_execution_profile(profile);
  ServiceExecutableRunOptions service_run_options(run_options);
  return executable->ExecuteAsyncOnStreamWrapper(&service_run_options,
                                                 std::move(inputs));
}

absl::Status GpuProfiler::CheckInputBuffers(InputBuffers& buffers) {
  if (options_.redzone_padding_bytes == 0) {
    return absl::OkStatus();
  }
  absl::ReaderMutexLock gpu_lock(GetGpuMutex(stream_executor_));
  const GpuInputBuffers& gpu_buffers =
      absl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  ABSL_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
                   rz_buffers.RedzoneAllocator().CheckRedzones());
  if (rz_check_status.ok()) {
    return absl::OkStatus();
  }
  LOG(ERROR) << "Red zone modified";
  return absl::InternalError(rz_check_status.RedzoneFailureMsg());
}

absl::Status GpuProfiler::CheckOutputBuffer(ScopedShapedBuffer& output,
                                            ScopedShapedBuffer& reference,
                                            float rtol) {
  absl::ReaderMutexLock gpu_lock(GetGpuMutex(stream_executor_));
  return ShapeUtil::ForEachLeafShapeWithStatus(
      reference.on_device_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        std::string diff_report;
        BufferComparator comparator(subshape, rtol,
                                    /*verbose=*/false);

        ABSL_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream_, output.buffer(index),
                                    reference.buffer(index), &diff_report));
        if (outputs_match) {
          return absl::OkStatus();
        }
        if (!diff_report.empty()) {
          return absl::InternalError(absl::StrFormat(
              "Output buffer does not match reference buffer at leaf index "
              "%s:\n%s",
              index.ToString(), diff_report));
        }
        return absl::InternalError(absl::StrFormat(
            "Output buffer does not match reference buffer at leaf index %s.",
            index.ToString()));
      });
}

}  // namespace gpu

}  // namespace xla
