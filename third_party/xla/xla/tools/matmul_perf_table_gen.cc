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

#include "xla/tools/matmul_perf_table_gen.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/matmul_interpolator_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Defines how many times do we profile a matrix multiplication of interest.
constexpr size_t kNumProfilingRuns = 5;

template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

struct ProfilingResult {
  std::string device_info;
  HloInstructionProto hlo_proto;
  std::vector<HloInstructionProto> operands;
  std::string fingerprint;
  int64_t clock_cycles;
  int64_t flops;

  struct Hash {
    size_t operator()(const ProfilingResult& profiling_result) const {
      return absl::HashOf(profiling_result.device_info,
                          profiling_result.fingerprint);
    }
  };

  struct Eq {
    bool operator()(const ProfilingResult& lhs,
                    const ProfilingResult& rhs) const {
      return lhs.device_info == rhs.device_info &&
             lhs.fingerprint == rhs.fingerprint;
    }
  };
};

struct ExplicitSpec {
  std::unique_ptr<HloModule> module;
};

struct PathSpec {
  std::string filepath;
};

struct StaticSpec {
  int b;
  int m;
  int n;
  int k;
  std::string dtype_lhs;
  std::string dtype_rhs;
  std::string dtype_out;

  static absl::StatusOr<StaticSpec> FromDotProfile(
      const HloInstructionProfile& profile) {
    const HloInstructionProto& instr = profile.instruction();
    CHECK_EQ(instr.opcode(), HloOpcodeString(HloOpcode::kDot));
    const DotDimensionNumbers& dot_dims = instr.dot_dimension_numbers();
    TF_ASSIGN_OR_RETURN(Shape lhs,
                        Shape::FromProto(profile.operands(0).shape()));
    TF_ASSIGN_OR_RETURN(Shape rhs,
                        Shape::FromProto(profile.operands(1).shape()));
    int b = 1, m = 1, n = 1, k = 1;
    for (int dim : dot_dims.lhs_batch_dimensions()) {
      b *= ShapeUtil::GetDimension(lhs, dim);
    }
    for (int dim : dot_dims.lhs_contracting_dimensions()) {
      k *= ShapeUtil::GetDimension(lhs, dim);
    }
    for (int dim : GetNonContractingDims(lhs.dimensions().size(),
                                         dot_dims.lhs_contracting_dimensions(),
                                         dot_dims.lhs_batch_dimensions())) {
      m *= ShapeUtil::GetDimension(lhs, dim);
    }
    for (int dim : GetNonContractingDims(rhs.dimensions().size(),
                                         dot_dims.rhs_contracting_dimensions(),
                                         dot_dims.rhs_batch_dimensions())) {
      n *= ShapeUtil::GetDimension(rhs, dim);
    }

    StaticSpec spec;
    spec.b = b;
    spec.m = m;
    spec.n = n;
    spec.k = k;
    spec.dtype_lhs =
        primitive_util::LowercasePrimitiveTypeName(lhs.element_type());
    spec.dtype_rhs =
        primitive_util::LowercasePrimitiveTypeName(rhs.element_type());
    spec.dtype_out = primitive_util::LowercasePrimitiveTypeName(
        profile.instruction().shape().element_type());
    return spec;
  }
};

using EntrySpec = std::variant<StaticSpec, PathSpec>;

void ReportProgress(absl::string_view prefix, int i, int size) {
  if (i % (size / std::min(size, 10)) == 0) {
    LOG(INFO) << prefix << ": " << 100 * i / size << "%.";
  }
}

std::unique_ptr<HloModule> GetModule(absl::string_view lhs_dtype,
                                     absl::string_view rhs_dtype,
                                     absl::string_view out_dtype, int b, int m,
                                     int n, int k) {
  std::string text =
      absl::Substitute(R"(
    HloModule m

    ENTRY e {
      lhs = $0[$6,$3,$5] parameter(0)
      rhs = $1[$6,$5,$4] parameter(1)
      ROOT _ = $2[$6,$3,$4] dot(lhs,rhs), lhs_contracting_dims={2},
        rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
    }
  )",
                       lhs_dtype, rhs_dtype, out_dtype, m, n, k, b);

  auto parsed = ParseAndReturnUnverifiedModule(text);
  CHECK_OK(parsed.status());
  return *std::move(parsed);
}

void Measure(HloRunner& runner, OpaqueExecutable* executable,
             const std::vector<Literal>& args_small,
             const std::vector<Literal>& args_large) {
  CHECK_OK(runner.ExecuteWithExecutable(executable, args_small).status());
  CHECK_OK(runner.ExecuteWithExecutable(executable, args_large).status());
}

void AddDotsFromStaticSpec(const MatmulPerfTableGen::Config& config,
                           std::vector<EntrySpec>& specs) {
  auto inc = [](uint32_t i, const MatmulPerfTableGen::StepSpec& spec) {
    if (spec.step > 0) {
      return i + spec.step;
    }
    if (spec.factor > 0) {
      return i * spec.factor;
    }
    return i;
  };

  MatmulPerfTableGen::StepSpec b_spec = config.b_spec;
  MatmulPerfTableGen::StepSpec m_spec = config.m_spec;
  MatmulPerfTableGen::StepSpec n_spec = config.n_spec;
  MatmulPerfTableGen::StepSpec k_spec = config.k_spec;
  for (const MatmulPerfTableGen::DataTypeSpec& dtype : config.dtypes) {
    for (int b = b_spec.start; b <= b_spec.stop; b = inc(b, b_spec)) {
      for (int m = m_spec.start; m <= m_spec.stop; m = inc(m, m_spec)) {
        for (int n = n_spec.start; n <= n_spec.stop; n = inc(n, n_spec)) {
          for (int k = k_spec.start; k <= k_spec.stop; k = inc(k, k_spec)) {
            StaticSpec spec;
            spec.b = b;
            spec.m = m;
            spec.k = k;
            spec.n = n;
            spec.dtype_lhs = dtype.lhs_dtype;
            spec.dtype_rhs = dtype.rhs_dtype;
            spec.dtype_out = dtype.out_dtype;
            specs.push_back(spec);
          }
        }
      }
    }
  }
}

void AddDotsFromHlos(const std::string& hlo_scan_path,
                     std::vector<EntrySpec>& specs) {
  if (hlo_scan_path.empty()) {
    return;
  }

  // `IsDirectory` returns FAILED_PRECONDITION if file exists but is not a
  // directory. Add it to scanning.
  if (auto is_dir = tsl::Env::Default()->IsDirectory(hlo_scan_path);
      absl::IsFailedPrecondition(is_dir)) {
    PathSpec spec;
    spec.filepath = hlo_scan_path;
    specs.push_back(spec);
    return;
  }

  std::vector<std::string> filenames;
  CHECK_OK(tsl::Env::Default()->GetChildren(hlo_scan_path, &filenames));
  for (const std::string& filename : filenames) {
    PathSpec spec;
    spec.filepath = absl::StrCat(hlo_scan_path, "/", filename);
    specs.push_back(spec);
  }
}

std::unique_ptr<HloModule> GetModule(const std::string& hlo) {
  auto module = ParseAndReturnUnverifiedModule(hlo);
  if (!module.ok()) {
    LOG(ERROR) << "Cannot parse: " << hlo;
    return nullptr;
  }
  return std::move(*module);
}

std::unique_ptr<HloModule> CreateDotModule(HloInstruction* instr) {
  // Create module.
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  auto module = std::make_unique<HloModule>("module", config);

  // Create entry computation with dot.
  HloComputation::Builder entry_builder("entry");
  HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, dot->operand(0)->shape(), "p0"));
  HloInstruction* p1 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(1, dot->operand(1)->shape(), "p1"));
  entry_builder.AddInstruction(HloInstruction::CreateDot(
      dot->shape(), p0, p1, dot->dot_dimension_numbers(),
      dot->precision_config()));
  module->AddEntryComputation(entry_builder.Build());

  return module;
}

std::vector<ExplicitSpec> GetExplicitSpecs(
    const std::vector<EntrySpec>& entry_specs) {
  std::vector<ExplicitSpec> specs;
  for (int i = 0; i < entry_specs.size(); i++) {
    const EntrySpec& entry_spec = entry_specs[i];
    std::visit(
        Overload{
            [&specs](const PathSpec& spec) {
              std::string hlo;
              CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), spec.filepath,
                                             &hlo));
              std::unique_ptr<HloModule> model_module = GetModule(hlo);
              if (model_module == nullptr) {
                return;
              }
              hlo_query::ForEachInstructionWithOpcode(
                  *model_module, HloOpcode::kDot,
                  [&specs](HloInstruction* instr) {
                    specs.emplace_back(ExplicitSpec{CreateDotModule(instr)});
                  });
            },
            [&specs](const StaticSpec spec) {
              specs.emplace_back(ExplicitSpec{
                  GetModule(spec.dtype_lhs, spec.dtype_rhs, spec.dtype_out,
                            spec.b, spec.m, spec.n, spec.k)});
            }},
        entry_spec);
    ReportProgress("Parsing modules progress", i + 1, entry_specs.size());
  }
  return specs;
}

std::string CanonicalKey(HloModule& module) {
  return module.GetFingerprint128();
}

std::vector<ExplicitSpec> Deduplicate(std::vector<ExplicitSpec>& specs) {
  absl::flat_hash_set<std::string> seen_keys;
  std::vector<ExplicitSpec> deduplicated_specs;
  for (ExplicitSpec& spec : specs) {
    std::string key = CanonicalKey(*(spec.module));
    if (seen_keys.contains(key)) {
      continue;
    }
    seen_keys.insert(key);
    deduplicated_specs.push_back(std::move(spec));
  }
  return deduplicated_specs;
}

// Gets # of FMAs instructions from a `dot`.
int64_t GetFlops(const HloDotInstruction& dot) {
  int64_t fmas = 1;

  auto dim_size = [](const HloInstruction& instr, int idx) {
    return ShapeUtil::GetDimension(instr.shape(), idx);
  };

  const DotDimensionNumbers& dot_dims = dot.dot_dimension_numbers();
  const HloInstruction& lhs = *dot.operand(0);
  const HloInstruction& rhs = *dot.operand(1);

  // Get non-contracting dims
  for (int dim : GetNonContractingDims(lhs.shape().dimensions().size(),
                                       dot_dims.lhs_contracting_dimensions(),
                                       dot_dims.lhs_batch_dimensions())) {
    fmas *= dim_size(lhs, dim);
  }
  for (int dim : GetNonContractingDims(rhs.shape().dimensions().size(),
                                       dot_dims.rhs_contracting_dimensions(),
                                       dot_dims.rhs_batch_dimensions())) {
    fmas *= dim_size(rhs, dim);
  }

  // Get contracting dim.
  for (int dim : dot.dot_dimension_numbers().lhs_contracting_dimensions()) {
    fmas *= dim_size(lhs, dim);
  }

  // Get batch dim
  for (int dim : dot.dot_dimension_numbers().lhs_batch_dimensions()) {
    CHECK_EQ(dim_size(lhs, dim), dim_size(rhs, dim));
    fmas *= dim_size(lhs, dim);
  }

  return fmas * 2;  // Every FMA is 2 floating point ops.
}

}  // namespace

std::unique_ptr<OpaqueExecutable> MatmulPerfTableGen::Compile(
    std::unique_ptr<HloModule> module) {
  auto compiled =
      runner_.CreateExecutable(std::move(module), /*run_hlo_passes=*/true);
  CHECK_OK(compiled.status());
  return std::move(*compiled);
}

absl::Duration MatmulPerfTableGen::Profile(std::unique_ptr<HloModule> module) {
  VLOG(1) << "Profiling module: " << module->ToString();

  // Flip flop between arguments to prevent caching.
  std::minstd_rand0 engine;

  std::vector<Literal> args_small = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/false)
                                        .value();
  std::vector<Literal> args_large = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/true)
                                        .value();

  std::unique_ptr<OpaqueExecutable> compiled = Compile(std::move(module));

  // First run to warm up stuff.
  CHECK_OK(runner_.ExecuteWithExecutable(compiled.get(), args_small).status());

  // Run matrix multiplications but do not trace.
  if (config_.dry_run) {
    for (int i = 0; i < kNumProfilingRuns; i++) {
      Measure(runner_, compiled.get(), args_small, args_large);
    }
    return absl::Nanoseconds(42);
  }

  // Trace `kNumProfilingRuns` times to get decent measurement.
  std::unique_ptr<HloOpProfiler::KernelTracer> tracer =
      HloOpProfiler::GetKernelTracer();
  for (int i = 0; i < kNumProfilingRuns; i++) {
    Measure(runner_, compiled.get(), args_small, args_large);
  }

  return absl::Nanoseconds(std::move(*tracer).getMedianKernelTimeNs());
}

absl::StatusOr<DeviceHloInstructionProfiles> MatmulPerfTableGen::Merge(
    absl::string_view filepath) {
  DeviceHloInstructionProfiles result;
  std::vector<std::string> filenames;
  CHECK_OK(tsl::Env::Default()->GetChildren(std::string(filepath), &filenames));

  absl::flat_hash_set<ProfilingResult, ProfilingResult::Hash,
                      ProfilingResult::Eq>
      profiling_results;
  uint64_t profiling_results_counter = 0;
  for (const std::string& filename : filenames) {
    // Read file.
    std::string profile_path = absl::StrCat(filepath, "/", filename);
    DeviceHloInstructionProfiles partial_profile;

    CHECK_OK(tsl::Env::Default()->FileExists(profile_path));
    if (!tsl::ReadTextOrBinaryProto(tsl::Env::Default(), profile_path,
                                    &partial_profile)
             .ok()) {
      LOG(WARNING) << "Cannot read :" << profile_path;
      continue;
    }

    for (auto& [device_descriptor, data] : partial_profile.entries()) {
      for (const HloInstructionProfile& profile : data.entries()) {
        CHECK(!profile.fingerprint().empty())
            << "Expected fingerprint to deduplicate: " << profile.DebugString();

        ProfilingResult profiling_result{
            device_descriptor,
            std::move(profile.instruction()),
            {
                profile.operands().begin(),
                profile.operands().end(),
            },
            std::move(profile.fingerprint()),
            profile.clock_cycles(),
            profile.flops(),
        };
        profiling_results.insert(profiling_result);
        profiling_results_counter++;
      }
    }
  }
  LOG(INFO) << "Merging and deduplication entries count. Before "
            << profiling_results_counter << ", after "
            << profiling_results.size() << ".";
  for (const ProfilingResult& profiling_result : profiling_results) {
    std::string device_descriptor = profiling_result.device_info;
    if (!result.mutable_entries()->contains(device_descriptor)) {
      result.mutable_entries()->insert({device_descriptor, {}});
    }

    HloInstructionProfile profile_proto;
    *profile_proto.mutable_instruction() =
        std::move(profiling_result.hlo_proto);
    for (auto op : profiling_result.operands) {
      *profile_proto.add_operands() = std::move(op);
    }
    profile_proto.set_flops(profiling_result.flops);
    profile_proto.set_clock_cycles(profiling_result.clock_cycles);
    profile_proto.set_fingerprint(profiling_result.fingerprint);

    *result.mutable_entries()->at(device_descriptor).add_entries() =
        std::move(profile_proto);
  }

  return result;
}

DeviceHloInstructionProfiles MatmulPerfTableGen::ComputeTable() {
  gpu::DeviceHloInstructionProfiles device_profiles;
  gpu::HloInstructionProfileList profile_list;

  std::vector<EntrySpec> entry_specs;

  // Sweep over statically defined search space.
  AddDotsFromStaticSpec(config_, entry_specs);

  // Sweep over provided HLOs.
  AddDotsFromHlos(config_.hlo_scan_path, entry_specs);

  // Transform to explicit specs.
  std::vector<ExplicitSpec> specs = GetExplicitSpecs(entry_specs);
  entry_specs.clear();

  LOG(INFO) << "Specs size before deduplication: " << specs.size();
  specs = Deduplicate(specs);
  LOG(INFO) << "Specs size after deduplication: " << specs.size();

  std::minstd_rand0 engine;
  std::shuffle(specs.begin(), specs.end(), engine);

  auto& device_info =
      runner_.backend().stream_executors()[0]->GetDeviceDescription();

  for (int i = 0; i < specs.size(); i++) {
    ExplicitSpec& spec = specs[i];

    std::unique_ptr<HloModule> module = std::move(spec.module);
    CHECK_NOTNULL(module);
    CHECK_EQ(module->entry_computation()->root_instruction()->opcode(),
             HloOpcode::kDot);

    HloInstruction* instr = module->entry_computation()->root_instruction();
    HloInstructionProto instr_proto = instr->ToProto();

    HloInstructionProfile entry;
    *entry.mutable_fingerprint() = CanonicalKey(*module);
    *entry.mutable_instruction() = instr_proto;
    for (auto* operand : instr->operands()) {
      *entry.add_operands() = operand->ToProto();
    }

    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    int64_t fmas = GetFlops(*dot);
    absl::Duration time = Profile(std::move(module));
    entry.set_clock_cycles(device_info.clock_rate_ghz() *
                           absl::ToInt64Nanoseconds(time));
    entry.set_flops(fmas * 1e9 / absl::ToInt64Nanoseconds(time));

    *profile_list.add_entries() = entry;

    ReportProgress("Profiling progress", i + 1, specs.size());
  }
  std::string device_key = gpu::HloOpProfiles::GetProfileName(device_info);
  device_profiles.mutable_entries()->insert({device_key, profile_list});
  return device_profiles;
}

/*static*/ absl::StatusOr<GemmPerfTable> MatmulPerfTableGen::Compact(
    const DeviceHloInstructionProfiles& profiles) {
  GemmPerfTable result;
  for (const auto& [device_info, profile_list] : profiles.entries()) {
    if (!result.entries().contains(device_info)) {
      result.mutable_entries()->insert({device_info, {}});
    }
    absl::flat_hash_map<std::array<int64_t, 4>, GemmPerfTableEntry>
        gemm_perf_table_entry;
    for (const HloInstructionProfile& profile : profile_list.entries()) {
      TF_ASSIGN_OR_RETURN(StaticSpec spec, StaticSpec::FromDotProfile(profile));

      std::array<int64_t, 4> key = {spec.b, spec.m, spec.k, spec.n};
      if (!gemm_perf_table_entry.contains(key)) {
        GemmPerfTableEntry entry;
        entry.set_b(spec.b);
        entry.set_m(spec.m);
        entry.set_k(spec.k);
        entry.set_n(spec.n);
        gemm_perf_table_entry[key] = std::move(entry);
      }

      std::string dtype_key =
          MatmulDTypeKey(spec.dtype_lhs, spec.dtype_rhs, spec.dtype_out)
              .KeyString();

      GemmPerfTableEntry& entry = gemm_perf_table_entry[key];
      entry.mutable_flops()->insert({dtype_key, profile.flops()});
    }

    for (const auto& [_, entry] : gemm_perf_table_entry) {
      *result.mutable_entries()->at(device_info).add_entries() =
          std::move(entry);
    }
  }
  return result;
}

absl::Status MatmulPerfTableGen::Dump(
    const DeviceHloInstructionProfiles& table) {
  if (config_.output == "stdout") {
    LOG(INFO) << table.DebugString();
    return absl::OkStatus();
  }

  DeviceHloInstructionProfiles file;
  if (tsl::Env::Default()->FileExists(config_.output).ok()) {
    TF_RETURN_IF_ERROR(
        tsl::ReadTextOrBinaryProto(tsl::Env::Default(), config_.output, &file));
  }

  for (const auto& [sm_ver, entries] : table.entries()) {
    if (file.entries().contains(sm_ver)) {
      file.mutable_entries()->at(sm_ver).MergeFrom(entries);
    } else {
      file.MergeFrom(table);
    }

    if (absl::StrContains(config_.output, ".pbtxt")) {
      TF_RETURN_IF_ERROR(
          tsl::WriteTextProto(tsl::Env::Default(), config_.output, file));
      continue;
    }
    if (absl::StrContains(config_.output, ".pb")) {
      TF_RETURN_IF_ERROR(
          tsl::WriteBinaryProto(tsl::Env::Default(), config_.output, file));
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported file: ", config_.output,
                     ". Expecting .pb or .pbtxt suffix."));
  }
  return absl::OkStatus();
}

absl::Status MatmulPerfTableGen::Dump(const GemmPerfTable& table) {
  if (config_.output == "stdout") {
    LOG(INFO) << table.DebugString();
    return absl::OkStatus();
  }
  if (absl::StrContains(config_.output, ".pbtxt")) {
    return tsl::WriteTextProto(tsl::Env::Default(), config_.output, table);
  }
  if (absl::StrContains(config_.output, ".pb")) {
    return tsl::WriteBinaryProto(tsl::Env::Default(), config_.output, table);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported file: ", config_.output,
                   ". Expecting .pb or .pbtxt suffix."));
}

}  // namespace xla::gpu
