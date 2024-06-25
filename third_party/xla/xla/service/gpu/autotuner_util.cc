/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuner_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// Bump this version whenever you change the structure of the results.
// LINT.IfChange(version)
constexpr int kVersion = 3;
// LINT.ThenChange()

}  // namespace

using AutotuneCacheMap = absl::flat_hash_map<AutotuneCacheKey, AutotuneResult>;

static absl::Mutex autotune_cache_mu(absl::kConstInit);
static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new AutotuneCacheMap();

absl::StatusOr<std::string> GetBase64EncodedSha256Hash(absl::string_view s) {
  llvm::SHA256 sha256;
  sha256.update(llvm::StringRef(s));
  std::array<uint8_t, 32> hash = sha256.final();
  // C++ strict aliasing rules allow reinterpret casting to (const) char*.
  absl::string_view hash_view(reinterpret_cast<const char*>(hash.data()),
                              hash.size());
  std::string base64_encoded_hash;
  TF_RETURN_IF_ERROR(tsl::Base64Encode(hash_view, &base64_encoded_hash));
  return base64_encoded_hash;
}

namespace {

// Get the path corresponding to the given key.
absl::StatusOr<std::string> GetCacheFilePath(absl::string_view cache_dir,
                                             const AutotuneCacheKey& key) {
  if (cache_dir.empty()) {
    return absl::InvalidArgumentError("autotune_cache_dir should not be empty");
  }

  TF_ASSIGN_OR_RETURN(std::string key_hash,
                      GetBase64EncodedSha256Hash(key.ToString()));
  return tsl::io::JoinPath(cache_dir, absl::StrCat(key_hash, ".textproto"));
}

struct ResultAndInserted {
  // The result that ended up in the cache. This is the existing result if
  // inserted is false, and the new result if inserted is true.
  //
  // We return a value, not a pointer, for thread safety reasons.
  AutotuneResult result;
  // Did we insert the given result into the cache?
  bool inserted;
};

ResultAndInserted AddResultToInMemoryCache(const AutotuneCacheKey& key,
                                           AutotuneResult result)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(&autotune_cache_mu);
  auto [it, inserted] = autotune_cache.emplace(key, std::move(result));
  return {it->second, inserted};
}

absl::Status AddResultToFileBasedCacheIfEnabled(const AutotuneCacheKey& key,
                                                AutotuneResult result,
                                                std::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  if (cache_dir.empty()) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key));

  VLOG(1) << "Writing autotune result to file: " << file_path;

  std::string result_str;
  if (!tsl::protobuf::TextFormat::PrintToString(result, &result_str)) {
    return absl::InternalError("Failed to serialize autotune result.");
  }

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple processes are writing to the same
  // file. Also avoids reading incomplete files. (This may not work on all file
  // systems.)
  std::string temp_file_path = tsl::io::GetTempFilename(".textproto");
  tsl::Env* default_env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(default_env, temp_file_path, result_str));
  return default_env->RenameFile(temp_file_path, file_path);
}

absl::StatusOr<ResultAndInserted> AddResultToCaches(const AutotuneCacheKey& key,
                                                    AutotuneResult result,
                                                    std::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  ResultAndInserted result_and_inserted = AddResultToInMemoryCache(key, result);
  if (result_and_inserted.inserted) {
    TF_RETURN_IF_ERROR(AddResultToFileBasedCacheIfEnabled(
        key, result_and_inserted.result, cache_dir));
  }
  return result_and_inserted;
}

std::optional<AutotuneResult> TryToFindInInMemoryCache(
    const AutotuneCacheKey& key) ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(&autotune_cache_mu);
  auto it = autotune_cache.find(key);
  if (it == autotune_cache.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::StatusOr<std::optional<AutotuneResult>>
TryToFindInFileBasedCacheIfEnabled(const AutotuneCacheKey& key,
                                   absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  if (cache_dir.empty()) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key));
  if (!tsl::Env::Default()->FileExists(file_path).ok()) {
    VLOG(1) << "Autotune result file not found: " << file_path;
    return std::nullopt;
  }

  VLOG(1) << "Autotune result file found: " << file_path;
  std::string autotune_result_str;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), file_path,
                                           &autotune_result_str));
  AutotuneResult result;
  if (!tsl::protobuf::TextFormat::ParseFromString(autotune_result_str,
                                                  &result)) {
    return absl::InvalidArgumentError("Failed to parse autotune result.");
  }
  return result;
}

// Sort the results so that they're deterministic.
void SortAutotuneResults(AutotuneResults* results) {
  std::sort(results->mutable_results()->pointer_begin(),
            results->mutable_results()->pointer_end(),
            [](const auto* a, const auto* b) {
              return std::make_pair(absl::string_view(a->device()),
                                    absl::string_view(a->hlo())) <
                     std::make_pair(absl::string_view(b->device()),
                                    absl::string_view(b->hlo()));
            });
}

}  // namespace

// Serialize `results` to string as a proto.
absl::StatusOr<std::string> AutotuneResultsToString(
    const AutotuneResults& results, bool as_textproto) {
  if (as_textproto) {
    std::string textproto;
    if (tsl::protobuf::TextFormat::PrintToString(results, &textproto)) {
      return textproto;
    } else {
      return Internal("Failed to serialize autotune results.");
    }
  }
  return results.SerializeAsString();
}

namespace {
// Serialize a single entry to `results`.
void SerializeAutotuneEntry(AutotuneResults* results, const AutotuneCacheKey& k,
                            const AutotuneResult* res) {
  auto& entry = *results->add_results();
  entry.set_device(std::string(k.GetModelStr()));
  entry.set_hlo(std::string(k.GetHlo()));
  *entry.mutable_result() = *res;
}
}  // namespace

/*static*/ absl::Status AutotunerUtil::SerializeAutotuneResults(
    AutotuneResults* results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const auto& [k, result] : autotune_cache) {
    SerializeAutotuneEntry(results, k, &result);
  }

  results->set_version(kVersion);
  SortAutotuneResults(results);

  return absl::OkStatus();
}

/*static*/ absl::Status AutotunerUtil::LoadAutotuneResults(
    const AutotuneResults& results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const AutotuneResults::Entry& result : results.results()) {
    if (auto [it, inserted] = autotune_cache.emplace(
            AutotuneCacheKey(result.device(), result.hlo()), result.result());
        !inserted) {
      return absl::InternalError(absl::StrCat(
          "Duplicate autotuning result for ", it->first.ToString()));
    }
  }
  return absl::OkStatus();
}

/*static*/ void AutotunerUtil::ClearAutotuneResults() {
  absl::MutexLock lock(&autotune_cache_mu);
  autotune_cache.clear();
}

/*static*/ bool AutotunerUtil::ResultCacheIsEmpty() {
  absl::MutexLock lock(&autotune_cache_mu);
  return autotune_cache.empty();
}

/* static*/ absl::StatusOr<se::DeviceMemoryBase> AutotunerUtil::CreateBuffer(
    se::RedzoneAllocator& allocator, const Shape& shape,
    const AutotuneConfig& config, int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      allocator.AllocateBytes(ShapeUtil::ByteSizeOf(shape)));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), shape.element_type(), &rng_state,
                     buffer);
  }
  return buffer;
}

namespace {
std::string ToCanonicalString(const HloInstruction* instr) {
  auto options = HloPrintOptions::Canonical();
  if (instr->opcode() != HloOpcode::kFusion) {
    options.set_print_backend_config(true);
    return instr->ToString(options);
  }
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  options.set_print_infeed_outfeed_config(false);
  options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  options.set_canonicalize_computations(true);

  // TODO(b/266210099): This is unsound. We should probably do the fingerprint
  // of the HLO computation proto instead.
  return instr->called_computations()[0]->ToString(options);
}

}  // namespace

AutotuneCacheKey::AutotuneCacheKey(absl::string_view model_str,
                                   const HloInstruction& instr)
    : AutotuneCacheKey(model_str, ToCanonicalString(&instr)) {}

namespace {
absl::StatusOr<std::optional<AutotuneResult>> TryFindInCache(
    const AutotuneCacheKey& key, absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  std::optional<AutotuneResult> opt_result = TryToFindInInMemoryCache(key);
  if (opt_result.has_value()) {
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "In-memory autotune cache hit";
    } else if (VLOG_IS_ON(2)) {
      LOG(INFO) << "In-memory autotune cache hit: key = " << key.ToString();
    }
    return opt_result;
  }

  TF_ASSIGN_OR_RETURN(opt_result,
                      TryToFindInFileBasedCacheIfEnabled(key, cache_dir));
  if (opt_result.has_value()) {
    AddResultToInMemoryCache(key, opt_result.value());

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "File-based autotune cache hit";
    } else if (VLOG_IS_ON(2)) {
      LOG(INFO) << "File-based autotune cache hit: key = " << key.ToString();
    }
    return opt_result;
  }

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "Autotune cache miss";
  } else if (VLOG_IS_ON(2)) {
    LOG(INFO) << "Autotune cache miss: key = " << key.ToString();
  }
  return std::nullopt;
}
}  // namespace

/*static*/ AutotuneCacheKey AutotunerUtil::GetKey(
    const HloInstruction* instr, const AutotuneConfig& config) {
  return AutotuneCacheKey(config.GetModelStr(), *instr);
}

/*static*/ absl::StatusOr<bool> AutotunerUtil::IsInCache(
    const AutotuneCacheKey& key, const AutotuneConfig& config) {
  TF_ASSIGN_OR_RETURN(std::optional<AutotuneResult> opt_res,
                      TryFindInCache(key, config.autotune_cache_dir()));
  return opt_res.has_value();
}

/*static*/ absl::StatusOr<bool> AutotunerUtil::AddResult(
    const AutotuneCacheKey& key, AutotuneResult result,
    const AutotuneConfig& config) {
  TF_ASSIGN_OR_RETURN(
      ResultAndInserted result_and_inserted,
      AddResultToCaches(key, std::move(result), config.autotune_cache_dir()));
  return result_and_inserted.inserted;
}

/*static*/ absl::StatusOr<AutotuneResult> AutotunerUtil::Autotune(
    const HloInstruction* instr, const AutotuneConfig& config,
    const AutotuneNoCacheFn& autotune_fn) {
  const AutotuneCacheKey key = GetKey(instr, config);
  TF_ASSIGN_OR_RETURN(std::optional<AutotuneResult> opt_res,
                      TryFindInCache(key, config.autotune_cache_dir()));
  if (opt_res.has_value()) {
    return opt_res.value();
  }

  // Cache miss.
  if (config.should_require_complete_aot_autotune_results()) {
    return NotFound(
        "Complete XLA AOT autotuning results are required, but no AOT result "
        "was found for key: %s",
        key.ToString());
  }

  TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result, autotune_fn());

  TF_ASSIGN_OR_RETURN(ResultAndInserted result_and_inserted,
                      AddResultToCaches(key, std::move(autotune_result),
                                        config.autotune_cache_dir()));
  return result_and_inserted.result;
}

namespace {

bool IsTextProtoPath(absl::string_view file_path) {
  return absl::EndsWith(file_path, ".txt") ||
         absl::EndsWith(file_path, ".textproto") ||
         absl::EndsWith(file_path, ".prototxt") ||
         absl::EndsWith(file_path, ".pbtxt");
}

}  // anonymous namespace

/*static*/ absl::Status AutotunerUtil::LoadAutotuneResults(
    absl::string_view data, bool as_textproto) {
  AutotuneResults results;
  // The cast here is necessary for MacOS builds.
  bool parse_success =
      as_textproto ? tsl::protobuf::TextFormat::ParseFromString(
                         std::string(data), &results)             // NOLINT
                   : results.ParseFromString(std::string(data));  // NOLINT
  if (!parse_success) {
    return absl::InvalidArgumentError(
        "Failed to parse autotune results string.");
  }
  if (results.version() != kVersion) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Version mismatch in autotune results. Expected %d but was %d",
        kVersion, results.version()));
  }

  TF_RETURN_IF_ERROR(LoadAutotuneResults(results));
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<std::string> AutotunerUtil::SerializeAutotuneResults(
    bool as_textproto) {
  AutotuneResults results;
  TF_RETURN_IF_ERROR(SerializeAutotuneResults(&results));
  return AutotuneResultsToString(results, as_textproto);
}

/*static*/ absl::Status AutotunerUtil::SerializeAutotuneResultsToFile(
    const AutotuneResults& results, absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());
  TF_RET_CHECK(results.version() > 0)
      << "Did you call SerializeAutotuneResults to get this AutotuneResults?";

  std::string resolved_path;
  if (!tsl::io::ResolveTestPrefixes(file_path, resolved_path)) {
    return FailedPrecondition("File path can not be resolved: %s", file_path);
  }

  TF_ASSIGN_OR_RETURN(
      std::string autotune_results_str,
      AutotuneResultsToString(results, IsTextProtoPath(resolved_path)));
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(), resolved_path,
                                            autotune_results_str));
  LOG(INFO) << "Autotune results serialized to file: " << resolved_path;

  return absl::OkStatus();
}

/*static*/ absl::Status AutotunerUtil::SerializeAutotuneResultsToFile(
    absl::string_view file_path) {
  AutotuneResults results;
  TF_RETURN_IF_ERROR(SerializeAutotuneResults(&results));
  return SerializeAutotuneResultsToFile(results, file_path);
}

/*static*/ absl::Status AutotunerUtil::LoadAutotuneResultsFromFile(
    absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());

  std::string resolved_path;
  if (!tsl::io::ResolveTestPrefixes(file_path, resolved_path)) {
    return FailedPrecondition("File path can not be resolved: %s", file_path);
  }

  if (!tsl::Env::Default()->FileExists(resolved_path).ok()) {
    return FailedPrecondition("Autotune results file does not exist: %s",
                              resolved_path);
  }
  std::string autotune_results_str;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), resolved_path,
                                           &autotune_results_str));

  TF_RETURN_IF_ERROR(LoadAutotuneResults(autotune_results_str,
                                         IsTextProtoPath(resolved_path)));

  LOG(INFO) << "Autotune results loaded from file: " << resolved_path;

  return absl::OkStatus();
}

/*static*/ absl::StatusOr<se::RedzoneAllocator>
AutotunerUtil::CreateRedzoneAllocator(const AutotuneConfig& config,
                                      const DebugOptions& opts) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, config.GetStream());
  return se::RedzoneAllocator(
      stream, config.GetAllocator(), PtxOptsFromDebugOptions(opts),
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      /*redzone_size=*/config.should_check_correctness()
          ? opts.xla_gpu_redzone_padding_bytes()
          : 0);
}

}  // namespace gpu
}  // namespace xla
