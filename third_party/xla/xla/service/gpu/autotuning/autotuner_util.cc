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

#include "xla/service/gpu/autotuning/autotuner_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "absl/time/clock.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {
namespace {

// Bump this version whenever you change the structure of the results.
// LINT.IfChange(version)
constexpr int kVersion = 3;
// LINT.ThenChange()

}  // namespace

using AutotuneCacheMap = absl::flat_hash_map<AutotuneCacheKey, AutotuneResult>;
using ResultAndInserted = AutotunerUtil::ResultAndInserted;

static absl::Mutex autotune_cache_mu(absl::kConstInit);
static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new AutotuneCacheMap();
static AutotunerUtil::CacheStats autotune_cache_stats
    ABSL_GUARDED_BY(autotune_cache_mu);

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
                                             absl::string_view key_hash) {
  if (cache_dir.empty()) {
    return absl::InvalidArgumentError("autotune_cache_dir should not be empty");
  }

  return tsl::io::JoinPath(cache_dir, absl::StrCat(key_hash, ".textproto"));
}

ResultAndInserted AddResultToInMemoryCache(const AutotuneCacheKey& key,
                                           AutotuneResult result)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(autotune_cache_mu);
  auto [it, inserted] = autotune_cache.emplace(key, std::move(result));
  return {it->second, inserted};
}

absl::Status AddResultToFileBasedCacheIfEnabled(
    const AutotuneCacheKey& key, AutotuneResult result,
    absl::string_view cache_dir,
    DebugOptions::AutotuneCacheMode autotune_cache_mode)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  if (cache_dir.empty() ||
      autotune_cache_mode == DebugOptions::AUTOTUNE_CACHE_MODE_READ) {
    return absl::OkStatus();
  }

  tsl::Env* default_env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(CreateDirIfNeeded(std::string(cache_dir), default_env));

  TF_ASSIGN_OR_RETURN(std::string key_hash,
                      GetBase64EncodedSha256Hash(key.ToString()));

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key_hash));

  VLOG(1) << "Writing autotune result to file: " << file_path;

  std::string result_str;
  if (!tsl::protobuf::TextFormat::PrintToString(result, &result_str)) {
    return absl::InternalError("Failed to serialize autotune result.");
  }

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple threads are writing to the same
  // file. Also avoids reading incomplete files. (This may not work on all file
  // systems.)
  std::string tmp_dir = tsl::io::JoinPath(cache_dir, "tmp");
  TF_RETURN_IF_ERROR(CreateDirIfNeeded(tmp_dir, default_env));
  int64_t time_stamp = absl::GetCurrentTimeNanos();

  std::string temp_file_path = tsl::io::JoinPath(
      tmp_dir, absl::StrCat("tmp_per_fusion_cache_", key_hash, "_",
                            std::to_string(time_stamp), ".textproto"));

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(default_env, temp_file_path, result_str));
  return default_env->RenameFile(temp_file_path, file_path);
}

std::optional<AutotuneResult> TryToFindInInMemoryCache(
    const AutotuneCacheKey& key) ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  absl::MutexLock lock(autotune_cache_mu);
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

  TF_ASSIGN_OR_RETURN(std::string key_hash,
                      GetBase64EncodedSha256Hash(key.ToString()));

  TF_ASSIGN_OR_RETURN(const std::string file_path,
                      GetCacheFilePath(cache_dir, key_hash));
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
    }
    return Internal("Failed to serialize autotune results.");
  }
  return results.SerializeAsString();
}

namespace {
// Serialize a single entry to `results`.
void SerializeAutotuneEntry(AutotuneResults* results, const AutotuneCacheKey& k,
                            const AutotuneResult* res) {
  auto& entry = *results->add_results();
  entry.set_device(k.GetModelStr());
  entry.set_hlo(k.GetHlo());
  entry.set_version(k.GetVersion());
  *entry.mutable_result() = *res;
}
}  // namespace

/*static*/ absl::Status AutotunerUtil::SerializeAutotuneResults(
    AutotuneResults* results, std::optional<const AutotuneCacheKeySet*> keys) {
  absl::MutexLock lock(autotune_cache_mu);
  for (const auto& [k, result] : autotune_cache) {
    if (!keys.has_value() || keys.value()->contains(k)) {
      SerializeAutotuneEntry(results, k, &result);
    }
  }

  results->set_version(kVersion);
  SortAutotuneResults(results);

  return absl::OkStatus();
}

/*static*/ absl::Status AutotunerUtil::LoadAutotuneResults(
    const AutotuneResults& results, bool allow_override) {
  absl::MutexLock lock(autotune_cache_mu);
  for (const AutotuneResults::Entry& result : results.results()) {
    AutotuneCacheKey key(result.device(), result.hlo(), result.version());
    if (allow_override) {
      autotune_cache.insert_or_assign(key, result.result());
    } else {
      if (auto [it, inserted] = autotune_cache.emplace(key, result.result());
          !inserted) {
        return absl::InternalError(absl::StrCat(
            "Duplicate autotuning result for ", it->first.ToString()));
      }
    }
  }
  return absl::OkStatus();
}

/*static*/ void AutotunerUtil::ClearAutotuneResults() {
  absl::MutexLock lock(autotune_cache_mu);
  autotune_cache.clear();
}

/*static*/ bool AutotunerUtil::ResultCacheIsEmpty() {
  absl::MutexLock lock(autotune_cache_mu);
  return autotune_cache.empty();
}

namespace {
enum class CacheType { kNone, kInMemory, kOnDisk };

absl::StatusOr<std::pair<CacheType, std::optional<AutotuneResult>>>
TryFindInAllCacheTypes(const AutotuneCacheKey& key, absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  std::optional<AutotuneResult> opt_result = TryToFindInInMemoryCache(key);
  if (opt_result.has_value()) {
    return std::make_pair(CacheType::kInMemory, opt_result);
  }

  TF_ASSIGN_OR_RETURN(opt_result,
                      TryToFindInFileBasedCacheIfEnabled(key, cache_dir));
  if (opt_result.has_value()) {
    AddResultToInMemoryCache(key, opt_result.value());
    return std::make_pair(CacheType::kOnDisk, opt_result);
  }

  return std::make_pair(CacheType::kNone, std::nullopt);
}
}  // namespace

absl::StatusOr<AutotuneConfig> AutotuneConfig::FromDebugOptions(
    const DeviceOrDevicelessConfig& config, const DebugOptions& opts) {
  int autotune_level = opts.xla_gpu_autotune_level();

  bool should_init_buffers = autotune_level >= 2;
  bool should_reinit_output_buffer = autotune_level >= 3;
  bool should_check_correctness = autotune_level >= 4;
  bool should_skip_wrong_results = autotune_level >= 5;

  bool should_crash_on_check_failure =
      opts.xla_gpu_crash_on_verification_failures();

  bool exhaustive_tiling_search = opts.xla_gpu_exhaustive_tiling_search();

  bool should_require_complete_aot_autotune_results =
      opts.xla_gpu_require_complete_aot_autotune_results();

  std::string autotune_cache_dir = opts.xla_gpu_per_fusion_autotune_cache_dir();
  DebugOptions_AutotuneCacheMode autotune_cache_mode =
      opts.xla_gpu_experimental_autotune_cache_mode();

  std::optional<std::vector<AutotuneResult::TritonGemmKey>>
      gemm_config_overrides;
  const std::string& override_file =
      opts.xla_gpu_gemm_autotuner_override_file();
  if (!override_file.empty()) {
    std::string file_content;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), override_file,
                                             &file_content));
    TritonGemmConfigsProto configs;
    if (!tsl::protobuf::TextFormat::ParseFromString(file_content, &configs)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not parse override file: ", override_file));
    }
    gemm_config_overrides.emplace();
    absl::c_copy(configs.config(), std::back_inserter(*gemm_config_overrides));
    LOG(INFO) << "Loaded " << gemm_config_overrides->size()
              << " gemm config overrides from " << override_file;
  }

  return AutotuneConfig(
      config, should_init_buffers, should_reinit_output_buffer,
      should_check_correctness, should_skip_wrong_results,
      should_crash_on_check_failure, exhaustive_tiling_search,
      should_require_complete_aot_autotune_results, autotune_cache_dir,
      autotune_cache_mode, gemm_config_overrides);
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
      AddResultToCaches(key, std::move(result), config.autotune_cache_dir(),
                        config.autotune_cache_mode()));
  return result_and_inserted.inserted;
}

/*static*/ absl::StatusOr<AutotuneResult> AutotunerUtil::Autotune(
    const HloInstruction* instr, const AutotuneConfig& config,
    const AutotuneNoCacheFn& autotune_fn) {
  const AutotuneCacheKey key(config.GetDeviceDescription(), *instr);
  TF_ASSIGN_OR_RETURN(std::optional<AutotuneResult> opt_res,
                      TryFindInCache(key, config.autotune_cache_dir()));
  if (opt_res.has_value()) {
    return opt_res.value();
  }

  // Cache miss.
  if (config.should_require_complete_aot_autotune_results()) {
    absl::Status s = NotFound(
        "Complete XLA AOT autotuning results are required, but no AOT result "
        "was found for key: %s",
        key.ToString());
    tsl::errors::InsertPayloads(
        s, {{std::string(kAutotuneCacheRequiredErrorPayloadKey), ""}});
    return s;
  }

  TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result, autotune_fn());

  TF_ASSIGN_OR_RETURN(ResultAndInserted result_and_inserted,
                      AddResultToCaches(key, std::move(autotune_result),
                                        config.autotune_cache_dir(),
                                        config.autotune_cache_mode()));
  return result_and_inserted.result;
}

absl::StatusOr<std::optional<AutotuneResult>> AutotunerUtil::TryFindInCache(
    const AutotuneCacheKey& key, absl::string_view cache_dir)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  TF_ASSIGN_OR_RETURN(auto cached, TryFindInAllCacheTypes(key, cache_dir));

  if (VLOG_IS_ON(1)) {
    std::string logged_key =
        (VLOG_IS_ON(2)) ? absl::StrCat(": key = ", key.ToString()) : "";
    switch (cached.first) {
      case CacheType::kNone:
        LOG(INFO) << "Autotune cache miss" << logged_key;
        break;
      case CacheType::kInMemory:
        LOG(INFO) << "In-memory autotune cache hit" << logged_key;
        break;
      case CacheType::kOnDisk:
        LOG(INFO) << "File-based autotune cache hit" << logged_key;
        break;
    }
  }

  {
    auto cache_hit = cached.second.has_value();
    absl::MutexLock lock(autotune_cache_mu);
    autotune_cache_stats.cache_hits += cache_hit ? 1 : 0;
    autotune_cache_stats.cache_misses += cache_hit ? 0 : 1;
  }
  return std::move(cached.second);
}

absl::StatusOr<ResultAndInserted> AutotunerUtil::AddResultToCaches(
    const AutotuneCacheKey& key, AutotuneResult result,
    absl::string_view cache_dir,
    DebugOptions::AutotuneCacheMode autotune_cache_mode)
    ABSL_LOCKS_EXCLUDED(autotune_cache_mu) {
  ResultAndInserted result_and_inserted = AddResultToInMemoryCache(key, result);
  if (result_and_inserted.inserted) {
    TF_RETURN_IF_ERROR(AddResultToFileBasedCacheIfEnabled(
        key, result_and_inserted.result, cache_dir, autotune_cache_mode));
  }
  return result_and_inserted;
}

namespace {

bool IsTextProtoPath(absl::string_view file_path) {
  return absl::EndsWith(file_path, ".txt") ||
         absl::EndsWith(file_path, ".txtpb") ||
         absl::EndsWith(file_path, ".textproto") ||
         absl::EndsWith(file_path, ".prototxt") ||
         absl::EndsWith(file_path, ".pbtxt");
}

}  // anonymous namespace

/*static*/ absl::Status AutotunerUtil::LoadAutotuneResults(
    absl::string_view data, bool as_textproto, bool allow_override) {
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

  AddVersionToAutotuneResults(results);
  TF_RETURN_IF_ERROR(LoadAutotuneResults(results, allow_override));
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

/*static*/ AutotunerUtil::CacheStats AutotunerUtil::GetCacheStats() {
  absl::MutexLock lock(autotune_cache_mu);
  return autotune_cache_stats;
}

/*static*/ void AutotunerUtil::ClearCacheStats() {
  absl::MutexLock lock(autotune_cache_mu);
  autotune_cache_stats = CacheStats();
}

void AddVersionToAutotuneResults(AutotuneResults& results) {
  for (auto& result : *results.mutable_results()) {
    if (result.version() == 0) {
      // Set to current version if we don't have one specified.
      result.set_version(AutotuneCacheKey::kCurrentVersion);
    }
  }
}

}  // namespace gpu
}  // namespace xla
