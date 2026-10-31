/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/test_utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_value.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/constraint_propagator.h"
#include "xla/tests/constraint_state.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"

namespace xla {

namespace {

// Reduce, ReduceWindow, and SelectAndScatter ops may need a non-random
// initialization value.
bool NeedsInitValue(const HloUse& use) {
  const HloInstruction* const instruction = use.instruction;
  const HloOpcode opcode = instruction->opcode();
  const int64_t op_num = use.operand_number;
  return ((opcode == HloOpcode::kReduceWindow && op_num == 1) ||
          (opcode == HloOpcode::kSelectAndScatter && op_num == 2) ||
          (opcode == HloOpcode::kReduce &&
           op_num >= instruction->operand_count() / 2));
}

bool IsDataFormattingOp(const HloInstruction* instruction,
                        bool treat_gte_as_data_formatting) {
  switch (instruction->opcode()) {
    case HloOpcode::kConvert:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kBitcast:
      return true;
    case HloOpcode::kCustomCall:
      return instruction->custom_call_target() == "AssumeGatherIndicesInBound";
    case HloOpcode::kGetTupleElement:
      return treat_gte_as_data_formatting;
    default:
      return false;
  }
}

void FindConstrainedUsesHelper(
    const HloDataflowAnalysis& dataflow, const HloInstruction& instruction,
    bool treat_gte_as_data_formatting,
    absl::flat_hash_set<const HloInstruction*>& visited,
    std::vector<HloUse>& constrained_uses) {
  auto [it, inserted] = visited.insert(&instruction);
  if (!inserted) {
    return;
  }

  for (const auto& pair : dataflow.GetInstructionValueSet(&instruction)) {
    for (const HloValue* value : pair.second.values()) {
      for (const HloUse& use : value->GetUses()) {
        HloInstruction* const user = use.instruction;
        const HloOpcode opcode = user->opcode();
        const int64_t op_num = use.operand_number;

        if ((opcode == HloOpcode::kDynamicSlice && op_num >= 1) ||
            (opcode == HloOpcode::kDynamicUpdateSlice && op_num >= 2)) {
          constrained_uses.push_back(use);
        } else if ((opcode == HloOpcode::kGather ||
                    opcode == HloOpcode::kScatter) &&
                   op_num == 1) {
          constrained_uses.push_back(use);
        } else if (opcode == HloOpcode::kFusion) {
          // Fusions can have unused operands with no corresponding parameter.
          if (op_num < user->fused_parameters().size()) {
            const HloInstruction* const to_analyze =
                user->fused_parameter(op_num);
            FindConstrainedUsesHelper(dataflow, *to_analyze,
                                      treat_gte_as_data_formatting, visited,
                                      constrained_uses);
          }
        } else if (NeedsInitValue(use)) {
          constrained_uses.push_back(use);
        } else if (IsDataFormattingOp(user, treat_gte_as_data_formatting)) {
          FindConstrainedUsesHelper(dataflow, *user,
                                    treat_gte_as_data_formatting, visited,
                                    constrained_uses);
        } else if (opcode == HloOpcode::kSort &&
                   (user->operand_count() >= 2 ||
                    Cast<const HloSortInstruction>(user)->is_stable()) &&
                   op_num == 0) {
          // Operand 0 of sort is the array of keys used for key/value
          // (two-operand) kSort instructions. Since sort stability is not
          // guaranteed, constrain keys of key-value sort not to have
          // duplicates, since otherwise the value order may legitimately
          // differ.
          constrained_uses.push_back(use);
        }
      }
    }
  }
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of
// instructions that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloUse> FindConstrainedUses(const HloDataflowAnalysis& dataflow,
                                        const HloInstruction& param,
                                        bool treat_gte_as_data_formatting) {
  std::vector<HloUse> constrained_uses;
  absl::flat_hash_set<const HloInstruction*> visited;
  FindConstrainedUsesHelper(dataflow, param, treat_gte_as_data_formatting,
                            visited, constrained_uses);
  return constrained_uses;
}

// Process-wide in-memory and optional on-disk cache for generated fake argument
// literals keyed by canonical shape, layout, constraint profile, base seed, and
// variant index.
//
// In-memory entries are capped at 8 GiB by default with LRU eviction so
// long-lived benchmark runners reuse tensors across sequential module runs
// without unbounded memory growth. When `XLA_ARGUMENT_CACHE_DIR` is set,
// entries up to 4 GiB also persist as `.xlatensor` files with a 4096-byte
// page-aligned header before raw host buffer bytes, capped at 8 GiB of tracked
// disk usage by default. This layout supports parallel striped `pread` without
// 2 GiB protobuf limits.
class FakeArgumentCache {
 public:
  static FakeArgumentCache& Instance() {
    static absl::NoDestructor<FakeArgumentCache> instance;
    return *instance;
  }

  // Looks up a cached literal by `key`. Checks the in-memory LRU cache first,
  // then falls back to `XLA_ARGUMENT_CACHE_DIR` if configured. Promotes hits to
  // the front of the LRU list and populates the in-memory cache on disk hits.
  std::shared_ptr<const Literal> Lookup(absl::string_view key,
                                        const Shape& normalized_shape)
      ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock lock(mu_);
      auto it = entries_.find(key);
      if (it != entries_.end()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        auto disk_it = disk_entries_.find(key);
        if (disk_it != disk_entries_.end()) {
          disk_lru_list_.splice(disk_lru_list_.begin(), disk_lru_list_,
                                disk_it->second);
        }
        return it->second->literal;
      }
    }

    std::string cache_dir = GetCacheDir();
    if (!cache_dir.empty()) {
      const std::string file_path =
          GetFilePath(cache_dir, key, normalized_shape);
      if (absl::StatusOr<Literal> loaded =
              TryLoadFromDisk(file_path, normalized_shape);
          loaded.ok()) {
        auto shared_lit = std::make_shared<const Literal>(*std::move(loaded));
        InsertInMemory(key, shared_lit);
        RecordDiskEntryAndEvict(key, file_path, shared_lit->size_bytes());
        return shared_lit;
      }
    }
    return nullptr;
  }

  // Inserts a newly generated literal into the in-memory LRU cache and, when
  // `XLA_ARGUMENT_CACHE_DIR` is set, persists it as a `.xlatensor` file.
  void Insert(absl::string_view key, std::shared_ptr<const Literal> literal)
      ABSL_LOCKS_EXCLUDED(mu_) {
    InsertInMemory(key, literal);
    std::string cache_dir = GetCacheDir();
    if (!cache_dir.empty()) {
      TrySaveToDisk(cache_dir, key, *literal);
    }
  }

  // Clears all in-memory entries and tracked disk LRU state, and resets
  // capacity limits to their defaults.
  void Clear() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    entries_.clear();
    lru_list_.clear();
    total_bytes_ = 0;
    max_in_memory_bytes_ = kDefaultMaxInMemoryCacheBytes;
    disk_entries_.clear();
    disk_lru_list_.clear();
    total_disk_bytes_ = 0;
    max_disk_bytes_ = kDefaultMaxDiskCacheBytes;
  }

  // Overrides the in-memory and on-disk byte capacity limits for unit testing.
  void SetLimitsForTest(int64_t max_in_memory_bytes, int64_t max_disk_bytes)
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    max_in_memory_bytes_ = max_in_memory_bytes;
    max_disk_bytes_ = max_disk_bytes;
  }

 private:
  struct MemoryCacheEntry {
    std::string key;
    std::shared_ptr<const Literal> literal;
  };

  struct DiskCacheEntry {
    std::string key;
    std::string file_path;
    int64_t bytes = 0;
  };

  // Binary layout of a `.xlatensor` cache file:
  //   [0..8)    : magic bytes "XLATNSR1"
  //   [8..16)   : host-endian uint64_t format version (kFormatVersion)
  //   [16..24)  : host-endian uint64_t payload size in bytes
  //   [24..4096): zero padding to 4096-byte page boundary
  //   [4096..)  : raw host Literal buffer bytes
  static constexpr absl::string_view kFileMagic = "XLATNSR1";
  static constexpr uint64_t kFormatVersion = 1;
  static constexpr size_t kHeaderVersionOffset = 8;
  static constexpr size_t kHeaderPayloadBytesOffset = 16;
  static constexpr size_t kHeaderUsedBytes = 24;
  static constexpr int64_t kPayloadOffset = 4096;
  static constexpr int64_t kIoChunkBytes = 16LL << 20;                 // 16 MiB
  static constexpr int64_t kDefaultMaxInMemoryCacheBytes = 8LL << 30;  // 8 GiB
  static constexpr int64_t kDefaultMaxDiskCacheBytes = 8LL << 30;      // 8 GiB
  static constexpr int64_t kMaxSingleEntryBytes = 4LL << 30;           // 4 GiB

  static_assert(kFileMagic.size() == kHeaderVersionOffset);
  static_assert(kHeaderUsedBytes <= kPayloadOffset);

  static std::string GetCacheDir() {
    const char* env_dir = std::getenv("XLA_ARGUMENT_CACHE_DIR");
    return env_dir != nullptr ? std::string(env_dir) : std::string();
  }

  // Computes the deterministic sharded path for a cached tensor:
  // `<cache_dir>/v1/<type>_<dims>/<2-char-shard>/<32-char-fp128>.xlatensor`.
  static std::string GetFilePath(absl::string_view cache_dir,
                                 absl::string_view key, const Shape& shape) {
    tsl::Fprint128 fp = tsl::Fingerprint128(key);
    std::string fp_hex = absl::StrFormat("%016x%016x", fp.high64, fp.low64);
    std::string shape_dir = absl::StrCat(
        primitive_util::LowercasePrimitiveTypeName(shape.element_type()), "_",
        absl::StrJoin(shape.dimensions(), "x"));
    return tsl::io::JoinPath(cache_dir, "v1", shape_dir, fp_hex.substr(0, 2),
                             absl::StrCat(fp_hex, ".xlatensor"));
  }

  // Loads a `.xlatensor` file from `file_path`, validating its magic header,
  // version, and exact byte size. Uses parallel 16 MiB striped `pread`s for
  // payloads of at least 32 MiB. Deletes corrupted or truncated files
  // automatically.
  static absl::StatusOr<Literal> TryLoadFromDisk(
      const std::string& file_path, const Shape& normalized_shape) {
    tsl::Env* env = tsl::Env::Default();
    std::unique_ptr<tsl::RandomAccessFile> file;
    ABSL_RETURN_IF_ERROR(env->NewRandomAccessFile(file_path, &file));

    absl::StatusOr<Literal> result = [&]() -> absl::StatusOr<Literal> {
      char header_buf[kHeaderUsedBytes];
      absl::string_view header_piece;
      ABSL_RETURN_IF_ERROR(file->Read(
          0, header_piece, absl::MakeSpan(header_buf, sizeof(header_buf))));
      if (header_piece.size() != sizeof(header_buf) ||
          header_piece.substr(0, kFileMagic.size()) != kFileMagic) {
        return InvalidArgument("Invalid xlatensor header in %s", file_path);
      }
      uint64_t version = 0;
      uint64_t payload_bytes = 0;
      std::memcpy(&version, header_piece.data() + kHeaderVersionOffset,
                  sizeof(uint64_t));
      std::memcpy(&payload_bytes,
                  header_piece.data() + kHeaderPayloadBytesOffset,
                  sizeof(uint64_t));

      ABSL_ASSIGN_OR_RETURN(Literal literal, Literal::Make(normalized_shape));
      const int64_t expected_bytes = literal.size_bytes();
      if (expected_bytes < 0 || version != kFormatVersion ||
          payload_bytes != static_cast<uint64_t>(expected_bytes)) {
        return InvalidArgument("Mismatched xlatensor size in %s", file_path);
      }

      char* dst = static_cast<char*>(literal.untyped_data());
      auto read_chunk = [&](int64_t offset, int64_t len) -> absl::Status {
        absl::string_view chunk_piece;
        ABSL_RETURN_IF_ERROR(file->Read(
            static_cast<uint64_t>(kPayloadOffset + offset), chunk_piece,
            absl::MakeSpan(dst + offset, static_cast<size_t>(len))));
        if (chunk_piece.size() != static_cast<size_t>(len)) {
          return InvalidArgument("Short read from %s", file_path);
        }
        if (chunk_piece.data() != dst + offset) {
          std::memcpy(dst + offset, chunk_piece.data(),
                      static_cast<size_t>(len));
        }
        return absl::OkStatus();
      };

      if (expected_bytes >= kIoChunkBytes * 2) {
        const int64_t num_chunks =
            CeilOfRatio<int64_t>(expected_bytes, kIoChunkBytes);
        Shape work_shape = ShapeUtil::MakeShape(U8, {num_chunks});
        ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachIndexParallelWithStatus(
            work_shape,
            [&](absl::Span<const int64_t> idx,
                int /*thread_id*/) -> absl::StatusOr<bool> {
              const int64_t offset = idx[0] * kIoChunkBytes;
              const int64_t len =
                  std::min<int64_t>(kIoChunkBytes, expected_bytes - offset);
              ABSL_RETURN_IF_ERROR(read_chunk(offset, len));
              return true;
            }));
      } else if (expected_bytes > 0) {
        ABSL_RETURN_IF_ERROR(read_chunk(/*offset=*/0, expected_bytes));
      }
      return literal;
    }();
    if (!result.ok()) {
      file.reset();
      env->DeleteFile(file_path).IgnoreError();
    }
    return result;
  }

  // Atomically writes `literal` to a temporary file and renames it into place
  // so concurrent readers never observe partial writes.
  void TrySaveToDisk(absl::string_view cache_dir, absl::string_view key,
                     const Literal& literal) ABSL_LOCKS_EXCLUDED(mu_) {
    const int64_t bytes = literal.size_bytes();
    int64_t max_disk_bytes = 0;
    {
      absl::MutexLock lock(mu_);
      max_disk_bytes = max_disk_bytes_;
    }
    if (bytes <= 0 || bytes > kMaxSingleEntryBytes || bytes > max_disk_bytes) {
      return;
    }
    std::string file_path = GetFilePath(cache_dir, key, literal.shape());
    tsl::Env* env = tsl::Env::Default();
    if (env->FileExists(file_path).ok()) {
      RecordDiskEntryAndEvict(key, file_path, bytes);
      return;
    }
    std::string dir_path(tsl::io::Dirname(file_path));
    if (!env->RecursivelyCreateDir(dir_path).ok()) {
      return;
    }

    static std::atomic<uint64_t> next_tmp_seq{1};
    absl::BitGen bitgen;
    const uint64_t nonce = absl::Uniform<uint64_t>(bitgen);
    const uint64_t seq = next_tmp_seq.fetch_add(1, std::memory_order_relaxed);
    std::string tmp_path = absl::StrFormat("%s.tmp.%016x.%u.%u", file_path,
                                           nonce, env->NowMicros(), seq);
    std::unique_ptr<tsl::WritableFile> out;
    if (!env->NewWritableFile(tmp_path, &out).ok()) {
      return;
    }

    std::string header(static_cast<size_t>(kPayloadOffset), '\0');
    std::memcpy(header.data(), kFileMagic.data(), kFileMagic.size());
    const uint64_t version = kFormatVersion;
    const uint64_t payload_bytes = static_cast<uint64_t>(bytes);
    std::memcpy(header.data() + kHeaderVersionOffset, &version,
                sizeof(uint64_t));
    std::memcpy(header.data() + kHeaderPayloadBytesOffset, &payload_bytes,
                sizeof(uint64_t));
    absl::Status status = out->Append(header);
    const char* src = static_cast<const char*>(literal.untyped_data());
    for (int64_t offset = 0; status.ok() && offset < bytes;
         offset += kIoChunkBytes) {
      const int64_t chunk_len =
          std::min<int64_t>(kIoChunkBytes, bytes - offset);
      status = out->Append(
          absl::string_view(src + offset, static_cast<size_t>(chunk_len)));
    }
    if (status.ok()) {
      status = out->Close();
    } else {
      out->Close().IgnoreError();
    }
    if (!status.ok()) {
      env->DeleteFile(tmp_path).IgnoreError();
      return;
    }
    if (!env->RenameFile(tmp_path, file_path).ok()) {
      env->DeleteFile(tmp_path).IgnoreError();
      return;
    }
    RecordDiskEntryAndEvict(key, file_path, bytes);
  }

  // Tracks a `.xlatensor` file in the on-disk LRU list and deletes the oldest
  // tracked files outside the mutex lock when total usage exceeds
  // `max_disk_bytes_`.
  void RecordDiskEntryAndEvict(absl::string_view key,
                               const std::string& file_path, int64_t bytes)
      ABSL_LOCKS_EXCLUDED(mu_) {
    std::vector<std::string> files_to_delete;
    {
      absl::MutexLock lock(mu_);
      if (bytes <= 0 || bytes > max_disk_bytes_) {
        return;
      }
      auto it = disk_entries_.find(key);
      if (it != disk_entries_.end()) {
        disk_lru_list_.splice(disk_lru_list_.begin(), disk_lru_list_,
                              it->second);
        return;
      }
      while (total_disk_bytes_ + bytes > max_disk_bytes_ &&
             !disk_lru_list_.empty()) {
        const DiskCacheEntry& victim = disk_lru_list_.back();
        files_to_delete.push_back(victim.file_path);
        total_disk_bytes_ -= victim.bytes;
        disk_entries_.erase(victim.key);
        disk_lru_list_.pop_back();
      }
      disk_lru_list_.push_front(
          DiskCacheEntry{std::string(key), file_path, bytes});
      disk_entries_.emplace(disk_lru_list_.front().key, disk_lru_list_.begin());
      total_disk_bytes_ += bytes;
    }
    tsl::Env* env = tsl::Env::Default();
    for (const std::string& victim_path : files_to_delete) {
      env->DeleteFile(victim_path).IgnoreError();
    }
  }

  // Inserts `literal` into the in-memory LRU cache, evicting least-recently
  // used entries until the new entry fits within `max_in_memory_bytes_`.
  void InsertInMemory(absl::string_view key,
                      std::shared_ptr<const Literal> literal)
      ABSL_LOCKS_EXCLUDED(mu_) {
    const int64_t bytes = literal->size_bytes();
    if (bytes <= 0 || bytes > kMaxSingleEntryBytes) {
      return;
    }
    absl::MutexLock lock(mu_);
    if (bytes > max_in_memory_bytes_) {
      return;
    }
    auto existing = entries_.find(key);
    if (existing != entries_.end()) {
      lru_list_.splice(lru_list_.begin(), lru_list_, existing->second);
      return;
    }
    while (total_bytes_ + bytes > max_in_memory_bytes_ && !lru_list_.empty()) {
      const MemoryCacheEntry& victim = lru_list_.back();
      total_bytes_ -= victim.literal->size_bytes();
      entries_.erase(victim.key);
      lru_list_.pop_back();
    }
    lru_list_.push_front(
        MemoryCacheEntry{std::string(key), std::move(literal)});
    entries_.emplace(lru_list_.front().key, lru_list_.begin());
    total_bytes_ += bytes;
  }

  absl::Mutex mu_;
  std::list<MemoryCacheEntry> lru_list_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::list<MemoryCacheEntry>::iterator>
      entries_ ABSL_GUARDED_BY(mu_);
  int64_t total_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t max_in_memory_bytes_ ABSL_GUARDED_BY(mu_) =
      kDefaultMaxInMemoryCacheBytes;

  std::list<DiskCacheEntry> disk_lru_list_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::list<DiskCacheEntry>::iterator>
      disk_entries_ ABSL_GUARDED_BY(mu_);
  int64_t total_disk_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t max_disk_bytes_ ABSL_GUARDED_BY(mu_) = kDefaultMaxDiskCacheBytes;
};

// Per-invocation state for `MakeFakeArguments` /
// `MakeDataflowConstrainedArguments` when argument caching is enabled. Tracks
// how many times each canonical `(shape, constraint)` key has been requested in
// the current module so repeated parameters cycle through `0 ..
// max_variants_per_shape - 1`.
struct FakeArgumentCacheContext {
  uint32_t base_seed_token = 0;
  int max_variants_per_shape = 2;
  absl::flat_hash_map<std::string, int> occurrence_counts;
};

// Initializes a `FakeArgumentCacheContext` from `options` and `engine` without
// mutating `engine` (a copy is used to sample `base_seed_token`).
std::optional<FakeArgumentCacheContext> MaybeCreateCacheContext(
    const FakeArgumentsOptions& options, const std::minstd_rand0* engine) {
  if (!options.cache_arguments || engine == nullptr ||
      options.max_cached_variants_per_shape <= 0) {
    return std::nullopt;
  }
  FakeArgumentCacheContext cache_ctx;
  std::minstd_rand0 copy = *engine;
  cache_ctx.base_seed_token = static_cast<uint32_t>(copy());
  cache_ctx.max_variants_per_shape = options.max_cached_variants_per_shape;
  return cache_ctx;
}

// Clones an array literal, parallelizing the `memcpy` across worker threads for
// buffers of at least 32 MiB.
absl::StatusOr<Literal> CloneArrayLiteralParallel(const Literal& src) {
  ABSL_ASSIGN_OR_RETURN(Literal dst, Literal::Make(src.shape()));
  const int64_t total_bytes = dst.size_bytes();
  CHECK_EQ(src.size_bytes(), total_bytes);
  if (total_bytes <= 0) {
    return dst;
  }
  const char* src_ptr = static_cast<const char*>(src.untyped_data());
  char* dst_ptr = static_cast<char*>(dst.untyped_data());
  constexpr int64_t kParallelCopyChunkBytes = 16LL << 20;  // 16 MiB
  if (total_bytes >= kParallelCopyChunkBytes * 2 &&
      ShapeUtil::GetForEachIndexParallelThreadCount() > 1) {
    const int64_t num_chunks =
        CeilOfRatio<int64_t>(total_bytes, kParallelCopyChunkBytes);
    Shape work_shape = ShapeUtil::MakeShape(U8, {num_chunks});
    ShapeUtil::ForEachIndexParallel(
        work_shape,
        [&](absl::Span<const int64_t> idx,
            int /*thread_id*/) -> absl::StatusOr<bool> {
          const int64_t offset = idx[0] * kParallelCopyChunkBytes;
          const int64_t len =
              std::min<int64_t>(kParallelCopyChunkBytes, total_bytes - offset);
          std::memcpy(dst_ptr + offset, src_ptr + offset,
                      static_cast<size_t>(len));
          return true;
        });
  } else {
    std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(total_bytes));
  }
  return dst;
}

// Strips device-specific tiling, tail padding, element-size-in-bits (`E(n)`),
// memory-space tags, and split configs from `shape` so host `Literal` buffers
// are allocated and cached in a canonical untiled host layout.
Shape NormalizeShapeForArgumentCache(const Shape& shape) {
  Shape normalized_shape = ShapeUtil::DeviceShapeToHostShape(shape);
  if (!normalized_shape.has_layout()) {
    LayoutUtil::SetToDefaultLayout(&normalized_shape);
  }
  normalized_shape.mutable_layout()->clear_split_configs();
  return normalized_shape;
}

// Restores a non-default layout memory space (such as
// `Layout::kHostMemorySpace` for pinned host parameters) from `original_shape`
// onto a cached host literal.
void RestoreOriginalMemorySpace(const Shape& original_shape, Literal* literal) {
  if (original_shape.has_layout() &&
      original_shape.layout().memory_space() != Layout::kDefaultMemorySpace) {
    literal->mutable_shape_do_not_use()->mutable_layout()->set_memory_space(
        original_shape.layout().memory_space());
  }
}

// Formats the canonical base cache key encoding the normalized shape, layout,
// dataflow/value constraints, and base RNG seed token.
std::string BuildFakeArgumentCacheKey(
    const Shape& normalized_shape,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision,
    std::optional<int64_t> index_alignment,
    std::optional<uint64_t> index_known_zeroes,
    std::optional<ConstraintInterval> interval, uint32_t base_seed_token) {
  std::string interval_key = "none";
  if (interval.has_value() && !interval->IsUnconstrained()) {
    interval_key = absl::StrFormat("%016x:%016x:%d",
                                   absl::bit_cast<uint64_t>(interval->min),
                                   absl::bit_cast<uint64_t>(interval->max),
                                   interval->exclude_zero ? 1 : 0);
  }
  return absl::StrFormat(
      "v1|s=%s|sorted=%d|nd=%d|lr=%d|mb=%s|lim=%s|al=%s|kz=%s|iv=%s|seed=%u",
      ShapeUtil::HumanStringWithLayout(normalized_shape), is_sorted,
      no_duplicates, use_large_range,
      max_bits_of_precision.has_value() ? absl::StrCat(*max_bits_of_precision)
                                        : "none",
      limit.has_value() ? absl::StrCat(limit->first, ":", limit->second)
                        : "none",
      index_alignment.has_value() ? absl::StrCat(*index_alignment) : "none",
      index_known_zeroes.has_value()
          ? absl::StrFormat("%x", *index_known_zeroes)
          : "none",
      interval_key, base_seed_token);
}

// Generates or retrieves a cached fake literal for `shape` when `cache_ctx` is
// non-null.
//
// Tuple parameters are decomposed recursively so leaf array subshapes are
// cached individually. For static arrays of at least 64 KiB, device-specific
// tiling, tail padding, and sub-byte packing attributes are stripped to form a
// canonical host shape. Up to `max_variants_per_shape` distinct random variants
// are retained per unique shape and constraint key. Binary ops on identical
// parameter shapes such as `subtract(p0, p1)` therefore do not cancel out,
// while deep models generate at most `max_variants_per_shape` tensors per
// weight shape.
absl::StatusOr<Literal> MakeCachedFakeLiteral(
    const Shape& shape, std::minstd_rand0* engine,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision,
    std::optional<int64_t> index_alignment,
    std::optional<uint64_t> index_known_zeroes,
    FakeArgumentCacheContext* cache_ctx,
    std::optional<ConstraintInterval> interval = std::nullopt) {
  if (shape.IsTuple()) {
    std::vector<Literal> elements;
    const std::vector<Shape>& shape_tuple_shapes = shape.tuple_shapes();
    elements.reserve(shape_tuple_shapes.size());
    for (const Shape& element_shape : shape_tuple_shapes) {
      ABSL_ASSIGN_OR_RETURN(
          Literal element,
          MakeCachedFakeLiteral(element_shape, engine, limit, is_sorted,
                                no_duplicates, use_large_range,
                                max_bits_of_precision, index_alignment,
                                index_known_zeroes, cache_ctx, interval));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }

  constexpr int64_t kMinCacheableBytes = 64 * 1024;  // 64 KiB
  if (engine == nullptr || cache_ctx == nullptr ||
      cache_ctx->max_variants_per_shape <= 0 ||
      !shape.IsArrayExcludingBuffer() || !shape.is_static()) {
    return MakeFakeLiteral(shape, engine, limit, is_sorted, no_duplicates,
                           use_large_range, max_bits_of_precision,
                           index_alignment, index_known_zeroes,
                           /*float_generator=*/nullptr, interval);
  }

  Shape normalized_shape = NormalizeShapeForArgumentCache(shape);
  if (ShapeUtil::ByteSizeOf(normalized_shape) < kMinCacheableBytes) {
    return MakeFakeLiteral(shape, engine, limit, is_sorted, no_duplicates,
                           use_large_range, max_bits_of_precision,
                           index_alignment, index_known_zeroes,
                           /*float_generator=*/nullptr, interval);
  }

  const std::string base_key = BuildFakeArgumentCacheKey(
      normalized_shape, limit, is_sorted, no_duplicates, use_large_range,
      max_bits_of_precision, index_alignment, index_known_zeroes, interval,
      cache_ctx->base_seed_token);
  const int variant_idx = (cache_ctx->occurrence_counts[base_key]++) %
                          cache_ctx->max_variants_per_shape;
  const std::string full_key = absl::StrCat(base_key, "|v=", variant_idx);

  if (std::shared_ptr<const Literal> cached =
          FakeArgumentCache::Instance().Lookup(full_key, normalized_shape)) {
    // Advance `engine` by a fixed 1 step so subsequent uncached small
    // parameters observe the exact same RNG state on both cache hits and
    // misses.
    engine->discard(1);
    ABSL_ASSIGN_OR_RETURN(Literal cloned, CloneArrayLiteralParallel(*cached));
    RestoreOriginalMemorySpace(shape, &cloned);
    return cloned;
  }

  // Derive a deterministic per-key leaf RNG engine from the 64-bit fingerprint
  // of `full_key` so generated contents depend only on the canonical key and
  // not on parameter ordering within the HLO module.
  const uint64_t key_fp = tsl::Fingerprint64(full_key);
  const auto leaf_seed = static_cast<std::minstd_rand0::result_type>(
      (key_fp % (std::minstd_rand0::modulus - 1)) + 1);
  std::minstd_rand0 leaf_engine(leaf_seed);
  ABSL_ASSIGN_OR_RETURN(
      Literal generated,
      MakeFakeLiteral(normalized_shape, &leaf_engine, limit, is_sorted,
                      no_duplicates, use_large_range, max_bits_of_precision,
                      index_alignment, index_known_zeroes,
                      /*float_generator=*/nullptr, interval));
  engine->discard(1);
  auto shared_lit = std::make_shared<const Literal>(std::move(generated));
  FakeArgumentCache::Instance().Insert(full_key, shared_lit);
  ABSL_ASSIGN_OR_RETURN(Literal result, CloneArrayLiteralParallel(*shared_lit));
  RestoreOriginalMemorySpace(shape, &result);
  return result;
}

// Given a parameter, generate a random Literal to use as input if there exist
// no constrained uses in the dataflow graph.  If such constraints exist,
// generate a constrained literal (either bounded in the case of indices, or
// zero in the case of init_values for reductions).
absl::StatusOr<Literal> CreateLiteralForConstrainedUses(
    const absl::Span<const HloUse> constrained_uses,
    const HloInstruction& param, const Shape& param_shape,
    std::minstd_rand0* engine, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision,
    bool generate_aligned_ds_indices,
    GetIndexKnownZeroesFn get_index_known_zeroes = nullptr,
    FakeArgumentCacheContext* cache_ctx = nullptr) {
  int64_t index_bound = INT64_MAX;
  // Used for operations like DUS / DS which need to be aligned when they
  // appear in a fusion.
  std::optional<int64_t> index_alignment = std::nullopt;
  bool no_duplicates = false;
  bool needs_constant = false;
  bool needs_sorted_indices = false;
  std::optional<uint64_t> index_known_zeroes = std::nullopt;
  IdentityElementType identity_type = IdentityElementType::kUnknown;
  for (const HloUse& hlo_use : constrained_uses) {
    HloInstruction* use = hlo_use.instruction;
    switch (use->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        const Shape& indexed_shape = use->operand(0)->shape();
        const Shape& slice_shape = use->opcode() == HloOpcode::kDynamicSlice
                                       ? use->shape()
                                       : use->operand(1)->shape();
        const int64_t first_index =
            Cast<HloDynamicIndexInstruction>(use)->first_index_operand_number();
        const int64_t sliced_dim = hlo_use.operand_number - first_index;
        if (hlo_use.operand_number >= first_index) {
          index_bound =
              std::min(index_bound,
                       ShapeUtil::GetDimension(indexed_shape, sliced_dim) -
                           ShapeUtil::GetDimension(slice_shape, sliced_dim));
          const int64_t physical_sliced_dim = PositionInContainer(
              use->shape().layout().minor_to_major(), sliced_dim);
          switch (physical_sliced_dim) {
            // Lanes
            case 0:
              index_alignment = std::max(index_alignment.value_or(1),
                                         static_cast<int64_t>(128));
              break;
            // Sublanes
            case 1:
              index_alignment = std::max(index_alignment.value_or(1),
                                         static_cast<int64_t>(8));
              break;
            default:
              break;
          }
          if (get_index_known_zeroes != nullptr) {
            if (std::optional<uint64_t> current_known_zeroes =
                    get_index_known_zeroes(use, sliced_dim)) {
              if (index_known_zeroes.has_value()) {
                // If we have multiple uses with different masks, we take the
                // union of known zeroes.
                index_known_zeroes =
                    *index_known_zeroes | *current_known_zeroes;
              } else {
                index_known_zeroes = current_known_zeroes;
              }
            }
          }
        }
        break;
      }
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        const Shape& operand_shape = use->operand(0)->shape();
        auto index_map = use->opcode() == HloOpcode::kGather
                             ? use->gather_dimension_numbers().start_index_map()
                             : use->scatter_dimension_numbers()
                                   .scatter_dims_to_operand_dims();
        for (const auto dim_in_operand : index_map) {
          index_bound = std::min(index_bound,
                                 operand_shape.dimensions(dim_in_operand) - 1);
        }
        if (use->opcode() == HloOpcode::kScatter) {
          needs_sorted_indices |=
              Cast<const HloScatterInstruction>(use)->indices_are_sorted();
        } else {
          needs_sorted_indices |=
              Cast<const HloGatherInstruction>(use)->indices_are_sorted();
        }
        break;
      }
      case HloOpcode::kReduce:
      case HloOpcode::kReduceWindow:
        needs_constant = true;
        identity_type = GetReductionIdentityElementType(*use->to_apply());
        break;

      case HloOpcode::kSelectAndScatter:
        needs_constant = true;
        identity_type = GetReductionIdentityElementType(*use->scatter());
        break;

      case HloOpcode::kSort:
        if (ShapeUtil::ElementIsIntegral(use->operand(0)->shape())) {
          // Turn on no_duplicates for integer keys. It's basically shuffled
          // iota from [0, N) for unsigned, or [-N/2, N/2) for signed.
          no_duplicates = true;
        }
        break;

      default:
        return Unimplemented(
            "Constrained operand generation not implemented for %s.",
            use->ToString());
    }
  }
  if (!generate_aligned_ds_indices) {
    index_alignment = std::nullopt;
  }
  int constraint_count = 0;
  constraint_count += no_duplicates ? 1 : 0;
  constraint_count += (index_bound != INT64_MAX) ? 1 : 0;
  constraint_count += needs_constant ? 1 : 0;
  if (constraint_count > 1) {
    return Unimplemented("Conflicting operand generation constraints.");
  }
  if (index_bound != INT64_MAX) {
    return MakeCachedFakeLiteral(
        param_shape, engine, std::pair<int64_t, int64_t>(0, index_bound),
        needs_sorted_indices, no_duplicates, use_large_range,
        max_bits_of_precision, index_alignment, index_known_zeroes, cache_ctx);
  }
  if (needs_constant) {
    switch (identity_type) {
      case IdentityElementType::kZero:
        return LiteralUtil::Zero(param_shape.element_type());
      case IdentityElementType::kOne:
        return LiteralUtil::One(param_shape.element_type());
      case IdentityElementType::kMinimum:
        return LiteralUtil::MinValue(param_shape.element_type());
      case IdentityElementType::kMaximum:
        return LiteralUtil::MaxValue(param_shape.element_type());
      case IdentityElementType::kUnknown:
        // We want the identity element for the computation, but we don't
        // really know what it is - so any value we generate will be just as
        // wrong.
        return MakeCachedFakeLiteral(
            param_shape, engine, /*limit=*/std::nullopt,
            /*is_sorted=*/needs_sorted_indices,
            /*no_duplicates=*/false, use_large_range, max_bits_of_precision,
            /*index_alignment=*/std::nullopt,
            /*index_known_zeroes=*/std::nullopt, cache_ctx);
    }
  }
  return MakeCachedFakeLiteral(param_shape, engine, /*limit=*/std::nullopt,
                               /*is_sorted=*/needs_sorted_indices,
                               no_duplicates, use_large_range,
                               max_bits_of_precision,
                               /*index_alignment=*/std::nullopt,
                               /*index_known_zeroes=*/std::nullopt, cache_ctx);
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
absl::StatusOr<Literal> MakeConstrainedArgument(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    const Shape& param_shape, std::minstd_rand0* engine, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision,
    bool generate_aligned_ds_indices,
    GetIndexKnownZeroesFn get_index_known_zeroes = nullptr,
    FakeArgumentCacheContext* cache_ctx = nullptr) {
  const auto constrained_uses =
      FindConstrainedUses(dataflow, param, treat_gte_as_data_formatting);
  return CreateLiteralForConstrainedUses(
      constrained_uses, param, param_shape, engine, use_large_range,
      max_bits_of_precision, generate_aligned_ds_indices,
      get_index_known_zeroes, cache_ctx);
}

}  // namespace

void ClearFakeArgumentCacheForTest() { FakeArgumentCache::Instance().Clear(); }

void SetFakeArgumentCacheLimitsForTest(int64_t max_in_memory_bytes,
                                       int64_t max_disk_bytes) {
  FakeArgumentCache::Instance().SetLimitsForTest(max_in_memory_bytes,
                                                 max_disk_bytes);
}

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, const FakeArgumentsOptions& options) {
  if (!options.parameter_ranges.empty()) {
    return Unimplemented(
        "parameter_ranges is not currently supported in MakeFakeArguments.");
  }

  std::unique_ptr<std::minstd_rand0> default_engine;
  std::minstd_rand0* engine = options.engine;
  if (!options.pseudo_random) {
    engine = nullptr;
  } else if (engine == nullptr) {
    default_engine = std::make_unique<std::minstd_rand0>();
    engine = default_engine.get();
  }

  std::optional<FakeArgumentCacheContext> cache_ctx =
      MaybeCreateCacheContext(options, engine);
  FakeArgumentCacheContext* cache_ctx_ptr =
      cache_ctx.has_value() ? &*cache_ctx : nullptr;

  ABSL_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    const HloModuleConfig& module_config = module->config();
    const Shape& param_shape = (module_config.has_entry_computation_layout() &&
                                module_config.entry_computation_layout()
                                    .parameter_layout(i)
                                    .shape()
                                    .is_static())
                                   ? module_config.entry_computation_layout()
                                         .parameter_layout(i)
                                         .shape()
                                   : params[i]->shape();

    ABSL_ASSIGN_OR_RETURN(
        arguments[i],
        MakeConstrainedArgument(
            *dataflow, *params[i], param_shape, engine, options.use_large_range,
            options.treat_gte_as_data_formatting, options.max_bits_of_precision,
            options.generate_aligned_ds_indices, options.get_index_known_zeroes,
            cache_ctx_ptr));
  }
  return std::move(arguments);
}

// Validates options for dataflow-constrained fake argument generation.
absl::Status ValidateDataflowConstrainedOptions(
    const FakeArgumentsOptions& options, int64_t total_flat_indices) {
  if (options.max_bits_of_precision.has_value()) {
    return Unimplemented(
        "max_bits_of_precision is not supported in "
        "MakeDataflowConstrainedArguments.");
  }

  // NOLINTNEXTLINE
  for (const auto& [flat_idx, range] : options.parameter_ranges) {
    if (flat_idx < 0 || flat_idx >= total_flat_indices) {
      return InvalidArgument(
          "parameter_ranges index %d is out of bounds; module has %d "
          "flattened parameter indices.",
          flat_idx, total_flat_indices);
    }
    if (std::isnan(range.first) || std::isnan(range.second) ||
        range.first > range.second) {
      return InvalidArgument(
          "Invalid range [%f, %f] specified for flattened parameter index %d.",
          range.first, range.second, flat_idx);
    }
  }
  return absl::OkStatus();
}

// Applies optional user-specified parameter range by intersecting it with the
// graph-derived constraint interval. Returns an InvalidArgument error if the
// intersection is empty.
absl::StatusOr<ConstraintInterval> ApplyUserParameterRange(
    const ConstraintInterval& graph_interval,
    const absl::flat_hash_map<int64_t, std::pair<double, double>>&
        parameter_ranges,
    int64_t flat_idx, absl::string_view target_name) {
  auto range_it = parameter_ranges.find(flat_idx);
  if (range_it == parameter_ranges.end()) {
    return graph_interval;
  }

  ConstraintInterval user_interval{range_it->second.first,
                                   range_it->second.second,
                                   /*exclude_zero=*/false};
  ConstraintInterval candidate = graph_interval.Intersect(user_interval);
  if (candidate.IsEmpty()) {
    return InvalidArgument(
        "User-provided range [%f, %f] for parameter %s (flat index %d) "
        "conflicts with required graph constraint [%f, %f]%s.",
        user_interval.min, user_interval.max, target_name, flat_idx,
        graph_interval.min, graph_interval.max,
        graph_interval.exclude_zero ? " (excl 0)" : "");
  }
  return candidate;
}

// Converts a continuous ConstraintInterval into an exact discrete [min, max]
// range for integral types, checking for boundary conditions and preventing
// double-to-int64 overflow undefined behavior.
absl::StatusOr<std::pair<int64_t, int64_t>> DiscretizeIntegerInterval(
    const ConstraintInterval& interval, absl::string_view target_name,
    int64_t flat_idx) {
  // Use exact hexadecimal floating-point literals 0x1.0p63 (2^63) and
  // -0x1.0p63 (-2^63) for boundary comparisons. INT64_MAX (2^63 - 1)
  // cannot be exactly represented in a 53-bit mantissa double and rounds
  // up to 2^63 when cast. Because powers of 2 are exact in IEEE 754,
  // checking interval.min < kMaxInt64AsDouble and
  // interval.max >= kMinInt64AsDouble guarantees that any double strictly less
  // than 0x1.0p63 is at most 2^63 - 2048 < INT64_MAX, safely fitting in
  // int64_t without overflow UB.
  constexpr double kMaxInt64AsDouble = 0x1.0p63;   // 2^63
  constexpr double kMinInt64AsDouble = -0x1.0p63;  // -2^63

  if (std::isnan(interval.min) || std::isnan(interval.max) ||
      interval.min >= kMaxInt64AsDouble || interval.max < kMinInt64AsDouble) {
    return InvalidArgument(
        "Unsatisfiable integer constraint interval [%f, %f]%s for "
        "parameter %s (flat index %d): outside representable range of int64_t.",
        interval.min, interval.max, interval.exclude_zero ? " (excl 0)" : "",
        target_name, flat_idx);
  }

  int64_t min_val = interval.min <= kMinInt64AsDouble
                        ? std::numeric_limits<int64_t>::min()
                        : static_cast<int64_t>(std::ceil(interval.min));
  int64_t max_val = interval.max >= kMaxInt64AsDouble
                        ? std::numeric_limits<int64_t>::max()
                        : static_cast<int64_t>(std::floor(interval.max));

  if (interval.exclude_zero && min_val == 0) {
    min_val = 1;
  }
  if (interval.exclude_zero && max_val == 0) {
    max_val = -1;
  }

  if (min_val > max_val) {
    return InvalidArgument(
        "Unsatisfiable integer constraint interval [%f, %f]%s for "
        "parameter %s (flat index %d): collapsed to empty discrete range [%d, "
        "%d].",
        interval.min, interval.max, interval.exclude_zero ? " (excl 0)" : "",
        target_name, flat_idx, min_val, max_val);
  }

  return std::make_pair(min_val, max_val);
}

// Generates a single fake literal for a given shape, constraint state, and flat
// parameter index.
absl::StatusOr<Literal> MakeDataflowConstrainedLiteral(
    const Shape& shape, const ConstraintState& state,
    absl::string_view target_name, int64_t flat_idx,
    const FakeArgumentsOptions& options, std::minstd_rand0* engine,
    FakeArgumentCacheContext* cache_ctx) {
  ABSL_ASSIGN_OR_RETURN(
      ConstraintInterval interval,
      ApplyUserParameterRange(state.GetConstraintInterval(),
                              options.parameter_ranges, flat_idx, target_name));

  StructuralConstraints structure = state.GetStructuralConstraints();
  if (!options.generate_aligned_ds_indices) {
    structure.alignment = std::nullopt;
  }

  std::optional<std::pair<int64_t, int64_t>> limit = std::nullopt;
  if (ShapeUtil::ElementIsIntegral(shape) && !interval.IsUnconstrained() &&
      !interval.IsEmpty()) {
    ABSL_ASSIGN_OR_RETURN(
        limit, DiscretizeIntegerInterval(interval, target_name, flat_idx));
  }

  return MakeCachedFakeLiteral(
      shape, engine, limit, structure.needs_sorted_indices,
      structure.no_duplicates, options.use_large_range,
      /*max_bits_of_precision=*/std::nullopt, structure.alignment,
      structure.known_zeroes_mask, cache_ctx, interval);
}

// Generates a literal for a parameter instruction (either array or tuple).
// For tuple parameters, searches for get-tuple-element users to extract
// per-element propagated constraints and generates an owned tuple literal.
absl::StatusOr<Literal> GenerateParameterLiteral(
    const HloInstruction* param, const Shape& param_shape,
    const absl::flat_hash_map<const HloInstruction*, ConstraintState>&
        constraint_states,
    const FakeArgumentsOptions& options, std::minstd_rand0* engine,
    int64_t& current_flat_idx, FakeArgumentCacheContext* cache_ctx) {
  if (param_shape.IsTuple()) {
    std::vector<Literal> elements;
    elements.reserve(param_shape.tuple_shapes().size());
    for (int64_t j = 0; j < param_shape.tuple_shapes().size(); ++j) {
      ConstraintState elem_state;
      for (const HloInstruction* user : param->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            user->tuple_index() == j) {
          auto it = constraint_states.find(user);
          if (it != constraint_states.end()) {
            elem_state.Merge(it->second);
          }
        }
      }
      int64_t flat_idx = current_flat_idx++;
      ABSL_ASSIGN_OR_RETURN(Literal elem_lit,
                       MakeDataflowConstrainedLiteral(
                           param_shape.tuple_shapes(j), elem_state,
                           absl::StrFormat("%s (element %d)", param->name(), j),
                           flat_idx, options, engine, cache_ctx));
      elements.push_back(std::move(elem_lit));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }

  int64_t flat_idx = current_flat_idx++;
  auto it = constraint_states.find(param);
  const ConstraintState& state =
      (it != constraint_states.end()) ? it->second : ConstraintState();
  return MakeDataflowConstrainedLiteral(param_shape, state, param->name(),
                                        flat_idx, options, engine, cache_ctx);
}

absl::StatusOr<std::vector<Literal>> MakeDataflowConstrainedArguments(
    const HloModule* module, const FakeArgumentsOptions& options) {
  std::unique_ptr<std::minstd_rand0> default_engine;
  std::minstd_rand0* engine = options.engine;
  if (!options.pseudo_random) {
    engine = nullptr;
  } else if (engine == nullptr) {
    default_engine = std::make_unique<std::minstd_rand0>();
    engine = default_engine.get();
  }

  std::optional<FakeArgumentCacheContext> cache_ctx =
      MaybeCreateCacheContext(options, engine);
  FakeArgumentCacheContext* cache_ctx_ptr =
      cache_ctx.has_value() ? &*cache_ctx : nullptr;

  const auto params = module->entry_computation()->parameter_instructions();
  const HloModuleConfig& module_config = module->config();
  auto get_param_shape = [&](int i) -> const Shape& {
    if (module_config.has_entry_computation_layout() &&
        module_config.entry_computation_layout()
            .parameter_layout(i)
            .shape()
            .is_static()) {
      return module_config.entry_computation_layout()
          .parameter_layout(i)
          .shape();
    }
    return params[i]->shape();
  };

  // Count total top-level elements across all parameters.
  int64_t total_flat_indices = 0;
  for (int i = 0; i < params.size(); ++i) {
    const Shape& param_shape = get_param_shape(i);
    if (param_shape.IsTuple()) {
      total_flat_indices += param_shape.tuple_shapes().size();
    } else {
      total_flat_indices += 1;
    }
  }

  ABSL_RETURN_IF_ERROR(
      ValidateDataflowConstrainedOptions(options, total_flat_indices));

  ABSL_ASSIGN_OR_RETURN(
      auto constraint_states,
      ConstraintPropagator::Run(*module, options.get_index_known_zeroes));

  int64_t current_flat_idx = 0;
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(arguments[i],
                     GenerateParameterLiteral(
                         params[i], get_param_shape(i), constraint_states,
                         options, engine, current_flat_idx, cache_ctx_ptr));
  }
  return std::move(arguments);
}

absl::Status VerifyHloModule(HloModule* const module, bool layout_sensitive,
                             bool allow_mixed_precision) {
  return HloVerifier(/*layout_sensitive=*/layout_sensitive,
                     /*allow_mixed_precision=*/allow_mixed_precision)
      .Run(module)
      .status();
}

std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs) {
  CHECK_LE(lhs->shape().dimensions().size(), 2);
  CHECK_LE(rhs->shape().dimensions().size(), 2);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->resize(
      2, PrecisionConfig::DEFAULT);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(
      lhs->shape().dimensions().size() > 1 ? 1 : 0);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  return std::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dot_dimension_numbers, precision_config);
}

}  // namespace xla
