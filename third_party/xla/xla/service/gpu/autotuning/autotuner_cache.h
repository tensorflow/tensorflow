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
#ifndef XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_CACHE_H_
#define XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_CACHE_H_

#include <functional>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {


using AutotuneNoCacheFn = std::function<absl::StatusOr<AutotuneResult>()>;

class AutotunerCache {
 public:
  // Used in the new autotuner to provide current cache compatibility.
  static absl::StatusOr<std::optional<AutotuneResult>> TryFindInCache(
      const AutotuneCacheKey& key, absl::string_view cache_dir);

  // Used in the new autotuner to provide current cache compatibility.
  struct ResultAndInserted {
    // The result that ended up in the cache. This is the existing result if
    // inserted is false, and the new result if inserted is true.
    //
    // We return a value, not a pointer, for thread safety reasons.
    AutotuneResult result;
    // Did we insert the given result into the cache?
    bool inserted;
  };

  // Used in the new autotuner to provide current cache compatibility.
  static absl::StatusOr<ResultAndInserted> AddResultToCaches(
      const AutotuneCacheKey& key, AutotuneResult result,
      absl::string_view cache_dir,
      DebugOptions::AutotuneCacheMode autotune_cache_mode);

  // Functions to save/load XLA's autotuning results.
  //
  // This is used for ahead-of-time autotuning.  Specifically:
  //
  // When XLA calls cublas (for matmuls, aka "gemm" or "dot") or cudnn (for
  // convolutions), it usually has to choose an "algorithm" for the particular
  // dot/conv.  XLA queries cublas/cudnn for a list of candidate algorithms.
  // Then it runs all of them and picks the fastest one.  This is what we call
  // "autotuning". It happens in GemmAlgorithmPicker and GpuConvAlgorithmPicker.
  //
  // Autotuning is necessary to get good performance for dot/conv.  But it also
  // has some disadvantages.
  //
  //  - Because it relies on timing data, it is fundamentally nondeterministic.
  //    But even if two algorithms have similar runtimes, our choice of
  //    algorithm may be visible to the user: Different algorithms can have
  //    different numerics, and sometimes they can even have different bugs!
  //
  //  - Trying all the candidate algorithms can be slow, especially if when some
  //    of the candidates are "very bad" and run especially slowly compared to
  //    the optimal candidate.  This slows down compilation.
  //
  // To address the disadvantages above, we allow users to save/restore the
  // autotuning choices that XLA has made, using the functions below.
  //
  // Loading autotuning results does not erase existing autotuning choices, but
  // in the event of a disagreement between the existing data and the new data,
  // the new algorithm is chosen.
  //
  // Note that even if you call LoadAutotuneResults(), if XLA encounters a
  // dot/conv that is *not* covered by the loaded data, it will go ahead and
  // autotune it like normal.  In other words, the behavior of XLA should be
  // identical with or without ahead-of-time autotuning, modulo nondeterminism.
  //
  // This is important if you want to be able to use the same autotuning file
  // with different versions of XLA, because as XLA changes, exactly which
  // dots/convs it wants to run can also change.  For example, XLA might change
  // the conv padding heuristics it uses, and we don't want that to mean that
  // all users of ahead-of-time autotuning are broken.
  static absl::StatusOr<std::string> SerializeAutotuneResults(
      bool as_textproto = false);

  // Serializes autotune results into the given proto. If optional keys are
  // provided, serializes results only for these keys.
  static absl::Status SerializeAutotuneResults(
      AutotuneResults* results,
      std::optional<const AutotuneCacheKeySet*> keys = {});

  // Loads autotune results from the given string of bytes.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static absl::Status LoadAutotuneResults(absl::string_view data,
                                          bool as_textproto = false,
                                          bool allow_override = false);

  // Loads autotune results from the given proto.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static absl::Status LoadAutotuneResults(const AutotuneResults& results,
                                          bool allow_override = false);

  // Serializes autotune results into a file.
  //
  // If `file_path` ends with ".txt" or ".textproto", then the textproto format
  // is used, otherwise the binary protobuf format.
  static absl::Status SerializeAutotuneResultsToFile(
      absl::string_view file_path);

  // As above, but if you already called SerializeAutotuneResults to get a
  // proto.
  static absl::Status SerializeAutotuneResultsToFile(
      const AutotuneResults& results, absl::string_view file_path);

  // Loads autotune results from a file.
  //
  // If `file_path` ends with ".txt" or ".textproto", then the file is
  // considered to be in the textproto format, otherwise the binary protobuf
  // format.
  //
  // Warning: The results are only loaded to the in-memory cache.
  static absl::Status LoadAutotuneResultsFromFile(absl::string_view file_path);

  // Warning: This only clears the in-memory cache. If you use a file based
  // cache you're responsible for clearing the cache directory when you want to.
  static void ClearAutotuneResults();

  // Warning: This only checks the in-memory cache. If you use a file based
  // cache, you're responsible for checking whether the cache directory is
  // empty.
  static bool ResultCacheIsEmpty();
};

absl::StatusOr<std::string> AutotuneResultsToString(
    const AutotuneResults& results, bool as_textproto);

// Returns the SHA-256 hash of the input string, encoded in base64.
//
// SHA-256 was chosen to follow industry best practices and avoid collisions.
// Git is also transitioning to SHA-256. This is probably better than
// tsl::Fingerprint128.
absl::StatusOr<std::string> GetBase64EncodedSha256Hash(absl::string_view s);

// Adds version information to each entry in AutotuneResults. Useful for unit
// tests involving hard-coded AutotuneResults (including those read from files,
// which happens automatically), as the entry version changes much more often
// than the overall structure version of the AutotuneResults itself, so it's
// nice to only have to change one place to update it.
void AddVersionToAutotuneResults(AutotuneResults& results);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_CACHE_H_
