/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace stream_executor {
class RedzoneAllocator;
}  // namespace stream_executor

namespace xla {
class AutotuneResult;
}  // namespace xla

namespace tensorflow {

class NodeDef;
using xla::AutotuneResult;

template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// Return whether the redzone check is disabled.
//
// Controlled by the TF_DISABLE_RZ_CHECK environment variable.
bool RedzoneCheckDisabled();

// Return an allocated buffer with redzones the size of `buffer`. Does
// *not* copy the contents of the `buffer` into the newly allocated buffer:
// assumes that buffer is a pure out-parameter.
//
// Returns `buffer` if RedzoneCheckDisabled() is true.
//
// On error, return `buffer`, and log an error message (once).
se::DeviceMemoryBase WrapRedzoneBestEffort(se::RedzoneAllocator* rz_allocator,
                                           se::DeviceMemoryBase buffer);

// Check the passed allocator for redzone violations.
// If violations have occurred, mark the corresponding autotune result
// as a failure.
void CheckRedzones(const se::RedzoneAllocator& rz_allocator,
                   AutotuneResult* autotune_result);

template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// Returns whether cuBLASLt is enabled.
//
// Controlled by the TF_USE_CUBLASLT environment variable.
bool EnableCublasLtGemm();

namespace internal {

template <typename Parameters>
struct AutotuneMapHasher {
  std::size_t operator()(const Parameters& parameter) const {
    return parameter.hash();
  }
};

}  // namespace internal

// A helper class that looks up the best autotuned config from parameters.
// Due to the noisy nature of autotune, especially with multiple devices, it
// only accepts a config if its margin exceeds a threshold.
// For the same shape configs, if a new best config matches the previous best,
// they get promoted; otherwise, the winner gets demoted. This process stops
// when the winner's score exceeds the threshold.
// In a bad case when two configs are very close to each other and flips
// back and forth randomly, the expected number of experiments before autotune
// settles is O(threshold ^ 2). So we recommend that number of warmup runs
// for any benchmarks.
template <typename Parameters, typename Config,
          typename Hasher = internal::AutotuneMapHasher<Parameters>>
class AutotuneMap {
 public:
  bool Find(const Parameters& params, Config* config) const {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end() ||
        (iter->second.score < min_score_threshold_ &&
         iter->second.count <= max_autotune_count_)) {
      return false;
    }
    *config = iter->second.config;
    return true;
  }
  void Insert(const Parameters& params, const Config& config) {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    int new_score = 0;
    if (iter == params_config_map_.end()) {
      // Create a new entry if params is new.
      VLOG(1) << GetActionSummary("creates", params, config);
      params_config_map_.insert(
          std::make_pair(params, ValueType{config, 1, 1}));
      new_score = 1;
    } else if (iter->second.score < min_score_threshold_ &&
               iter->second.count <= max_autotune_count_) {
      DCHECK_GT(iter->second.score, 0);
      if (iter->second.config != config) {
        // If it is different from the current winner, demotes the winner.
        VLOG(1) << GetActionSummary("demotes", params, config);
        new_score = --iter->second.score;
        ++iter->second.count;
        if (new_score <= 0) {
          VLOG(1) << GetActionSummary("erases", params, config);
          params_config_map_.erase(iter);
        }
      } else {
        // If it is the same as the current winner, promotes the winner.
        VLOG(1) << GetActionSummary("promotes", params, config);
        new_score = ++iter->second.score;
        ++iter->second.count;
      }
    }
    if (new_score >= min_score_threshold_) {
      VLOG(1) << GetActionSummary("accepts", params, config);
    } else if (autotune_global_count_ >= max_autotune_global_count_) {
      // The autotuning exceeds the max iteration threshold and we accept the
      // the winner if it exists in the map, otherwise we accept the current
      // winner.
      auto winner = params_config_map_.find(params);
      if (winner == params_config_map_.end()) {
        VLOG(1) << GetActionSummary("creates", params, config);
        for (int i = 0; i < min_score_threshold_; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        params_config_map_.insert(
            std::make_pair(params, ValueType{config, min_score_threshold_, 1}));
      } else {
        int promotes_times = min_score_threshold_ - winner->second.score;
        for (int i = 0; i < promotes_times; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        winner->second.score = min_score_threshold_;
      }
      VLOG(1) << GetActionSummary("accepts", params, config);
    }
    autotune_global_count_++;
  }

  std::unordered_map<Parameters, Config, Hasher> GetMap() const {
    mutex_lock lock(mu_);
    std::unordered_map<Parameters, Config, Hasher> map;
    for (const auto& entry : params_config_map_) {
      map.insert(std::make_pair(entry.first, entry.second.config));
    }
    return map;
  }

  // Only for testing
  void ClearMap() {
    mutex_lock lock(mu_);
    params_config_map_.clear();
  }

 private:
  // Underlying data structure of values in the map.
  struct ValueType {
    Config config;
    int32 score;
    int32 count;
  };
  AutotuneMap(const std::string& name) : name_(name) {
    min_score_threshold_ = 1;
    int min_warmup_iterations = 10;
    const char* threshold_str = getenv("TF_AUTOTUNE_THRESHOLD");
    if (threshold_str != nullptr) {
      VLOG(1) << "TF_AUTOTUNE_THRESHOLD = " << threshold_str;
      strings::safe_strto32(threshold_str, &min_score_threshold_);
    }
    const char* min_warmup_iteration_str =
        getenv("TF_AUTOTUNE_MIN_WARMUP_ITERATIONS");
    if (min_warmup_iteration_str != nullptr) {
      strings::safe_strto32(min_warmup_iteration_str, &min_warmup_iterations);
    }
    min_score_threshold_ = std::max(min_score_threshold_, 1);
    max_autotune_count_ = std::max(
        5 * min_score_threshold_ * min_score_threshold_, min_warmup_iterations);
    max_autotune_global_count_ = 2 * max_autotune_count_;
    autotune_global_count_ = 0;
  }

  template <class Group, class Params, class Cfg, class Hash>
  friend class AutotuneSingleton;

  std::string GetActionSummary(StringPiece action, const Parameters& params,
                               const Config& config) {
    return strings::Printf("autotune_map %s %s: %s -> (%s)", name_.c_str(),
                           string(action).c_str(), params.ToString().c_str(),
                           config.ToString().c_str());
  }

  mutable mutex mu_;

  std::unordered_map<Parameters, ValueType, Hasher> params_config_map_
      TF_GUARDED_BY(mu_);
  std::string name_;
  int32 min_score_threshold_;
  int32 max_autotune_count_;
  int32 max_autotune_global_count_;
  int32 autotune_global_count_;

  AutotuneMap(const AutotuneMap&) = delete;
  void operator=(const AutotuneMap&) = delete;
};

// A Singleton helper that manages the global autotune results by groups.
// The caller specified arbitrary Group type that can distinguish between
// different autotune results, even if their Parameters and Configs are the
// same.
template <class Group, typename Parameters, typename Config,
          typename Hasher = internal::AutotuneMapHasher<Parameters>>
class AutotuneSingleton {
 public:
  typedef AutotuneMap<Parameters, Config, Hasher> AutotuneType;
  static AutotuneType* GetInstance() {
    static AutotuneType* instance = new AutotuneType(Group::name());
    return instance;
  }
};

// Logs convolution results to customized back-storage.
void LogConvAutotuneResults(se::dnn::ConvolutionKind kind,
                            se::dnn::DataType element_type,
                            se::DeviceMemoryBase input_buffer,
                            se::DeviceMemoryBase filter_buffer,
                            se::DeviceMemoryBase output_buffer,
                            const se::dnn::BatchDescriptor& input_desc,
                            const se::dnn::FilterDescriptor& filter_desc,
                            const se::dnn::BatchDescriptor& output_desc,
                            const se::dnn::ConvolutionDescriptor& conv_desc,
                            se::StreamExecutor* stream_exec,
                            absl::Span<const AutotuneResult> results);

// Logs fused convolution results to customized back-storage.
void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results);

// Logs fused matmul results to customized back-storage.
void LogFusedMatmulAutotuneResults(
    se::dnn::DataType ab_dtype, se::dnn::DataType c_dtype,
    se::DeviceMemoryBase a_buffer, se::DeviceMemoryBase b_buffer,
    se::DeviceMemoryBase c_buffer, se::DeviceMemoryBase bias_buffer,
    bool trans_a, bool trans_b, uint32_t m, uint32_t n, uint32_t k, int32_t lda,
    int32_t ldb, int32_t ldc, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results);

// Autotuning map entry for cuDNN-frontend-capable APIs.
//
// The longer-term intent is to remove the AlgorithmConfig variant and make this
// contain only the two LazyOpRunners, but for the time being ROCm is stuck on
// the legacy API and requires an AlgorithmConfig.
template <typename Op>
class AutotuneEntry {
 public:
  AutotuneEntry() : is_algorithm_config_(true) {}

  // Initialize with legacy-API AlgorithmConfig; used for the ROCm backend only.
  explicit AutotuneEntry(se::dnn::AlgorithmConfig config)
      : is_algorithm_config_(true), algorithm_config_(std::move(config)) {}

  AutotuneEntry(std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary,
                std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback)
      : is_algorithm_config_(false),
        op_runners_{std::move(primary), std::move(no_scratch_fallback)} {}

  // Initialize from config data, without pre-cached runners, such as when
  // loading AoT autotuning maps.
  AutotuneEntry(se::dnn::AlgorithmDesc primary,
                absl::optional<se::dnn::AlgorithmDesc> no_scratch_fallback)
      : AutotuneEntry(std::make_shared<se::dnn::LazyOpRunner<Op>>(primary),
                      no_scratch_fallback
                          ? std::make_shared<se::dnn::LazyOpRunner<Op>>(
                                *no_scratch_fallback)
                          : nullptr) {}

  // Initialize with pre-cached OpRunners, such as during autotuning.
  static StatusOr<AutotuneEntry> FromOpRunners(
      std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>> primary,
      std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>>
          no_cache_fallback) {
    TF_ASSIGN_OR_RETURN(
        auto primary_cache,
        se::dnn::LazyOpRunner<Op>::FromOpRunner(std::move(primary)));

    if (no_cache_fallback) {
      TF_ASSIGN_OR_RETURN(auto fallback_cache,
                          se::dnn::LazyOpRunner<Op>::FromOpRunner(
                              std::move(no_cache_fallback)));
      return AutotuneEntry(std::move(primary_cache), std::move(fallback_cache));

    } else {
      return AutotuneEntry(std::move(primary_cache), nullptr);
    }
  }

  struct OpRunners {
    OpRunners() = default;

    OpRunners(std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary_,
              std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback_)
        : primary(std::move(primary_)),
          no_scratch_fallback(std::move(no_scratch_fallback_)) {}

    // Null iff this 'OpRunners' is default-constructed as part of the
    // fake-variant in AutotuneEntry; users outside gpu_utils.h itself should
    // never see primary = nullptr.
    std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary;
    std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback;  // Nullable

    bool operator==(const OpRunners& other) const {
      return *primary == *other.primary &&
             ((!no_scratch_fallback && !other.no_scratch_fallback) ||
              (no_scratch_fallback && other.no_scratch_fallback &&
               *no_scratch_fallback == *other.no_scratch_fallback));
    }
  };

  bool is_algorithm_config() const { return is_algorithm_config_; }

  const se::dnn::AlgorithmConfig& GetAlgorithmConfig() const {
    DCHECK(is_algorithm_config_);
    return algorithm_config_;
  }

  const OpRunners& GetOpRunners() const {
    DCHECK(!is_algorithm_config_);
    return op_runners_;
  }

  // AutotuneMap needs to test equality to keep track of the number of times an
  // algorithm has won autotuning; for this purpose, we can use ToString to
  // determine whether runners are equal.
  bool operator==(const AutotuneEntry<Op>& other) const {
    if (is_algorithm_config_) {
      return other.is_algorithm_config_ &&
             algorithm_config_ == other.algorithm_config_;
    }

    return !other.is_algorithm_config_ && op_runners_ == other.op_runners_;
  }

  bool operator!=(const AutotuneEntry<Op>& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
    if (is_algorithm_config_) {
      return algorithm_config_.ToString();
    }
    return absl::StrCat("{", op_runners_.primary->ToString(), ", ",
                        (op_runners_.no_scratch_fallback
                             ? op_runners_.no_scratch_fallback->ToString()
                             : "(op_runners have no fallback)"),
                        "}");
  }

 private:
  // NVCC is broken, so we can't use absl::variant here.  Just fake it with a
  // bool and both fields.
  bool is_algorithm_config_;
  se::dnn::AlgorithmConfig algorithm_config_;
  OpRunners op_runners_;
};

namespace internal {
StatusOr<std::tuple<int, int>> BestCudnnConvAlgorithmIndices(
    absl::Span<const AutotuneResult> results);
}  // namespace internal

// Returns the best algorithms for the config, one is the fastest, the other is
// other is fastest with 0 scratch space. Unsuccessful autotuning results are
// allowed and ignored.
StatusOr<se::dnn::AlgorithmConfig> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results);

// Explicitly-instantiated with ConvOp and FusedConvOp.
//
// The definition can't be in the header because including .pb.h files in
// headers is forbidden.
template <typename Op>
StatusOr<AutotuneEntry<Op>> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results,
    std::vector<
        std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>>>
        runners);

// Get the Dnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64_t GetDnnWorkspaceLimit(const string& envvar_in_mb,
                             int64_t default_value_in_bytes);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
