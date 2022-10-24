/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <array>
#include <memory>
#include <set>

#include "absl/types/optional.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Implements core algorithm selection logic in a testable manner. The policy
// implemented depends on the given TRT version. We have this class because TRT
// interfaces make it difficult to directly test an IAlgorithmSelector
// implementation.
class AlgorithmSelectorImpl {
 public:
  using TRTVersion = std::array<int, 4>;
  using ImplementationID = int64_t;
  using TacticID = int64_t;

  static constexpr TRTVersion CompileTimeTRTVersion() {
    return TRTVersion{NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
                      NV_TENSORRT_BUILD};
  }

  explicit AlgorithmSelectorImpl(
      const TRTVersion& version = CompileTimeTRTVersion())
      : version_(version) {}

  bool IsShuffleLayer(ImplementationID id) const;

  bool IsBannedTactic(TacticID id) const;

  // Returns true if the algorithm implementing the IShuffleLayer is acceptable.
  bool AllowShuffleAlgorithm(TacticID tactic, nvinfer1::DataType input_dtype,
                             nvinfer1::TensorFormat input_format) const;

  bool IsTrtVersionGE(const TRTVersion& version) const;

  // Returns true if we know at compile time that the algorithm selector
  // should be required. This is a conservative estimate.
  bool IsAlgorithmSelectorRequired() const;

  static std::set<TacticID> GetBannedTRT72TuringTactics();

 private:
  TRTVersion version_;
};

// Impelements the TRT IAlgorithmSelector interface. The method
// "selectAlgorithms" selects allowable algorithms for each layer, and
// "reportAlgorithms" summarizes the algorithms selected by TensorRT.
class TftrtAlgorithmSelector : public nvinfer1::IAlgorithmSelector {
 private:
  using TacticID = AlgorithmSelectorImpl::TacticID;

  // An index we should choose for all algorithms. Used for debugging.
  std::optional<int32_t> fixed_algorithm_idx_;

  AlgorithmSelectorImpl selector_;

 public:
  TftrtAlgorithmSelector();

  // If the environment variable TF_TRT_FIXED_ALGORITHM_ID is empty, this
  // function returns nullopt. Otherwise, it returns the specified number.
  static std::optional<int64_t> GetFixedAlgorithmID();

  // Returns true if the algorithm associated with context is acceptable.
  bool AlgorithmPolicy(const nvinfer1::IAlgorithmContext& context,
                       const nvinfer1::IAlgorithm& alg) const;

  // This function fills the array "selection" with the indices of selected
  // algorithm candidates from "algoChoices", each of which is an implementation
  // for the kernel described by the given IAlgorithmContext. It should return a
  // number in [0, nbChoices] indicating the number of selected indices. If 0 is
  // returned, TensorRT will use its default selection mechanism.
  int32_t selectAlgorithms(const nvinfer1::IAlgorithmContext& algoContext,
                           const nvinfer1::IAlgorithm* const* algoChoices,
                           int32_t nbChoices,
                           int32_t* selection) noexcept override;

  // Called by TensorRT to report choices it made.
  void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
                        const nvinfer1::IAlgorithm* const* algoChoices,
                        int32_t nbAlgorithms) noexcept override;

  bool IsRequired() const {
    return selector_.IsAlgorithmSelectorRequired() ||
           fixed_algorithm_idx_ != std::nullopt;
  }
};

// Returns an initialized AlgorithmSelector if an algorithm selector is required
// for the current TRT version. Otherwise, returns nullptr.
std::unique_ptr<TftrtAlgorithmSelector> MaybeCreateAlgorithmSelector();

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
