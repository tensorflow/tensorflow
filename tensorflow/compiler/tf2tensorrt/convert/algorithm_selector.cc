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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h"

#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"

// getAlgorithmIOInfo is deprecated in TRT >= 8, replaced by
// getAlgorithmIOInfoByIndex.
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
#define ALGORITHM_IO_INFO_BY_IDX(alg, idx) *(alg).getAlgorithmIOInfoByIndex(idx)
#else
#define ALGORITHM_IO_INFO_BY_IDX(alg, idx) (alg).getAlgorithmIOInfo(idx)
#endif

namespace nvinfer1 {

std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::IAlgorithmContext& ctx) {
  os << "AlgorithmContext(name=" << ctx.getName()
     << ",nbInputs=" << ctx.getNbInputs() << ",nbOutputs=" << ctx.getNbOutputs()
     << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const nvinfer1::IAlgorithm& alg) {
  const nvinfer1::IAlgorithmVariant& variant = alg.getAlgorithmVariant();
  os << "Algorithm(" << "variant.implementation=" << variant.getImplementation()
     << ",variant.tactic=" << variant.getTactic()
     << ",timingMSec=" << alg.getTimingMSec()
     << ",workspaceSize=" << alg.getWorkspaceSize() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::IAlgorithmIOInfo& info) {
  os << "IOTensor(format=" << info.getTensorFormat()
     << ",dtype=" << info.getDataType() << ",strides=" << info.getStrides()
     << ")";
  return os;
}
}  // namespace nvinfer1

namespace tensorflow {
namespace tensorrt {
namespace convert {

bool operator>=(const AlgorithmSelectorImpl::TRTVersion& lhs,
                const AlgorithmSelectorImpl::TRTVersion& rhs) {
  if (lhs[0] > rhs[0]) return true;
  if (lhs[0] == rhs[0] && lhs[1] > rhs[1]) return true;
  if (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] > rhs[2]) return true;
  if (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
      lhs[3] >= rhs[3]) {
    return true;
  }
  return false;
}

bool AlgorithmSelectorImpl::IsTrtVersionGE(const TRTVersion& version) const {
  return version_ >= version;
}

bool AlgorithmSelectorImpl::IsShuffleLayer(ImplementationID id) const {
  if (IsTrtVersionGE({8, 2, 0, 0})) {
    return id == 0x80000000 + 13;
  }
  if (IsTrtVersionGE({8, 0, 0, 0})) {
    return id == 0x80000000 + 14;
  }
  if (IsTrtVersionGE({7, 2, 0, 0})) {
    return id == 0x80000000 + 16;
  }
  return id == 18;
}

std::set<AlgorithmSelectorImpl::TacticID>
AlgorithmSelectorImpl::GetBannedTRT72TuringTactics() {
  static const std::set<TacticID> banned_turing_72{
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_medium_nhwc_gelu_tn_v1
      -5927686925093575778,
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_interior_nhwc_gelu_tn_v1
      -3848538574386518527,
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_small_nhwc_gelu_tn_v1
      -959009792490796596};
  return banned_turing_72;
}

bool AlgorithmSelectorImpl::IsBannedTactic(TacticID id) const {
  // Disable problematic FP16-Turing tactics in TensorRT 7.2.
  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    auto banned_turing_72 = GetBannedTRT72TuringTactics();
    return banned_turing_72.find(id) != banned_turing_72.end();
  }
  return false;
}

bool AlgorithmSelectorImpl::AllowShuffleAlgorithm(
    TacticID tactic, nvinfer1::DataType input_dtype,
    nvinfer1::TensorFormat input_format) const {
  if (IsTrtVersionGE({8, 0, 0, 0}) && !IsTrtVersionGE({8, 0, 3, 0})) {
    // Reject shuffle node when input format is linear row major INT8
    // format in TensorRT 8.0 GA.
    return !(input_format == nvinfer1::TensorFormat::kLINEAR &&
             input_dtype == nvinfer1::DataType::kINT8);
  }

  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    // For TRT 7.2, accept shuffle node when input format is not 32-wide
    // channel vectorized row major FP32 format
    return !(input_format == nvinfer1::TensorFormat::kCHW32 &&
             input_dtype == nvinfer1::DataType::kFLOAT);
  }
  return true;
}

bool AlgorithmSelectorImpl::IsAlgorithmSelectorRequired() const {
  // If we are in turing for TensorRT 7.2, we need the  selector for shuffle and
  // avoiding specific Turing tactics.
  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    return true;
  }

  // If we are in TensorRT 8.0 GA, we want to reject certain types of shuffles.
  if (IsTrtVersionGE({8, 0, 0, 0}) && !IsTrtVersionGE({8, 0, 3, 0})) {
    return true;
  }

  return false;
}

namespace {

string FormatAlgorithmList(const nvinfer1::IAlgorithmContext& ctx,
                           absl::Span<const nvinfer1::IAlgorithm* const> algs) {
  return absl::StrFormat(
      "%s:\n\t%s", absl::FormatStreamed(ctx),
      absl::StrJoin(
          algs, "\n\t",
          [&ctx](std::string* out, const nvinfer1::IAlgorithm* const alg) {
            absl::StrAppendFormat(out, "%s", absl::FormatStreamed(*alg));
            for (int i = 0; i < ctx.getNbInputs() + ctx.getNbOutputs(); i++) {
              absl::StrAppendFormat(
                  out, "\n\t\t%s",
                  absl::FormatStreamed(ALGORITHM_IO_INFO_BY_IDX(*alg, i)));
            }
          }));
}

}  // namespace

TftrtAlgorithmSelector::TftrtAlgorithmSelector()
    : fixed_algorithm_idx_(GetFixedAlgorithmID()),
      selector_(AlgorithmSelectorImpl::CompileTimeTRTVersion()) {}

std::optional<int64_t> TftrtAlgorithmSelector::GetFixedAlgorithmID() {
  int64_t trt_algorithm_idx = 0;
  constexpr auto null_idx =
      std::numeric_limits<decltype(trt_algorithm_idx)>::min();
  Status status = tensorflow::ReadInt64FromEnvVar("TF_TRT_FIXED_ALGORITHM_ID",
                                                  /*default_val=*/null_idx,
                                                  &trt_algorithm_idx);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return std::nullopt;
  }
  if (trt_algorithm_idx != null_idx) {
    return std::max(static_cast<int32_t>(trt_algorithm_idx), 0);
  }
  return std::nullopt;
}

bool TftrtAlgorithmSelector::AlgorithmPolicy(
    const nvinfer1::IAlgorithmContext& context,
    const nvinfer1::IAlgorithm& alg) const {
  const nvinfer1::IAlgorithmVariant& variant = alg.getAlgorithmVariant();

  // Check if this tactic ID is banned.
  TacticID tactic_id = variant.getTactic();
  if (selector_.IsBannedTactic(tactic_id)) {
    return false;
  }

  if (selector_.IsShuffleLayer(variant.getImplementation())) {
    return selector_.AllowShuffleAlgorithm(
        tactic_id, alg.getAlgorithmIOInfo(0).getDataType(),
        alg.getAlgorithmIOInfo(0).getTensorFormat());
  }
  return true;
}

int32_t TftrtAlgorithmSelector::selectAlgorithms(
    const nvinfer1::IAlgorithmContext& algoContext,
    const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbChoices,
    int32_t* selection) noexcept {
  if (fixed_algorithm_idx_) {
    LOG(WARNING) << "Forcing TRT algorithm selection to: ID = "
                 << *fixed_algorithm_idx_;
    selection[0] = std::min(*fixed_algorithm_idx_, nbChoices - 1);
    return 1;
  }

  int num_selections = 0;

  VLOG(1) << "Algorithm selection choices: "
          << FormatAlgorithmList(algoContext,
                                 absl::MakeSpan(algoChoices, nbChoices));

  for (int i = 0; i < nbChoices; i++) {
    const nvinfer1::IAlgorithm& alg = *algoChoices[i];

    // Check layer-specific issues.
    if (!AlgorithmPolicy(algoContext, alg)) {
      LOG(WARNING) << absl::StrFormat("Rejecting Algorithm: %s ",
                                      absl::FormatStreamed(alg));
      continue;
    }
    selection[num_selections++] = i;
  }
  return num_selections;
}

// Called by TensorRT to report choices it made.
void TftrtAlgorithmSelector::reportAlgorithms(
    const nvinfer1::IAlgorithmContext* const* algoContexts,
    const nvinfer1::IAlgorithm* const* algoChoices,
    int32_t nbAlgorithms) noexcept {
  if (VLOG_IS_ON(1)) {
    string selection_msg = "Algorithms selected:\n";
    for (int i = 0; i < nbAlgorithms; i++) {
      absl::StrAppend(&selection_msg,
                      FormatAlgorithmList(*algoContexts[i],
                                          absl::MakeSpan(algoChoices + i, 1)));
    }
    VLOG(1) << selection_msg;
  }
}

std::unique_ptr<TftrtAlgorithmSelector> MaybeCreateAlgorithmSelector() {
  auto selector = std::make_unique<TftrtAlgorithmSelector>();

  if (selector->IsRequired()) {
    return selector;
  }

  return nullptr;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
