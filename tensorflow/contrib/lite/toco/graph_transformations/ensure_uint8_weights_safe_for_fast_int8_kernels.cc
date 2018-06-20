/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// === Summary ===
//
// TLDR: Some of our 8-bit arithmetic operations require uint8 weight values
// to avoid the value 0, thus ranging only in [1, 255]. This enables faster
// runtime arithmetic kernels on ARM NEON. This is not relevant on most
// other hardware architectures, and will cease to be relevant on ARM NEON
// in the future. These topics are elaborated below ("Context").
//
// Having just one isolated uint8 value equal to 0 is fine. The bad case is when
// two uint8 values are both zero and are less than 16 bytes apart.
//
// By default, toco generates a fatal error when that happens. The user may opt
// in to more lax behavior by passing
//   --allow_nudging_weights_to_use_fast_gemm_kernel.
// This causes toco to nudge such bad 0 values into the value 1, thus avoiding
// the problem in exchange for compromising on accuracy.
//
// The present graph transformation implements both the default fatal-erroring
// behavior, and, when allow_nudging_weights is set, also the lax nudging
// behavior.
//
//
// === Context ===
//
// Since March 2017, we have been using a trick to perform faster
// 8bit matrix multiplications, to our knowledge first implemented in gemmlowp
// here:
//   https://github.com/google/gemmlowp/commit/25b2989415b99e797e1ab977837111b2e231f81f
//
// This trick is explained in Appendix B of our paper,
//   https://arxiv.org/abs/1712.05877
//
// Here is the relevant paragraph:
//
//      For efficient NEON implementation of the matrix multiplication’s
//      core accumulation, we use the following trick.
//      In the multiply-add operation in (10), we first change the
//      operands’ type from uint8 to int8 (which can be done by
//      subtracting 128 from the quantized values and zero-points).
//      Thus the core multiply-add becomes
//
//            int32 += int8 * int8. (B.1)
//
//      As mentioned in section 3, with a minor tweak of the quantized
//      training process, we can ensure that the weights, once
//      quantized as int8 values, never take the value −128. Hence,
//      the product in (B.1) is never −128 ∗ −128, and is therefore
//      always less than 2^14 in absolute value. Hence, (B.1)
//      can accumulate two products on a local int16 accumulator
//      before that needs to be accumulated into the true int32 accumulator.
//      This allows the use of an 8-way SIMD multiplication
//      (SMULL on int8 operands), followed by an 8-way
//      SIMD multiply-add (SMLAL on int8 operands), followed
//      by a pairwise-add-and-accumulate into the int32 accumulators
//      (SADALP).
//
// As that paragraph notes, quantized training should be suitably modified to
// ensure that quantized uint8 weights value only range in [1, 255]. So the
// problem that we are dealing with is only about the existing 8-bit quantized
// models that haven't been trained specifically to get 8-bit weights only in
// [1, 255].
//
// This spreadsheet shows the speed benefit of this trick across many existing
// ARM-architecture CPUs:
//
//    https://docs.google.com/spreadsheets/d/1-0LjdMvW0XtH1bYknC0bQINoFaxjTuL9eplZZcitykI/edit?usp=sharing
//
// Compare Row 18 (fast int8 trick) to Row 20 (regular uint8 kernel).
//
// The introduction of the 'dotprod' extension to ARM NEON, specifically the
// SDOT instruction, renders this eventually moot. See the experimental
// kernels contributed by ARM here,
//
//     https://github.com/google/gemmlowp/pull/116
//
// However, as of April 2018, there don't seem to be any commercially available
// CPU supporting these instructions (yet); we are waiting for
// Cortex-A{75,55}-r1 to become available; the "-r1" is key here. Even if such
// CPUs become available soon, it will presumably take years for them to
// overtake the large volume of existing CPUs not supporting these new
// instructions, especially in current and future low-end devices. All in all,
// we can foresee these 'fast int8 kernels' to remain important to have into
// the 2020s.
//
bool EnsureUint8WeightsSafeForFastInt8Kernels::Run(Model* model,
                                                   std::size_t op_index) {
  const auto& op = *model->operators[op_index];
  int weights_index = 0;
  switch (op.type) {
    case OperatorType::kConv:
      weights_index = 1;
      break;
    case OperatorType::kLstmCell:
      weights_index = 2;
      break;
    case OperatorType::kFullyConnected: {
      weights_index = 1;
      const auto& fc_op = static_cast<const toco::FullyConnectedOperator&>(op);
      CHECK(!fc_op.experimental_shuffled_weights)
          << "This graph transformation expects to run before FC weights get "
             "shuffled.";
      break;
    }
    default:
      // Other operator types are unaffected by this graph transformation,
      // because their runtime implementations don't use the fast int8 trick.
      // In particular that's the case of DepthwiseConv at the moment.
      // We have to update this logic when that changes, e.g. if in the future
      // some DepthwiseConv kernel wants to use the trick.
      //
      // The reason why that's not so likely, hence why it's fairly safe to
      // stay conservative in the list of operators that we handle here, is that
      // the fast int8 kernel trick is only applicable to ops that either are
      // implemented as a GEMM, or use symmetric ranges for both weights and
      // activations. The reason why GEMM is special (can use the trick even
      // without symmetric ranges) is that it is so arithmetic-intense that
      // it can use techniques reducing its implementation to the symmetric
      // ranges case, with limited relative overhead (O(N^2) overhead vs
      // O(N^3) GEMM cost). See https://arxiv.org/pdf/1712.05877, section
      // 2.3 Efficient handling of zero-points.
      //
      // That's why at the moment we only handle operators that use a GEMM
      // (Conv, fully-connected --- note that LSTM merely wraps a
      // fully-connected operator).
      return false;
  }

  const string& name = op.inputs[weights_index];
  auto& array = model->GetArray(name);
  if (!array.buffer) {
    return false;
  }
  if (array.data_type != ArrayDataType::kUint8) {
    return false;
  }
  auto& buffer_data = array.GetMutableBuffer<ArrayDataType::kUint8>().data;

  int count_bad = 0;
  int index_of_previous_bad_value = 0;
  bool changed = false;

  for (int i = 0; i < buffer_data.size(); i++) {
    if (buffer_data[i] == 0) {
      count_bad++;
      if (count_bad > 1) {
        const int distance = i - index_of_previous_bad_value;
        // Semi-arbitrary threshold. The idea is that trouble only occurs
        // when two bad values are very close to each other so that they
        // are jointly used within registers inside some GEMM kernel.
        // The details of that depend on the kernel. Our current fast ARM64
        // kernel, for instance, only has an issue when the distance between
        // consecutive bad values is exactly 8. We do not want to track such
        // kernel details too closely here, so we pick a threshold that's
        // a bit larger than that, to give us room to change kernels in the
        // future without worrying.
        static constexpr int kMinDistanceBetweenBadValues = 16;
        if (distance < kMinDistanceBetweenBadValues) {
          if (allow_nudging_weights()) {
            buffer_data[i] = 1;
            changed = true;
            continue;
          }
          LOG(FATAL) << "Bad value for " << name << " at index " << i
                     << ", previous bad value at index "
                     << index_of_previous_bad_value << ", distance=" << distance
                     << ", kMinDistanceBetweenBadValues="
                     << kMinDistanceBetweenBadValues << ". Consider passing "
                     << "--allow_nudging_weights_to_use_fast_gemm_kernel "
                     << "if you don't care about accuracy.";
        }
      }
      index_of_previous_bad_value = i;
    }
  }

  if (changed) {
    AddMessageF("Tweaked weights values for %s", LogName(op));
  }

  return changed;
}

}  // namespace toco
