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

#ifndef TENSORFLOW_COMPILER_XLA_AUTOTUNE_SERIALIZE_H_
#define TENSORFLOW_COMPILER_XLA_AUTOTUNE_SERIALIZE_H_

#include <string>

#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Functions to save/load XLA's autotuning results.
//
// This is used for ahead-of-time autotuning.  Specifically:
//
// When XLA calls cublas (for matmuls, aka "gemm" or "dot") or cudnn (for
// convolutions), it usually has to choose an "algorithm" for the particular
// dot/conv.  XLA queries cublas/cudnn for a list of candidate algorithms.  Then
// it runs all of them and picks the fastest one.  This is what we call
// "autotuning". It happens in GemmAlgorithmPicker and GpuConvAlgorithmPicker.
//
// Autotuning is necessary to get good performance for dot/conv.  But it also
// has some disadvantages.
//
//  - Because it relies on timing data, it is fundamentally nondeterministic.
//    But even if two algorithms have similar runtimes, our choice of algorithm
//    may be visible to the user: Different algorithms can have different
//    numerics, and sometimes they can even have different bugs!
//
//  - Trying all the candidate algorithms can be slow, especially if when some
//    of the candidates are "very bad" and run especially slowly compared to the
//    optimal candidate.  This slows down compilation.
//
// To address the disadvantages above, we allow users to save/restore the
// autotuning choices that XLA has made, using the functions below.
//
// Loading autotuning results does not erase existing autotuning choices, but in
// the event of a disagreement between the existing data and the new data, the
// new algorithm is chosen.
//
// Note that even if you call LoadAutotuneResults(), if XLA encounters a
// dot/conv that is *not* covered by the loaded data, it will go ahead and
// autotune it like normal.  In other words, the behavior of XLA should be
// identical with or without ahead-of-time autotuning, modulo nondeterminism.
//
// This is important if you want to be able to use the same autotuning file with
// different versions of XLA, because as XLA changes, exactly which dots/convs
// it wants to run can also change.  For example, XLA might change the conv
// padding heuristics it uses, and we don't want that to mean that all users of
// ahead-of-time autotuning are broken.
//
StatusOr<std::string> SerializeAutotuneResults();
Status LoadAutotuneResults(absl::string_view data);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_AUTOTUNE_SERIALIZE_H_
