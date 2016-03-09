/* Copyright 2016 Google Inc. All Rights Reserved.

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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_

#include "tensorflow/core/util/ctc/ctc_beam_entry.h"

namespace tensorflow {
namespace ctc {

// BeamScorerInterface can be subclassed and provided as a template argument to
// CTCBeamSearchDecoder, if complex scoring is required. Its main purpose is to
// provide a thin layer for integrating language model scoring easily.
template <typename CTCBeamState>
class BeamScorerInterface {
 public:
  virtual ~BeamScorerInterface() {}

  // State initialization.
  virtual inline void InitializeState(CTCBeamState* root) const = 0;

  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam.
  virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                           CTCBeamState* to_state, int to_label) const = 0;

  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  virtual void ExpandStateEnd(CTCBeamState* state) const = 0;

  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability.
  virtual float GetStateExpansionScore(const CTCBeamState& state,
                                       float previous_score) const = 0;

  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  virtual float GetStateEndExpansionScore(const CTCBeamState& state) const = 0;
};

// Base implementation of BeamScorer used by default by the decoder.
template <typename CTCBeamState>
class BaseBeamScorer : public BeamScorerInterface<CTCBeamState> {
 public:
  ~BaseBeamScorer() override {}

  // In the simplest case, no state expansion is done.
  void InitializeState(CTCBeamState* root) const override {}
  void ExpandState(const CTCBeamState& from_state, int from_label,
                   CTCBeamState* to_state, int to_label) const override {}
  void ExpandStateEnd(CTCBeamState* state) const override {}
  // As there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const CTCBeamState& state,
                               float previous_score) const override {
    return previous_score;
  }
  float GetStateEndExpansionScore(const CTCBeamState& state) const override {
    return 0;
  }
};

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
