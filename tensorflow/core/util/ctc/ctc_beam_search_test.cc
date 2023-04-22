/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This test illustrates how to make use of the CTCBeamSearchDecoder using a
// custom BeamScorer and BeamState based on a dictionary with a few artificial
// words.
#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include <cmath>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace {

template <class T>
using TestData = std::vector<std::vector<std::vector<T>>>;

// The HistoryBeamState is used to keep track of the current candidate and
// caches the expansion score (needed by the scorer).
template <class T>
struct HistoryBeamState {
  T score;
  std::vector<int> labels;
};

// DictionaryBeamScorer essentially favors candidates that can still become
// dictionary words. As soon as a beam candidate is not a dictionary word or
// a prefix of a dictionary word it gets a low probability at each step.
//
// The dictionary itself is hard-coded a static const variable of the class.
template <class T, class BeamState>
class DictionaryBeamScorer
    : public tensorflow::ctc::BaseBeamScorer<T, BeamState> {
 public:
  DictionaryBeamScorer()
      : tensorflow::ctc::BaseBeamScorer<T, BeamState>(),
        dictionary_({{3}, {3, 1}}) {}

  void InitializeState(BeamState* root) const override { root->score = 0; }

  void ExpandState(const BeamState& from_state, int from_label,
                   BeamState* to_state, int to_label) const override {
    // Keep track of the current complete candidate by storing the labels along
    // the expansion path in the beam state.
    to_state->labels.push_back(to_label);
    SetStateScoreAccordingToDict(to_state);
  }

  void ExpandStateEnd(BeamState* state) const override {
    SetStateScoreAccordingToDict(state);
  }

  T GetStateExpansionScore(const BeamState& state,
                           T previous_score) const override {
    return previous_score + state.score;
  }

  T GetStateEndExpansionScore(const BeamState& state) const override {
    return state.score;
  }

  // Simple dictionary used when scoring the beams to check if they are prefixes
  // of dictionary words (see SetStateScoreAccordingToDict below).
  const std::vector<std::vector<int>> dictionary_;

 private:
  void SetStateScoreAccordingToDict(BeamState* state) const;
};

template <class T, class BeamState>
void DictionaryBeamScorer<T, BeamState>::SetStateScoreAccordingToDict(
    BeamState* state) const {
  // Check if the beam can still be a dictionary word (e.g. prefix of one).
  const std::vector<int>& candidate = state->labels;
  for (int w = 0; w < dictionary_.size(); ++w) {
    const std::vector<int>& word = dictionary_[w];
    // If the length of the current beam is already larger, skip.
    if (candidate.size() > word.size()) {
      continue;
    }
    if (std::equal(word.begin(), word.begin() + candidate.size(),
                   candidate.begin())) {
      state->score = std::log(T(1.0));
      return;
    }
  }
  // At this point, the candidate certainly can't be in the dictionary.
  state->score = std::log(T(0.01));
}

template <class T>
void ctc_beam_search_decoding_with_and_without_dictionary() {
  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 3;
  const int num_classes = 6;

  // Plain decoder using hibernating beam search algorithm.
  typename tensorflow::ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer
      default_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T> decoder(num_classes, 10 * top_paths,
                                                   &default_scorer);

  // Dictionary decoder, allowing only two dictionary words : {3}, {3, 1}.
  DictionaryBeamScorer<T, HistoryBeamState<T>> dictionary_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T, HistoryBeamState<T>>
      dictionary_decoder(num_classes, top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }

  // Plain output, without any additional scoring.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> expected_output =
      {
          {{1, 3}, {1, 3, 1}, {3, 1, 3}},
      };

  // Dictionary outputs: preference for dictionary candidates. The
  // second-candidate is there, despite it not being a dictionary word, due to
  // stronger probability in the input to the decoder.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_dict_output = {
          {{3}, {1, 3}, {3, 1}},
      };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output[0][path]);
  }

  // Prepare dictionary outputs.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> dict_outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : dict_outputs) {
    output.resize(batch_size);
  }
  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(dict_outputs[path][0], expected_dict_output[0][path]);
  }
}

template <class T>
void ctc_beam_search_decoding_all_beam_elements_have_finite_scores() {
  const int batch_size = 1;
  const int timesteps = 1;
  const int top_paths = 3;
  const int num_classes = 6;

  // Plain decoder using hibernating beam search algorithm.
  typename tensorflow::ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer
      default_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T> decoder(num_classes, top_paths,
                                                   &default_scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{0.4, 0.3, 0, 0, 0, 0.5}}};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  // Make sure all scores are finite.
  for (int path = 0; path < top_paths; ++path) {
    LOG(INFO) << "path " << path;
    EXPECT_FALSE(std::isinf(score[0][path]));
  }
}

// A beam decoder to test label selection. It simply models N labels with
// rapidly dropping off log-probability.

typedef int LabelState;  // The state is simply the final label.

template <class T>
class RapidlyDroppingLabelScorer
    : public tensorflow::ctc::BaseBeamScorer<T, LabelState> {
 public:
  void InitializeState(LabelState* root) const override {}

  void ExpandState(const LabelState& from_state, int from_label,
                   LabelState* to_state, int to_label) const override {
    *to_state = to_label;
  }

  void ExpandStateEnd(LabelState* state) const override {}

  T GetStateExpansionScore(const LabelState& state,
                           T previous_score) const override {
    // Drop off rapidly for later labels.
    const T kRapidly = 100;
    return previous_score - kRapidly * state;
  }

  T GetStateEndExpansionScore(const LabelState& state) const override {
    return T(0);
  }
};

template <class T>
void ctc_beam_search_label_selection() {
  const int batch_size = 1;
  const int timesteps = 3;
  const int top_paths = 5;
  const int num_classes = 6;

  // Decoder which drops off log-probabilities for labels 0 >> 1 >> 2 >> 3.
  RapidlyDroppingLabelScorer<T> scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T, LabelState> decoder(
      num_classes, top_paths, &scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  // Log probabilities, slightly preferring later labels, this decision
  // should be overridden by the scorer which strongly prefers earlier labels.
  // The last one is empty label, and for simplicity  we give it an extremely
  // high cost to ignore it. We also use the first label to break up the
  // repeated label sequence.
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{-1e6, 1, 2, 3, 4, -1e6}},
      {{1e6, 0, 0, 0, 0, -1e6}},  // force label 0 to break up repeated
      {{-1e6, 1.1, 2.2, 3.3, 4.4, -1e6}},
  };

  // Expected output without label selection
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_default_output = {
          {{1, 0, 1}, {1, 0, 2}, {2, 0, 1}, {1, 0, 3}, {2, 0, 2}},
      };

  // Expected output with label selection limiting to 2 items
  // this is suboptimal because only labels 3 and 4 were allowed to be seen.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_output_size2 = {
          {{3, 0, 3}, {3, 0, 4}, {4, 0, 3}, {4, 0, 4}, {3}},
      };

  // Expected output with label width of 2.0. This would permit three labels at
  // the first timestep, but only two at the last.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_output_width2 = {
          {{2, 0, 3}, {2, 0, 4}, {3, 0, 3}, {3, 0, 4}, {4, 0, 3}},
      };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_default_output[0][path]);
  }

  // Try label selection size 2
  decoder.SetLabelSelectionParameters(2, T(-1));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_size2[0][path]);
  }

  // Try label selection width 2.0
  decoder.SetLabelSelectionParameters(0, T(2.0));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_width2[0][path]);
  }

  // Try both size 2 and width 2.0: the former is more constraining, so
  // it's equivalent to that.
  decoder.SetLabelSelectionParameters(2, T(2.0));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_size2[0][path]);
  }

  // Size 4 and width > 3.3 are equivalent to no label selection
  decoder.SetLabelSelectionParameters(4, T(3.3001));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_default_output[0][path]);
  }
}

TEST(CtcBeamSearch, FloatDecodingWithAndWithoutDictionary) {
  ctc_beam_search_decoding_with_and_without_dictionary<float>();
}

TEST(CtcBeamSearch, DoubleDecodingWithAndWithoutDictionary) {
  ctc_beam_search_decoding_with_and_without_dictionary<double>();
}

TEST(CtcBeamSearch, FloatAllBeamElementsHaveFiniteScores) {
  ctc_beam_search_decoding_all_beam_elements_have_finite_scores<float>();
}

TEST(CtcBeamSearch, DoubleAllBeamElementsHaveFiniteScores) {
  ctc_beam_search_decoding_all_beam_elements_have_finite_scores<double>();
}

TEST(CtcBeamSearch, FloatLabelSelection) {
  ctc_beam_search_label_selection<float>();
}

TEST(CtcBeamSearch, DoubleLabelSelection) {
  ctc_beam_search_label_selection<double>();
}

}  // namespace
