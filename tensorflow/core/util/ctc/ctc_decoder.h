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
// LINT.IfChange

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_DECODER_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_DECODER_H_

#include <memory>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace ctc {

// The CTCDecoder is an abstract interface to be implemented when providing a
// decoding method on the timestep output of a RNN trained with CTC loss.
//
// The two types of decoding available are:
//   - greedy path, through the CTCGreedyDecoder
//   - beam search, through the CTCBeamSearchDecoder
template <class T>
class CTCDecoder {
 public:
  typedef Eigen::Map<const Eigen::ArrayXi> SequenceLength;
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      Input;
  typedef std::vector<std::vector<int>> Output;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ScoreOutput;

  CTCDecoder(int num_classes, int batch_size, bool merge_repeated)
      : num_classes_(num_classes),
        blank_index_(num_classes - 1),
        batch_size_(batch_size),
        merge_repeated_(merge_repeated) {}

  virtual ~CTCDecoder() {}

  // Dimensionality of the input/output is expected to be:
  //  - seq_len[b] - b = 0 to batch_size_
  //  - input[t].rows(b) - t = 0 to timesteps; b = 0 t batch_size_
  //  - output.size() specifies the number of beams to be returned.
  //  - scores(b, i) - b = 0 to batch_size; i = 0 to output.size()
  virtual Status Decode(const SequenceLength& seq_len,
                        const std::vector<Input>& input,
                        std::vector<Output>* output, ScoreOutput* scores) = 0;

  int batch_size() { return batch_size_; }
  int num_classes() { return num_classes_; }

 protected:
  int num_classes_;
  int blank_index_;
  int batch_size_;
  bool merge_repeated_;
};

// CTCGreedyDecoder is an implementation of the simple best path decoding
// algorithm, selecting at each timestep the most likely class at each timestep.
template <class T>
class CTCGreedyDecoder : public CTCDecoder<T> {
 public:
  typedef CTCDecoder<T> Decoder;
  CTCGreedyDecoder(int num_classes, int batch_size, bool merge_repeated)
      : CTCDecoder<T>(num_classes, batch_size, merge_repeated) {}

  Status Decode(const typename CTCDecoder<T>::SequenceLength& seq_len,
                const std::vector<typename CTCDecoder<T>::Input>& input,
                std::vector<typename CTCDecoder<T>::Output>* output,
                typename CTCDecoder<T>::ScoreOutput* scores) override {
    if (output->empty() || (*output)[0].size() < Decoder::batch_size_) {
      return errors::InvalidArgument(
          "output needs to be of size at least (1, batch_size).");
    }
    if (scores->rows() < Decoder::batch_size_ || scores->cols() == 0) {
      return errors::InvalidArgument(
          "scores needs to be of size at least (batch_size, 1).");
    }
    // For each batch entry, identify the transitions
    for (int b = 0; b < Decoder::batch_size_; ++b) {
      int seq_len_b = seq_len[b];
      // Only writing to beam 0
      std::vector<int>& output_b = (*output)[0][b];

      int prev_class_ix = -1;
      (*scores)(b, 0) = 0;
      for (int t = 0; t < seq_len_b; ++t) {
        auto row = input[t].row(b);
        int max_class_ix;
        (*scores)(b, 0) += -row.maxCoeff(&max_class_ix);
        if (max_class_ix != Decoder::blank_index_ &&
            !(Decoder::merge_repeated_ && max_class_ix == prev_class_ix)) {
          output_b.push_back(max_class_ix);
        }
        prev_class_ix = max_class_ix;
      }
    }
    return Status::OK();
  }
};

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_DECODER_H_
// LINT.ThenChange(//tensorflow/lite/kernels/ctc/ctc_decoder.h)
