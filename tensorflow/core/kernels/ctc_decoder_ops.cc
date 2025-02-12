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

// See docs in ../ops/ctc_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/ctc/ctc_beam_search.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
inline T RowMax(const typename TTypes<T>::UnalignedConstMatrix& m, int r,
                int* c) {
  *c = 0;
  CHECK_LT(0, m.dimension(1));
  auto p = m(r, 0);
  for (int i = 1; i < m.dimension(1); ++i) {
    if (m(r, i) > p) {
      p = m(r, i);
      *c = i;
    }
  }
  return p;
}

class CTCDecodeHelper {
 public:
  CTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  absl::Status ValidateInputsGenerateOutputs(
      OpKernelContext* ctx, const Tensor** inputs, const Tensor** seq_len,
      Tensor** log_prob, OpOutputList* decoded_indices,
      OpOutputList* decoded_values, OpOutputList* decoded_shape) const {
    absl::Status status = ctx->input("inputs", inputs);
    if (!status.ok()) return status;
    status = ctx->input("sequence_length", seq_len);
    if (!status.ok()) return status;

    const TensorShape& inputs_shape = (*inputs)->shape();

    if (inputs_shape.dims() != 3) {
      return errors::InvalidArgument("inputs is not a 3-Tensor");
    }
    if (inputs_shape.num_elements() == 0) {
      return errors::InvalidArgument("inputs must not be empty");
    }

    const int64_t max_time = inputs_shape.dim_size(0);
    const int64_t batch_size = inputs_shape.dim_size(1);

    if (max_time == 0) {
      return errors::InvalidArgument("max_time is 0");
    }
    if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
      return errors::InvalidArgument("sequence_length is not a vector");
    }

    if (!(batch_size == (*seq_len)->dim_size(0))) {
      return errors::FailedPrecondition(
          "len(sequence_length) != batch_size.  ",
          "len(sequence_length):  ", (*seq_len)->dim_size(0),
          " batch_size: ", batch_size);
    }

    auto seq_len_t = (*seq_len)->vec<int32>();

    for (int b = 0; b < batch_size; ++b) {
      if (!(seq_len_t(b) <= max_time)) {
        return errors::FailedPrecondition("sequence_length(", b,
                                          ") <= ", max_time);
      }
    }

    absl::Status s = ctx->allocate_output(
        "log_probability", TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;

    s = ctx->output_list("decoded_indices", decoded_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", decoded_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", decoded_shape);
    if (!s.ok()) return s;

    return absl::OkStatus();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  absl::Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int>>>& sequences,
      OpOutputList* decoded_indices, OpOutputList* decoded_values,
      OpOutputList* decoded_shape) const {
    // Calculate the total number of entries for each path
    const int64_t batch_size = sequences.size();
    std::vector<int64_t> num_entries(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto& batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      Tensor* p_indices = nullptr;
      Tensor* p_values = nullptr;
      Tensor* p_shape = nullptr;

      const int64_t p_num = num_entries[p];

      absl::Status s =
          decoded_indices->allocate(p, TensorShape({p_num, 2}), &p_indices);
      if (!s.ok()) return s;
      s = decoded_values->allocate(p, TensorShape({p_num}), &p_values);
      if (!s.ok()) return s;
      s = decoded_shape->allocate(p, TensorShape({2}), &p_shape);
      if (!s.ok()) return s;

      auto indices_t = p_indices->matrix<int64_t>();
      auto values_t = p_values->vec<int64_t>();
      auto shape_t = p_shape->vec<int64_t>();

      int64_t max_decoded = 0;
      int64_t offset = 0;

      for (int64_t b = 0; b < batch_size; ++b) {
        auto& p_batch = sequences[b][p];
        int64_t num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        if (num_decoded > 0) {
          DCHECK_NE(values_t.data(), nullptr)
              << "values_t should not be nullptr: p_num=" << p_num
              << " num_decoded=" << num_decoded;
          DCHECK_LT(offset, values_t.size())
              << "offset should be smaller than values_t.size()";
          std::copy_n(p_batch.begin(), num_decoded, &values_t(offset));
        }
        for (int64_t t = 0; t < num_decoded; ++t, ++offset) {
          indices_t(offset, 0) = b;
          indices_t(offset, 1) = t;
        }
      }

      shape_t(0) = batch_size;
      shape_t(1) = max_decoded;
    }
    return absl::OkStatus();
  }

 private:
  int top_paths_;
  CTCDecodeHelper(const CTCDecodeHelper&) = delete;
  void operator=(const CTCDecodeHelper&) = delete;
};

template <typename T>
class CTCGreedyDecoderOp : public OpKernel {
 public:
  explicit CTCGreedyDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* seq_len;
    Tensor* log_prob = nullptr;
    OpOutputList decoded_indices;
    OpOutputList decoded_values;
    OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    const TensorShape& inputs_shape = inputs->shape();

    std::vector<typename TTypes<T>::UnalignedConstMatrix> input_list_t;
    const int64_t max_time = inputs_shape.dim_size(0);
    const int64_t batch_size = inputs_shape.dim_size(1);
    const int64_t num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    auto inputs_t = inputs->tensor<T, 3>();

    input_list_t.reserve(max_time);
    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
    }
    auto seq_len_t = seq_len->vec<int32>();
    auto log_prob_t = log_prob->matrix<T>();

    log_prob_t.setZero();

    int blank_index =
        (blank_index_ < 0) ? num_classes + blank_index_ : blank_index_;
    OP_REQUIRES(ctx, FastBoundsCheck(blank_index, num_classes),
                errors::InvalidArgument("blank_index expected to be between ",
                                        -num_classes, " and ", num_classes - 1,
                                        " but was ", blank_index_));

    // Perform best path decoding
    std::vector<std::vector<std::vector<int> > > sequences(batch_size);
    auto decode = [&](const int64_t begin, const int64_t end) {
      for (int b = begin; b < end; ++b) {
        sequences[b].resize(1);
        auto &sequence = sequences[b][0];
        int prev_indices = -1;
        for (int t = 0; t < seq_len_t(b); ++t) {
          int max_class_indices;
          OP_REQUIRES(ctx, input_list_t[t].dimension(1) > 0,
                      errors::InvalidArgument("Invalid input dimensions."));
          log_prob_t(b, 0) +=
              -RowMax<T>(input_list_t[t], b, &max_class_indices);
          if (max_class_indices != blank_index &&
              !(merge_repeated_ && max_class_indices == prev_indices)) {
            sequence.push_back(max_class_indices);
          }
          prev_indices = max_class_indices;
        }
      }
    };

    const int64_t kCostPerUnit = 50 * max_time * num_classes;
    const int64_t total = batch_size;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, total,
          kCostPerUnit, decode);

    OP_REQUIRES_OK(
        ctx, decode_helper_.StoreAllDecodedSequences(
                 sequences, &decoded_indices, &decoded_values, &decoded_shape));
  }

 private:
  CTCDecodeHelper decode_helper_;
  bool merge_repeated_;
  int blank_index_;

  CTCGreedyDecoderOp(const CTCGreedyDecoderOp&) = delete;
  void operator=(const CTCGreedyDecoderOp&) = delete;
};

#define REGISTER_CPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CTCGreedyDecoder").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CTCGreedyDecoderOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

// CTC beam search
template <typename T>
class CTCBeamSearchDecoderOp : public OpKernel {
 public:
  explicit CTCBeamSearchDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
    int top_paths;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
    decode_helper_.SetTopPaths(top_paths);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* seq_len;
    Tensor* log_prob = nullptr;
    OpOutputList decoded_indices;
    OpOutputList decoded_values;
    OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    auto inputs_t = inputs->tensor<T, 3>();
    auto seq_len_t = seq_len->vec<int32>();
    auto log_prob_t = log_prob->matrix<T>();

    const TensorShape& inputs_shape = inputs->shape();

    const int64_t max_time = inputs_shape.dim_size(0);
    const int64_t batch_size = inputs_shape.dim_size(1);
    const int64_t num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    log_prob_t.setZero();

    std::vector<typename TTypes<T>::UnalignedConstMatrix> input_list_t;

    input_list_t.reserve(max_time);
    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
    }

    ctc::CTCBeamSearchDecoder<T> beam_search(num_classes, beam_width_,
                                             &beam_scorer_, 1 /* batch_size */,
                                             merge_repeated_);
    Tensor input_chip(DataTypeToEnum<T>::v(), TensorShape({num_classes}));
    auto input_chip_t = input_chip.flat<T>();

    std::vector<std::vector<std::vector<int> > > best_paths(batch_size);
    std::vector<T> log_probs;

    // Assumption: the blank index is num_classes - 1
    for (int b = 0; b < batch_size; ++b) {
      auto& best_paths_b = best_paths[b];
      best_paths_b.resize(decode_helper_.GetTopPaths());
      for (int t = 0; t < seq_len_t(b); ++t) {
        input_chip_t = input_list_t[t].chip(b, 0);
        auto input_bi = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
            input_chip_t.data(), num_classes);
        beam_search.Step(input_bi);
      }
      OP_REQUIRES_OK(
          ctx, beam_search.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b,
                                    &log_probs, merge_repeated_));

      beam_search.Reset();

      for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
        log_prob_t(b, bp) = log_probs[bp];
      }
    }

    OP_REQUIRES_OK(ctx, decode_helper_.StoreAllDecodedSequences(
                            best_paths, &decoded_indices, &decoded_values,
                            &decoded_shape));
  }

 private:
  CTCDecodeHelper decode_helper_;
  typename ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer beam_scorer_;
  bool merge_repeated_;
  int beam_width_;
  CTCBeamSearchDecoderOp<T>(const CTCBeamSearchDecoderOp<T>&) = delete;
  void operator=(const CTCBeamSearchDecoderOp<T>&) = delete;
};

#define REGISTER_CPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("CTCBeamSearchDecoder").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CTCBeamSearchDecoderOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

}  // end namespace tensorflow
