/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kDatasetType[] = "WeightedFlatMap";
constexpr const char kInputNumElements[] = "input_num_elements";
constexpr const char kWeightedFlatMapDataset[] = "WeightedFlatMapDataset";
constexpr const char kOutputTypes[] = "output_types";
constexpr const char kOutputShapes[] = "output_shapes";

// Computes the index of the `cardinalities` where the `index` appears in. The
// `cardinalities` vector is a vector of ranges represented by the cumulative
// endpoints of the ranges where the endpoints are exclusive. For example, [3,
// 6] defines 2 ranges: {0, 1, 2} and {3, 4, 5}. When called with 4 for this
// example, it returns 1.
size_t IntervalIndex(const std::vector<uint64_t>& cardinalities,
                     uint64_t index) {
  DCHECK(index < cardinalities.back())
      << "index: " << index << " cardinalities.back() " << cardinalities.back();
  return std::upper_bound(cardinalities.begin(), cardinalities.end(), index) -
         cardinalities.begin();
}

// Normalizes input cardinalities with respect to the given weights. The
// normalized cardinalities are the number of elements from each input dataset
// that this dataset reads from. They are computed in such a way at least one of
// the input datasets is exhausted while respecting the weights. For example, if
// the cardinalities of the input datasets are {100, 100, 10} and the weights
// are {0.2, 0.3, 0.5}, then the cardinalities are {4, 6, 10} because the third
// input runs out of elements after reading 4 from the first dataset, 6 from the
// second, and 10 from the third.
Status NormalizeInputCardinalities(const std::vector<double>& weights,
                                   std::vector<uint64_t>* input_cardinalities) {
  double max_weight = 0.0;
  for (const double weight : weights) {
    max_weight = std::max(max_weight, weight);
  }
  DCHECK_GT(max_weight, 0.0);
  double min_cardinality = std::numeric_limits<double>::max();
  size_t min_cardinality_index;
  for (size_t i = 0; i < input_cardinalities->size(); ++i) {
    const double cardinality = static_cast<double>((*input_cardinalities)[i]) *
                               weights[i] / max_weight;
    if (min_cardinality > cardinality) {
      min_cardinality = cardinality;
      min_cardinality_index = i;
    }
  }
  for (size_t i = 0; i < input_cardinalities->size(); ++i) {
    uint64_t cardinality = static_cast<uint64_t>(
        min_cardinality * weights[i] / weights[min_cardinality_index]);
    if (cardinality > (*input_cardinalities)[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input ", i, " needs to have at least ", cardinality,
          " elements. It only has ", (*input_cardinalities)[i], " elements."));
    }
    (*input_cardinalities)[i] = cardinality;
  }
  return absl::OkStatus();
}

}  // namespace

// A dataset kernel that fetches elements from its inputs and flattens the
// results according to the weights of its inputs.
class WeightedFlatMapDatasetOp : public DatasetOpKernel {
 public:
  explicit WeightedFlatMapDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class WeightedFlatMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::vector<DatasetBase*>& inputs,
          const std::vector<double>& weights,
          const std::vector<uint64_t>& input_cardinalities,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        inputs_(std::move(inputs)),
        weights_(std::move(weights)),
        input_cardinalities_(std::move(input_cardinalities)),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    random_indexing_compatible_ = absl::OkStatus();
    for (auto input : inputs_) {
      input->Ref();
      random_indexing_compatible_.Update(input->RandomIndexingCompatible());
    }
  }

  std::vector<uint64_t> ComputeCumulativeInputCardinalities() const {
    std::vector<uint64_t> cumulative_input_cardinalities(
        input_cardinalities_.size());
    cumulative_input_cardinalities[0] = input_cardinalities_[0];
    for (size_t i = 1; i < inputs_.size(); ++i) {
      cumulative_input_cardinalities[i] +=
          cumulative_input_cardinalities[i - 1] + input_cardinalities_[i];
    }
    return cumulative_input_cardinalities;
  }

  ~Dataset() override {
    for (auto input : inputs_) {
      input->Unref();
    }
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return std::accumulate(input_cardinalities_.begin(),
                           input_cardinalities_.end(), 0UL);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    for (const auto& input : inputs_) {
      inputs->push_back(input);
    }
    return absl::OkStatus();
  }

  Status CheckExternalState() const override {
    for (const auto& input : inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override {
    return std::make_unique<Dataset::Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> input_nodes;
    input_nodes.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
      input_nodes.push_back(input_node);
    }

    std::vector<Node*> weight_nodes;
    weight_nodes.reserve(weights_.size());
    for (const double weight : weights_) {
      Node* weight_node;
      TF_RETURN_IF_ERROR(b->AddScalar(weight, &weight_node));
      weight_nodes.push_back(weight_node);
    }

    auto s = b->AddDataset(
        this,
        /*inputs=*/{},
        /*list_inputs=*/
        {std::make_pair(0, input_nodes), std::make_pair(1, weight_nodes)},
        /*attrs=*/{}, output);
    return s;
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          input_impls_(dataset()->inputs_.size()),
          element_count_(0),
          inputs_element_count_(dataset()->inputs_.size(), 0),
          next_positions_(dataset()->inputs_.size(), 0),
          cumulative_input_cardinalities_(
              dataset()->ComputeCumulativeInputCardinalities()) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override
        ABSL_LOCKS_EXCLUDED(mu_) {
      absl::MutexLock l(&mu_);
      for (int i = 0; i < dataset()->inputs_.size(); ++i) {
        TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
            ctx, this, prefix(), &input_impls_[i]));
      }
      return absl::OkStatus();
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override
        ABSL_LOCKS_EXCLUDED(mu_) {
      absl::MutexLock l(&mu_);
      if (element_count_ >= cumulative_input_cardinalities_.back()) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
      size_t input_dataset_index;
      if (ctx->index_mapper() == nullptr) {
        input_dataset_index =
            IntervalIndex(cumulative_input_cardinalities_, element_count_);
        TF_RETURN_IF_ERROR(input_impls_[input_dataset_index]->GetNext(
            ctx, out_tensors, end_of_sequence));
      } else {
        TF_ASSIGN_OR_RETURN(auto parent_index,
                            ctx->index_mapper()(element_count_));
        input_dataset_index =
            IntervalIndex(cumulative_input_cardinalities_, parent_index);
        IteratorContext::Params params(ctx);
        params.index_mapper = GetWeightedFlatMapIndexMapper(
            ctx->index_mapper(), input_dataset_index);
        IteratorContext global_shuffle_ctx(params);
        TF_RETURN_IF_ERROR(input_impls_[input_dataset_index]->GetNext(
            &global_shuffle_ctx, out_tensors, end_of_sequence));
        ctx->MergeCheckpoint(global_shuffle_ctx.checkpoint());
      }
      ++inputs_element_count_[input_dataset_index];
      ++element_count_;
      return absl::OkStatus();
    }

    // Returns the index mapper for an input given its `input_dataset_index`.
    IndexMapperFn GetWeightedFlatMapIndexMapper(
        IndexMapperFn parent_index_mapper, size_t input_dataset_index = 0)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      size_t last_position = this->cumulative_input_cardinalities_.back();
      return [this, parent_index_mapper, input_dataset_index, last_position](
                 size_t element_position) -> absl::StatusOr<size_t> {
        // This index mapper function scans the position of the
        // `WeightedFlatMap` elements to find the first element that matches the
        // `input_dataset_index`. It updates this position each time the
        // function is called so that it does not start from the beginning the
        // next time it is called. For example, if there are 2 inputs: input0
        // and input1 with elements [0, 1, 2], and [10, 11, 12] and the output
        // is shuffled to return [1, 12, 10, 2, 11, 0]. The first time each
        // input is called, the following is what each variable has before
        // returning.
        //                       input0       input1
        //   element_position      0            0
        //   index                 1            1
        //   next_position         1            2 (next_position = 1 is skipped
        //                                         because it is for input0)
        //   index                 1 (for 1)    2 (for 12)
        // The second time around, input0 will start scanning from
        // `next_positions_[0]`, which is 1, and input1 will start scanning from
        // `next_positions_[1]`, which is 2.
        while (this->next_positions_[input_dataset_index] < last_position) {
          // `index` is the shuffled index of this dataset, not any of the
          // inputs.
          size_t index = this->next_positions_[input_dataset_index];
          if (parent_index_mapper != nullptr) {
            TF_ASSIGN_OR_RETURN(index, parent_index_mapper(index));
          }
          ++(this->next_positions_[input_dataset_index]);
          // Finds the shuffled `index` comes from dataset
          // `input_dataset_index`, computes the local offset to the input and
          // return the offset. If not, iterate to continue scanning.
          if (IntervalIndex(this->cumulative_input_cardinalities_, index) ==
              input_dataset_index) {
            // Finds the offset in input `input_dataset_index`.
            if (input_dataset_index > 0) {
              index -= cumulative_input_cardinalities_[input_dataset_index - 1];
            }
            return index;
          }
        }
        return last_position;
      };
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      absl::MutexLock l(&mu_);
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kInputNumElements, element_count_));
      for (int i = 0; i < inputs_element_count_.size(); ++i) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), absl::StrCat(kInputNumElements, "[", i, "]"),
            inputs_element_count_[i]));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      absl::MutexLock l(&mu_);
      if (ctx->restored_element_count().has_value()) {
        element_count_ = *ctx->restored_element_count();
        // Restores all input's element counts and next positions.
        std::fill(inputs_element_count_.begin(), inputs_element_count_.end(),
                  0);
        std::fill(next_positions_.begin(), next_positions_.end(), 0);
        for (int64_t count = 0; count < element_count_; ++count) {
          if (element_count_ >= cumulative_input_cardinalities_.back()) {
            break;
          }
          auto parent_index = count;
          if (ctx->index_mapper() != nullptr) {
            TF_ASSIGN_OR_RETURN(parent_index, ctx->index_mapper()(count));
          }
          auto input_dataset_index =
              IntervalIndex(cumulative_input_cardinalities_, parent_index);
          ++inputs_element_count_[input_dataset_index];
          next_positions_[input_dataset_index] = count + 1;
        }
        // Restores all inputs.
        for (int i = 0; i < inputs_element_count_.size(); ++i) {
          IteratorContext::Params params(ctx);
          params.restored_element_count = inputs_element_count_[i];
          IteratorContext ctx_copy(params);
          TF_RETURN_IF_ERROR(RestoreInput(&ctx_copy, reader, input_impls_[i]));
          ctx->MergeCheckpoint(ctx_copy.checkpoint());
        }
        return absl::OkStatus();
      }
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kInputNumElements, &element_count_));
      for (int i = 0; i < inputs_element_count_.size(); ++i) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impls_[i]));
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            prefix(), absl::StrCat(kInputNumElements, "[", i, "]"),
            &inputs_element_count_[i]));
      }
      return absl::OkStatus();
    }

   private:
    mutable absl::Mutex mu_;
    std::vector<std::unique_ptr<IteratorBase>> input_impls_
        ABSL_GUARDED_BY(mu_);
    // Counts the number of elements this iterator has produced.
    int64_t element_count_ ABSL_GUARDED_BY(mu_) = 0;
    // Counts the number of elements each input iterator has produced.
    std::vector<int64_t> inputs_element_count_ ABSL_GUARDED_BY(mu_);
    // Keeps track of the position of this iterator that each input starts to
    // scan for its next index.
    std::vector<size_t> next_positions_;
    std::vector<uint64_t> cumulative_input_cardinalities_;
  };

  const std::vector<DatasetBase*> inputs_;
  std::vector<double> weights_;
  std::vector<uint64_t> input_cardinalities_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_;
};

WeightedFlatMapDatasetOp::WeightedFlatMapDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void WeightedFlatMapDatasetOp::MakeDataset(OpKernelContext* ctx,
                                           DatasetBase** output) {
  OpInputList input_list;
  OP_REQUIRES_OK(ctx, ctx->input_list("input_datasets", &input_list));
  OP_REQUIRES(ctx, input_list.size() > 1,
              absl::InvalidArgumentError(
                  "WeightedFlatMap must have at least two input datasets."));
  OpInputList weight_list;
  OP_REQUIRES_OK(ctx, ctx->input_list("weights", &weight_list));
  OP_REQUIRES(ctx, weight_list.size() == input_list.size(),
              absl::InvalidArgumentError(
                  absl::StrCat("`input_datasets` and `weights` of the "
                               "WeightedFlatMap must have the same size. Got ",
                               input_list.size(), " datasets and ",
                               weight_list.size(), " weights.")));

  std::vector<DatasetBase*> inputs;
  for (const auto& tensor : input_list) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &input));
    inputs.push_back(input);
  }

  for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
    OP_REQUIRES(ctx, inputs[i]->output_dtypes() == output_types_,
                absl::InvalidArgumentError(absl::StrCat(
                    "All inputs to WeightedFlatMap must have the same output "
                    "types. Input ",
                    i, " has output types: ",
                    DataTypeVectorString(inputs[i]->output_dtypes()),
                    ". Expected: ", DataTypeVectorString(output_types_), ".")));
  }

  // Merge the output shapes of all the input datasets, returning an
  // error if any of them are incompatible.
  for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
    OP_REQUIRES(ctx, inputs[i]->output_shapes().size() == output_shapes_.size(),
                absl::InvalidArgumentError(
                    absl::StrCat("All inputs to WeightedFlatMap must have "
                                 "compatible outputs. Input ",
                                 i, " has ", inputs[i]->output_shapes().size(),
                                 " components. Expected to have ",
                                 output_shapes_.size(), " components.")));
    for (size_t j = 0, num_components = output_shapes_.size();
         j < num_components; ++j) {
      PartialTensorShape result;
      OP_REQUIRES(
          ctx,
          output_shapes_[j]
              .MergeWith(inputs[i]->output_shapes().at(j), &result)
              .ok(),
          absl::InvalidArgumentError(absl::StrCat(
              "All inputs to WeightedFlatMap must have compatible output "
              "shapes. Component ",
              j, " of input ", i,
              " has shape: ", inputs[i]->output_shapes().at(j).DebugString(),
              ". Expected to be compatible with shape: ",
              output_shapes_[j].DebugString(), ".")));
      output_shapes_[j] = std::move(result);
    }
  }

  // Checks that none of the input dataset has unknown or infinite cardinality.
  for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    const auto cardinality = inputs[i]->Cardinality();
    OP_REQUIRES(
        ctx,
        (cardinality != kUnknownCardinality) &&
            (cardinality != kInfiniteCardinality),
        absl::InvalidArgumentError(absl::StrCat(
            "Cardinalities of the inputs must be known. Input ", i, " has ",
            (cardinality == kInfiniteCardinality ? "INFINITE" : "UNKNOWN"),
            " cardinality.")));
  }

  std::vector<double> weights;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const auto weight = weight_list[i].scalar<double>()();
    OP_REQUIRES(ctx, weight > 0.0,
                absl::InvalidArgumentError(
                    absl::StrCat("`weights` must be greater than 0.0. Input ",
                                 i, " has a weight of ", weight)));
    weights.emplace_back(weight);
  }
  std::vector<uint64_t> input_cardinalities(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_cardinalities[i] = inputs[i]->Cardinality();
  }
  OP_REQUIRES_OK(ctx,
                 NormalizeInputCardinalities(weights, &input_cardinalities));
  *output = new Dataset(ctx, std::move(inputs), std::move(weights),
                        std::move(input_cardinalities), output_types_,
                        output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name(kWeightedFlatMapDataset).Device(DEVICE_CPU),
                        WeightedFlatMapDatasetOp);
}

}  // namespace data
}  // namespace tensorflow
