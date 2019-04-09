/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/histogram/histogram.h"

namespace tensorflow {
namespace data {
namespace {

static const double kPercentile = 90.0;

// Each instance of this class wraps an iterator. Whenever an iterator created
// for this dataset invokes the `GetNext` method, the call is delegated to the
// wrapped iterator's `GetNext` method.
class WrapperDataset : public DatasetBase {
 public:
  WrapperDataset(DatasetContext::Params params,
                 const DataTypeVector* output_dtypes,
                 const std::vector<PartialTensorShape>* output_shapes,
                 IteratorBase* iterator)
      : DatasetBase(DatasetContext(std::move(params))),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes),
        real_iterator_(iterator) {}

  const DataTypeVector& output_dtypes() const override {
    return *output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return *output_shapes_;
  }

  string DebugString() const override { return "WrapperDataset"; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
    return errors::Unimplemented(DebugString(), "::AsGraphDefInternal");
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    // MakeIterator should only be called once per WrapperDataset. However,
    // since this function expects an iterator return value, we raise the
    // error only at iterator initialization time.
    bool error = iterator_created_;
    iterator_created_ = true;
    return absl::make_unique<WrapperIterator>(
        WrapperIterator::Params{this, strings::StrCat(prefix, "::Wrapper")},
        error);
  }

 private:
  class WrapperIterator : public DatasetIterator<WrapperDataset> {
   public:
    explicit WrapperIterator(const Params& params, bool error)
        : DatasetIterator<WrapperDataset>(params), error_(error) {}

    Status Initialize(IteratorContext* ctx) override {
      if (error_) {
        return errors::InvalidArgument(
            "Cannot create more than one WrapperIterator per WrapperDataset. "
            "Make sure the branches to ChooseFastestDataset do not expect the "
            "input to repeat.");
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      return dataset()->real_iterator_->GetNext(ctx, out_tensors,
                                                end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1.0);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return Status::OK();
    }

   private:
    const bool error_;
  };

  mutable bool iterator_created_ = false;
  const DataTypeVector* const output_dtypes_;
  const std::vector<PartialTensorShape>* const output_shapes_;
  IteratorBase* const real_iterator_;  // not owned.
};

// This Dataset picks between some dataset function branches. Each function is
// expected to input a dataset and output a dataset. The datasets in the
// branches are expected to be stateless. For each iterator that can be produced
// by a functions output, it is expected to call the input dataset's
// MakeIterator method at most once; otherwise, undefined behavior may occur.
class ChooseFastestBranchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ChooseFastestBranchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("branches", &funcs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_elements_per_branch",
                                     &num_elements_per_branch_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("other_arguments_lengths",
                                     &other_arguments_lengths_));
    OP_REQUIRES(
        ctx, funcs_.size() == other_arguments_lengths_.size(),
        errors::InvalidArgument(
            "branches and other_arguments_lengths must have the same length."));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "ratio_numerator",
                                                   &ratio_numerator_));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "ratio_denominator",
                                                   &ratio_denominator_));
    OP_REQUIRES(ctx, ratio_numerator_ > 0,
                errors::InvalidArgument(
                    "`ratio_numerator` must be greater than zero."));
    OP_REQUIRES(ctx, ratio_denominator_ > 0,
                errors::InvalidArgument(
                    "`ratio_denominator` must be greater than zero."));
    OP_REQUIRES(ctx, num_elements_per_branch_ % ratio_denominator_ == 0,
                errors::InvalidArgument("`num_elements_per_branch` must be "
                                        "divisible by `ratio_denominator`."));

    std::vector<std::unique_ptr<CapturedFunction>> captured_funcs(
        funcs_.size());
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));

    // Keeps track of starting index into other_arguments for a given function.
    int index = 0;
    for (int i = 0; i < funcs_.size(); ++i) {
      std::vector<Tensor> captured_args;
      captured_args.reserve(other_arguments_lengths_[i]);
      int end_index = index + other_arguments_lengths_[i];
      for (; index < end_index; ++index) {
        captured_args.push_back(inputs[index]);
      }
      OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                              funcs_[i], ctx, std::move(captured_args),
                              /*params=*/{}, &captured_funcs[i]));
    }
    *output =
        new Dataset(ctx, input, funcs_, std::move(captured_funcs),
                    output_types_, output_shapes_, num_elements_per_branch_,
                    ratio_numerator_, ratio_denominator_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, DatasetBase* input,
            const std::vector<NameAttrList>& funcs,
            std::vector<std::unique_ptr<CapturedFunction>> captured_funcs,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            int64 num_elements_per_branch, int64 ratio_numerator,
            int64 ratio_denominator)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          funcs_(funcs),
          captured_funcs_(std::move(captured_funcs)),
          output_types_(output_types),
          output_shapes_(output_shapes),
          num_elements_per_branch_(num_elements_per_branch),
          ratio_numerator_(ratio_numerator),
          ratio_denominator_(ratio_denominator) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<ChooseFastestIterator>(
          ChooseFastestIterator::Params{
              this, strings::StrCat(prefix, "::ChooseFastestBranch")});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ChooseFastestBranchDatasetOp::Dataset";
    }

    int64 Cardinality() const override {
      int64 n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      // TODO(rachelim): this might be wrong if the ratio is not fixed, for
      // example, from a BatchDataset with drop_remainder = False
      return static_cast<double>(n) * ratio_numerator_ / ratio_denominator_;
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* ratio_numerator_node;
      TF_RETURN_IF_ERROR(b->AddScalar(ratio_numerator_, &ratio_numerator_node));
      Node* ratio_denominator_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(ratio_denominator_, &ratio_denominator_node));

      std::vector<int32> other_arguments_lengths;
      other_arguments_lengths.reserve(captured_funcs_.size());
      int num_captured_inputs = 0;
      for (const auto& func : captured_funcs_) {
        num_captured_inputs += func->captured_inputs().size();
        other_arguments_lengths.push_back(func->captured_inputs().size());
      }
      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(num_captured_inputs);
      other_arguments.reserve(num_captured_inputs);
      for (const auto& captured_func : captured_funcs_) {
        TF_RETURN_IF_ERROR(captured_func->AddToGraph(ctx, b, &other_arguments,
                                                     &other_arguments_types));
      }

      // Targuments
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      // num_elements_per_branch
      AttrValue num_elements_per_branch_attr;
      b->BuildAttrValue(num_elements_per_branch_,
                        &num_elements_per_branch_attr);

      // branches
      AttrValue branches_attr;
      b->BuildAttrValue(funcs_, &branches_attr);

      // other_arguments_lengths
      AttrValue other_arguments_lengths_attr;
      b->BuildAttrValue(other_arguments_lengths, &other_arguments_lengths_attr);

      return b->AddDataset(
          this,
          /*inputs=*/
          {std::make_pair(0, input_graph_node),
           std::make_pair(1, ratio_numerator_node),
           std::make_pair(2, ratio_denominator_node)},
          /*list_inputs=*/{std::make_pair(3, other_arguments)},
          /*attrs=*/
          {std::make_pair("Targuments", other_arguments_types_attr),
           std::make_pair("num_elements_per_branch",
                          num_elements_per_branch_attr),
           std::make_pair("branches", branches_attr),
           std::make_pair("other_arguments_lengths",
                          other_arguments_lengths_attr)},
          output);
    }

   private:
    // This iterator picks the fastest of dataset branches by running
    // experiments for the first dataset()->num_elements_per_branch_ *
    // num_branches iterations.
    class ChooseFastestIterator : public DatasetIterator<Dataset> {
     public:
      explicit ChooseFastestIterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            instantiated_captured_funcs_(dataset()->funcs_.size()),
            histograms_(dataset()->funcs_.size()) {}

      Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));

        for (int i = 0; i < dataset()->funcs_.size(); ++i) {
          TF_RETURN_IF_ERROR(dataset()->captured_funcs_[i]->Instantiate(
              ctx, &instantiated_captured_funcs_[i]));
        }

        return Status::OK();
      }

      // The first num_elements_per_branch * num_branches iterations, we run
      // experiments on the branches, using (branch_index_, experiment_counter_)
      // to keep track of which experiment we're on.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        {  // Locking scope
          mutex_lock l(mu_);
          if (branch_index_ < dataset()->funcs_.size()) {
            // Still running experiments
            if (!current_iterator_) {
              TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, branch_index_,
                                                     /*is_experiment=*/true));
            }

            Status s = GetNextFromExperiment(ctx, out_tensors, end_of_sequence);
            experiment_counter_++;

            if (experiment_counter_ >= dataset()->num_elements_per_branch_) {
              // Done experimenting with this branch. Increment the branch index
              // so that on the next iteration, we will draw from the next
              // branch.
              experiment_counter_ = 0;
              branch_index_++;
              current_iterator_.reset();
            }
            return s;
          }
          if (!current_iterator_) {
            SelectFastestInputIndex();
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, fastest_index_,
                                                   /*is_experiment=*/false));
          }
        }

        return current_iterator_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(
            std::move(args),
            /*ratio=*/static_cast<double>(dataset()->ratio_numerator_) /
                dataset()->ratio_denominator_);
      }

      // TODO(rachelim): Save and restore histogram state as well. Currently,
      // if an iterator is saved and restored, the histograms start recording
      // from scratch.
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("experiment_counter"),
                                               experiment_counter_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("branch_index"), branch_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("fastest_index"), fastest_index_));
        if (current_iterator_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, current_iterator_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("experiment_counter"),
                                              &experiment_counter_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("branch_index"), &branch_index_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("fastest_index"), &fastest_index_));

        // Restore state of `current_iterator_` if it exists.
        if (!reader->Contains(full_name("input_impl_empty"))) {
          if (branch_index_ < dataset()->funcs_.size()) {
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, branch_index_,
                                                   /*is_experiment=*/true));
          } else {
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, fastest_index_,
                                                   /*is_experiment=*/false));
          }
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_iterator_));
        }
        return Status::OK();
      }

     private:
      Status GetNextFromExperiment(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DCHECK_GE(branch_index_, 0);
        DCHECK_LT(branch_index_, histograms_.size());

        int64 start = ctx->env()->NowNanos();
        Status s =
            current_iterator_->GetNext(ctx, out_tensors, end_of_sequence);

        if (experiment_counter_ > 0) {
          // Ignore the first experiment when benchmarking. It may be an outlier
          // due to session set up time and other overheads.
          histograms_[branch_index_].Add(
              static_cast<double>(ctx->env()->NowNanos() - start));
        }
        return s;
      }

      // Select the fastest input to use based on the histograms of timings
      // of the completed iterations. The input with the best 90th percentile
      // iteration time is selected.
      void SelectFastestInputIndex() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        fastest_index_ = 0;

        VLOG(2) << "90.0 percentile iteration time:";
        double best_percentile = histograms_[0].Percentile(kPercentile);
        VLOG(2) << "Branch 0: " << best_percentile;
        for (size_t i = 1, num_inputs = histograms_.size(); i < num_inputs;
             ++i) {
          double percentile = histograms_[i].Percentile(kPercentile);
          VLOG(2) << "Branch " << i << ": " << percentile;
          if (percentile <= best_percentile) {
            best_percentile = percentile;
            fastest_index_ = i;
          }
        }
        VLOG(1) << "Selecting index " << fastest_index_
                << " as the fastest index.";
      }

      Status MakeCurrentIterator(IteratorContext* ctx, int64 branch_index,
                                 bool is_experiment)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DCHECK_GE(branch_index, 0);
        DCHECK_LT(branch_index, histograms_.size());

        // `StoreDatasetInVariantTensor` transfers ownership of the dataset
        // to the tensor, so the tensor must persist between iterations.
        wrapper_dataset_tensor_ =
            absl::make_unique<Tensor>(DT_VARIANT, TensorShape({}));

        DatasetContext::Params params;
        params.type_string = "ChooseFastestBranch_Wrapper";
        params.node_name = strings::StrCat(params.type_string, branch_index);
        DatasetBase* temp_dataset =
            new WrapperDataset(std::move(params), &dataset()->output_types_,
                               &dataset()->output_shapes_, input_impl_.get());

        if (is_experiment) {
          // When running experiment iterations, we add a TakeDataset in between
          // the input and the function datasets. This is so that function
          // datasets with prefetching behavior won't consume more input
          // elements than they actually use to produce output.
          DatasetContext::Params take_dataset_params;
          take_dataset_params.type_string = "ChooseFastestBranch_Take";
          take_dataset_params.node_name =
              strings::StrCat(take_dataset_params.type_string, branch_index);
          int64 count = dataset()->num_elements_per_branch_ *
                        dataset()->ratio_numerator_ /
                        dataset()->ratio_denominator_;
          temp_dataset = new TakeDataset(std::move(take_dataset_params), count,
                                         temp_dataset);
        }

        TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(
            temp_dataset, wrapper_dataset_tensor_.get()));

        TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
            ctx, {*wrapper_dataset_tensor_}, branch_index,
            *instantiated_captured_funcs_[branch_index], prefix(),
            &current_iterator_));

        return Status::OK();
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<std::unique_ptr<InstantiatedCapturedFunction>>
          instantiated_captured_funcs_ GUARDED_BY(mu_);

      // For tracking the time taken for each input's iterations.
      std::vector<histogram::Histogram> histograms_ GUARDED_BY(mu_);
      int64 fastest_index_ = -1;
      std::unique_ptr<Tensor> wrapper_dataset_tensor_;
      std::unique_ptr<IteratorBase> current_iterator_;

      // Keeps track of which (branch, experiment) the next iteration is on.
      int64 branch_index_ GUARDED_BY(mu_) = 0;
      int64 experiment_counter_ GUARDED_BY(mu_) = 0;
    };  // class Iterator

    const DatasetBase* const input_;
    std::vector<NameAttrList> funcs_;
    const std::vector<std::unique_ptr<CapturedFunction>> captured_funcs_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const int64 num_elements_per_branch_;
    const int64 ratio_numerator_;
    const int64 ratio_denominator_;
  };  // class Dataset

  int64 ratio_numerator_;
  int64 ratio_denominator_;
  int64 num_elements_per_branch_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::vector<NameAttrList> funcs_;
  std::vector<int32> other_arguments_lengths_;
};  // class ChooseFastestBranchDatasetOp

// Register the kernel implementation for ChooseFastestBranchDataset.
REGISTER_KERNEL_BUILDER(Name("ChooseFastestBranchDataset").Device(DEVICE_CPU),
                        ChooseFastestBranchDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
