/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <map>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/window_dataset.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class GroupByWindowDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByWindowDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, "key_func", /*params=*/{},
                                                 &key_func_metadata_));
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "reduce_func", /*params=*/{},
                                            &reduce_func_metadata_));
    OP_REQUIRES_OK(
        ctx, FunctionMetadata::Create(ctx, "window_size_func", /*params=*/{},
                                      &window_size_func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, key_func_metadata_,
                                                 "key_func_other_arguments",
                                                 &captured_key_func));

    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, reduce_func_metadata_,
                                                 "reduce_func_other_arguments",
                                                 &captured_reduce_func));

    std::unique_ptr<CapturedFunction> captured_window_size_func;
    OP_REQUIRES_OK(ctx,
                   CapturedFunction::Create(ctx, window_size_func_metadata_,
                                            "window_size_func_other_arguments",
                                            &captured_window_size_func));

    *output = new Dataset(ctx, input, std::move(captured_key_func),
                          std::move(captured_reduce_func),
                          std::move(captured_window_size_func), output_types_,
                          output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_window_size_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_key_func_(std::move(captured_key_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
          captured_window_size_func_(std::move(captured_window_size_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::GroupByWindow")});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "GroupByWindowDatasetOp::Dataset";
    }

    int64_t CardinalityInternal(CardinalityOptions options) const override {
      int64_t n = input_->Cardinality(options);
      if (n == kInfiniteCardinality) {
        return n;
      }
      return kUnknownCardinality;
    }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
    }

    absl::Status CheckExternalState() const override {
      TF_RETURN_IF_ERROR(captured_key_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_reduce_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_window_size_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      std::vector<Node*> key_func_other_arguments_node;
      DataTypeVector key_func_other_arguments_types;
      TF_RETURN_IF_ERROR(
          captured_key_func_->AddToGraph(ctx, b, &key_func_other_arguments_node,
                                         &key_func_other_arguments_types));

      std::vector<Node*> reduce_func_other_arguments_node;
      DataTypeVector reduce_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_reduce_func_->AddToGraph(
          ctx, b, &reduce_func_other_arguments_node,
          &reduce_func_other_arguments_types));

      std::vector<Node*> window_size_func_other_arguments_node;
      DataTypeVector window_size_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_window_size_func_->AddToGraph(
          ctx, b, &window_size_func_other_arguments_node,
          &window_size_func_other_arguments_types));

      AttrValue key_func;
      b->BuildAttrValue(captured_key_func_->func(), &key_func);
      AttrValue reduce_func;
      b->BuildAttrValue(captured_reduce_func_->func(), &reduce_func);
      AttrValue window_size_func;
      b->BuildAttrValue(captured_window_size_func_->func(), &window_size_func);

      AttrValue key_func_other_arguments_types_attr;
      b->BuildAttrValue(key_func_other_arguments_types,
                        &key_func_other_arguments_types_attr);
      AttrValue reduce_func_other_arguments_types_attr;
      b->BuildAttrValue(reduce_func_other_arguments_types,
                        &reduce_func_other_arguments_types_attr);
      AttrValue window_size_func_other_arguments_types_attr;
      b->BuildAttrValue(window_size_func_other_arguments_types,
                        &window_size_func_other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}},
          {{1, key_func_other_arguments_node},
           {2, reduce_func_other_arguments_node},
           {3, window_size_func_other_arguments_node}},
          {{"key_func", key_func},
           {"reduce_func", reduce_func},
           {"window_size_func", window_size_func},
           {"Tkey_func_other_arguments", key_func_other_arguments_types_attr},
           {"Treduce_func_other_arguments",
            reduce_func_other_arguments_types_attr},
           {"Twindow_size_func_other_arguments",
            window_size_func_other_arguments_types_attr}},
          output));
      return absl::OkStatus();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      absl::Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Instantiate(
            ctx, &instantiated_key_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Instantiate(
            ctx, &instantiated_reduce_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_window_size_func_->Instantiate(
            ctx, &instantiated_window_size_func_));
        return absl::OkStatus();
      }

      absl::Status GetNextInternal(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          if (current_group_iterator_) {
            // We are currently processing a group, so try to get the
            // next element.
            bool end_of_group;
            TF_RETURN_IF_ERROR(current_group_iterator_->GetNext(
                MakeNestedIteratorContext(ctx), out_tensors, &end_of_group));
            if (!end_of_group) {
              // Produce the subelement as output.
              *end_of_sequence = false;
              return absl::OkStatus();
            }
            // We have reached the end of the current group, so maybe move on
            // to the next group.
            current_group_iterator_.reset();
            groups_.erase(current_key_);
          }

          // Iterate through the input dataset until we get a full
          // group, or reach the end.
          while (!end_of_input_) {
            std::vector<Tensor> next_input_element;
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(MakeNestedIteratorContext(ctx),
                                     &next_input_element, &end_of_input_));

            if (!end_of_input_) {
              // Run the key function on the input element to identify its
              // group.
              std::vector<Tensor> key_func_output;
              TF_RETURN_IF_ERROR(instantiated_key_func_->RunWithBorrowedArgs(
                  ctx, next_input_element, &key_func_output, model_node()));

              if (key_func_output.size() != 1 ||
                  key_func_output[0].dtype() != DT_INT64 ||
                  key_func_output[0].NumElements() != 1) {
                // TODO(b/78665031): Support non-int64 keys.
                return errors::InvalidArgument(
                    "`key_func` must return a scalar int64.");
              }
              const int64_t key = key_func_output[0].scalar<int64_t>()();

              if (window_sizes_.find(key) == window_sizes_.end()) {
                // Run the window size function on the key to identify its
                // window size.
                std::vector<Tensor> window_size_func_output;
                TF_RETURN_IF_ERROR(instantiated_window_size_func_->Run(
                    ctx, std::move(key_func_output), &window_size_func_output,
                    model_node()));

                if (window_size_func_output.size() != 1 ||
                    window_size_func_output[0].dtype() != DT_INT64 ||
                    window_size_func_output[0].NumElements() != 1) {
                  // TODO(mrry): Support non-int64 window sizes.
                  return errors::InvalidArgument(
                      "`window_size_func` must return a scalar int64.");
                }
                const int64_t window_size =
                    window_size_func_output[0].scalar<int64_t>()();
                if (window_size <= 0) {
                  return errors::InvalidArgument(
                      "Window size must be greater than zero, but got ",
                      window_size, ".");
                }
                window_sizes_[key] = window_size;
              }

              const int64_t window_size = window_sizes_[key];

              std::vector<std::vector<Tensor>>& group = groups_[key];
              group.push_back(std::move(next_input_element));

              if (group.size() == window_size) {
                current_key_ = key;
                TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, key));
                break;
              }
            }
          }

          if (end_of_input_) {
            if (!groups_.empty()) {
              // We have consumed all of the input, so flush an
              // arbitrarily chosen group.
              current_key_ = groups_.begin()->first;
              TF_RETURN_IF_ERROR(
                  StartFlushingGroup(ctx, groups_.begin()->first));
            }
          }
        } while (current_group_iterator_ || !end_of_input_);

        *end_of_sequence = true;
        return absl::OkStatus();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeUnknownRatioNode(std::move(args));
      }

      absl::Status SaveInternal(SerializationContext* ctx,
                                IteratorStateWriter* writer) override {
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_key_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_reduce_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_window_size_func_->CheckExternalState()));
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));

        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }

        // Saving groups_
        if (!groups_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("groups_size"), groups_.size()));
          int idx = 0;
          for (auto it = groups_.begin(); it != groups_.end(); it++) {
            int64_t key = it->first;
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("groups_[", idx, "]->key")), key));
            TF_RETURN_IF_ERROR(SaveGroup(
                writer, full_name(strings::StrCat("groups_[", idx, "]")),
                it->second));
            idx++;
          }
        }

        // Saving window_sizes_
        if (!window_sizes_.empty()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("window_sizes_size"),
                                                 window_sizes_.size()));
          int idx = 0;
          for (auto it = window_sizes_.begin(); it != window_sizes_.end();
               it++) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->key")),
                it->first));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->value")),
                it->second));
            idx++;
          }
        }

        if (current_group_iterator_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, current_group_iterator_));

          // Saving current_key_
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("current_key"), current_key_));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name("current_iterator_not_initialized"), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("group_counter"),
                                               group_counter_ - 1));
        return absl::OkStatus();
      }

      absl::Status RestoreInternal(IteratorContext* ctx,
                                   IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

        // Restoring groups_
        if (reader->Contains(full_name("groups_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("groups_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64_t key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("groups_[", idx, "]->key")), &key));
            std::vector<std::vector<Tensor>> group;
            TF_RETURN_IF_ERROR(RestoreGroup(
                ctx, reader, full_name(strings::StrCat("groups_[", idx, "]")),
                &group));
            groups_[key] = group;
          }
        }

        // Restoring window_sizes_
        if (reader->Contains(full_name("window_sizes_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("window_sizes_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64_t key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->key")),
                &key));
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->value")),
                &window_sizes_[key]));
          }
        }

        // Group counter needs to be restored before current group iterator.
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("group_counter"), &group_counter_));

        if (reader->Contains(full_name("current_iterator_not_initialized"))) {
          current_group_iterator_.reset();
        } else {
          // Restore current_key_
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("current_key"), &current_key_));

          // Initialize current_group_iterator_
          TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, current_key_));
          // Restore current_group_iterator_ state
          TF_RETURN_IF_ERROR(
              RestoreInput(ctx, reader, current_group_iterator_));
        }
        return absl::OkStatus();
      }

     private:
      absl::Status SaveGroup(IteratorStateWriter* writer, const string& name,
                             const std::vector<std::vector<Tensor>>& group)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(strings::StrCat(name, "_size"), group.size()));
        for (int i = 0; i < group.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              strings::StrCat(name, "[", i, "]_size"), group[i].size()));
          for (int j = 0; j < group[i].size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                strings::StrCat(name, "[", i, "][", j, "]"), group[i][j]));
          }
        }
        return absl::OkStatus();
      }

      absl::Status RestoreGroup(IteratorContext* ctx,
                                IteratorStateReader* reader, const string& name,
                                std::vector<std::vector<Tensor>>* group)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        int64_t group_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(strings::StrCat(name, "_size"), &group_size));
        group->resize(group_size);
        for (int i = 0; i < group_size; i++) {
          int64_t vector_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              strings::StrCat(name, "[", i, "]_size"), &vector_size));
          group->at(i).resize(vector_size);
          for (int j = 0; j < vector_size; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), strings::StrCat(name, "[", i, "][", j, "]"),
                &group->at(i)[j]));
          }
        }
        return absl::OkStatus();
      }

      absl::Status StartFlushingGroup(IteratorContext* ctx, int64_t key)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DatasetBase* group_dataset;
        TF_RETURN_IF_ERROR(
            NewWindow(groups_[key], dataset()->input_->output_dtypes(),
                      dataset()->input_->output_shapes(), &group_dataset));

        Tensor key_arg(DT_INT64, TensorShape({}));
        key_arg.scalar<int64_t>()() = key;

        Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

        std::vector<Tensor> args(
            {std::move(key_arg), std::move(group_dataset_arg)});
        std::vector<Tensor> return_values;
        // If not restoring, pass the model node of this iterator in order to
        // exclude captured function run time from being added to the processing
        // time of the node. If restoring, pass nullptr to not record processing
        // time because iterator modeling is only used to model Iterator's
        // GetNext() resource usage.
        auto status = instantiated_reduce_func_->Run(
            ctx, std::move(args), &return_values,
            ctx->is_restoring() ? nullptr : model_node());
        if (!status.ok()) {
          return absl::InternalError(absl::StrFormat(
              "Got error code %s and message: {\n%s\n} \nfrom running "
              "user-defined function %s: ",
              absl::StatusCodeToString(status.code()), status.message(),
              instantiated_reduce_func_->func_name()));
        }

        if (!(return_values.size() == 1 &&
              return_values[0].dtype() == DT_VARIANT &&
              TensorShapeUtils::IsScalar(return_values[0].shape()))) {
          return errors::InvalidArgument(
              "`reduce_func` must return a single scalar of dtype "
              "DT_VARIANT.");
        }

        // Retrieve the dataset that was created in `f`.
        // `returned_dataset` is borrowed from the `return_values[0]`.
        DatasetBase* returned_dataset;
        TF_RETURN_IF_ERROR(
            GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

        // Create an iterator for the dataset that was returned by `f`.
        return returned_dataset->MakeIterator(
            MakeNestedIteratorContext(ctx), this,
            strings::StrCat(prefix(), "[", group_counter_++, "]"),
            &current_group_iterator_);
      }

      mutex mu_;
      int64_t group_counter_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      // TODO(mrry): Optimize for dense key space if appropriate.
      bool end_of_input_ TF_GUARDED_BY(mu_) = false;
      int64_t current_key_ TF_GUARDED_BY(mu_);
      std::map<int64_t, std::vector<std::vector<Tensor>>> groups_
          TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_group_iterator_ TF_GUARDED_BY(mu_);
      std::map<int64_t, int64_t> window_sizes_ TF_GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_key_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_reduce_func_;
      std::unique_ptr<InstantiatedCapturedFunction>
          instantiated_window_size_func_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_window_size_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  std::shared_ptr<FunctionMetadata> key_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> reduce_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> window_size_func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByWindowDataset").Device(DEVICE_CPU),
                        GroupByWindowDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalGroupByWindowDataset").Device(DEVICE_CPU),
    GroupByWindowDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("GroupByWindowDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalGroupByWindowDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
