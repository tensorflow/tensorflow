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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/window_dataset.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
class GroupByWindowDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByWindowDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        lib_def_(std::make_shared<FunctionLibraryDefinition>(
            ctx->function_library()
                ->GetFunctionLibraryDefinition()
                ->default_registry(),
            FunctionDefLibrary{})) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_func", &key_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_func", &reduce_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size_func", &window_size_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));

    for (const auto& func : {key_func_, reduce_func_, window_size_func_}) {
      std::shared_ptr<FunctionLibraryDefinition> result;
      OP_REQUIRES_OK(
          ctx, CreateFunctionLibraryDefinition(
                   ctx->function_library()->GetFunctionLibraryDefinition(),
                   func.name(), &result));
      OP_REQUIRES_OK(ctx, lib_def_->AddLibrary(*result));
    }
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    CapturedFunction::Params params;
    params.lib_def = lib_def_;

    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(key_func_, ctx,
                                                 "key_func_other_arguments",
                                                 params, &captured_key_func));
    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            reduce_func_, ctx, "reduce_func_other_arguments",
                            params, &captured_reduce_func));
    std::unique_ptr<CapturedFunction> captured_window_size_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(window_size_func_, ctx,
                                      "window_size_func_other_arguments",
                                      params, &captured_window_size_func));

    *output = new Dataset(
        ctx, input, key_func_, reduce_func_, window_size_func_,
        std::move(captured_key_func), std::move(captured_reduce_func),
        std::move(captured_window_size_func), output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& key_func, const NameAttrList& reduce_func,
            const NameAttrList& window_size_func,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_window_size_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          key_func_(key_func),
          reduce_func_(reduce_func),
          window_size_func_(window_size_func),
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
      return absl::make_unique<Iterator>(
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

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
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
      b->BuildAttrValue(key_func_, &key_func);
      AttrValue reduce_func;
      b->BuildAttrValue(reduce_func_, &reduce_func);
      AttrValue window_size_func;
      b->BuildAttrValue(window_size_func_, &window_size_func);

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
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Instantiate(
            ctx, &instantiated_key_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Instantiate(
            ctx, &instantiated_reduce_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_window_size_func_->Instantiate(
            ctx, &instantiated_window_size_func_));
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          if (current_group_iterator_) {
            // We are currently processing a group, so try to get the
            // next element.
            bool end_of_group;
            TF_RETURN_IF_ERROR(current_group_iterator_->GetNext(
                ctx, out_tensors, &end_of_group));
            if (!end_of_group) {
              // Produce the subelement as output.
              *end_of_sequence = false;
              return Status::OK();
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
                input_impl_->GetNext(ctx, &next_input_element, &end_of_input_));

            if (!end_of_input_) {
              // Run the key function on the input element to identify its
              // group.
              std::vector<Tensor> key_func_output;
              TF_RETURN_IF_ERROR(instantiated_key_func_->RunWithBorrowedArgs(
                  ctx, next_input_element, &key_func_output));

              if (key_func_output.size() != 1 ||
                  key_func_output[0].dtype() != DT_INT64 ||
                  key_func_output[0].NumElements() != 1) {
                // TODO(b/78665031): Support non-int64 keys.
                return errors::InvalidArgument(
                    "`key_func` must return a scalar int64.");
              }
              const int64 key = key_func_output[0].scalar<int64>()();

              if (window_sizes_.find(key) == window_sizes_.end()) {
                // Run the window size function on the key to identify its
                // window size.
                std::vector<Tensor> window_size_func_output;
                TF_RETURN_IF_ERROR(instantiated_window_size_func_->Run(
                    ctx, std::move(key_func_output), &window_size_func_output));

                if (window_size_func_output.size() != 1 ||
                    window_size_func_output[0].dtype() != DT_INT64 ||
                    window_size_func_output[0].NumElements() != 1) {
                  // TODO(mrry): Support non-int64 window sizes.
                  return errors::InvalidArgument(
                      "`window_size_func` must return a scalar int64.");
                }
                const int64 window_size =
                    window_size_func_output[0].scalar<int64>()();
                if (window_size <= 0) {
                  return errors::InvalidArgument(
                      "Window size must be greater than zero, but got ",
                      window_size, ".");
                }
                window_sizes_[key] = window_size;
              }

              const int64 window_size = window_sizes_[key];

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
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeUnknownRatioNode(std::move(args));
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));

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
            int64 key = it->first;
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
          TF_RETURN_IF_ERROR(SaveInput(writer, current_group_iterator_));

          // Saving current_key_
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("current_key"), current_key_));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name("current_iterator_not_initialized"), ""));
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

        // Restoring groups
        if (reader->Contains(full_name("groups_size"))) {
          int64 size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("groups_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64 key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("groups_[", idx, "]->key")), &key));
            std::vector<std::vector<Tensor>> group;
            TF_RETURN_IF_ERROR(RestoreGroup(
                reader, full_name(strings::StrCat("groups_[", idx, "]")),
                &group));
            groups_[key] = group;
          }
        }

        // Restoring Windows
        if (reader->Contains(full_name("window_sizes_size"))) {
          int64 size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("window_sizes_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64 key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->key")),
                &key));
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->value")),
                &window_sizes_[key]));
          }
        }

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
        return Status::OK();
      }

     private:
      Status SaveGroup(IteratorStateWriter* writer, const string& name,
                       const std::vector<std::vector<Tensor>>& group)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
        return Status::OK();
      }

      Status RestoreGroup(IteratorStateReader* reader, const string& name,
                          std::vector<std::vector<Tensor>>* group)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        int64 group_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(strings::StrCat(name, "_size"), &group_size));
        group->resize(group_size);
        for (int i = 0; i < group_size; i++) {
          int64 vector_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              strings::StrCat(name, "[", i, "]_size"), &vector_size));
          group->at(i).resize(vector_size);
          for (int j = 0; j < vector_size; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                strings::StrCat(name, "[", i, "][", j, "]"), &group->at(i)[j]));
          }
        }
        return Status::OK();
      }

      Status StartFlushingGroup(IteratorContext* ctx, int64 key)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DatasetBase* group_dataset;
        TF_RETURN_IF_ERROR(NewWindowDataset(
            groups_[key], dataset()->input_->output_dtypes(),
            dataset()->input_->output_shapes(), &group_dataset));

        Tensor key_arg(DT_INT64, TensorShape({}));
        key_arg.scalar<int64>()() = key;

        Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

        std::vector<Tensor> args(
            {std::move(key_arg), std::move(group_dataset_arg)});
        std::vector<Tensor> return_values;
        TF_RETURN_IF_ERROR(instantiated_reduce_func_->Run(ctx, std::move(args),
                                                          &return_values));

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
        return returned_dataset->MakeIterator(ctx, prefix(),
                                              &current_group_iterator_);
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      // TODO(mrry): Optimize for dense key space if appropriate.
      bool end_of_input_ GUARDED_BY(mu_) = false;
      int64 current_key_ GUARDED_BY(mu_);
      std::map<int64, std::vector<std::vector<Tensor>>> groups_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_group_iterator_ GUARDED_BY(mu_);
      std::map<int64, int64> window_sizes_ GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_key_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_reduce_func_;
      std::unique_ptr<InstantiatedCapturedFunction>
          instantiated_window_size_func_;
    };

    const DatasetBase* const input_;
    const NameAttrList key_func_;
    const NameAttrList reduce_func_;
    const NameAttrList window_size_func_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_window_size_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList key_func_;
  NameAttrList reduce_func_;
  NameAttrList window_size_func_;
  std::shared_ptr<FunctionLibraryDefinition> lib_def_;
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalGroupByWindowDataset").Device(DEVICE_CPU),
    GroupByWindowDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
