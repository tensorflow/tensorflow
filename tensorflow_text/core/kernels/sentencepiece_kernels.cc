// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "src/sentencepiece_model.pb.h"
#include "src/sentencepiece.pb.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace text {

namespace {

// Our resource object that will hold the SentencePiece processor.
struct SentencepieceResource : public ResourceBase {
  sentencepiece::SentencePieceProcessor processor;
  int64 memory_used;
  bool add_bos = false;
  bool add_eos = false;
  bool reverse = false;
  mutable absl::Mutex mu;

  string DebugString() const override { return "Sentencepiece Resource"; }

  int64 MemoryUsed() const override { return memory_used; }

  bool SameOptions(bool add_bos, bool add_eos, bool reverse) const {
    return (add_bos == this->add_bos) && (add_eos == this->add_eos) &&
           (reverse == this->reverse);
  }

  Status AsGraphDef(GraphDefBuilder* builder, Node** out) const override {
    absl::ReaderMutexLock l(&mu);
    // We set use_node_name_sharing with a unique node name so that the resource
    // can outlive the kernel. This means that the lifetime of the re-created
    // resource will be tied to the lifetime of the resource manager it is
    // created in.
    static std::atomic<int64> counter(0);
    std::string unique_node_name = strings::StrCat(
        "SentencepieceResourceFromGraphDef", "/", counter.fetch_add(1));
    std::string model = processor.model_proto().SerializeAsString();
    *out = ops::SourceOp(
        "SentencepieceOp",
        builder->opts()
            .WithName(unique_node_name)
            .WithAttr("model", model)
            .WithAttr("use_node_name_sharing", true));
    return absl::OkStatus();
  }
};

// According to .../tensorflow/core/util/work_sharder.cc, this values determines
// how much to shard. It assumes each cost unit is 1ns, and the minimum cost
// per shard is 10000 (10us).
// TODO(broken) Determine a medium cost of a call to the SentencePiece processor
constexpr int64 kCostPerUnit = 10000;

::tensorflow::Status ToTFStatus(const sentencepiece::util::Status& s) {
  if (s.ok()) return ::tensorflow::Status();
  return ::tensorflow::Status(static_cast<::absl::StatusCode>(s.code()),
                              ::tensorflow::string(s.message()));
}

template <typename T>
T GetPieceOrId(const sentencepiece::SentencePieceText::SentencePiece& sp);

template <>
tensorflow::tstring GetPieceOrId<tensorflow::tstring>(
    const sentencepiece::SentencePieceText::SentencePiece& sp) {
  return sp.piece();
}

template <>
int32 GetPieceOrId<int32>(
    const sentencepiece::SentencePieceText::SentencePiece& sp) {
  return sp.id();
}

tensorflow::Status HandleExtraOptions(OpKernelContext* ctx,
                                      SentencepieceResource* sp) {
  const Tensor* add_bos_tensor = nullptr;
  TF_RETURN_IF_ERROR(ctx->input("add_bos", &add_bos_tensor));
  const bool add_bos = add_bos_tensor->scalar<bool>()();

  const Tensor* add_eos_tensor = nullptr;
  TF_RETURN_IF_ERROR(ctx->input("add_eos", &add_eos_tensor));
  const bool add_eos = add_eos_tensor->scalar<bool>()();

  const Tensor* reverse_tensor = nullptr;
  TF_RETURN_IF_ERROR(ctx->input("reverse", &reverse_tensor));
  const bool reverse = reverse_tensor->scalar<bool>()();

  {
    // Because we expect most of the time no change in these options, we grab
    // the reader lock once and do a quick check first.
    absl::ReaderMutexLock l(&sp->mu);
    if (sp->SameOptions(add_bos, add_eos, reverse)) {
      return absl::OkStatus();
    }
  }

  absl::WriterMutexLock lock(&sp->mu);
  if (sp->SameOptions(add_bos, add_eos, reverse)) {
    return absl::OkStatus();
  }
  string options;
  sp->add_bos = add_bos;
  if (sp->add_bos) {
    absl::StrAppend(&options, "bos");
  }
  sp->add_eos = add_eos;
  if (sp->add_eos) {
    if (!options.empty()) {
      absl::StrAppend(&options, ":");
    }
    absl::StrAppend(&options, "eos");
  }
  sp->reverse = reverse;
  if (sp->reverse) {
    if (!options.empty()) {
      absl::StrAppend(&options, ":");
    }
    absl::StrAppend(&options, "reverse");
  }

  TF_RETURN_IF_ERROR(ToTFStatus(sp->processor.SetEncodeExtraOptions(options)));
  TF_RETURN_IF_ERROR(ToTFStatus(sp->processor.SetDecodeExtraOptions(options)));

  return absl::OkStatus();
}

}  // namespace

class SentencepieceOp : public OpKernel {
 public:
  explicit SentencepieceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), sp_set_(false) {
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                           tensorflow::TensorShape({2}), &sp_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  ~SentencepieceOp() override {
    // If the table object was not shared, delete it.
    if (sp_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<SentencepieceResource>(cinfo_.container(),
                                                        cinfo_.name())
               .ok()) {
        // Do nothing; the resource may have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock lock(&mu_);

    if (!sp_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](SentencepieceResource** resource)
            ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              SentencepieceResource* sp = new SentencepieceResource();

              string model_proto_attr;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(this->def(), "model", &model_proto_attr));

              if (TF_PREDICT_FALSE(model_proto_attr.empty())) {
                return Status(tensorflow::errors::InvalidArgument(
                    "Model argument must be specified."));
              }
              // Loads serialized sentencepiece model proto to enable embedding
              // the relatively small sentencepiece model proto into the
              // tensorflow graph such that the tensorflow graph is
              // self-contained.
              TF_RETURN_IF_ERROR(ToTFStatus(
                  sp->processor.LoadFromSerializedProto(model_proto_attr)));
              // TODO(broken): Determine a better computation of what the memory
              // requirements for the processor are.
              sp->memory_used = model_proto_attr.size();

              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(sp->MemoryUsed());
              }

              *resource = sp;
              return absl::OkStatus();
            };

    // Register the ResourceType alias.
    SentencepieceResource* resource = nullptr;
    OP_REQUIRES_OK(
        ctx, cinfo_.resource_manager()
                 ->template LookupOrCreate<SentencepieceResource>(
                     cinfo_.container(), cinfo_.name(), &resource, creator));
    core::ScopedUnref unref_me(resource);

    // Put a handle to resource in the output tensor (the other aliases will
    // have the same handle).
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() =
        MakeResourceHandle<SentencepieceResource>(ctx, cinfo_.container(),
                                                  cinfo_.name());
    sp_set_ = true;
  }

 private:
  absl::Mutex mu_;
  Tensor sp_ ABSL_GUARDED_BY(mu_);
  bool sp_set_ ABSL_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;
  TF_DISALLOW_COPY_AND_ASSIGN(SentencepieceOp);
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceOp").Device(DEVICE_CPU),
                        tensorflow::text::SentencepieceOp);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceOp");

template <typename T, typename Tsplits>
class SentencepieceTokenizeOp : public OpKernel {
 public:
  explicit SentencepieceTokenizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    ctx->GetAttr("return_nbest", &return_nbest_).IgnoreError();
  }

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    const Tensor& input_values_tensor = ctx->input(1);
    const auto input_values_flat =
        input_values_tensor.flat<tensorflow::tstring>();
    const int64 num_of_input_values = input_values_flat.size();

    const Tensor* nbest_size_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("nbest_size", &nbest_size_tensor));
    const Tensor* alpha_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));

    OP_REQUIRES_OK(ctx, HandleExtraOptions(ctx, sp));

    if (return_nbest_) {
      OP_REQUIRES(ctx, nbest_size_tensor->dims() == 0,
                  errors::InvalidArgument(
                      "When return_nbest is true nbest_size must "
                      "be a scalar; got",
                      nbest_size_tensor->shape().DebugString(), "instead"));
      OP_REQUIRES(ctx, nbest_size_tensor->scalar<int32>()() >= 1,
                  errors::InvalidArgument(
                      "When return_nbest is true nbest_size must be >= 1; got ",
                      nbest_size_tensor->scalar<int32>()()));
    }

    std::vector<std::vector<typename std::conditional<
        std::is_same<T, tstring>::value, std::string, T>::type>>
        tokens(return_nbest_ ? 0 : num_of_input_values);
    std::vector<std::vector<std::vector<typename std::conditional<
        std::is_same<T, tstring>::value, std::string, T>::type>>>
        nbest_tokens(return_nbest_ ? num_of_input_values : 0);
    if (num_of_input_values > 0) {
      const bool return_nbest = return_nbest_;
      const auto& worker_threads =
          *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(
          worker_threads.num_threads,  // max parallelism
          worker_threads.workers,      // thread pool
          num_of_input_values,         // total number of data to process.
          kCostPerUnit,                // cost per unit
          [ctx, sp, &input_values_flat, &tokens, &nbest_tokens,
          &nbest_size_tensor, &alpha_tensor,
          return_nbest](int64 start, int64 limit) {
            absl::ReaderMutexLock lock(&sp->mu);
            for (int i = start; i < limit; ++i) {
              const int32 nbest_size = nbest_size_tensor->dims() == 1
                                         ? nbest_size_tensor->vec<int32>()(i)
                                         : nbest_size_tensor->scalar<int32>()();
              if (return_nbest) {
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.NBestEncode(
                                        input_values_flat(i), nbest_size,
                                        &nbest_tokens[i])));
              } else if (nbest_size == 0 || nbest_size == 1) {
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.Encode(
                                        input_values_flat(i), &tokens[i])));
              } else {
                const float alpha = alpha_tensor->dims() == 1
                                        ? alpha_tensor->vec<float>()(i)
                                        : alpha_tensor->scalar<float>()();
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.SampleEncode(
                                        input_values_flat(i), nbest_size, alpha,
                                        &tokens[i])));
              }
            }
          });
    }

    if (return_nbest_) {
      for (auto& col : nbest_tokens) {
        for (auto& row : col) {
          tokens.push_back(std::move(row));
        }
      }
      nbest_tokens.clear();
    }
    int64 total_tokens = 0;
    for (auto& tokens_row : tokens) {
      total_tokens += tokens_row.size();
    }

    Tensor* output_values_tensor = nullptr;
    Tensor* output_splits_tensor = nullptr;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {total_tokens}, &output_values_tensor));
    int64 splits_size = tokens.size() + 1;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {splits_size}, &output_splits_tensor));

    auto values_tensor_flat = output_values_tensor->vec<T>();
    auto splits_tensor_flat = output_splits_tensor->vec<Tsplits>();

    int i = 0;
    splits_tensor_flat(0) = 0;
    for (int row = 0; row < tokens.size(); ++row) {
      for (int col = 0; col < tokens[row].size(); ++col, ++i) {
        values_tensor_flat(i) = tokens[row][col];
      }
      splits_tensor_flat(row + 1) = i;
    }
  }

  bool return_nbest_{false};
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type")
                            .TypeConstraint<int32>("Tsplits"),
                        SentencepieceTokenizeOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tensorflow::tstring>("out_type")
                            .TypeConstraint<int32>("Tsplits"),
                        SentencepieceTokenizeOp<tensorflow::tstring, int32>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type")
                            .TypeConstraint<int64>("Tsplits"),
                        SentencepieceTokenizeOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tensorflow::tstring>("out_type")
                            .TypeConstraint<int64>("Tsplits"),
                        SentencepieceTokenizeOp<tensorflow::tstring, int64>);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceTokenizeOp");

template <typename T, typename Tsplits>
class SentencepieceTokenizeWithOffsetsOp : public OpKernel {
 public:
  explicit SentencepieceTokenizeWithOffsetsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    ctx->GetAttr("return_nbest", &return_nbest_).IgnoreError();
  }

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    const Tensor& input_values_tensor = ctx->input(1);
    const auto input_values_flat =
        input_values_tensor.flat<tensorflow::tstring>();
    const int64 num_of_input_values = input_values_flat.size();

    const Tensor* nbest_size_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("nbest_size", &nbest_size_tensor));
    const Tensor* alpha_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));

    OP_REQUIRES_OK(ctx, HandleExtraOptions(ctx, sp));

    if (return_nbest_) {
      OP_REQUIRES(ctx, nbest_size_tensor->dims() == 0,
                  errors::InvalidArgument(
                      "When return_nbest is true nbest_size must "
                      "be a scalar; got",
                      nbest_size_tensor->shape().DebugString(), "instead"));
      OP_REQUIRES(ctx, nbest_size_tensor->scalar<int32>()() >= 1,
                  errors::InvalidArgument(
                      "When return_nbest is true nbest_size must be >= 1; got ",
                      nbest_size_tensor->scalar<int32>()()));
    }

    std::vector<sentencepiece::SentencePieceText> results(
        return_nbest_ ? 0 : num_of_input_values);
    std::vector<sentencepiece::NBestSentencePieceText> nbest_results(
        return_nbest_ ? num_of_input_values : 0);
    if (num_of_input_values > 0) {
      const bool return_nbest = return_nbest_;
      const auto& worker_threads =
          *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(
          worker_threads.num_threads,  // max parallelism
          worker_threads.workers,      // thread pool
          num_of_input_values,         // total number of data to process.
          kCostPerUnit,
          [ctx, sp, &input_values_flat, &results, &nbest_results,
          &nbest_size_tensor, &alpha_tensor,
          return_nbest](int64 start, int64 limit) {
            absl::ReaderMutexLock lock(&sp->mu);
            for (int i = start; i < limit; ++i) {
              const int32 nbest_size = nbest_size_tensor->dims() == 1
                                         ? nbest_size_tensor->vec<int32>()(i)
                                         : nbest_size_tensor->scalar<int32>()();
              if (return_nbest) {
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.NBestEncode(
                                        input_values_flat(i), nbest_size,
                                        &nbest_results[i])));
              } else if (nbest_size == 0 || nbest_size == 1) {
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.Encode(
                                        input_values_flat(i), &results[i])));
              } else {
                const float alpha = alpha_tensor->dims() == 1
                                        ? alpha_tensor->vec<float>()(i)
                                        : alpha_tensor->scalar<float>()();
                OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.SampleEncode(
                                        input_values_flat(i), nbest_size, alpha,
                                        &results[i])));
              }
            }
          });
    }

    if (return_nbest_) {
      for (auto& nbest : nbest_results) {
        for (auto& result : nbest.nbests()) {
          results.push_back(std::move(result));
        }
      }
    }
    int64 total_tokens = 0;
    for (auto& sp_result : results) {
      total_tokens += sp_result.pieces_size();
    }

    Tensor* output_values_tensor = nullptr;
    Tensor* output_splits_tensor = nullptr;
    Tensor* output_starts_tensor = nullptr;
    Tensor* output_limits_tensor = nullptr;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {total_tokens}, &output_values_tensor));
    int64 splits_size = results.size() + 1;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {splits_size}, &output_splits_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, {total_tokens}, &output_starts_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(3, {total_tokens}, &output_limits_tensor));

    auto values_tensor_flat = output_values_tensor->vec<T>();
    auto splits_tensor_flat = output_splits_tensor->vec<Tsplits>();
    auto starts_tensor_flat = output_starts_tensor->vec<int64>();
    auto limits_tensor_flat = output_limits_tensor->vec<int64>();

    int i = 0;
    splits_tensor_flat(0) = 0;
    for (int row = 0; row < results.size(); ++row) {
      for (auto& sp : results[row].pieces()) {
        values_tensor_flat(i) = GetPieceOrId<T>(sp);
        starts_tensor_flat(i) = sp.begin();
        limits_tensor_flat(i) = sp.end();
        ++i;
      }
      splits_tensor_flat(row + 1) = i;
    }
  }

  bool return_nbest_{false};
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeWithOffsetsOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type")
                            .TypeConstraint<int32>("Tsplits"),
                        SentencepieceTokenizeWithOffsetsOp<int32, int32>);
REGISTER_KERNEL_BUILDER(
    Name("SentencepieceTokenizeWithOffsetsOp")
        .Device(DEVICE_CPU)
        .TypeConstraint<tensorflow::tstring>("out_type")
        .TypeConstraint<int32>("Tsplits"),
    SentencepieceTokenizeWithOffsetsOp<tensorflow::tstring, int32>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceTokenizeWithOffsetsOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type")
                            .TypeConstraint<int64>("Tsplits"),
                        SentencepieceTokenizeWithOffsetsOp<int32, int64>);
REGISTER_KERNEL_BUILDER(
    Name("SentencepieceTokenizeWithOffsetsOp")
        .Device(DEVICE_CPU)
        .TypeConstraint<tensorflow::tstring>("out_type")
        .TypeConstraint<int64>("Tsplits"),
    SentencepieceTokenizeWithOffsetsOp<tensorflow::tstring, int64>);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceTokenizeWithOffsetsOp");

template <typename T, typename Tsplits>
class SentencepieceDetokenizeOp : public OpKernel {
 public:
  explicit SentencepieceDetokenizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    const Tensor& input_values_tensor = ctx->input(1);
    const auto input_values_flat = input_values_tensor.flat<T>();
    const Tensor& input_splits_tensor = ctx->input(2);
    const auto input_splits_flat = input_splits_tensor.flat<Tsplits>();
    const int64 num_of_sentences = input_splits_flat.size() - 1;

    OP_REQUIRES_OK(ctx, HandleExtraOptions(ctx, sp));

    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {num_of_sentences}, &output_tensor));
    auto output_flat = output_tensor->flat<tensorflow::tstring>();

    if (input_values_flat.size() > 0) {
      const auto& worker_threads =
          *(ctx->device()->tensorflow_cpu_worker_threads());
      ::tensorflow::Shard(
          worker_threads.num_threads,  // max parallelism
          worker_threads.workers,      // thread pool
          num_of_sentences,            // total number of data to process.
          kCostPerUnit,
          [ctx, sp, &input_values_flat, &input_splits_flat, &output_flat](
              int64 start, int64 limit) {
            absl::ReaderMutexLock lock(&sp->mu);
            for (int i = start; i < limit; ++i) {
              if (i + 1 >= input_splits_flat.size()) {
                ctx->CtxFailure(errors::OutOfRange("Invalid splits; ", i));
                return;
              }
              if (input_splits_flat(i) > input_values_flat.size()) {
                ctx->CtxFailure(errors::OutOfRange(
                    "Splits and values do not match; split ",
                    input_splits_flat(i), "but values size is ",
                    input_values_flat.size()));
                return;
              }
              const std::vector<typename std::conditional<
                  std::is_same<T, tstring>::value, std::string, T>::type>
                  pieces(&input_values_flat(input_splits_flat(i)),
                        &input_values_flat(input_splits_flat(i + 1)));
              std::string output_flat_str;
              OP_REQUIRES_OK(ctx, ToTFStatus(sp->processor.Decode(
                                      pieces, &output_flat_str)));
              output_flat(i) = output_flat_str;
            }
          });
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceDetokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tsplits"),
                        SentencepieceDetokenizeOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceDetokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tensorflow::tstring>("T")
                            .TypeConstraint<int32>("Tsplits"),
                        SentencepieceDetokenizeOp<tensorflow::tstring, int32>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceDetokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("Tsplits"),
                        SentencepieceDetokenizeOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("SentencepieceDetokenizeOp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tensorflow::tstring>("T")
                            .TypeConstraint<int64>("Tsplits"),
                        SentencepieceDetokenizeOp<tensorflow::tstring, int64>);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceDetokenizeOp");

class SentencepieceVocabSizeOp : public OpKernel {
 public:
  explicit SentencepieceVocabSizeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output_tensor));
    output_tensor->scalar<int32>()() = sp->processor.GetPieceSize();
  }
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceVocabSizeOp").Device(DEVICE_CPU),
                        SentencepieceVocabSizeOp);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceVocabSizeOp");

class SentencepieceIdToStringOp : public OpKernel {
 public:
  explicit SentencepieceIdToStringOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    const Tensor& input_tensor = ctx->input(1);
    const auto input_tensor_flat = input_tensor.flat<int32>();
    Tensor* output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_tensor_flat = output_tensor->flat<tensorflow::tstring>();

    absl::ReaderMutexLock lock(&sp->mu);
    for (int i = 0; i < input_tensor_flat.size(); ++i) {
      output_tensor_flat(i) = sp->processor.IdToPiece(input_tensor_flat(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceIdToStringOp").Device(DEVICE_CPU),
                        SentencepieceIdToStringOp);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceIdToStringOp");

class SentencepieceStringToIdOp : public OpKernel {
 public:
  explicit SentencepieceStringToIdOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SentencepieceResource* sp;
    const Tensor& resource_tensor = ctx->input(0);
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup<SentencepieceResource>(
                 resource_handle.container(), resource_handle.name(), &sp));
    core::ScopedUnref unref_me(sp);

    const Tensor& input_tensor = ctx->input(1);
    const auto input_tensor_flat = input_tensor.flat<tensorflow::tstring>();
    Tensor* output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_tensor_flat = output_tensor->flat<int32>();

    absl::ReaderMutexLock lock(&sp->mu);
    for (int i = 0; i < input_tensor_flat.size(); ++i) {
      output_tensor_flat(i) = sp->processor.PieceToId(input_tensor_flat(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SentencepieceStringToIdOp").Device(DEVICE_CPU),
                        SentencepieceStringToIdOp);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("SentencepieceStringToIdOp");

}  // namespace text
}  // namespace tensorflow
