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
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

static mutex* get_counters_map_lock() {
  static mutex counters_map_lock(LINKER_INITIALIZED);
  return &counters_map_lock;
}

static std::unordered_map<string, monitoring::Counter<1>*>* get_counters_map() {
  static std::unordered_map<string, monitoring::Counter<1>*>* counters_map =
      new std::unordered_map<string, monitoring::Counter<1>*>;
  return counters_map;
}

class StatsAggregatorImpl : public StatsAggregator {
 public:
  StatsAggregatorImpl() {}

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      const int64_t steps) override {
    mutex_lock l(mu_);
    histogram::Histogram& histogram = histograms_[name];
    for (double value : values) {
      histogram.Add(value);
    }
  }

  void AddScalar(const string& name, float value,
                 const int64_t steps) override {
    mutex_lock l(mu_);
    scalars_[name] = value;
  }

  void EncodeToProto(Summary* out_summary) override {
    mutex_lock l(mu_);
    for (const auto& pair : histograms_) {
      const string& name = pair.first;
      const histogram::Histogram& histogram = pair.second;

      Summary::Value* value = out_summary->add_value();
      value->set_tag(name);
      histogram.EncodeToProto(value->mutable_histo(),
                              false /* doesn't preserve zero buckets */);
    }
    for (const auto& pair : scalars_) {
      Summary::Value* value = out_summary->add_value();
      value->set_tag(pair.first);
      value->set_simple_value(pair.second);
    }
  }

  // StatsAggregator implementation for V2 is based on push-based summary, no-op
  // in V1.
  Status SetSummaryWriter(
      SummaryWriterInterface* summary_writer_interface) override {
    return OkStatus();
  }

  void IncrementCounter(const string& name, const string& label,
                        int64_t val) override {
    mutex_lock l(*get_counters_map_lock());
    auto counters_map = get_counters_map();
    if (counters_map->find(name) == counters_map->end()) {
      counters_map->emplace(
          name,
          monitoring::Counter<1>::New(
              /*streamz name*/ name,
              /*streamz description*/
              strings::StrCat(name, " generated or consumed by the component."),
              /*streamz label name*/ "component_descriptor"));
    }
    counters_map->at(name)->GetCell(label)->IncrementBy(val);
  }

 private:
  mutex mu_;
  std::unordered_map<string, histogram::Histogram> histograms_
      TF_GUARDED_BY(mu_);
  std::unordered_map<string, float> scalars_ TF_GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorImpl);
};

class StatsAggregatorHandleOp
    : public ResourceOpKernel<StatsAggregatorResource> {
 public:
  explicit StatsAggregatorHandleOp(OpKernelConstruction* ctx)
      : ResourceOpKernel<StatsAggregatorResource>(ctx) {}

 private:
  Status CreateResource(StatsAggregatorResource** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret = new StatsAggregatorResource(std::make_unique<StatsAggregatorImpl>());
    return OkStatus();
  }
};

class StatsAggregatorImplV2 : public StatsAggregator {
 public:
  StatsAggregatorImplV2() {}

  ~StatsAggregatorImplV2() override {
    if (summary_writer_interface_) {
      summary_writer_interface_->Unref();
    }
  }

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      const int64_t steps) override {
    mutex_lock l(mu_);
    histogram::Histogram& histogram = histograms_[name];
    for (double value : values) {
      histogram.Add(value);
    }
    AddToEvents(name, steps, histogram);
  }

  void AddScalar(const string& name, float value,
                 const int64_t steps) override {
    mutex_lock l(mu_);
    AddToEvents(name, steps, value);
  }

  // TODO(b/116314787): expose this is public API to manually flush summary.
  Status Flush() {
    mutex_lock l(mu_);
    if (summary_writer_interface_)
      TF_RETURN_IF_ERROR(summary_writer_interface_->Flush());
    return OkStatus();
  }

  void IncrementCounter(const string& name, const string& label,
                        int64_t val) override {
    mutex_lock l(*get_counters_map_lock());
    auto counters_map = get_counters_map();
    if (counters_map->find(name) == counters_map->end()) {
      counters_map->emplace(
          name, monitoring::Counter<1>::New(
                    /*streamz name*/ "/tensorflow/" + name,
                    /*streamz description*/
                    name + " generated or consumed by the component.",
                    /*streamz label name*/ "component_descriptor"));
    }
    counters_map->at(name)->GetCell(label)->IncrementBy(val);
  }

  // StatsAggregator implementation for V1 is based on pull-based summary, no-op
  // in V2.
  void EncodeToProto(Summary* out_summary) override {}

  Status SetSummaryWriter(
      SummaryWriterInterface* summary_writer_interface) override {
    mutex_lock l(mu_);
    if (summary_writer_interface_) {
      summary_writer_interface_->Unref();
      // If we create stats_aggregator twice in a program, we would end up with
      // already existing resource. In this case emitting an error if a
      // `summary_writer_resource` is present is not the intended behavior, we
      // could either Unref the existing summary_writer_resource or not set the
      // new resource at all.
    }
    summary_writer_interface_ = summary_writer_interface;
    summary_writer_interface_->Ref();
    return OkStatus();
  }

 private:
  void AddToEvents(const string& name, const int64_t steps,
                   const float scalar_value) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (summary_writer_interface_ == nullptr) {
      return;
    }
    std::unique_ptr<Event> e{new Event};
    e->set_step(steps);
    e->set_wall_time(EnvTime::NowMicros() / 1.0e6);
    // maybe expose GetWallTime in SummaryWriterInterface
    Summary::Value* v = e->mutable_summary()->add_value();
    v->set_tag(name);
    v->set_simple_value(scalar_value);
    TF_CHECK_OK(summary_writer_interface_->WriteEvent(std::move(e)));
  }

  void AddToEvents(const string& name, const int64_t steps,
                   const histogram::Histogram& histogram)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (summary_writer_interface_ == nullptr) {
      return;
    }
    std::unique_ptr<Event> e{new Event};
    e->set_step(steps);
    e->set_wall_time(EnvTime::NowMicros() / 1.0e6);
    Summary::Value* v = e->mutable_summary()->add_value();
    v->set_tag(name);
    histogram.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);
    TF_CHECK_OK(summary_writer_interface_->WriteEvent(std::move(e)));
  }

  mutex mu_;
  SummaryWriterInterface* summary_writer_interface_ TF_GUARDED_BY(mu_) =
      nullptr;
  // not owned, we might be associating the default summary_writer from the
  // context
  std::unordered_map<string, histogram::Histogram> histograms_
      TF_GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorImplV2);
};

class StatsAggregatorHandleOpV2
    : public ResourceOpKernel<StatsAggregatorResource> {
 public:
  explicit StatsAggregatorHandleOpV2(OpKernelConstruction* ctx)
      : ResourceOpKernel<StatsAggregatorResource>(ctx) {}

 private:
  Status CreateResource(StatsAggregatorResource** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret =
        new StatsAggregatorResource(std::make_unique<StatsAggregatorImplV2>());
    return OkStatus();
  }
};

class StatsAggregatorSummaryOp : public OpKernel {
 public:
  explicit StatsAggregatorSummaryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    Tensor* summary_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &summary_t));
    Summary summary;
    resource->stats_aggregator()->EncodeToProto(&summary);
    summary_t->scalar<tstring>()() = summary.SerializeAsString();
  }
};

class StatsAggregatorSetSummaryWriterOp : public OpKernel {
 public:
  explicit StatsAggregatorSetSummaryWriterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    const Tensor& summary_resource_handle_t = ctx->input(1);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(summary_resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));
    core::RefCountPtr<SummaryWriterInterface> summary_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &summary_resource));
    TF_CHECK_OK(
        resource->stats_aggregator()->SetSummaryWriter(summary_resource.get()));
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorHandle").Device(DEVICE_CPU),
                        StatsAggregatorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalStatsAggregatorHandle").Device(DEVICE_CPU),
    StatsAggregatorHandleOp);

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorHandleV2").Device(DEVICE_CPU),
                        StatsAggregatorHandleOpV2);

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorSummary").Device(DEVICE_CPU),
                        StatsAggregatorSummaryOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalStatsAggregatorSummary").Device(DEVICE_CPU),
    StatsAggregatorSummaryOp);

REGISTER_KERNEL_BUILDER(
    Name("StatsAggregatorSetSummaryWriter").Device(DEVICE_CPU),
    StatsAggregatorSetSummaryWriterOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
