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

#include "tensorflow/contrib/tensorboard/db/summary_db_writer.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("SummaryWriter").Device(DEVICE_CPU),
                        ResourceHandleOp<SummaryWriterInterface>);

class CreateSummaryFileWriterOp : public OpKernel {
 public:
  explicit CreateSummaryFileWriterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("logdir", &tmp));
    const string logdir = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("max_queue", &tmp));
    const int32 max_queue = tmp->scalar<int32>()();
    OP_REQUIRES_OK(ctx, ctx->input("flush_millis", &tmp));
    const int32 flush_millis = tmp->scalar<int32>()();
    OP_REQUIRES_OK(ctx, ctx->input("filename_suffix", &tmp));
    const string filename_suffix = tmp->scalar<string>()();
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, CreateSummaryWriter(max_queue, flush_millis, logdir,
                                            filename_suffix, ctx->env(), &s));
    OP_REQUIRES_OK(ctx, CreateResource(ctx, HandleFromInput(ctx, 0), s));
  }
};
REGISTER_KERNEL_BUILDER(Name("CreateSummaryFileWriter").Device(DEVICE_CPU),
                        CreateSummaryFileWriterOp);

class CreateSummaryDbWriterOp : public OpKernel {
 public:
  explicit CreateSummaryDbWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("db_uri", &tmp));
    const string db_uri = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("experiment_name", &tmp));
    const string experiment_name = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("run_name", &tmp));
    const string run_name = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("user_name", &tmp));
    const string user_name = tmp->scalar<string>()();
    SummaryWriterInterface* s;
    auto db = Sqlite::Open(db_uri);
    OP_REQUIRES_OK(ctx, db.status());
    db.ValueOrDie()->UseWriteAheadLogWithReducedDurabilityIfPossible();
    OP_REQUIRES_OK(
        ctx, CreateSummaryDbWriter(std::move(db.ValueOrDie()), experiment_name,
                                   run_name, user_name, ctx->env(), &s));
    OP_REQUIRES_OK(ctx, CreateResource(ctx, HandleFromInput(ctx, 0), s));
  }
};
REGISTER_KERNEL_BUILDER(Name("CreateSummaryDbWriter").Device(DEVICE_CPU),
                        CreateSummaryDbWriterOp);

class FlushSummaryWriterOp : public OpKernel {
 public:
  explicit FlushSummaryWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    OP_REQUIRES_OK(ctx, s->Flush());
  }
};
REGISTER_KERNEL_BUILDER(Name("FlushSummaryWriter").Device(DEVICE_CPU),
                        FlushSummaryWriterOp);

class CloseSummaryWriterOp : public OpKernel {
 public:
  explicit CloseSummaryWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, DeleteResource<SummaryWriterInterface>(
                            ctx, HandleFromInput(ctx, 0)));
  }
};
REGISTER_KERNEL_BUILDER(Name("CloseSummaryWriter").Device(DEVICE_CPU),
                        CloseSummaryWriterOp);

class WriteSummaryOp : public OpKernel {
 public:
  explicit WriteSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64 step = tmp->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("summary_metadata", &tmp));
    const string& serialized_metadata = tmp->scalar<string>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx, s->WriteTensor(step, *t, tag, serialized_metadata));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteSummary").Device(DEVICE_CPU),
                        WriteSummaryOp);

class ImportEventOp : public OpKernel {
 public:
  explicit ImportEventOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("event", &t));
    std::unique_ptr<Event> event{new Event};
    if (!ParseProtoUnlimited(event.get(), t->scalar<string>()())) {
      ctx->CtxFailureWithWarning(
          errors::DataLoss("Bad tf.Event binary proto tensor string"));
      return;
    }
    OP_REQUIRES_OK(ctx, s->WriteEvent(std::move(event)));
  }
};
REGISTER_KERNEL_BUILDER(Name("ImportEvent").Device(DEVICE_CPU), ImportEventOp);

class WriteScalarSummaryOp : public OpKernel {
 public:
  explicit WriteScalarSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64 step = tmp->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<string>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("value", &t));

    OP_REQUIRES_OK(ctx, s->WriteScalar(step, *t, tag));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteScalarSummary").Device(DEVICE_CPU),
                        WriteScalarSummaryOp);

class WriteHistogramSummaryOp : public OpKernel {
 public:
  explicit WriteHistogramSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64 step = tmp->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<string>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("values", &t));

    OP_REQUIRES_OK(ctx, s->WriteHistogram(step, *t, tag));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteHistogramSummary").Device(DEVICE_CPU),
                        WriteHistogramSummaryOp);

class WriteImageSummaryOp : public OpKernel {
 public:
  explicit WriteImageSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    int64 max_images_tmp;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_images", &max_images_tmp));
    OP_REQUIRES(ctx, max_images_tmp < (1LL << 31),
                errors::InvalidArgument("max_images must be < 2^31"));
    max_images_ = static_cast<int32>(max_images_tmp);
  }

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64 step = tmp->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<string>()();
    const Tensor* bad_color;
    OP_REQUIRES_OK(ctx, ctx->input("bad_color", &bad_color));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(bad_color->shape()),
        errors::InvalidArgument("bad_color must be a vector, got shape ",
                                bad_color->shape().DebugString()));

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx, s->WriteImage(step, *t, tag, max_images_, *bad_color));
  }

 private:
  int32 max_images_;
};
REGISTER_KERNEL_BUILDER(Name("WriteImageSummary").Device(DEVICE_CPU),
                        WriteImageSummaryOp);

class WriteAudioSummaryOp : public OpKernel {
 public:
  explicit WriteAudioSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_outputs", &max_outputs_));
    OP_REQUIRES(ctx, max_outputs_ > 0,
                errors::InvalidArgument("max_outputs must be > 0"));
  }

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64 step = tmp->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<string>()();
    OP_REQUIRES_OK(ctx, ctx->input("sample_rate", &tmp));
    const float sample_rate = tmp->scalar<float>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx,
                   s->WriteAudio(step, *t, tag, max_outputs_, sample_rate));
  }

 private:
  int max_outputs_;
  bool has_sample_rate_attr_;
  float sample_rate_attr_;
};
REGISTER_KERNEL_BUILDER(Name("WriteAudioSummary").Device(DEVICE_CPU),
                        WriteAudioSummaryOp);

class WriteGraphSummaryOp : public OpKernel {
 public:
  explicit WriteGraphSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SummaryWriterInterface* s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    core::ScopedUnref unref(s);
    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("step", &t));
    const int64 step = t->scalar<int64>()();
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));
    std::unique_ptr<GraphDef> graph{new GraphDef};
    if (!ParseProtoUnlimited(graph.get(), t->scalar<string>()())) {
      ctx->CtxFailureWithWarning(
          errors::DataLoss("Bad tf.GraphDef binary proto tensor string"));
      return;
    }
    OP_REQUIRES_OK(ctx, s->WriteGraph(step, std::move(graph)));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteGraphSummary").Device(DEVICE_CPU),
                        WriteGraphSummaryOp);

}  // namespace tensorflow
