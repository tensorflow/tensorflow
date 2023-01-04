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
#include "tensorflow/core/summary/summary_file_writer.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_converter.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

class SummaryFileWriter : public SummaryWriterInterface {
 public:
  SummaryFileWriter(int max_queue, int flush_millis, Env* env)
      : SummaryWriterInterface(),
        is_initialized_(false),
        max_queue_(max_queue),
        flush_millis_(flush_millis),
        env_(env) {}

  Status Initialize(const string& logdir, const string& filename_suffix) {
    const Status is_dir = env_->IsDirectory(logdir);
    if (!is_dir.ok()) {
      if (is_dir.code() != tensorflow::error::NOT_FOUND) {
        return is_dir;
      }
      TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(logdir));
    }
    // Embed PID plus a unique counter as the leading portion of the filename
    // suffix to help prevent filename collisions between and within processes.
    int32_t pid = env_->GetProcessId();
    static std::atomic<int64_t> file_id_counter(0);
    // Precede filename_suffix with "." if it doesn't already start with one.
    string sep = absl::StartsWith(filename_suffix, ".") ? "" : ".";
    const string uniquified_filename_suffix = absl::StrCat(
        ".", pid, ".", file_id_counter.fetch_add(1), sep, filename_suffix);
    mutex_lock ml(mu_);
    events_writer_ =
        tensorflow::MakeUnique<EventsWriter>(io::JoinPath(logdir, "events"));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        events_writer_->InitWithSuffix(uniquified_filename_suffix),
        "Could not initialize events writer.");
    last_flush_ = env_->NowMicros();
    is_initialized_ = true;
    return OkStatus();
  }

  Status Flush() override {
    mutex_lock ml(mu_);
    if (!is_initialized_) {
      return errors::FailedPrecondition("Class was not properly initialized.");
    }
    return InternalFlush();
  }

  ~SummaryFileWriter() override {
    (void)Flush();  // Ignore errors.
  }

  Status WriteTensor(int64_t global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    Summary::Value* v = e->mutable_summary()->add_value();

    if (t.dtype() == DT_STRING) {
      // Treat DT_STRING specially, so that tensor_util.MakeNdarray in Python
      // can convert the TensorProto to string-type numpy array. MakeNdarray
      // does not work with strings encoded by AsProtoTensorContent() in
      // tensor_content.
      t.AsProtoField(v->mutable_tensor());
    } else {
      t.AsProtoTensorContent(v->mutable_tensor());
    }
    v->set_tag(tag);
    if (!serialized_metadata.empty()) {
      v->mutable_metadata()->ParseFromString(serialized_metadata);
    }
    return WriteEvent(std::move(e));
  }

  Status WriteScalar(int64_t global_step, Tensor t,
                     const string& tag) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(
        AddTensorAsScalarToSummary(t, tag, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteHistogram(int64_t global_step, Tensor t,
                        const string& tag) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(
        AddTensorAsHistogramToSummary(t, tag, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteImage(int64_t global_step, Tensor t, const string& tag,
                    int max_images, Tensor bad_color) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(AddTensorAsImageToSummary(t, tag, max_images, bad_color,
                                                 e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteAudio(int64_t global_step, Tensor t, const string& tag,
                    int max_outputs, float sample_rate) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(AddTensorAsAudioToSummary(
        t, tag, max_outputs, sample_rate, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteGraph(int64_t global_step,
                    std::unique_ptr<GraphDef> graph) override {
    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    graph->SerializeToString(e->mutable_graph_def());
    return WriteEvent(std::move(e));
  }

  Status WriteEvent(std::unique_ptr<Event> event) override {
    mutex_lock ml(mu_);
    queue_.emplace_back(std::move(event));
    if (queue_.size() > max_queue_ ||
        env_->NowMicros() - last_flush_ > 1000 * flush_millis_) {
      return InternalFlush();
    }
    return OkStatus();
  }

  string DebugString() const override { return "SummaryFileWriter"; }

 private:
  double GetWallTime() {
    return static_cast<double>(env_->NowMicros()) / 1.0e6;
  }

  Status InternalFlush() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    for (const std::unique_ptr<Event>& e : queue_) {
      events_writer_->WriteEvent(*e);
    }
    queue_.clear();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(events_writer_->Flush(),
                                    "Could not flush events file.");
    last_flush_ = env_->NowMicros();
    return OkStatus();
  }

  bool is_initialized_;
  const int max_queue_;
  const int flush_millis_;
  uint64 last_flush_;
  Env* env_;
  mutex mu_;
  std::vector<std::unique_ptr<Event>> queue_ TF_GUARDED_BY(mu_);
  // A pointer to allow deferred construction.
  std::unique_ptr<EventsWriter> events_writer_ TF_GUARDED_BY(mu_);
  std::vector<std::pair<string, SummaryMetadata>> registered_summaries_
      TF_GUARDED_BY(mu_);
};

}  // namespace

Status CreateSummaryFileWriter(int max_queue, int flush_millis,
                               const string& logdir,
                               const string& filename_suffix, Env* env,
                               SummaryWriterInterface** result) {
  SummaryFileWriter* w = new SummaryFileWriter(max_queue, flush_millis, env);
  const Status s = w->Initialize(logdir, filename_suffix);
  if (!s.ok()) {
    w->Unref();
    *result = nullptr;
    return s;
  }
  *result = w;
  return OkStatus();
}

}  // namespace tensorflow
