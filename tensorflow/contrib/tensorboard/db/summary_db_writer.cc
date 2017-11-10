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

#include "tensorflow/contrib/tensorboard/db/schema.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace {

int64 MakeRandomId() {
  int64 id = static_cast<int64>(random::New64() & ((1ULL << 63) - 1));
  if (id == 0) {
    ++id;
  }
  return id;
}

class SummaryDbWriter : public SummaryWriterInterface {
 public:
  SummaryDbWriter(Env* env, std::shared_ptr<Sqlite> db)
      : SummaryWriterInterface(), env_(env), db_(std::move(db)), run_id_(-1) {}
  ~SummaryDbWriter() override {}

  Status Initialize(const string& experiment_name, const string& run_name,
                    const string& user_name) {
    mutex_lock ml(mu_);
    insert_tensor_ = db_->Prepare(R"sql(
      INSERT OR REPLACE INTO Tensors (tag_id, step, computed_time, tensor)
      VALUES (?, ?, ?, ?)
    )sql");
    update_metadata_ = db_->Prepare(R"sql(
      UPDATE Tags SET metadata = ? WHERE tag_id = ?
    )sql");
    experiment_name_ = experiment_name;
    run_name_ = run_name;
    user_name_ = user_name;
    return Status::OK();
  }

  // TODO(@jart): Use transactions that COMMIT on Flush()
  // TODO(@jart): Retry Commit() on SQLITE_BUSY with exponential back-off.
  Status Flush() override { return Status::OK(); }

  Status WriteTensor(int64 global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override {
    mutex_lock ml(mu_);
    TF_RETURN_IF_ERROR(InitializeParents());
    // TODO(@jart): Memoize tag_id.
    int64 tag_id;
    TF_RETURN_IF_ERROR(GetTagId(run_id_, tag, &tag_id));
    if (!serialized_metadata.empty()) {
      // TODO(@jart): Only update metadata for first tensor.
      update_metadata_.BindBlobUnsafe(1, serialized_metadata);
      update_metadata_.BindInt(2, tag_id);
      TF_RETURN_IF_ERROR(update_metadata_.StepAndReset());
    }
    // TODO(@jart): Lease blocks of rowids and *_ids to minimize fragmentation.
    // TODO(@jart): Check for random ID collisions without needing txn retry.
    insert_tensor_.BindInt(1, tag_id);
    insert_tensor_.BindInt(2, global_step);
    insert_tensor_.BindDouble(3, GetWallTime());
    switch (t.dtype()) {
      case DT_INT64:
        insert_tensor_.BindInt(4, t.scalar<int64>()());
        break;
      case DT_DOUBLE:
        insert_tensor_.BindDouble(4, t.scalar<double>()());
        break;
      default:
        TF_RETURN_IF_ERROR(BindTensor(t));
        break;
    }
    TF_RETURN_IF_ERROR(insert_tensor_.StepAndReset());
    return Status::OK();
  }

  Status WriteEvent(std::unique_ptr<Event> e) override {
    // TODO(@jart): This will be used to load event logs.
    return errors::Unimplemented("WriteEvent");
  }

  Status WriteScalar(int64 global_step, Tensor t, const string& tag) override {
    // TODO(@jart): Unlike WriteTensor, this method would be granted leniency
    //              to change the dtype if it saves storage space. For example,
    //              DT_UINT32 would be stored in the database as an INTEGER
    //              rather than a serialized BLOB. But when reading it back,
    //              the dtype would become DT_INT64.
    return errors::Unimplemented("WriteScalar");
  }

  Status WriteHistogram(int64 global_step, Tensor t,
                        const string& tag) override {
    return errors::Unimplemented(
        "SummaryDbWriter::WriteHistogram not supported. Please use ",
        "tensorboard.summary.histogram() instead.");
  }

  Status WriteImage(int64 global_step, Tensor tensor, const string& tag,
                    int max_images, Tensor bad_color) override {
    return errors::Unimplemented(
        "SummaryDbWriter::WriteImage not supported. Please use ",
        "tensorboard.summary.image() instead.");
  }

  Status WriteAudio(int64 global_step, Tensor tensor, const string& tag,
                    int max_outputs, float sample_rate) override {
    return errors::Unimplemented(
        "SummaryDbWriter::WriteAudio not supported. Please use ",
        "tensorboard.summary.audio() instead.");
  }

  string DebugString() override { return "SummaryDbWriter"; }

 private:
  double GetWallTime() {
    // TODO(@jart): Follow precise definitions for time laid out in schema.
    // TODO(@jart): Use monotonic clock from gRPC codebase.
    return static_cast<double>(env_->NowMicros()) / 1.0e6;
  }

  Status BindTensor(const Tensor& t) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // TODO(@jart): Make portable between little and big endian systems.
    // TODO(@jart): Use TensorChunks with minimal copying for big tensors.
    TensorProto p;
    t.AsProtoTensorContent(&p);
    string encoded;
    if (!p.SerializeToString(&encoded)) {
      return errors::DataLoss("SerializeToString failed");
    }
    // TODO(@jart): Put byte at beginning of blob to indicate encoding.
    // TODO(@jart): Allow crunch tool to re-compress with zlib instead.
    string compressed;
    if (!port::Snappy_Compress(encoded.data(), encoded.size(), &compressed)) {
      return errors::FailedPrecondition("TensorBase needs Snappy");
    }
    insert_tensor_.BindBlobUnsafe(4, compressed);
    return Status::OK();
  }

  Status InitializeParents() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (run_id_ >= 0) {
      return Status::OK();
    }
    int64 user_id;
    TF_RETURN_IF_ERROR(GetUserId(user_name_, &user_id));
    int64 experiment_id;
    TF_RETURN_IF_ERROR(
        GetExperimentId(user_id, experiment_name_, &experiment_id));
    TF_RETURN_IF_ERROR(GetRunId(experiment_id, run_name_, &run_id_));
    return Status::OK();
  }

  Status GetUserId(const string& user_name, int64* user_id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (user_name.empty()) {
      *user_id = 0LL;
      return Status::OK();
    }
    SqliteStatement get_user_id = db_->Prepare(R"sql(
      SELECT user_id FROM Users WHERE user_name = ?
    )sql");
    get_user_id.BindText(1, user_name);
    bool is_done;
    TF_RETURN_IF_ERROR(get_user_id.Step(&is_done));
    if (!is_done) {
      *user_id = get_user_id.ColumnInt(0);
    } else {
      *user_id = MakeRandomId();
      SqliteStatement insert_user = db_->Prepare(R"sql(
        INSERT INTO Users (user_id, user_name, inserted_time) VALUES (?, ?, ?)
      )sql");
      insert_user.BindInt(1, *user_id);
      insert_user.BindText(2, user_name);
      insert_user.BindDouble(3, GetWallTime());
      TF_RETURN_IF_ERROR(insert_user.StepAndReset());
    }
    return Status::OK();
  }

  Status GetExperimentId(int64 user_id, const string& experiment_name,
                         int64* experiment_id) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // TODO(@jart): Compute started_time.
    return GetId("Experiments", "user_id", user_id, "experiment_name",
                 experiment_name, "experiment_id", experiment_id);
  }

  Status GetRunId(int64 experiment_id, const string& run_name, int64* run_id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // TODO(@jart): Compute started_time.
    return GetId("Runs", "experiment_id", experiment_id, "run_name", run_name,
                 "run_id", run_id);
  }

  Status GetTagId(int64 run_id, const string& tag_name, int64* tag_id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return GetId("Tags", "run_id", run_id, "tag_name", tag_name, "tag_id",
                 tag_id);
  }

  Status GetId(const char* table, const char* parent_id_field, int64 parent_id,
               const char* name_field, const string& name, const char* id_field,
               int64* id) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (name.empty()) {
      *id = 0LL;
      return Status::OK();
    }
    SqliteStatement select = db_->Prepare(
        strings::Printf("SELECT %s FROM %s WHERE %s = ? AND %s = ?", id_field,
                        table, parent_id_field, name_field));
    if (parent_id > 0) {
      select.BindInt(1, parent_id);
    }
    select.BindText(2, name);
    bool is_done;
    TF_RETURN_IF_ERROR(select.Step(&is_done));
    if (!is_done) {
      *id = select.ColumnInt(0);
    } else {
      *id = MakeRandomId();
      SqliteStatement insert = db_->Prepare(strings::Printf(
          "INSERT INTO %s (%s, %s, %s, inserted_time) VALUES (?, ?, ?, ?)",
          table, parent_id_field, id_field, name_field));
      if (parent_id > 0) {
        insert.BindInt(1, parent_id);
      }
      insert.BindInt(2, *id);
      insert.BindText(3, name);
      insert.BindDouble(4, GetWallTime());
      TF_RETURN_IF_ERROR(insert.StepAndReset());
    }
    return Status::OK();
  }

  mutex mu_;
  Env* env_;
  std::shared_ptr<Sqlite> db_ GUARDED_BY(mu_);
  SqliteStatement insert_tensor_ GUARDED_BY(mu_);
  SqliteStatement update_metadata_ GUARDED_BY(mu_);
  string user_name_ GUARDED_BY(mu_);
  string experiment_name_ GUARDED_BY(mu_);
  string run_name_ GUARDED_BY(mu_);
  int64 run_id_ GUARDED_BY(mu_);
};

}  // namespace

Status CreateSummaryDbWriter(std::shared_ptr<Sqlite> db,
                             const string& experiment_name,
                             const string& run_name, const string& user_name,
                             Env* env, SummaryWriterInterface** result) {
  TF_RETURN_IF_ERROR(SetupTensorboardSqliteDb(db));
  SummaryDbWriter* w = new SummaryDbWriter(env, std::move(db));
  const Status s = w->Initialize(experiment_name, run_name, user_name);
  if (!s.ok()) {
    w->Unref();
    *result = nullptr;
    return s;
  }
  *result = w;
  return Status::OK();
}

}  // namespace tensorflow
