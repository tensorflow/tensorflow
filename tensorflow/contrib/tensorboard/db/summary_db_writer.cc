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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

double GetWallTime(Env* env) {
  // TODO(@jart): Follow precise definitions for time laid out in schema.
  // TODO(@jart): Use monotonic clock from gRPC codebase.
  return static_cast<double>(env->NowMicros()) / 1.0e6;
}

int64 MakeRandomId() {
  // TODO(@jart): Try generating ID in 2^24 space, falling back to 2^63
  //              https://sqlite.org/src4/doc/trunk/www/varint.wiki
  int64 id = static_cast<int64>(random::New64() & ((1ULL << 63) - 1));
  if (id == 0) {
    ++id;
  }
  return id;
}

Status Serialize(const protobuf::MessageLite& proto, string* output) {
  output->clear();
  if (!proto.SerializeToString(output)) {
    return errors::DataLoss("SerializeToString failed");
  }
  return Status::OK();
}

Status Compress(const string& data, string* output) {
  output->clear();
  if (!port::Snappy_Compress(data.data(), data.size(), output)) {
    return errors::FailedPrecondition("TensorBase needs Snappy");
  }
  return Status::OK();
}

Status BindProto(SqliteStatement* stmt, int parameter,
                 const protobuf::MessageLite& proto) {
  string serialized;
  TF_RETURN_IF_ERROR(Serialize(proto, &serialized));
  string compressed;
  TF_RETURN_IF_ERROR(Compress(serialized, &compressed));
  stmt->BindBlob(parameter, compressed);
  return Status::OK();
}

Status BindTensor(SqliteStatement* stmt, int parameter, const Tensor& t) {
  // TODO(@jart): Make portable between little and big endian systems.
  // TODO(@jart): Use TensorChunks with minimal copying for big tensors.
  // TODO(@jart): Add field to indicate encoding.
  // TODO(@jart): Allow crunch tool to re-compress with zlib instead.
  TensorProto p;
  t.AsProtoTensorContent(&p);
  return BindProto(stmt, parameter, p);
}

// Tries to fudge shape and dtype to something with smaller storage.
Status CoerceScalar(const Tensor& t, Tensor* out) {
  switch (t.dtype()) {
    case DT_DOUBLE:
      *out = t;
      break;
    case DT_INT64:
      *out = t;
      break;
    case DT_FLOAT:
      *out = {DT_DOUBLE, {}};
      out->scalar<double>()() = t.scalar<float>()();
      break;
    case DT_HALF:
      *out = {DT_DOUBLE, {}};
      out->scalar<double>()() = static_cast<double>(t.scalar<Eigen::half>()());
      break;
    case DT_INT32:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<int32>()();
      break;
    case DT_INT16:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<int16>()();
      break;
    case DT_INT8:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<int8>()();
      break;
    case DT_UINT32:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<uint32>()();
      break;
    case DT_UINT16:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<uint16>()();
      break;
    case DT_UINT8:
      *out = {DT_INT64, {}};
      out->scalar<int64>()() = t.scalar<uint8>()();
      break;
    default:
      return errors::Unimplemented("Scalar summary for dtype ",
                                   DataTypeString(t.dtype()),
                                   " is not supported.");
  }
  return Status::OK();
}

class Transactor {
 public:
  explicit Transactor(std::shared_ptr<Sqlite> db)
      : db_(std::move(db)),
        begin_(db_->Prepare("BEGIN TRANSACTION")),
        commit_(db_->Prepare("COMMIT TRANSACTION")),
        rollback_(db_->Prepare("ROLLBACK TRANSACTION")) {}

  template <typename T, typename... Args>
  Status Transact(T callback, Args&&... args) {
    TF_RETURN_IF_ERROR(begin_.StepAndReset());
    Status s = callback(std::forward<Args>(args)...);
    if (s.ok()) {
      TF_RETURN_IF_ERROR(commit_.StepAndReset());
    } else {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(rollback_.StepAndReset(), s.ToString());
    }
    return s;
  }

 private:
  std::shared_ptr<Sqlite> db_;
  SqliteStatement begin_;
  SqliteStatement commit_;
  SqliteStatement rollback_;
};

class GraphSaver {
 public:
  static Status SaveToRun(Env* env, Sqlite* db, GraphDef* graph, int64 run_id) {
    auto get = db->Prepare("SELECT graph_id FROM Runs WHERE run_id = ?");
    get.BindInt(1, run_id);
    bool is_done;
    TF_RETURN_IF_ERROR(get.Step(&is_done));
    int64 graph_id = is_done ? 0 : get.ColumnInt(0);
    if (graph_id == 0) {
      graph_id = MakeRandomId();
      // TODO(@jart): Check for ID collision.
      auto set = db->Prepare("UPDATE Runs SET graph_id = ? WHERE run_id = ?");
      set.BindInt(1, graph_id);
      set.BindInt(2, run_id);
      TF_RETURN_IF_ERROR(set.StepAndReset());
    }
    return Save(env, db, graph, graph_id);
  }

  static Status Save(Env* env, Sqlite* db, GraphDef* graph, int64 graph_id) {
    GraphSaver saver{env, db, graph, graph_id};
    saver.MapNameToNodeId();
    TF_RETURN_IF_ERROR(saver.SaveNodeInputs());
    TF_RETURN_IF_ERROR(saver.SaveNodes());
    TF_RETURN_IF_ERROR(saver.SaveGraph());
    return Status::OK();
  }

 private:
  GraphSaver(Env* env, Sqlite* db, GraphDef* graph, int64 graph_id)
      : env_(env), db_(db), graph_(graph), graph_id_(graph_id) {}

  void MapNameToNodeId() {
    size_t toto = static_cast<size_t>(graph_->node_size());
    name_copies_.reserve(toto);
    name_to_node_id_.reserve(toto);
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      // Copy name into memory region, since we call clear_name() later.
      // Then wrap in StringPiece so we can compare slices without copy.
      name_copies_.emplace_back(graph_->node(node_id).name());
      name_to_node_id_.emplace(name_copies_.back(), node_id);
    }
  }

  Status SaveNodeInputs() {
    auto purge = db_->Prepare("DELETE FROM NodeInputs WHERE graph_id = ?");
    purge.BindInt(1, graph_id_);
    TF_RETURN_IF_ERROR(purge.StepAndReset());
    auto insert = db_->Prepare(R"sql(
      INSERT INTO NodeInputs (graph_id, node_id, idx, input_node_id, is_control)
      VALUES (?, ?, ?, ?, ?)
    )sql");
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      const NodeDef& node = graph_->node(node_id);
      for (int idx = 0; idx < node.input_size(); ++idx) {
        StringPiece name = node.input(idx);
        insert.BindInt(1, graph_id_);
        insert.BindInt(2, node_id);
        insert.BindInt(3, idx);
        if (!name.empty() && name[0] == '^') {
          name.remove_prefix(1);
          insert.BindInt(5, 1);
        }
        auto e = name_to_node_id_.find(name);
        if (e == name_to_node_id_.end()) {
          return errors::DataLoss("Could not find node: ", name);
        }
        insert.BindInt(4, e->second);
        TF_RETURN_WITH_CONTEXT_IF_ERROR(insert.StepAndReset(), node.name(),
                                        " -> ", name);
      }
    }
    return Status::OK();
  }

  Status SaveNodes() {
    auto purge = db_->Prepare("DELETE FROM Nodes WHERE graph_id = ?");
    purge.BindInt(1, graph_id_);
    TF_RETURN_IF_ERROR(purge.StepAndReset());
    auto insert = db_->Prepare(R"sql(
      INSERT INTO Nodes (graph_id, node_id, node_name, op, device, node_def)
      VALUES (?, ?, ?, ?, ?, ?)
    )sql");
    for (int node_id = 0; node_id < graph_->node_size(); ++node_id) {
      NodeDef* node = graph_->mutable_node(node_id);
      insert.BindInt(1, graph_id_);
      insert.BindInt(2, node_id);
      insert.BindText(3, node->name());
      node->clear_name();
      if (!node->op().empty()) {
        insert.BindText(4, node->op());
        node->clear_op();
      }
      if (!node->device().empty()) {
        insert.BindText(5, node->device());
        node->clear_device();
      }
      node->clear_input();
      TF_RETURN_IF_ERROR(BindProto(&insert, 6, *node));
      TF_RETURN_WITH_CONTEXT_IF_ERROR(insert.StepAndReset(), node->name());
    }
    return Status::OK();
  }

  Status SaveGraph() {
    auto insert = db_->Prepare(R"sql(
      INSERT OR REPLACE INTO Graphs (graph_id, inserted_time, graph_def)
      VALUES (?, ?, ?)
    )sql");
    insert.BindInt(1, graph_id_);
    insert.BindDouble(2, GetWallTime(env_));
    graph_->clear_node();
    TF_RETURN_IF_ERROR(BindProto(&insert, 3, *graph_));
    return insert.StepAndReset();
  }

  Env* env_;
  Sqlite* db_;
  GraphDef* graph_;
  int64 graph_id_;
  std::vector<string> name_copies_;
  std::unordered_map<StringPiece, int64, StringPieceHasher> name_to_node_id_;
};

class SummaryDbWriter : public SummaryWriterInterface {
 public:
  SummaryDbWriter(Env* env, std::shared_ptr<Sqlite> db)
      : SummaryWriterInterface(),
        env_(env),
        db_(std::move(db)),
        txn_(db_),
        run_id_{0LL} {}
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
    insert_tensor_.BindDouble(3, GetWallTime(env_));
    if (t.shape().dims() == 0 && t.dtype() == DT_INT64) {
      insert_tensor_.BindInt(4, t.scalar<int64>()());
    } else if (t.shape().dims() == 0 && t.dtype() == DT_DOUBLE) {
      insert_tensor_.BindDouble(4, t.scalar<double>()());
    } else {
      TF_RETURN_IF_ERROR(BindTensor(&insert_tensor_, 4, t));
    }
    return insert_tensor_.StepAndReset();
  }

  Status WriteScalar(int64 global_step, Tensor t, const string& tag) override {
    Tensor t2;
    TF_RETURN_IF_ERROR(CoerceScalar(t, &t2));
    // TODO(jart): Generate scalars plugin metadata on this value.
    return WriteTensor(global_step, std::move(t2), tag, "");
  }

  Status WriteGraph(int64 global_step, std::unique_ptr<GraphDef> g) override {
    mutex_lock ml(mu_);
    TF_RETURN_IF_ERROR(InitializeParents());
    return txn_.Transact(GraphSaver::SaveToRun, env_, db_.get(), g.get(),
                         run_id_);
  }

  Status WriteEvent(std::unique_ptr<Event> e) override {
    switch (e->what_case()) {
      case Event::WhatCase::kSummary: {
        mutex_lock ml(mu_);
        TF_RETURN_IF_ERROR(InitializeParents());
        const Summary& summary = e->summary();
        for (int i = 0; i < summary.value_size(); ++i) {
          TF_RETURN_IF_ERROR(WriteSummary(e.get(), summary.value(i)));
        }
        return Status::OK();
      }
      case Event::WhatCase::kGraphDef: {
        std::unique_ptr<GraphDef> graph{new GraphDef};
        if (!ParseProtoUnlimited(graph.get(), e->graph_def())) {
          return errors::DataLoss("parse event.graph_def failed");
        }
        return WriteGraph(e->step(), std::move(graph));
      }
      default:
        // TODO(@jart): Handle other stuff.
        return Status::OK();
    }
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
  Status InitializeParents() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (run_id_ > 0) {
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
      insert_user.BindDouble(3, GetWallTime(env_));
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
      insert.BindDouble(4, GetWallTime(env_));
      TF_RETURN_IF_ERROR(insert.StepAndReset());
    }
    return Status::OK();
  }

  Status WriteSummary(const Event* e, const Summary::Value& summary)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int64 tag_id;
    TF_RETURN_IF_ERROR(GetTagId(run_id_, summary.tag(), &tag_id));
    insert_tensor_.BindInt(1, tag_id);
    insert_tensor_.BindInt(2, e->step());
    insert_tensor_.BindDouble(3, e->wall_time());
    switch (summary.value_case()) {
      case Summary::Value::ValueCase::kSimpleValue:
        insert_tensor_.BindDouble(4, summary.simple_value());
        break;
      default:
        // TODO(@jart): Handle the rest.
        return Status::OK();
    }
    return insert_tensor_.StepAndReset();
  }

  mutex mu_;
  Env* env_;
  std::shared_ptr<Sqlite> db_ GUARDED_BY(mu_);
  Transactor txn_ GUARDED_BY(mu_);
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
