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

// https://www.sqlite.org/fileformat.html#record_format
const uint64 kIdTiers[] = {
    0x7fffffULL,        // 23-bit (3 bytes on disk)
    0x7fffffffULL,      // 31-bit (4 bytes on disk)
    0x7fffffffffffULL,  // 47-bit (5 bytes on disk)
                        // Remaining bits reserved for future use.
};
const int kMaxIdTier = sizeof(kIdTiers) / sizeof(uint64);
const int kIdCollisionDelayMicros = 10;
const int kMaxIdCollisions = 21;  // sum(2**i*10µs for i in range(21))~=21s
const int64 kAbsent = 0LL;
const int64 kReserved = 0x7fffffffffffffffLL;

double GetWallTime(Env* env) {
  // TODO(@jart): Follow precise definitions for time laid out in schema.
  // TODO(@jart): Use monotonic clock from gRPC codebase.
  return static_cast<double>(env->NowMicros()) / 1.0e6;
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

/// \brief Generates unique IDs randomly in the [1,2**63-2] range.
///
/// This class starts off generating IDs in the [1,2**23-1] range,
/// because it's human friendly and occupies 4 bytes max on disk with
/// SQLite's zigzag varint encoding. Then, each time a collision
/// happens, the random space is increased by 8 bits.
///
/// This class uses exponential back-off so writes will slow down as
/// the ID space becomes exhausted.
class IdAllocator {
 public:
  IdAllocator(Env* env, Sqlite* db)
      : env_{env}, inserter_{db->Prepare("INSERT INTO Ids (id) VALUES (?)")} {}

  Status CreateNewId(int64* id) {
    Status s;
    for (int i = 0; i < kMaxIdCollisions; ++i) {
      int64 tid = MakeRandomId();
      inserter_.BindInt(1, tid);
      s = inserter_.StepAndReset();
      if (s.ok()) {
        *id = tid;
        break;
      }
      // SQLITE_CONSTRAINT maps to INVALID_ARGUMENT in sqlite.cc
      if (s.code() != error::INVALID_ARGUMENT) break;
      if (tier_ < kMaxIdTier) {
        LOG(INFO) << "IdAllocator collision at tier " << tier_ << " (of "
                  << kMaxIdTier << ") so auto-adjusting to a higher tier";
        ++tier_;
      } else {
        LOG(WARNING) << "IdAllocator (attempt #" << i << ") "
                     << "resulted in a collision at the highest tier; this "
                        "is problematic if it happens often; you can try "
                        "pruning the Ids table; you can also file a bug "
                        "asking for the ID space to be increased; otherwise "
                        "writes will gradually slow down over time until they "
                        "become impossible";
      }
      env_->SleepForMicroseconds((1 << i) * kIdCollisionDelayMicros);
    }
    return s;
  }

 private:
  int64 MakeRandomId() {
    int64 id = static_cast<int64>(random::New64() & kIdTiers[tier_]);
    if (id == kAbsent) ++id;
    if (id == kReserved) --id;
    return id;
  }

  Env* env_;
  SqliteStatement inserter_;
  int tier_ = 0;
};

class GraphSaver {
 public:
  static Status Save(Env* env, Sqlite* db, IdAllocator* id_allocator,
                     GraphDef* graph, int64* graph_id) {
    TF_RETURN_IF_ERROR(id_allocator->CreateNewId(graph_id));
    GraphSaver saver{env, db, graph, *graph_id};
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
      INSERT INTO Graphs (graph_id, inserted_time, graph_def)
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

class RunWriter {
 public:
  RunWriter(Env* env, std::shared_ptr<Sqlite> db, const string& experiment_name,
            const string& run_name, const string& user_name)
      : env_{env},
        db_{std::move(db)},
        id_allocator_{env_, db_.get()},
        experiment_name_{experiment_name},
        run_name_{run_name},
        user_name_{user_name},
        insert_tensor_{db_->Prepare(R"sql(
          INSERT OR REPLACE INTO Tensors (tag_id, step, computed_time, tensor)
          VALUES (?, ?, ?, ?)
        )sql")} {}

  ~RunWriter() {
    if (run_id_ == kAbsent) return;
    auto update = db_->Prepare(R"sql(
      UPDATE Runs SET finished_time = ? WHERE run_id = ?
    )sql");
    update.BindDouble(1, GetWallTime(env_));
    update.BindInt(2, run_id_);
    Status s = update.StepAndReset();
    if (!s.ok()) {
      LOG(ERROR) << "Failed to set Runs[" << run_id_
                 << "].finish_time: " << s.ToString();
    }
  }

  Status InsertTensor(int64 tag_id, int64 step, double computed_time,
                      Tensor t) {
    insert_tensor_.BindInt(1, tag_id);
    insert_tensor_.BindInt(2, step);
    insert_tensor_.BindDouble(3, computed_time);
    if (t.shape().dims() == 0 && t.dtype() == DT_INT64) {
      insert_tensor_.BindInt(4, t.scalar<int64>()());
    } else if (t.shape().dims() == 0 && t.dtype() == DT_DOUBLE) {
      insert_tensor_.BindDouble(4, t.scalar<double>()());
    } else {
      TF_RETURN_IF_ERROR(BindTensor(&insert_tensor_, 4, t));
    }
    return insert_tensor_.StepAndReset();
  }

  Status InsertGraph(std::unique_ptr<GraphDef> g, double computed_time) {
    TF_RETURN_IF_ERROR(InitializeRun(computed_time));
    int64 graph_id;
    TF_RETURN_IF_ERROR(
        GraphSaver::Save(env_, db_.get(), &id_allocator_, g.get(), &graph_id));
    if (run_id_ != kAbsent) {
      auto set = db_->Prepare("UPDATE Runs SET graph_id = ? WHERE run_id = ?");
      set.BindInt(1, graph_id);
      set.BindInt(2, run_id_);
      TF_RETURN_IF_ERROR(set.StepAndReset());
    }
    return Status::OK();
  }

  Status GetTagId(double computed_time, const string& tag_name,
                  const SummaryMetadata& metadata, int64* tag_id) {
    TF_RETURN_IF_ERROR(InitializeRun(computed_time));
    auto e = tag_ids_.find(tag_name);
    if (e != tag_ids_.end()) {
      *tag_id = e->second;
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(id_allocator_.CreateNewId(tag_id));
    tag_ids_[tag_name] = *tag_id;
    if (!metadata.summary_description().empty()) {
      SqliteStatement insert_description = db_->Prepare(R"sql(
        INSERT INTO Descriptions (id, description) VALUES (?, ?)
      )sql");
      insert_description.BindInt(1, *tag_id);
      insert_description.BindText(2, metadata.summary_description());
      TF_RETURN_IF_ERROR(insert_description.StepAndReset());
    }
    SqliteStatement insert = db_->Prepare(R"sql(
      INSERT INTO Tags (
        run_id,
        tag_id,
        tag_name,
        inserted_time,
        display_name,
        plugin_name,
        plugin_data
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    )sql");
    if (run_id_ != kAbsent) insert.BindInt(1, run_id_);
    insert.BindInt(2, *tag_id);
    insert.BindText(3, tag_name);
    insert.BindDouble(4, GetWallTime(env_));
    if (!metadata.display_name().empty()) {
      insert.BindText(5, metadata.display_name());
    }
    if (!metadata.plugin_data().plugin_name().empty()) {
      insert.BindText(6, metadata.plugin_data().plugin_name());
    }
    if (!metadata.plugin_data().content().empty()) {
      insert.BindBlob(7, metadata.plugin_data().content());
    }
    return insert.StepAndReset();
  }

 private:
  Status InitializeUser() {
    if (user_id_ != kAbsent || user_name_.empty()) return Status::OK();
    SqliteStatement get = db_->Prepare(R"sql(
      SELECT user_id FROM Users WHERE user_name = ?
    )sql");
    get.BindText(1, user_name_);
    bool is_done;
    TF_RETURN_IF_ERROR(get.Step(&is_done));
    if (!is_done) {
      user_id_ = get.ColumnInt(0);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(id_allocator_.CreateNewId(&user_id_));
    SqliteStatement insert = db_->Prepare(R"sql(
      INSERT INTO Users (user_id, user_name, inserted_time) VALUES (?, ?, ?)
    )sql");
    insert.BindInt(1, user_id_);
    insert.BindText(2, user_name_);
    insert.BindDouble(3, GetWallTime(env_));
    TF_RETURN_IF_ERROR(insert.StepAndReset());
    return Status::OK();
  }

  Status InitializeExperiment(double computed_time) {
    if (experiment_name_.empty()) return Status::OK();
    if (experiment_id_ == kAbsent) {
      TF_RETURN_IF_ERROR(InitializeUser());
      SqliteStatement get = db_->Prepare(R"sql(
        SELECT
          experiment_id,
          started_time
        FROM
          Experiments
        WHERE
          user_id IS ?
          AND experiment_name = ?
      )sql");
      if (user_id_ != kAbsent) get.BindInt(1, user_id_);
      get.BindText(2, experiment_name_);
      bool is_done;
      TF_RETURN_IF_ERROR(get.Step(&is_done));
      if (!is_done) {
        experiment_id_ = get.ColumnInt(0);
        experiment_started_time_ = get.ColumnInt(1);
      } else {
        TF_RETURN_IF_ERROR(id_allocator_.CreateNewId(&experiment_id_));
        experiment_started_time_ = computed_time;
        SqliteStatement insert = db_->Prepare(R"sql(
          INSERT INTO Experiments (
            user_id,
            experiment_id,
            experiment_name,
            inserted_time,
            started_time
          ) VALUES (?, ?, ?, ?, ?)
        )sql");
        if (user_id_ != kAbsent) insert.BindInt(1, user_id_);
        insert.BindInt(2, experiment_id_);
        insert.BindText(3, experiment_name_);
        insert.BindDouble(4, GetWallTime(env_));
        insert.BindDouble(5, computed_time);
        TF_RETURN_IF_ERROR(insert.StepAndReset());
      }
    }
    if (computed_time < experiment_started_time_) {
      experiment_started_time_ = computed_time;
      SqliteStatement update = db_->Prepare(R"sql(
        UPDATE Experiments SET started_time = ? WHERE experiment_id = ?
      )sql");
      update.BindDouble(1, computed_time);
      update.BindInt(2, experiment_id_);
      TF_RETURN_IF_ERROR(update.StepAndReset());
    }
    return Status::OK();
  }

  Status InitializeRun(double computed_time) {
    if (run_name_.empty()) return Status::OK();
    TF_RETURN_IF_ERROR(InitializeExperiment(computed_time));
    if (run_id_ == kAbsent) {
      TF_RETURN_IF_ERROR(id_allocator_.CreateNewId(&run_id_));
      run_started_time_ = computed_time;
      SqliteStatement insert = db_->Prepare(R"sql(
        INSERT OR REPLACE INTO Runs (
          experiment_id,
          run_id,
          run_name,
          inserted_time,
          started_time
        ) VALUES (?, ?, ?, ?, ?)
      )sql");
      if (experiment_id_ != kAbsent) insert.BindInt(1, experiment_id_);
      insert.BindInt(2, run_id_);
      insert.BindText(3, run_name_);
      insert.BindDouble(4, GetWallTime(env_));
      insert.BindDouble(5, computed_time);
      TF_RETURN_IF_ERROR(insert.StepAndReset());
    }
    if (computed_time < run_started_time_) {
      run_started_time_ = computed_time;
      SqliteStatement update = db_->Prepare(R"sql(
        UPDATE Runs SET started_time = ? WHERE run_id = ?
      )sql");
      update.BindDouble(1, computed_time);
      update.BindInt(2, run_id_);
      TF_RETURN_IF_ERROR(update.StepAndReset());
    }
    return Status::OK();
  }

  Env* env_;
  std::shared_ptr<Sqlite> db_;
  IdAllocator id_allocator_;
  const string experiment_name_;
  const string run_name_;
  const string user_name_;
  int64 experiment_id_ = kAbsent;
  int64 run_id_ = kAbsent;
  int64 user_id_ = kAbsent;
  std::unordered_map<string, int64> tag_ids_;
  double experiment_started_time_ = 0.0;
  double run_started_time_ = 0.0;
  SqliteStatement insert_tensor_;
};

class SummaryDbWriter : public SummaryWriterInterface {
 public:
  SummaryDbWriter(Env* env, std::shared_ptr<Sqlite> db,
                  const string& experiment_name, const string& run_name,
                  const string& user_name)
      : SummaryWriterInterface(),
        env_{env},
        run_writer_{env, std::move(db), experiment_name, run_name, user_name} {}
  ~SummaryDbWriter() override {}

  Status Flush() override { return Status::OK(); }

  Status WriteTensor(int64 global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override {
    mutex_lock ml(mu_);
    SummaryMetadata metadata;
    if (!serialized_metadata.empty()) {
      metadata.ParseFromString(serialized_metadata);
    }
    double now = GetWallTime(env_);
    int64 tag_id;
    TF_RETURN_IF_ERROR(run_writer_.GetTagId(now, tag, metadata, &tag_id));
    return run_writer_.InsertTensor(tag_id, global_step, now, t);
  }

  Status WriteScalar(int64 global_step, Tensor t, const string& tag) override {
    Tensor t2;
    TF_RETURN_IF_ERROR(CoerceScalar(t, &t2));
    // TODO(jart): Generate scalars plugin metadata on this value.
    return WriteTensor(global_step, std::move(t2), tag, "");
  }

  Status WriteGraph(int64 global_step, std::unique_ptr<GraphDef> g) override {
    mutex_lock ml(mu_);
    return run_writer_.InsertGraph(std::move(g), GetWallTime(env_));
  }

  Status WriteEvent(std::unique_ptr<Event> e) override {
    switch (e->what_case()) {
      case Event::WhatCase::kSummary: {
        mutex_lock ml(mu_);
        Status s;
        for (const auto& value : e->summary().value()) {
          s.Update(WriteSummary(e.get(), value));
        }
        return s;
      }
      case Event::WhatCase::kGraphDef: {
        mutex_lock ml(mu_);
        std::unique_ptr<GraphDef> graph{new GraphDef};
        if (!ParseProtoUnlimited(graph.get(), e->graph_def())) {
          return errors::DataLoss("parse event.graph_def failed");
        }
        return run_writer_.InsertGraph(std::move(graph), e->wall_time());
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
  Status WriteSummary(const Event* e, const Summary::Value& summary)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    switch (summary.value_case()) {
      case Summary::Value::ValueCase::kSimpleValue: {
        int64 tag_id;
        TF_RETURN_IF_ERROR(run_writer_.GetTagId(e->wall_time(), summary.tag(),
                                                summary.metadata(), &tag_id));
        Tensor t{DT_DOUBLE, {}};
        t.scalar<double>()() = summary.simple_value();
        return run_writer_.InsertTensor(tag_id, e->step(), e->wall_time(), t);
      }
      default:
        // TODO(@jart): Handle the rest.
        return Status::OK();
    }
  }

  mutex mu_;
  Env* env_;
  RunWriter run_writer_ GUARDED_BY(mu_);
};

}  // namespace

Status CreateSummaryDbWriter(std::shared_ptr<Sqlite> db,
                             const string& experiment_name,
                             const string& run_name, const string& user_name,
                             Env* env, SummaryWriterInterface** result) {
  TF_RETURN_IF_ERROR(SetupTensorboardSqliteDb(db));
  *result = new SummaryDbWriter(env, std::move(db), experiment_name, run_name,
                                user_name);
  return Status::OK();
}

}  // namespace tensorflow
