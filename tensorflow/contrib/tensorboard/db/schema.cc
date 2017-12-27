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
#include "tensorflow/contrib/tensorboard/db/schema.h"

namespace tensorflow {
namespace {

class SqliteSchema {
 public:
  explicit SqliteSchema(std::shared_ptr<Sqlite> db) : db_(std::move(db)) {}

  /// \brief Creates Ids table.
  ///
  /// This table must be used to randomly allocate Permanent IDs for
  /// all top-level tables, in order to maintain an invariant where
  /// foo_id != bar_id for all IDs of any two tables.
  ///
  /// A row should only be deleted from this table if it can be
  /// guaranteed that it exists absolutely nowhere else in the entire
  /// system.
  ///
  /// Fields:
  ///   id: An ID that was allocated globally. This must be in the
  ///     range [1,2**47). 0 is assigned the same meaning as NULL and
  ///     shouldn't be stored; 2**63-1 is reserved for statically
  ///     allocating space in a page to UPDATE later; and all other
  ///     int64 values are reserved for future use.
  Status CreateIdsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Ids (
        id INTEGER PRIMARY KEY
      )
    )sql");
  }

  /// \brief Creates Descriptions table.
  ///
  /// This table allows TensorBoard to associate Markdown text with any
  /// object in the database that has a Permanent ID.
  ///
  /// Fields:
  ///   id: The Permanent ID of the associated object. This is also the
  ///     SQLite rowid.
  ///   description: Arbitrary Markdown text.
  Status CreateDescriptionsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Descriptions (
        id INTEGER PRIMARY KEY,
        description TEXT
      )
    )sql");
  }

  /// \brief Creates Tensors table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   tag_id: ID of associated Tag.
  ///   computed_time: Float UNIX timestamp with microsecond precision.
  ///     In the old summaries system that uses FileWriter, this is the
  ///     wall time around when tf.Session.run finished. In the new
  ///     summaries system, it is the wall time of when the tensor was
  ///     computed. On systems with monotonic clocks, it is calculated
  ///     by adding the monotonic run duration to Run.started_time.
  ///     This field is not indexed because, in practice, it should be
  ///     ordered the same or nearly the same as TensorIndex, so local
  ///     insertion sort might be more suitable.
  ///   step: User-supplied number, ordering this tensor in Tag.
  ///     If NULL then the Tag must have only one Tensor.
  ///   tensor: Can be an INTEGER (DT_INT64), FLOAT (DT_DOUBLE), or
  ///     BLOB. The structure of a BLOB is currently undefined, but in
  ///     essence it is a Snappy tf.TensorProto that spills over into
  ///     TensorChunks.
  Status CreateTensorsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Tensors (
        rowid INTEGER PRIMARY KEY,
        tag_id INTEGER NOT NULL,
        computed_time REAL,
        step INTEGER,
        tensor BLOB
      )
    )sql");
  }

  /// \brief Creates TensorChunks table.
  ///
  /// This table can be used to split up a tensor across many rows,
  /// which has the advantage of not slowing down table scans on the
  /// main table, allowing asynchronous fetching, minimizing copying,
  /// and preventing large buffers from being allocated.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   tag_id: ID of associated Tag.
  ///   step: Same as corresponding Tensors.step.
  ///   sequence: 1-indexed sequence number for ordering chunks. Please
  ///     note that the 0th index is Tensors.tensor.
  ///   chunk: Bytes of next chunk in tensor.
  Status CreateTensorChunksTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS TensorChunks (
        rowid INTEGER PRIMARY KEY,
        tag_id INTEGER NOT NULL,
        step INTEGER,
        sequence INTEGER,
        chunk BLOB
      )
    )sql");
  }

  /// \brief Creates Tags table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   tag_id: The Permanent ID of the Tag.
  ///   run_id: Optional ID of associated Run.
  ///   tag_name: The tag field in summary.proto, unique across Run.
  ///   inserted_time: Float UNIX timestamp with µs precision. This is
  ///     always the wall time of when the row was inserted into the
  ///     DB. It may be used as a hint for an archival job.
  ///   display_name: Optional for GUI and defaults to tag_name.
  ///   plugin_name: Arbitrary TensorBoard plugin name for dispatch.
  ///   plugin_data: Arbitrary data that plugin wants.
  Status CreateTagsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Tags (
        rowid INTEGER PRIMARY KEY,
        run_id INTEGER,
        tag_id INTEGER NOT NULL,
        tag_name TEXT,
        inserted_time DOUBLE,
        display_name TEXT,
        plugin_name TEXT,
        plugin_data BLOB
      )
    )sql");
  }

  /// \brief Creates Runs table.
  ///
  /// This table stores information about Runs. Each row usually
  /// represents a single attempt at training or testing a TensorFlow
  /// model, with a given set of hyper-parameters, whose summaries are
  /// written out to a single event logs directory with a monotonic step
  /// counter.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   run_id: The Permanent ID of the Run. This has a 1:1 mapping
  ///     with a SummaryWriter instance. If two writers spawn for a
  ///     given (user_name, run_name, run_name) then each should
  ///     allocate its own run_id and whichever writer puts it in the
  ///     database last wins. The Tags / Tensors associated with the
  ///     previous invocations will then enter limbo, where they may be
  ///     accessible for certain operations, but should be garbage
  ///     collected eventually.
  ///   experiment_id: Optional ID of associated Experiment.
  ///   run_name: User-supplied string, unique across Experiment.
  ///   inserted_time: Float UNIX timestamp with µs precision. This is
  ///     always the time the row was inserted into the database. It
  ///     does not change.
  ///   started_time: Float UNIX timestamp with µs precision. In the
  ///     old summaries system that uses FileWriter, this is
  ///     approximated as the first tf.Event.wall_time. In the new
  ///     summaries system, it is the wall time of when summary writing
  ///     started, from the perspective of whichever machine talks to
  ///     the database. This field will be mutated if the run is
  ///     restarted.
  ///   finished_time: Float UNIX timestamp with µs precision of when
  ///     SummaryWriter resource that created this run was destroyed.
  ///     Once this value becomes non-NULL a Run and its Tags and
  ///     Tensors should be regarded as immutable.
  ///   graph_id: ID of associated Graphs row.
  Status CreateRunsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Runs (
        rowid INTEGER PRIMARY KEY,
        experiment_id INTEGER,
        run_id INTEGER NOT NULL,
        run_name TEXT,
        inserted_time REAL,
        started_time REAL,
        finished_time REAL,
        graph_id INTEGER
      )
    )sql");
  }

  /// \brief Creates Experiments table.
  ///
  /// This table stores information about experiments, which are sets of
  /// runs.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   user_id: Optional ID of associated User.
  ///   experiment_id: The Permanent ID of the Experiment.
  ///   experiment_name: User-supplied string, unique across User.
  ///   inserted_time: Float UNIX timestamp with µs precision. This is
  ///     always the time the row was inserted into the database. It
  ///     does not change.
  ///   started_time: Float UNIX timestamp with µs precision. This is
  ///     the MIN(experiment.started_time, run.started_time) of each
  ///     Run added to the database, including Runs which have since
  ///     been overwritten.
  Status CreateExperimentsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Experiments (
        rowid INTEGER PRIMARY KEY,
        user_id INTEGER,
        experiment_id INTEGER NOT NULL,
        experiment_name TEXT,
        inserted_time REAL,
        started_time REAL
      )
    )sql");
  }

  /// \brief Creates Users table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   user_id: The Permanent ID of the User.
  ///   user_name: Unique user name.
  ///   email: Optional unique email address.
  ///   inserted_time: Float UNIX timestamp with µs precision. This is
  ///     always the time the row was inserted into the database. It
  ///     does not change.
  Status CreateUsersTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Users (
        rowid INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        user_name TEXT,
        email TEXT,
        inserted_time REAL
      )
    )sql");
  }

  /// \brief Creates Graphs table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   graph_id: The Permanent ID of the Graph.
  ///   inserted_time: Float UNIX timestamp with µs precision. This is
  ///     always the wall time of when the row was inserted into the
  ///     DB. It may be used as a hint for an archival job.
  ///   node_def: Contains Snappy tf.GraphDef proto. All fields will be
  ///     cleared except those not expressed in SQL.
  Status CreateGraphsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Graphs (
        rowid INTEGER PRIMARY KEY,
        graph_id INTEGER NOT NULL,
        inserted_time REAL,
        graph_def BLOB
      )
    )sql");
  }

  /// \brief Creates Nodes table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   graph_id: The Permanent ID of the associated Graph.
  ///   node_id: ID for this node. This is more like a 0-index within
  ///     the Graph. Please note indexes are allowed to be removed.
  ///   node_name: Unique name for this Node within Graph. This is
  ///     copied from the proto so it can be indexed. This is allowed
  ///     to be NULL to save space on the index, in which case the
  ///     node_def.name proto field must not be cleared.
  ///   op: Copied from tf.NodeDef proto.
  ///   device: Copied from tf.NodeDef proto.
  ///   node_def: Contains Snappy tf.NodeDef proto. All fields will be
  ///     cleared except those not expressed in SQL.
  Status CreateNodesTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS Nodes (
        rowid INTEGER PRIMARY KEY,
        graph_id INTEGER NOT NULL,
        node_id INTEGER NOT NULL,
        node_name TEXT,
        op TEXT,
        device TEXT,
        node_def BLOB
      )
    )sql");
  }

  /// \brief Creates NodeInputs table.
  ///
  /// Fields:
  ///   rowid: Ephemeral b-tree ID dictating locality.
  ///   graph_id: The Permanent ID of the associated Graph.
  ///   node_id: Index of Node in question. This can be considered the
  ///     'to' vertex.
  ///   idx: Used for ordering inputs on a given Node.
  ///   input_node_id: Nodes.node_id of the corresponding input node.
  ///     This can be considered the 'from' vertex.
  ///   is_control: If non-zero, indicates this input is a controlled
  ///     dependency, which means this isn't an edge through which
  ///     tensors flow. NULL means 0.
  Status CreateNodeInputsTable() {
    return Run(R"sql(
      CREATE TABLE IF NOT EXISTS NodeInputs (
        rowid INTEGER PRIMARY KEY,
        graph_id INTEGER NOT NULL,
        node_id INTEGER NOT NULL,
        idx INTEGER NOT NULL,
        input_node_id INTEGER NOT NULL,
        is_control INTEGER
      )
    )sql");
  }

  /// \brief Uniquely indexes (tag_id, step) on Tensors table.
  Status CreateTensorIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS TensorIndex
      ON Tensors (tag_id, step)
    )sql");
  }

  /// \brief Uniquely indexes (tag_id, step, sequence) on TensorChunks table.
  Status CreateTensorChunkIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS TensorChunkIndex
      ON TensorChunks (tag_id, step, sequence)
    )sql");
  }

  /// \brief Uniquely indexes tag_id on Tags table.
  Status CreateTagIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS TagIdIndex
      ON Tags (tag_id)
    )sql");
  }

  /// \brief Uniquely indexes run_id on Runs table.
  Status CreateRunIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS RunIdIndex
      ON Runs (run_id)
    )sql");
  }

  /// \brief Uniquely indexes experiment_id on Experiments table.
  Status CreateExperimentIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS ExperimentIdIndex
      ON Experiments (experiment_id)
    )sql");
  }

  /// \brief Uniquely indexes user_id on Users table.
  Status CreateUserIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS UserIdIndex
      ON Users (user_id)
    )sql");
  }

  /// \brief Uniquely indexes graph_id on Graphs table.
  Status CreateGraphIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS GraphIdIndex
      ON Graphs (graph_id)
    )sql");
  }

  /// \brief Uniquely indexes (graph_id, node_id) on Nodes table.
  Status CreateNodeIdIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS NodeIdIndex
      ON Nodes (graph_id, node_id)
    )sql");
  }

  /// \brief Uniquely indexes (graph_id, node_id, idx) on NodeInputs table.
  Status CreateNodeInputsIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS NodeInputsIndex
      ON NodeInputs (graph_id, node_id, idx)
    )sql");
  }

  /// \brief Uniquely indexes (run_id, tag_name) on Tags table.
  Status CreateTagNameIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS TagNameIndex
      ON Tags (run_id, tag_name)
      WHERE tag_name IS NOT NULL
    )sql");
  }

  /// \brief Uniquely indexes (experiment_id, run_name) on Runs table.
  Status CreateRunNameIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS RunNameIndex
      ON Runs (experiment_id, run_name)
      WHERE run_name IS NOT NULL
    )sql");
  }

  /// \brief Uniquely indexes (user_id, experiment_name) on Experiments table.
  Status CreateExperimentNameIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS ExperimentNameIndex
      ON Experiments (user_id, experiment_name)
      WHERE experiment_name IS NOT NULL
    )sql");
  }

  /// \brief Uniquely indexes user_name on Users table.
  Status CreateUserNameIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS UserNameIndex
      ON Users (user_name)
      WHERE user_name IS NOT NULL
    )sql");
  }

  /// \brief Uniquely indexes email on Users table.
  Status CreateUserEmailIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS UserEmailIndex
      ON Users (email)
      WHERE email IS NOT NULL
    )sql");
  }

  /// \brief Uniquely indexes (graph_id, node_name) on Nodes table.
  Status CreateNodeNameIndex() {
    return Run(R"sql(
      CREATE UNIQUE INDEX IF NOT EXISTS NodeNameIndex
      ON Nodes (graph_id, node_name)
      WHERE node_name IS NOT NULL
    )sql");
  }

  Status Run(const char* sql) {
    auto stmt = db_->Prepare(sql);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(stmt.StepAndReset(), sql);
    return Status::OK();
  }

 private:
  std::shared_ptr<Sqlite> db_;
};

}  // namespace

Status SetupTensorboardSqliteDb(std::shared_ptr<Sqlite> db) {
  SqliteSchema s(std::move(db));
  TF_RETURN_IF_ERROR(s.CreateIdsTable());
  TF_RETURN_IF_ERROR(s.CreateDescriptionsTable());
  TF_RETURN_IF_ERROR(s.CreateTensorsTable());
  TF_RETURN_IF_ERROR(s.CreateTensorChunksTable());
  TF_RETURN_IF_ERROR(s.CreateTagsTable());
  TF_RETURN_IF_ERROR(s.CreateRunsTable());
  TF_RETURN_IF_ERROR(s.CreateExperimentsTable());
  TF_RETURN_IF_ERROR(s.CreateUsersTable());
  TF_RETURN_IF_ERROR(s.CreateGraphsTable());
  TF_RETURN_IF_ERROR(s.CreateNodeInputsTable());
  TF_RETURN_IF_ERROR(s.CreateNodesTable());
  TF_RETURN_IF_ERROR(s.CreateTensorIndex());
  TF_RETURN_IF_ERROR(s.CreateTensorChunkIndex());
  TF_RETURN_IF_ERROR(s.CreateTagIdIndex());
  TF_RETURN_IF_ERROR(s.CreateRunIdIndex());
  TF_RETURN_IF_ERROR(s.CreateExperimentIdIndex());
  TF_RETURN_IF_ERROR(s.CreateUserIdIndex());
  TF_RETURN_IF_ERROR(s.CreateGraphIdIndex());
  TF_RETURN_IF_ERROR(s.CreateNodeIdIndex());
  TF_RETURN_IF_ERROR(s.CreateNodeInputsIndex());
  TF_RETURN_IF_ERROR(s.CreateTagNameIndex());
  TF_RETURN_IF_ERROR(s.CreateRunNameIndex());
  TF_RETURN_IF_ERROR(s.CreateExperimentNameIndex());
  TF_RETURN_IF_ERROR(s.CreateUserNameIndex());
  TF_RETURN_IF_ERROR(s.CreateUserEmailIndex());
  TF_RETURN_IF_ERROR(s.CreateNodeNameIndex());
  return Status::OK();
}

}  // namespace tensorflow
