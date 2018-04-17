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

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

Status Run(Sqlite* db, const char* sql) {
  SqliteStatement stmt;
  TF_RETURN_IF_ERROR(db->Prepare(sql, &stmt));
  return stmt.StepAndReset();
}

}  // namespace

Status SetupTensorboardSqliteDb(Sqlite* db) {
  // Note: GCC raw strings macros are broken.
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55971
  TF_RETURN_IF_ERROR(
      db->PrepareOrDie(strings::StrCat("PRAGMA application_id=",
                                       kTensorboardSqliteApplicationId))
          .StepAndReset());
  db->PrepareOrDie("PRAGMA user_version=0").StepAndResetOrDie();
  Status s;

  // Ids identify resources.
  //
  // This table can be used to efficiently generate Permanent IDs in
  // conjunction with a random number generator. Unlike rowids these
  // IDs safe to use in URLs and unique across tables.
  //
  // Within any given system, there can't be any foo_id == bar_id for
  // all rows of any two (Foos, Bars) tables. A row should only be
  // deleted from this table if there's a very high level of confidence
  // it exists nowhere else in the system.
  //
  // Fields:
  //   id: The system-wide ID. This must be in the range [1,2**47). 0
  //     is assigned the same meaning as NULL and shouldn't be stored
  //     and all other int64 values are reserved for future use. Please
  //     note that id is also the rowid.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Ids (
      id INTEGER PRIMARY KEY
    )
  )sql"));

  // Descriptions are Markdown text that can be associated with any
  // resource that has a Permanent ID.
  //
  // Fields:
  //   id: The foo_id of the associated row in Foos.
  //   description: Arbitrary NUL-terminated Markdown text.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Descriptions (
      id INTEGER PRIMARY KEY,
      description TEXT
    )
  )sql"));

  // Tensors are 0..n-dimensional numbers or strings.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   series: The Permanent ID of a different resource, e.g. tag_id. A
  //     tensor will be vacuumed if no series == foo_id exists for all
  //     rows of all Foos. When series is NULL this tensor may serve
  //     undefined purposes. This field should be set on placeholders.
  //   step: Arbitrary number to uniquely order tensors within series.
  //     The meaning of step is undefined when series is NULL. This may
  //     be set on placeholders to prepopulate index pages.
  //   computed_time: Float UNIX timestamp with microsecond precision.
  //     In the old summaries system that uses FileWriter, this is the
  //     wall time around when tf.Session.run finished. In the new
  //     summaries system, it is the wall time of when the tensor was
  //     computed. On systems with monotonic clocks, it is calculated
  //     by adding the monotonic run duration to Run.started_time.
  //   dtype: The tensorflow::DataType ID. For example, DT_INT64 is 9.
  //     When NULL or 0 this must be treated as a placeholder row that
  //     does not officially exist.
  //   shape: A comma-delimited list of int64 >=0 values representing
  //     length of each dimension in the tensor. This must be a valid
  //     shape. That means no -1 values and, in the case of numeric
  //     tensors, length(data) == product(shape) * sizeof(dtype). Empty
  //     means this is a scalar a.k.a. 0-dimensional tensor.
  //   data: Little-endian raw tensor memory. If dtype is DT_STRING and
  //     shape is empty, the nullness of this field indicates whether or
  //     not it contains the tensor contents; otherwise TensorStrings
  //     must be queried. If dtype is NULL then ZEROBLOB can be used on
  //     this field to reserve row space to be updated later.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Tensors (
      rowid INTEGER PRIMARY KEY,
      series INTEGER,
      step INTEGER,
      dtype INTEGER,
      computed_time REAL,
      shape TEXT,
      data BLOB
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS
      TensorSeriesStepIndex
    ON
      Tensors (series, step)
    WHERE
      series IS NOT NULL
      AND step IS NOT NULL
  )sql"));

  // TensorStrings are the flat contents of 1..n dimensional DT_STRING
  // Tensors.
  //
  // The number of rows associated with a Tensor must be equal to the
  // product of its Tensors.shape.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   tensor_rowid: References Tensors.rowid.
  //   idx: Index in flattened tensor, starting at 0.
  //   data: The string value at a particular index. NUL characters are
  //     permitted.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS TensorStrings (
      rowid INTEGER PRIMARY KEY,
      tensor_rowid INTEGER NOT NULL,
      idx INTEGER NOT NULL,
      data BLOB
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS TensorStringIndex
    ON TensorStrings (tensor_rowid, idx)
  )sql"));

  // Tags are series of Tensors.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   tag_id: The Permanent ID of the Tag.
  //   run_id: Optional ID of associated Run.
  //   inserted_time: Float UNIX timestamp with µs precision. This is
  //     always the wall time of when the row was inserted into the
  //     DB. It may be used as a hint for an archival job.
  //   tag_name: The tag field in summary.proto, unique across Run.
  //   display_name: Optional for GUI and defaults to tag_name.
  //   plugin_name: Arbitrary TensorBoard plugin name for dispatch.
  //   plugin_data: Arbitrary data that plugin wants.
  //
  // TODO(jart): Maybe there should be a Plugins table?
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Tags (
      rowid INTEGER PRIMARY KEY,
      run_id INTEGER,
      tag_id INTEGER NOT NULL,
      inserted_time DOUBLE,
      tag_name TEXT,
      display_name TEXT,
      plugin_name TEXT,
      plugin_data BLOB
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS TagIdIndex
    ON Tags (tag_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS
      TagRunNameIndex
    ON
      Tags (run_id, tag_name)
    WHERE
      run_id IS NOT NULL
      AND tag_name IS NOT NULL
  )sql"));

  // Runs are groups of Tags.
  //
  // Each Run usually represents a single attempt at training or testing
  // a TensorFlow model, with a given set of hyper-parameters, whose
  // summaries are written out to a single event logs directory with a
  // monotonic step counter.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   run_id: The Permanent ID of the Run. This has a 1:1 mapping
  //     with a SummaryWriter instance. If two writers spawn for a
  //     given (user_name, run_name, run_name) then each should
  //     allocate its own run_id and whichever writer puts it in the
  //     database last wins. The Tags / Tensors associated with the
  //     previous invocations will then enter limbo, where they may be
  //     accessible for certain operations, but should be garbage
  //     collected eventually.
  //   run_name: User-supplied string, unique across Experiment.
  //   experiment_id: Optional ID of associated Experiment.
  //   inserted_time: Float UNIX timestamp with µs precision. This is
  //     always the time the row was inserted into the database. It
  //     does not change.
  //   started_time: Float UNIX timestamp with µs precision. In the
  //     old summaries system that uses FileWriter, this is
  //     approximated as the first tf.Event.wall_time. In the new
  //     summaries system, it is the wall time of when summary writing
  //     started, from the perspective of whichever machine talks to
  //     the database. This field will be mutated if the run is
  //     restarted.
  //   finished_time: Float UNIX timestamp with µs precision of when
  //     SummaryWriter resource that created this run was destroyed.
  //     Once this value becomes non-NULL a Run and its Tags and
  //     Tensors should be regarded as immutable.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Runs (
      rowid INTEGER PRIMARY KEY,
      experiment_id INTEGER,
      run_id INTEGER NOT NULL,
      inserted_time REAL,
      started_time REAL,
      finished_time REAL,
      run_name TEXT
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS RunIdIndex
    ON Runs (run_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS RunNameIndex
    ON Runs (experiment_id, run_name)
    WHERE run_name IS NOT NULL
  )sql"));

  // Experiments are groups of Runs.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   user_id: Optional ID of associated User.
  //   experiment_id: The Permanent ID of the Experiment.
  //   experiment_name: User-supplied string, unique across User.
  //   inserted_time: Float UNIX timestamp with µs precision. This is
  //     always the time the row was inserted into the database. It
  //     does not change.
  //   started_time: Float UNIX timestamp with µs precision. This is
  //     the MIN(experiment.started_time, run.started_time) of each
  //     Run added to the database, including Runs which have since
  //     been overwritten.
  //   is_watching: A boolean indicating if someone is actively
  //     looking at this Experiment in the TensorBoard GUI. Tensor
  //     writers that do reservoir sampling can query this value to
  //     decide if they want the "keep last" behavior. This improves
  //     the performance of long running training while allowing low
  //     latency feedback in TensorBoard.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Experiments (
      rowid INTEGER PRIMARY KEY,
      user_id INTEGER,
      experiment_id INTEGER NOT NULL,
      inserted_time REAL,
      started_time REAL,
      is_watching INTEGER,
      experiment_name TEXT
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS ExperimentIdIndex
    ON Experiments (experiment_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS ExperimentNameIndex
    ON Experiments (user_id, experiment_name)
    WHERE experiment_name IS NOT NULL
  )sql"));

  // Users are people who love TensorBoard.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   user_id: The Permanent ID of the User.
  //   user_name: Unique user name.
  //   email: Optional unique email address.
  //   inserted_time: Float UNIX timestamp with µs precision. This is
  //     always the time the row was inserted into the database. It
  //     does not change.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Users (
      rowid INTEGER PRIMARY KEY,
      user_id INTEGER NOT NULL,
      inserted_time REAL,
      user_name TEXT,
      email TEXT
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserIdIndex
    ON Users (user_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserNameIndex
    ON Users (user_name)
    WHERE user_name IS NOT NULL
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserEmailIndex
    ON Users (email)
    WHERE email IS NOT NULL
  )sql"));

  // Graphs define how Tensors flowed in Runs.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   run_id: The Permanent ID of the associated Run. Only one Graph
  //     can be associated with a Run.
  //   graph_id: The Permanent ID of the Graph.
  //   inserted_time: Float UNIX timestamp with µs precision. This is
  //     always the wall time of when the row was inserted into the
  //     DB. It may be used as a hint for an archival job.
  //   graph_def: Contains the tf.GraphDef proto parts leftover which
  //     haven't been defined in SQL yet.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Graphs (
      rowid INTEGER PRIMARY KEY,
      run_id INTEGER,
      graph_id INTEGER NOT NULL,
      inserted_time REAL,
      graph_def BLOB
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS GraphIdIndex
    ON Graphs (graph_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS GraphRunIndex
    ON Graphs (run_id)
    WHERE run_id IS NOT NULL
  )sql"));

  // Nodes are the vertices in Graphs.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   graph_id: The Permanent ID of the associated Graph.
  //   node_id: ID for this node. This is more like a 0-index within
  //     the Graph. Please note indexes are allowed to be removed.
  //   node_name: Unique name for this Node within Graph. This is
  //     copied from the proto so it can be indexed. This is allowed
  //     to be NULL to save space on the index, in which case the
  //     node_def.name proto field must not be cleared.
  //   op: Copied from tf.NodeDef proto.
  //   device: Copied from tf.NodeDef proto.
  //   node_def: Contains the tf.NodeDef proto parts leftover which
  //     haven't been defined in SQL yet.
  //
  // TODO(jart): Make separate tables for op and device strings.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Nodes (
      rowid INTEGER PRIMARY KEY,
      graph_id INTEGER NOT NULL,
      node_id INTEGER NOT NULL,
      node_name TEXT,
      op TEXT,
      device TEXT,
      node_def BLOB
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeIdIndex
    ON Nodes (graph_id, node_id)
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeNameIndex
    ON Nodes (graph_id, node_name)
    WHERE node_name IS NOT NULL
  )sql"));

  // NodeInputs are directed edges between Nodes in Graphs.
  //
  // Fields:
  //   rowid: Ephemeral b-tree ID.
  //   graph_id: The Permanent ID of the associated Graph.
  //   node_id: Index of Node in question. This can be considered the
  //     'to' vertex.
  //   idx: Used for ordering inputs on a given Node.
  //   input_node_id: Nodes.node_id of the corresponding input node.
  //     This can be considered the 'from' vertex.
  //   input_node_idx: Since a Node can output multiple Tensors, this
  //     is the integer index of which of those outputs is our input.
  //     NULL is treated as 0.
  //   is_control: If non-zero, indicates this input is a controlled
  //     dependency, which means this isn't an edge through which
  //     tensors flow. NULL means 0.
  //
  // TODO(jart): Rename to NodeEdges.
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS NodeInputs (
      rowid INTEGER PRIMARY KEY,
      graph_id INTEGER NOT NULL,
      node_id INTEGER NOT NULL,
      idx INTEGER NOT NULL,
      input_node_id INTEGER NOT NULL,
      input_node_idx INTEGER,
      is_control INTEGER
    )
  )sql"));

  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeInputsIndex
    ON NodeInputs (graph_id, node_id, idx)
  )sql"));

  return s;
}

}  // namespace tensorflow
