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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

const float kTolerance = 1e-5;

Tensor MakeScalarInt64(int64 x) {
  Tensor t(DT_INT64, TensorShape({}));
  t.scalar<int64>()() = x;
  return t;
}

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {}
  void AdvanceByMillis(const uint64 millis) { current_millis_ += millis; }
  uint64 NowMicros() override { return current_millis_ * 1000; }
  uint64 NowSeconds() override { return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

class SummaryDbWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK(Sqlite::Open(":memory:", SQLITE_OPEN_READWRITE, &db_));
  }

  void TearDown() override {
    if (writer_ != nullptr) {
      writer_->Unref();
      writer_ = nullptr;
    }
    db_->Unref();
    db_ = nullptr;
  }

  int64 QueryInt(const string& sql) {
    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return -1;
    }
    return stmt.ColumnInt(0);
  }

  double QueryDouble(const string& sql) {
    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return -1;
    }
    return stmt.ColumnDouble(0);
  }

  string QueryString(const string& sql) {
    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return "MISSINGNO";
    }
    return stmt.ColumnString(0);
  }

  FakeClockEnv env_;
  Sqlite* db_ = nullptr;
  SummaryWriterInterface* writer_ = nullptr;
};

TEST_F(SummaryDbWriterTest, NothingWritten_NoRowsCreated) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  TF_ASSERT_OK(writer_->Flush());
  writer_->Unref();
  writer_ = nullptr;
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Ids"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Users"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, TensorsWritten_RowsGetInitialized) {
  SummaryMetadata metadata;
  metadata.set_display_name("display_name");
  metadata.set_summary_description("description");
  metadata.mutable_plugin_data()->set_plugin_name("plugin_name");
  metadata.mutable_plugin_data()->set_content("plugin_data");
  SummaryMetadata metadata_nope;
  metadata_nope.set_display_name("nope");
  metadata_nope.set_summary_description("nope");
  metadata_nope.mutable_plugin_data()->set_plugin_name("nope");
  metadata_nope.mutable_plugin_data()->set_content("nope");
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(2, MakeScalarInt64(314LL), "taggy",
                                    metadata_nope.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());

  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Users"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(2LL, QueryInt("SELECT COUNT(*) FROM Tensors"));

  int64 user_id = QueryInt("SELECT user_id FROM Users");
  int64 experiment_id = QueryInt("SELECT experiment_id FROM Experiments");
  int64 run_id = QueryInt("SELECT run_id FROM Runs");
  int64 tag_id = QueryInt("SELECT tag_id FROM Tags");
  EXPECT_LT(0LL, user_id);
  EXPECT_LT(0LL, experiment_id);
  EXPECT_LT(0LL, run_id);
  EXPECT_LT(0LL, tag_id);

  EXPECT_EQ("jart", QueryString("SELECT user_name FROM Users"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Users"));

  EXPECT_EQ(user_id, QueryInt("SELECT user_id FROM Experiments"));
  EXPECT_EQ("mad-science",
            QueryString("SELECT experiment_name FROM Experiments"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Experiments"));

  EXPECT_EQ(experiment_id, QueryInt("SELECT experiment_id FROM Runs"));
  EXPECT_EQ("train", QueryString("SELECT run_name FROM Runs"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Runs"));

  EXPECT_EQ(run_id, QueryInt("SELECT run_id FROM Tags"));
  EXPECT_EQ("taggy", QueryString("SELECT tag_name FROM Tags"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Tags"));

  EXPECT_EQ("display_name", QueryString("SELECT display_name FROM Tags"));
  EXPECT_EQ("plugin_name", QueryString("SELECT plugin_name FROM Tags"));
  EXPECT_EQ("plugin_data", QueryString("SELECT plugin_data FROM Tags"));
  EXPECT_EQ("description", QueryString("SELECT description FROM Descriptions"));

  EXPECT_EQ(tag_id, QueryInt("SELECT tag_id FROM Tensors WHERE step = 1"));
  EXPECT_EQ(0.023,
            QueryDouble("SELECT computed_time FROM Tensors WHERE step = 1"));
  EXPECT_FALSE(
      QueryString("SELECT tensor FROM Tensors WHERE step = 1").empty());

  EXPECT_EQ(tag_id, QueryInt("SELECT tag_id FROM Tensors WHERE step = 2"));
  EXPECT_EQ(0.046,
            QueryDouble("SELECT computed_time FROM Tensors WHERE step = 2"));
  EXPECT_FALSE(
      QueryString("SELECT tensor FROM Tensors WHERE step = 2").empty());
}

TEST_F(SummaryDbWriterTest, EmptyParentNames_NoParentsCreated) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy", ""));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Users"));
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, WriteEvent_Scalar) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  std::unique_ptr<Event> e{new Event};
  e->set_step(7);
  e->set_wall_time(123.456);
  Summary::Value* s = e->mutable_summary()->add_value();
  s->set_tag("π");
  s->set_simple_value(3.14f);
  s = e->mutable_summary()->add_value();
  s->set_tag("φ");
  s->set_simple_value(1.61f);
  TF_ASSERT_OK(writer_->WriteEvent(std::move(e)));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(2LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(2LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
  int64 tag1_id = QueryInt("SELECT tag_id FROM Tags WHERE tag_name = 'π'");
  int64 tag2_id = QueryInt("SELECT tag_id FROM Tags WHERE tag_name = 'φ'");
  EXPECT_GT(tag1_id, 0LL);
  EXPECT_GT(tag2_id, 0LL);
  EXPECT_EQ(123.456, QueryDouble(strings::StrCat(
                         "SELECT computed_time FROM Tensors WHERE tag_id = ",
                         tag1_id, " AND step = 7")));
  EXPECT_EQ(123.456, QueryDouble(strings::StrCat(
                         "SELECT computed_time FROM Tensors WHERE tag_id = ",
                         tag2_id, " AND step = 7")));
  EXPECT_NEAR(3.14,
              QueryDouble(strings::StrCat(
                  "SELECT tensor FROM Tensors WHERE tag_id = ", tag1_id,
                  " AND step = 7")),
              kTolerance);  // Summary::simple_value is float
  EXPECT_NEAR(1.61,
              QueryDouble(strings::StrCat(
                  "SELECT tensor FROM Tensors WHERE tag_id = ", tag2_id,
                  " AND step = 7")),
              kTolerance);
}

TEST_F(SummaryDbWriterTest, WriteGraph) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "R", "", &env_, &writer_));
  env_.AdvanceByMillis(23);
  GraphDef graph;
  NodeDef* node = graph.add_node();
  node->set_name("x");
  node->set_op("Placeholder");
  node = graph.add_node();
  node->set_name("y");
  node->set_op("Placeholder");
  node = graph.add_node();
  node->set_name("z");
  node->set_op("Love");
  node = graph.add_node();
  node->set_name("+");
  node->set_op("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_input("^z");
  node->set_device("tpu/lol");
  std::unique_ptr<Event> e{new Event};
  graph.SerializeToString(e->mutable_graph_def());
  TF_ASSERT_OK(writer_->WriteEvent(std::move(e)));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Graphs"));
  ASSERT_EQ(4LL, QueryInt("SELECT COUNT(*) FROM Nodes"));
  ASSERT_EQ(3LL, QueryInt("SELECT COUNT(*) FROM NodeInputs"));

  int64 graph_id = QueryInt("SELECT graph_id FROM Graphs");
  EXPECT_GT(graph_id, 0LL);
  EXPECT_EQ(graph_id, QueryInt("SELECT graph_id FROM Runs"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Graphs"));
  EXPECT_FALSE(QueryString("SELECT graph_def FROM Graphs").empty());

  EXPECT_EQ("x", QueryString("SELECT node_name FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("y", QueryString("SELECT node_name FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("z", QueryString("SELECT node_name FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("+", QueryString("SELECT node_name FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ("Placeholder",
            QueryString("SELECT op FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("Placeholder",
            QueryString("SELECT op FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("Love", QueryString("SELECT op FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("Add", QueryString("SELECT op FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("tpu/lol",
            QueryString("SELECT device FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(0LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(1LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(2LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(0LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(0LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(1LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 2"));
}

TEST_F(SummaryDbWriterTest, WriteScalarInt32_CoercesToInt64) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  Tensor t(DT_INT32, {});
  t.scalar<int32>()() = -17;
  TF_ASSERT_OK(writer_->WriteScalar(1, t, "t"));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(-17LL, QueryInt("SELECT tensor FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, WriteScalarInt8_CoercesToInt64) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  Tensor t(DT_INT8, {});
  t.scalar<int8>()() = static_cast<int8>(-17);
  TF_ASSERT_OK(writer_->WriteScalar(1, t, "t"));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(-17LL, QueryInt("SELECT tensor FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, WriteScalarUint8_CoercesToInt64) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  Tensor t(DT_UINT8, {});
  t.scalar<uint8>()() = static_cast<uint8>(254);
  TF_ASSERT_OK(writer_->WriteScalar(1, t, "t"));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(254LL, QueryInt("SELECT tensor FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, UsesIdsTable) {
  SummaryMetadata metadata;
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(4LL, QueryInt("SELECT COUNT(*) FROM Ids"));
  EXPECT_EQ(4LL, QueryInt(strings::StrCat(
                     "SELECT COUNT(*) FROM Ids WHERE id IN (",
                     QueryInt("SELECT user_id FROM Users"), ", ",
                     QueryInt("SELECT experiment_id FROM Experiments"), ", ",
                     QueryInt("SELECT run_id FROM Runs"), ", ",
                     QueryInt("SELECT tag_id FROM Tags"), ")")));
}

TEST_F(SummaryDbWriterTest, SetsRunFinishedTime) {
  SummaryMetadata metadata;
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(0.023, QueryDouble("SELECT started_time FROM Runs"));
  ASSERT_EQ(0.0, QueryDouble("SELECT finished_time FROM Runs"));
  env_.AdvanceByMillis(23);
  writer_->Unref();
  writer_ = nullptr;
  ASSERT_EQ(0.023, QueryDouble("SELECT started_time FROM Runs"));
  ASSERT_EQ(0.046, QueryDouble("SELECT finished_time FROM Runs"));
}

}  // namespace
}  // namespace tensorflow
