/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/lib/db/sqlite.h"

#include <array>
#include <climits>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class SqliteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK(Sqlite::Open(":memory:", SQLITE_OPEN_READWRITE, &db_));
    db_->PrepareOrDie("CREATE TABLE T (a BLOB, b BLOB)").StepAndResetOrDie();
  }

  void TearDown() override { db_->Unref(); }

  Sqlite* db_;
  bool is_done_;
};

TEST_F(SqliteTest, InsertAndSelectInt) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 3);
  stmt.BindInt(2, -7);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 123);
  stmt.BindInt(2, -123);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T ORDER BY b");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  ASSERT_FALSE(is_done_);
  EXPECT_EQ(123, stmt.ColumnInt(0));
  EXPECT_EQ(-123, stmt.ColumnInt(1));
  TF_ASSERT_OK(stmt.Step(&is_done_));
  ASSERT_FALSE(is_done_);
  EXPECT_EQ(3, stmt.ColumnInt(0));
  EXPECT_EQ(-7, stmt.ColumnInt(1));
  TF_ASSERT_OK(stmt.Step(&is_done_));
  ASSERT_TRUE(is_done_);
}

TEST_F(SqliteTest, InsertAndSelectDouble) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindDouble(1, 6.28318530);
  stmt.BindDouble(2, 1.61803399);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(6.28318530, stmt.ColumnDouble(0));
  EXPECT_EQ(1.61803399, stmt.ColumnDouble(1));
  EXPECT_EQ(6, stmt.ColumnInt(0));
  EXPECT_EQ(1, stmt.ColumnInt(1));
}

#ifdef DSQLITE_ENABLE_JSON1
TEST_F(SqliteTest, Json1Extension) {
  string s1 = "{\"key\": 42}";
  string s2 = "{\"key\": \"value\"}";
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindText(1, s1);
  stmt.BindText(2, s2);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT json_extract(a, '$.key'), json_extract(b, '$.key') FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(42, stmt.ColumnInt(0));
  EXPECT_EQ("value", stmt.ColumnString(1));
}
#endif //DSQLITE_ENABLE_JSON1

TEST_F(SqliteTest, NulCharsInString) {
  string s;  // XXX: Want to write {2, '\0'} but not sure why not.
  s.append(static_cast<size_t>(2), '\0');
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(2, stmt.ColumnSize(0));
  EXPECT_EQ(2, stmt.ColumnString(0).size());
  EXPECT_EQ('\0', stmt.ColumnString(0).at(0));
  EXPECT_EQ('\0', stmt.ColumnString(0).at(1));
  EXPECT_EQ(2, stmt.ColumnSize(1));
  EXPECT_EQ(2, stmt.ColumnString(1).size());
  EXPECT_EQ('\0', stmt.ColumnString(1).at(0));
  EXPECT_EQ('\0', stmt.ColumnString(1).at(1));
}

TEST_F(SqliteTest, Unicode) {
  string s = "要依法治国是赞美那些谁是公义的和惩罚恶人。 - 韩非";
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(s, stmt.ColumnString(0));
  EXPECT_EQ(s, stmt.ColumnString(1));
}

TEST_F(SqliteTest, StepAndResetClearsBindings) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 1);
  stmt.BindInt(2, 123);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 2);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT b FROM T ORDER BY a");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(123, stmt.ColumnInt(0));
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(SQLITE_NULL, stmt.ColumnType(0));
}

TEST_F(SqliteTest, SafeBind) {
  string s = "hello";
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  s.at(0) = 'y';
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ("hello", stmt.ColumnString(0));
  EXPECT_EQ("hello", stmt.ColumnString(1));
}

TEST_F(SqliteTest, UnsafeBind) {
  string s = "hello";
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlobUnsafe(1, s);
  stmt.BindTextUnsafe(2, s);
  s.at(0) = 'y';
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ("yello", stmt.ColumnString(0));
  EXPECT_EQ("yello", stmt.ColumnString(1));
}

TEST_F(SqliteTest, UnsafeColumn) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 1);
  stmt.BindText(2, "hello");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 2);
  stmt.BindText(2, "there");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT b FROM T ORDER BY a");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  StringPiece p = stmt.ColumnStringUnsafe(0);
  EXPECT_EQ('h', *p.data());
  TF_ASSERT_OK(stmt.Step(&is_done_));
  // This will actually happen, but it's not safe to test this behavior.
  // EXPECT_EQ('t', *p.data());
}

TEST_F(SqliteTest, NamedParameterBind) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a) VALUES (:a)");
  stmt.BindText(":a", "lol");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->PrepareOrDie("SELECT COUNT(*) FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(1, stmt.ColumnInt(0));
  stmt = db_->PrepareOrDie("SELECT a FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_FALSE(is_done_);
  EXPECT_EQ("lol", stmt.ColumnString(0));
}

TEST_F(SqliteTest, Statement_DefaultConstructor) {
  SqliteStatement stmt;
  EXPECT_FALSE(stmt);
  stmt = db_->PrepareOrDie("INSERT INTO T (a) VALUES (1)");
  EXPECT_TRUE(stmt);
  EXPECT_TRUE(stmt.StepAndReset().ok());
}

TEST_F(SqliteTest, Statement_MoveConstructor) {
  SqliteStatement stmt{db_->PrepareOrDie("INSERT INTO T (a) VALUES (1)")};
  EXPECT_TRUE(stmt.StepAndReset().ok());
}

TEST_F(SqliteTest, Statement_MoveAssignment) {
  SqliteStatement stmt1 = db_->PrepareOrDie("INSERT INTO T (a) VALUES (1)");
  SqliteStatement stmt2;
  EXPECT_TRUE(stmt1.StepAndReset().ok());
  EXPECT_FALSE(stmt2);
  stmt2 = std::move(stmt1);
  EXPECT_TRUE(stmt2.StepAndReset().ok());
}

TEST_F(SqliteTest, PrepareFailed) {
  SqliteLock lock(*db_);
  SqliteStatement stmt;
  Status s = db_->Prepare("SELECT", &stmt);
  ASSERT_FALSE(s.ok());
  EXPECT_NE(string::npos, s.message().find("SELECT"));
  EXPECT_EQ(SQLITE_ERROR, db_->errcode());
}

TEST_F(SqliteTest, BindFailed) {
  auto stmt = db_->PrepareOrDie("INSERT INTO T (a) VALUES (123)");
  stmt.BindInt(1, 123);
  Status s = stmt.StepOnce();
  EXPECT_NE(string::npos, s.message().find("INSERT INTO T (a) VALUES (123)"))
      << s.message();
}

TEST_F(SqliteTest, SnappyExtension) {
  auto stmt = db_->PrepareOrDie("SELECT UNSNAP(SNAP(?))");
  stmt.BindText(1, "hello");
  EXPECT_EQ("hello", stmt.StepOnceOrDie().ColumnString(0));
}

TEST_F(SqliteTest, SnappyBinaryCompatibility) {
  EXPECT_EQ(
      "today is the end of the republic",
      db_->PrepareOrDie("SELECT UNSNAP(X'03207C746F6461792069732074686520656E64"
                        "206F66207468652072657075626C6963')")
          .StepOnceOrDie()
          .ColumnString(0));
}

TEST(SqliteOpenTest, CloseConnectionBeforeStatement_KeepsConnectionOpen) {
  Sqlite* db;
  TF_ASSERT_OK(Sqlite::Open(":memory:", SQLITE_OPEN_READWRITE, &db));
  SqliteStatement stmt = db->PrepareOrDie("SELECT ? + ?");
  db->Unref();
  stmt.BindInt(1, 7);
  stmt.BindInt(2, 3);
  EXPECT_EQ(10, stmt.StepOnceOrDie().ColumnInt(0));
}

TEST_F(SqliteTest, TransactionRollback) {
  {
    SqliteTransaction txn(*db_);
    auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
    stmt.BindDouble(1, 6.28318530);
    stmt.BindDouble(2, 1.61803399);
    TF_ASSERT_OK(stmt.StepAndReset());
  }
  EXPECT_EQ(
      0,
      db_->PrepareOrDie("SELECT COUNT(*) FROM T").StepOnceOrDie().ColumnInt(0));
}

TEST_F(SqliteTest, TransactionCommit) {
  {
    SqliteTransaction txn(*db_);
    auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
    stmt.BindDouble(1, 6.28318530);
    stmt.BindDouble(2, 1.61803399);
    TF_ASSERT_OK(stmt.StepAndReset());
    TF_ASSERT_OK(txn.Commit());
  }
  EXPECT_EQ(
      1,
      db_->PrepareOrDie("SELECT COUNT(*) FROM T").StepOnceOrDie().ColumnInt(0));
}

TEST_F(SqliteTest, TransactionCommitMultipleTimes) {
  {
    SqliteTransaction txn(*db_);
    auto stmt = db_->PrepareOrDie("INSERT INTO T (a, b) VALUES (?, ?)");
    stmt.BindDouble(1, 6.28318530);
    stmt.BindDouble(2, 1.61803399);
    TF_ASSERT_OK(stmt.StepAndReset());
    TF_ASSERT_OK(txn.Commit());
    stmt.BindDouble(1, 6.28318530);
    stmt.BindDouble(2, 1.61803399);
    TF_ASSERT_OK(stmt.StepAndReset());
    TF_ASSERT_OK(txn.Commit());
  }
  EXPECT_EQ(
      2,
      db_->PrepareOrDie("SELECT COUNT(*) FROM T").StepOnceOrDie().ColumnInt(0));
}

}  // namespace
}  // namespace tensorflow
