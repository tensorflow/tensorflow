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

#include <limits.h>
#include <array>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class SqliteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    db_ = Sqlite::Open(":memory:").ValueOrDie();
    auto stmt = db_->Prepare("CREATE TABLE T (a BLOB, b BLOB)");
    TF_ASSERT_OK(stmt.StepAndReset());
  }
  std::shared_ptr<Sqlite> db_;
  bool is_done_;
};

TEST_F(SqliteTest, InsertAndSelectInt) {
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 3);
  stmt.BindInt(2, -7);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 123);
  stmt.BindInt(2, -123);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T ORDER BY b");
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
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindDouble(1, 6.28318530);
  stmt.BindDouble(2, 1.61803399);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(6.28318530, stmt.ColumnDouble(0));
  EXPECT_EQ(1.61803399, stmt.ColumnDouble(1));
  EXPECT_EQ(6, stmt.ColumnInt(0));
  EXPECT_EQ(1, stmt.ColumnInt(1));
}

TEST_F(SqliteTest, NulCharsInString) {
  string s;  // XXX: Want to write {2, '\0'} but not sure why not.
  s.append(static_cast<size_t>(2), '\0');
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T");
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
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(s, stmt.ColumnString(0));
  EXPECT_EQ(s, stmt.ColumnString(1));
}

TEST_F(SqliteTest, StepAndResetClearsBindings) {
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 1);
  stmt.BindInt(2, 123);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 2);
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT b FROM T ORDER BY a");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(123, stmt.ColumnInt(0));
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(SQLITE_NULL, stmt.ColumnType(0));
}

TEST_F(SqliteTest, CloseBeforeFinalizeFails) {
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  Status s = db_->Close();
  EXPECT_FALSE(s.ok());
}

// Rather than bothering to check the status code of creating a
// statement and every single bind call afterwards, SqliteStatement
// is designed to carry the first error state forward to Step().
TEST_F(SqliteTest, ErrorPuntingDoesNotReportLibraryAbuse) {
  auto stmt = db_->Prepare("lol cat");
  EXPECT_FALSE(stmt.status().ok());
  EXPECT_EQ(SQLITE_ERROR, stmt.error());
  stmt.BindInt(1, 1);
  stmt.BindInt(2, 2);
  Status s = stmt.Step(&is_done_);
  EXPECT_EQ(SQLITE_ERROR, stmt.error());  // first error of several
  EXPECT_FALSE(s.ok());
}

TEST_F(SqliteTest, SafeBind) {
  string s = "hello";
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlob(1, s);
  stmt.BindText(2, s);
  s.at(0) = 'y';
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ("hello", stmt.ColumnString(0));
  EXPECT_EQ("hello", stmt.ColumnString(1));
}

TEST_F(SqliteTest, UnsafeBind) {
  string s = "hello";
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindBlobUnsafe(1, s);
  stmt.BindTextUnsafe(2, s);
  s.at(0) = 'y';
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT a, b FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ("yello", stmt.ColumnString(0));
  EXPECT_EQ("yello", stmt.ColumnString(1));
}

TEST_F(SqliteTest, UnsafeColumn) {
  auto stmt = db_->Prepare("INSERT INTO T (a, b) VALUES (?, ?)");
  stmt.BindInt(1, 1);
  stmt.BindText(2, "hello");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt.BindInt(1, 2);
  stmt.BindText(2, "there");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT b FROM T ORDER BY a");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  const char* p = stmt.ColumnStringUnsafe(0);
  EXPECT_EQ('h', *p);
  TF_ASSERT_OK(stmt.Step(&is_done_));
  // This will actually happen, but it's not safe to test this behavior.
  // EXPECT_EQ('t', *p);
}

TEST_F(SqliteTest, NamedParameterBind) {
  auto stmt = db_->Prepare("INSERT INTO T (a) VALUES (:a)");
  stmt.BindText(":a", "lol");
  TF_ASSERT_OK(stmt.StepAndReset());
  stmt = db_->Prepare("SELECT COUNT(*) FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_EQ(1, stmt.ColumnInt(0));
  stmt = db_->Prepare("SELECT a FROM T");
  TF_ASSERT_OK(stmt.Step(&is_done_));
  EXPECT_FALSE(is_done_);
  EXPECT_EQ("lol", stmt.ColumnString(0));
}

TEST_F(SqliteTest, Statement_DefaultConstructor) {
  SqliteStatement stmt;
  EXPECT_FALSE(stmt);
  EXPECT_FALSE(stmt.StepAndReset().ok());
  stmt = db_->Prepare("INSERT INTO T (a) VALUES (1)");
  EXPECT_TRUE(stmt);
  EXPECT_TRUE(stmt.StepAndReset().ok());
}

TEST_F(SqliteTest, Statement_MoveConstructor) {
  SqliteStatement stmt{db_->Prepare("INSERT INTO T (a) VALUES (1)")};
  EXPECT_TRUE(stmt.StepAndReset().ok());
}

TEST_F(SqliteTest, Statement_MoveAssignment) {
  SqliteStatement stmt1 = db_->Prepare("INSERT INTO T (a) VALUES (1)");
  SqliteStatement stmt2;
  EXPECT_TRUE(stmt1.StepAndReset().ok());
  EXPECT_FALSE(stmt2.StepAndReset().ok());
  stmt2 = std::move(stmt1);
  EXPECT_TRUE(stmt2.StepAndReset().ok());
}

TEST_F(SqliteTest, PrepareFailed) {
  SqliteStatement s = db_->Prepare("SELECT");
  EXPECT_FALSE(s.status().ok());
  EXPECT_NE(string::npos, s.status().error_message().find("SELECT"));
}

TEST_F(SqliteTest, BindFailed) {
  SqliteStatement s = db_->Prepare("INSERT INTO T (a) VALUES (123)");
  EXPECT_TRUE(s.status().ok());
  EXPECT_EQ("", s.status().error_message());
  s.BindInt(1, 123);
  EXPECT_FALSE(s.status().ok());
  EXPECT_NE(string::npos,
            s.status().error_message().find("INSERT INTO T (a) VALUES (123)"));
}

}  // namespace
}  // namespace tensorflow
