#include "tensorflow/core/public/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include <gtest/gtest.h>

namespace tensorflow {

TEST(Status, OK) {
  EXPECT_EQ(Status::OK().code(), error::OK);
  EXPECT_EQ(Status::OK().error_message(), "");
  EXPECT_OK(Status::OK());
  ASSERT_OK(Status::OK());
  EXPECT_EQ(Status::OK(), Status());
  Status s;
  EXPECT_TRUE(s.ok());
}

TEST(DeathStatus, CheckOK) {
  Status status(errors::InvalidArgument("Invalid"));
  ASSERT_DEATH(TF_CHECK_OK(status), "Invalid");
}

TEST(Status, Set) {
  Status status;
  status = Status(error::CANCELLED, "Error message");
  EXPECT_EQ(status.code(), error::CANCELLED);
  EXPECT_EQ(status.error_message(), "Error message");
}

TEST(Status, Copy) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Update) {
  Status s;
  s.Update(Status::OK());
  ASSERT_TRUE(s.ok());
  Status a(errors::InvalidArgument("Invalid"));
  s.Update(a);
  ASSERT_EQ(s.ToString(), a.ToString());
  Status b(errors::Internal("Internal"));
  s.Update(b);
  ASSERT_EQ(s.ToString(), a.ToString());
  s.Update(Status::OK());
  ASSERT_EQ(s.ToString(), a.ToString());
  ASSERT_FALSE(s.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(Status::OK(), Status()); }

TEST(Status, EqualsSame) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(errors::InvalidArgument("Invalid"));
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const Status a(errors::InvalidArgument("Invalid"));
  const Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::Internal("message"));
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::InvalidArgument("another"));
  ASSERT_NE(a, b);
}

}  // namespace tensorflow
