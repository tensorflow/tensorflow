#ifndef TENSORFLOW_LIB_CORE_STATUS_TEST_UTIL_H_
#define TENSORFLOW_LIB_CORE_STATUS_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "tensorflow/core/public/status.h"

// Macros for testing the results of functions that return util::Status.

#define EXPECT_OK(statement) EXPECT_EQ(::tensorflow::Status::OK(), (statement))
#define ASSERT_OK(statement) ASSERT_EQ(::tensorflow::Status::OK(), (statement))

// There are no EXPECT_NOT_OK/ASSERT_NOT_OK macros since they would not
// provide much value (when they fail, they would just print the OK status
// which conveys no more information than EXPECT_FALSE(status.ok());
// If you want to check for particular errors, better alternatives are:
// EXPECT_EQ(::util::Status(...expected error...), status.StripMessage());
// EXPECT_THAT(status.ToString(), HasSubstr("expected error"));
// Also, see testing/lib/util/status_util.h.

#endif  // TENSORFLOW_LIB_CORE_STATUS_TEST_UTIL_H_
