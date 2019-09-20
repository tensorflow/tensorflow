#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H

#include "flatbuffers/base.h"
#include "flatbuffers/util.h"

// clang-format off

#ifdef __ANDROID__
  #include <android/log.h>
  #define TEST_OUTPUT_LINE(...) \
      __android_log_print(ANDROID_LOG_INFO, "FlatBuffers", __VA_ARGS__)
  #define FLATBUFFERS_NO_FILE_TESTS
#else
  #define TEST_OUTPUT_LINE(...) \
      { printf(__VA_ARGS__); printf("\n"); }
#endif

#define TEST_EQ(exp, val) TestEq(exp, val, #exp, __FILE__, __LINE__)
#define TEST_ASSERT(exp) TestEq(exp, true, #exp, __FILE__, __LINE__)
#define TEST_NOTNULL(exp) TestEq(exp == NULL, false, #exp, __FILE__, __LINE__)
#define TEST_EQ_STR(exp, val) TestEqStr(exp, val, #exp, __FILE__, __LINE__)

#ifdef _WIN32
  #define TEST_ASSERT_FUNC(exp) TestEq(exp, true, #exp, __FILE__, __LINE__, __FUNCTION__)
  #define TEST_EQ_FUNC(exp, val) TestEq(exp, val, #exp, __FILE__, __LINE__, __FUNCTION__)
#else
  #define TEST_ASSERT_FUNC(exp) TestEq(exp, true, #exp, __FILE__, __LINE__, __PRETTY_FUNCTION__)
  #define TEST_EQ_FUNC(exp, val) TestEq(exp, val, #exp, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#endif

// clang-format on

extern int testing_fails;

// Listener of TestFail, like 'gtest::OnTestPartResult' event handler.
// Called in TestFail after a failed assertion.
typedef bool (*TestFailEventListener)(const char *expval, const char *val,
                                      const char *exp, const char *file,
                                      int line, const char *func);

// Prepare test engine (MSVC assertion setup, etc).
// listener - this function will be notified on each TestFail call.
void InitTestEngine(TestFailEventListener listener = nullptr);

// Release all test-engine resources.
// Prints or schedule a debug report if all test passed.
// Returns 0 if all tests passed or 1 otherwise.
// Memory leak report: FLATBUFFERS_MEMORY_LEAK_TRACKING && _MSC_VER && _DEBUG.
int CloseTestEngine(bool force_report = false);

// Write captured state to a log and terminate test run.
void TestFail(const char *expval, const char *val, const char *exp,
              const char *file, int line, const char *func = 0);

void TestEqStr(const char *expval, const char *val, const char *exp,
               const char *file, int line);

template<typename T, typename U>
void TestEq(T expval, U val, const char *exp, const char *file, int line,
            const char *func = 0) {
  if (U(expval) != val) {
    TestFail(flatbuffers::NumToString(expval).c_str(),
             flatbuffers::NumToString(val).c_str(), exp, file, line, func);
  }
}

#endif  // !TEST_ASSERT_H
