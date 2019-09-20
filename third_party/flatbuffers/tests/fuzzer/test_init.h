
#ifndef FUZZER_TEST_INIT_H_
#define FUZZER_TEST_INIT_H_

#include "fuzzer_assert.h"
#include "test_assert.h"

static_assert(__has_feature(memory_sanitizer) ||
                  __has_feature(address_sanitizer),
              "sanitizer disabled");

// Utility for test run.
struct OneTimeTestInit {
  // Declare trap for the Flatbuffers test engine.
  // This hook terminate program both in Debug and Release.
  static bool TestFailListener(const char *expval, const char *val,
                               const char *exp, const char *file, int line,
                               const char *func = 0) {
    (void)expval;
    (void)val;
    (void)exp;
    (void)file;
    (void)line;
    (void)func;
    // FLATBUFFERS_ASSERT redefined to be fully independent of the Flatbuffers
    // library implementation (see test_assert.h for details).
    fuzzer_assert_impl(false);  // terminate
    return false;
  }

  OneTimeTestInit() : has_locale_(false) {
    // Fuzzer test should be independent of the test engine implementation.
    // This hook will terminate test if TEST_EQ/TEST_ASSERT asserted.
    InitTestEngine(OneTimeTestInit::TestFailListener);

    // Read a locale for the test.
    if (flatbuffers::ReadEnvironmentVariable("FLATBUFFERS_TEST_LOCALE",
                                             &test_locale_)) {
      TEST_OUTPUT_LINE("The environment variable FLATBUFFERS_TEST_LOCALE=%s",
                       test_locale_.c_str());
      test_locale_ = flatbuffers::RemoveStringQuotes(test_locale_);
      has_locale_ = true;
    }
  }

  static const char *test_locale() {
    return one_time_init_.has_locale_ ? nullptr
                                      : one_time_init_.test_locale_.c_str();
  }

  bool has_locale_;
  std::string test_locale_;
  static OneTimeTestInit one_time_init_;
};

#endif  // !FUZZER_TEST_INIT_H_