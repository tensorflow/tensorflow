#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"

#include <string>

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/core/platform/test.h"

namespace {

TEST(AbslErrorReporterTest, LongStringDoesNotOverflow) {
  mlir::TFL::AbslErrorReporter reporter;
  tflite::ErrorReporter* base = &reporter;

  // Construct a string that would overflow the old fixed 1024-byte buffer.
  std::string long_input(5000, 'A');

  // Should not crash or overflow; return value is implementation-defined but
  // expected to be 0 on success per ErrorReporter contract in this codebase.
  EXPECT_EQ(base->Report("%s", long_input.c_str()), 0);
}

}  // namespace
