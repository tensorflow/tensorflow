#include <algorithm>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/match.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {
namespace io {

static string Match(Env* env, const string& suffix_pattern) {
  std::vector<string> results;
  Status s = GetMatchingFiles(env, JoinPath(testing::TmpDir(), suffix_pattern),
                              &results);
  if (!s.ok()) {
    return s.ToString();
  } else {
    string r;
    std::sort(results.begin(), results.end());
    for (size_t i = 0; i < results.size(); i++) {
      strings::StrAppend(&r, (i > 0) ? "," : "", Basename(results[i]));
    }
    return r;
  }
}
TEST(GetMatchingFiles, Simple) {
  Env* env = Env::Default();
  EXPECT_EQ(Match(env, "thereisnosuchfile"), "");
  EXPECT_EQ(Match(env, "thereisnosuchfile*"), "");

  // Populate a few files
  EXPECT_OK(WriteStringToFile(Env::Default(),
                              JoinPath(testing::TmpDir(), "match-00"), ""));
  EXPECT_OK(WriteStringToFile(Env::Default(),
                              JoinPath(testing::TmpDir(), "match-0a"), ""));
  EXPECT_OK(WriteStringToFile(Env::Default(),
                              JoinPath(testing::TmpDir(), "match-01"), ""));
  EXPECT_OK(WriteStringToFile(Env::Default(),
                              JoinPath(testing::TmpDir(), "match-aaa"), ""));

  EXPECT_EQ(Match(env, "match-*"), "match-00,match-01,match-0a,match-aaa");
  EXPECT_EQ(Match(env, "match-0[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(env, "match-?[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(env, "match-?a*"), "match-0a,match-aaa");
  EXPECT_EQ(Match(env, "match-??"), "match-00,match-01,match-0a");
}

}  // namespace io
}  // namespace tensorflow
