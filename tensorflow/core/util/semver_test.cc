/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/public/version.h"

#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

bool IsDotOrIdentifierChar(char c) {
  if (c == '.') return true;
  if (c == '-') return true;
  if (c >= 'A' && c <= 'Z') return true;
  if (c >= 'a' && c <= 'z') return true;
  if (c >= '0' && c <= '9') return true;
  return false;
}

bool ConsumeDotSeparatedIdentifiers(StringPiece* s, const string& prefix,
                                    StringPiece* val) {
  if (!str_util::ConsumePrefix(s, prefix)) return false;
  size_t i;
  for (i = 0; i < s->size() && IsDotOrIdentifierChar((*s)[i]); ++i) {
    // Intentionally empty
  }
  *val = StringPiece(s->data(), i);
  s->remove_prefix(i);
  return i > 0;
}

// Test that TF_VERSION_STRING follows semantic versioning.
TEST(SemverTest, VersionStringFollowsSemver) {
  // Poor approximation of the semver 2.0 specification at www.semver.org.  Feel
  // free to refine further (for example, check for leading 0s in numbers), but
  // avoid adding dependencies.
  uint64 major, minor, patch;
  StringPiece prerelease, metadata;
  StringPiece semver(TF_VERSION_STRING);

  ASSERT_TRUE(str_util::ConsumeLeadingDigits(&semver, &major));
  ASSERT_TRUE(str_util::ConsumePrefix(&semver, "."));
  ASSERT_TRUE(str_util::ConsumeLeadingDigits(&semver, &minor));
  ASSERT_TRUE(str_util::ConsumePrefix(&semver, "."));
  // Till 0.11.0rc2, the prerelease version was (incorrectly) not separated from
  // the patch version number. Let that slide.
  // Remove this when TF_VERSION_STRING moves beyond 0.11.0rc2.
  if (major == 0 && minor <= 11) {
    return;
  }
  if (str_util::ConsumePrefix(&semver, "head")) {
    ASSERT_TRUE(semver.empty());
    return;
  }
  ASSERT_TRUE(str_util::ConsumeLeadingDigits(&semver, &patch));
  if (semver.empty()) return;
  if (semver[0] == '-') {
    ASSERT_TRUE(ConsumeDotSeparatedIdentifiers(&semver, "-", &prerelease));
  }
  if (semver.empty()) return;
  if (semver[0] == '+') {
    ASSERT_TRUE(ConsumeDotSeparatedIdentifiers(&semver, "+", &metadata));
  }
  ASSERT_TRUE(semver.empty());
}
}  // namespace
}  // namespace tensorflow
