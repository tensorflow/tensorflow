#include "tensorflow/core/lib/gtl/map_util.h"

#include <map>
#include <set>
#include <string>
#include "tensorflow/core/platform/port.h"

#include <gtest/gtest.h>

namespace tensorflow {

TEST(MapUtil, Find) {
  typedef std::map<string, string> Map;
  Map m;

  // Check that I can use a type that's implicitly convertible to the
  // key or value type, such as const char* -> string.
  EXPECT_EQ("", gtl::FindWithDefault(m, "foo", ""));
  m["foo"] = "bar";
  EXPECT_EQ("bar", gtl::FindWithDefault(m, "foo", ""));
  EXPECT_EQ("bar", *gtl::FindOrNull(m, "foo"));
  string str;
  EXPECT_TRUE(m.count("foo") > 0);
  EXPECT_EQ(m["foo"], "bar");
}

TEST(MapUtil, LookupOrInsert) {
  typedef std::map<string, string> Map;
  Map m;

  // Check that I can use a type that's implicitly convertible to the
  // key or value type, such as const char* -> string.
  EXPECT_EQ("xyz", gtl::LookupOrInsert(&m, "foo", "xyz"));
  EXPECT_EQ("xyz", gtl::LookupOrInsert(&m, "foo", "abc"));
}

TEST(MapUtil, InsertIfNotPresent) {
  // Set operations
  typedef std::set<int> Set;
  Set s;
  EXPECT_TRUE(gtl::InsertIfNotPresent(&s, 0));
  EXPECT_EQ(s.count(0), 1);
  EXPECT_FALSE(gtl::InsertIfNotPresent(&s, 0));
  EXPECT_EQ(s.count(0), 1);
}

}  // namespace tensorflow
