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

#include "tensorflow/core/platform/file_system.h"

#include <sys/stat.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class InterPlanetaryFileSystem : public NullFileSystem {
 public:
  Status IsDirectory(const string& dirname) override {
    if (dirname == "ipfs://solarsystem" ||
        dirname == "ipfs://solarsystem/Earth" ||
        dirname == "ipfs://solarsystem/Jupiter") {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    std::vector<string> celestial_bodies;
    if (dir == "ipfs://solarsystem") {
      celestial_bodies = {"Mercury",  "Venus",   "Earth",  "Mars",
                          "Jupiter",  "Saturn",  "Uranus", "Neptune",
                          ".PlanetX", "Planet0", "Planet1"};

    } else if (dir == "ipfs://solarsystem/Earth") {
      celestial_bodies = {"Moon"};
    } else if (dir == "ipfs://solarsystem/Jupiter") {
      celestial_bodies = {"Europa", "Io", "Ganymede"};
    }
    result->insert(result->end(), celestial_bodies.begin(),
                   celestial_bodies.end());
    return Status::OK();
  }
};

// Returns all the matched entries as a comma separated string removing the
// common prefix of BaseDir().
string Match(const string& base_dir, const string& suffix_pattern) {
  InterPlanetaryFileSystem fs;
  std::vector<string> results;
  Status s =
      fs.GetMatchingPaths(io::JoinPath(base_dir, suffix_pattern), &results);
  if (!s.ok()) {
    return s.ToString();
  } else {
    std::vector<StringPiece> trimmed_results;
    std::sort(results.begin(), results.end());
    for (const string& result : results) {
      StringPiece trimmed_result(result);
      EXPECT_TRUE(trimmed_result.Consume(base_dir + "/"));
      trimmed_results.push_back(trimmed_result);
    }
    return str_util::Join(trimmed_results, ",");
  }
}

TEST(TestFileSystem, IPFSMatch) {
  // Make sure we only get the 11 planets and not all their children.
  EXPECT_EQ(Match("ipfs://solarsystem", "*"),
            ".PlanetX,Earth,Jupiter,Mars,Mercury,Neptune,Planet0,Planet1,"
            "Saturn,Uranus,Venus");
  // Returns Jupiter's moons.
  EXPECT_EQ(Match("ipfs://solarsystem", "Jupiter/*"),
            "Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  // Returns Jupiter's and Earth's moons.
  EXPECT_EQ(Match("ipfs://solarsystem", "*/*"),
            "Earth/Moon,Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  EXPECT_EQ(Match("ipfs://solarsystem", "Planet[0-1]"), "Planet0,Planet1");
}

}  // namespace tensorflow
