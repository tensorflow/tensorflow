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
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static const char* const kPrefix = "ipfs://solarsystem";

// A file system that has Planets, Satellites and Sub Satellites. Sub satellites
// cannot have children further.
class InterPlanetaryFileSystem : public NullFileSystem {
 public:
  Status FileExists(const string& fname) override {
    string parsed_path;
    ParsePath(fname, &parsed_path);
    if (BodyExists(parsed_path)) {
      return Status::OK();
    }
    return Status(tensorflow::error::NOT_FOUND, "File does not exist");
  }

  // Adds the dir to the parent's children list and creates an entry for itself.
  Status CreateDir(const string& dirname) override {
    string parsed_path;
    ParsePath(dirname, &parsed_path);
    // If the directory already exists, throw an error.
    if (celestial_bodies_.find(parsed_path) != celestial_bodies_.end()) {
      return Status(tensorflow::error::ALREADY_EXISTS,
                    "dirname already exists.");
    }
    std::vector<string> split_path = str_util::Split(parsed_path, '/');
    // If the path is too long then we don't support it.
    if (split_path.size() > 3) {
      return Status(tensorflow::error::INVALID_ARGUMENT, "Bad dirname");
    }
    if (split_path.empty()) {
      return Status::OK();
    }
    if (split_path.size() == 1) {
      celestial_bodies_[""].insert(parsed_path);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    if (split_path.size() == 2) {
      if (!BodyExists(split_path[0])) {
        return Status(tensorflow::error::FAILED_PRECONDITION,
                      "Base dir not created");
      }
      celestial_bodies_[split_path[0]].insert(split_path[1]);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    if (split_path.size() == 3) {
      const string& parent_path = this->JoinPath(split_path[0], split_path[1]);
      if (!BodyExists(parent_path)) {
        return Status(tensorflow::error::FAILED_PRECONDITION,
                      "Base dir not created");
      }
      celestial_bodies_[parent_path].insert(split_path[2]);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Failed to create");
  }

  Status IsDirectory(const string& dirname) override {
    string parsed_path;
    ParsePath(dirname, &parsed_path);
    // Simulate evil_directory has bad permissions by throwing a LOG(FATAL)
    if (parsed_path == "evil_directory") {
      LOG(FATAL) << "evil_directory cannot be accessed";
    }
    std::vector<string> split_path = str_util::Split(parsed_path, '/');
    if (split_path.size() > 2) {
      return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
    }
    if (celestial_bodies_.find(parsed_path) != celestial_bodies_.end()) {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    TF_RETURN_IF_ERROR(IsDirectory(dir));
    string parsed_path;
    ParsePath(dir, &parsed_path);
    result->insert(result->begin(), celestial_bodies_[parsed_path].begin(),
                   celestial_bodies_[parsed_path].end());
    return Status::OK();
  }

 private:
  bool BodyExists(const string& name) {
    return celestial_bodies_.find(name) != celestial_bodies_.end();
  }

  void ParsePath(const string& name, string* parsed_path) {
    StringPiece scheme, host, path;
    this->ParseURI(name, &scheme, &host, &path);
    ASSERT_EQ(scheme, "ipfs");
    ASSERT_EQ(host, "solarsystem");
    absl::ConsumePrefix(&path, "/");
    *parsed_path = string(path);
  }

  std::map<string, std::set<string>> celestial_bodies_ = {
      std::pair<string, std::set<string>>(
          "", {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
               "Uranus", "Neptune"}),
      std::pair<string, std::set<string>>("Mercury", {}),
      std::pair<string, std::set<string>>("Venus", {}),
      std::pair<string, std::set<string>>("Earth", {"Moon"}),
      std::pair<string, std::set<string>>("Mars", {}),
      std::pair<string, std::set<string>>("Jupiter",
                                          {"Europa", "Io", "Ganymede"}),
      std::pair<string, std::set<string>>("Saturn", {}),
      std::pair<string, std::set<string>>("Uranus", {}),
      std::pair<string, std::set<string>>("Neptune", {}),
      std::pair<string, std::set<string>>("Earth/Moon", {}),
      std::pair<string, std::set<string>>("Jupiter/Europa", {}),
      std::pair<string, std::set<string>>("Jupiter/Io", {}),
      std::pair<string, std::set<string>>("Jupiter/Ganymede", {})};
};

// Returns all the matched entries as a comma separated string removing the
// common prefix of BaseDir().
string Match(InterPlanetaryFileSystem* ipfs, const string& suffix_pattern) {
  std::vector<string> results;
  Status s =
      ipfs->GetMatchingPaths(ipfs->JoinPath(kPrefix, suffix_pattern), &results);
  if (!s.ok()) {
    return s.ToString();
  } else {
    std::vector<StringPiece> trimmed_results;
    std::sort(results.begin(), results.end());
    for (const string& result : results) {
      StringPiece trimmed_result(result);
      EXPECT_TRUE(
          absl::ConsumePrefix(&trimmed_result, strings::StrCat(kPrefix, "/")));
      trimmed_results.push_back(trimmed_result);
    }
    return absl::StrJoin(trimmed_results, ",");
  }
}

TEST(InterPlanetaryFileSystemTest, IPFSMatch) {
  InterPlanetaryFileSystem ipfs;
  EXPECT_EQ(Match(&ipfs, "thereisnosuchfile"), "");
  EXPECT_EQ(Match(&ipfs, "*"),
            "Earth,Jupiter,Mars,Mercury,Neptune,Saturn,Uranus,Venus");
  // Returns Jupiter's moons.
  EXPECT_EQ(Match(&ipfs, "Jupiter/*"),
            "Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  // Returns Jupiter's and Earth's moons.
  EXPECT_EQ(Match(&ipfs, "*/*"),
            "Earth/Moon,Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "Planet0")));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "Planet1")));
  EXPECT_EQ(Match(&ipfs, "Planet[0-1]"), "Planet0,Planet1");
  EXPECT_EQ(Match(&ipfs, "Planet?"), "Planet0,Planet1");
}

TEST(InterPlanetaryFileSystemTest, MatchSimple) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-00")));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-0a")));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-01")));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-aaa")));

  EXPECT_EQ(Match(&ipfs, "match-*"), "match-00,match-01,match-0a,match-aaa");
  EXPECT_EQ(Match(&ipfs, "match-0[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(&ipfs, "match-?[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(&ipfs, "match-?a*"), "match-0a,match-aaa");
  EXPECT_EQ(Match(&ipfs, "match-??"), "match-00,match-01,match-0a");
}

// Create 2 directories abcd and evil_directory. Look for abcd and make sure
// that evil_directory isn't accessed.
TEST(InterPlanetaryFileSystemTest, MatchOnlyNeeded) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "abcd")));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "evil_directory")));

  EXPECT_EQ(Match(&ipfs, "abcd"), "abcd");
}

TEST(InterPlanetaryFileSystemTest, MatchDirectory) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-00/abc/x")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-0a/abc/x")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-01/abc/x")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-aaa/abc/x")));

  EXPECT_EQ(Match(&ipfs, "match-*/abc/x"),
            "match-00/abc/x,match-01/abc/x,match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-0[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?a*/abc/x"), "match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?[^a]/abc/x"), "match-00/abc/x,match-01/abc/x");
}

TEST(InterPlanetaryFileSystemTest, MatchMultipleWildcards) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-00/abc/00")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-00/abc/01")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-00/abc/09")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-01/abc/00")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-01/abc/04")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-01/abc/10")));
  TF_EXPECT_OK(
      ipfs.RecursivelyCreateDir(ipfs.JoinPath(kPrefix, "match-02/abc/00")));

  EXPECT_EQ(Match(&ipfs, "match-0[0-1]/abc/0[0-8]"),
            "match-00/abc/00,match-00/abc/01,match-01/abc/00,match-01/abc/04");
}

TEST(InterPlanetaryFileSystemTest, RecursivelyCreateAlreadyExistingDir) {
  InterPlanetaryFileSystem ipfs;
  const string dirname = ipfs.JoinPath(kPrefix, "match-00/abc/00");
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(dirname));
  // We no longer check for recursively creating the directory again because
  // `ipfs.IsDirectory` is badly implemented, fixing it will break other tests
  // in this suite and we already test creating the directory again in
  // env_test.cc as well as in the modular filesystem tests.
}

TEST(InterPlanetaryFileSystemTest, HasAtomicMove) {
  InterPlanetaryFileSystem ipfs;
  const string dirname = io::JoinPath(kPrefix, "match-00/abc/00");
  bool has_atomic_move;
  TF_EXPECT_OK(ipfs.HasAtomicMove(dirname, &has_atomic_move));
  EXPECT_EQ(has_atomic_move, true);
}

// A simple file system with a root directory and a single file underneath it.
class TestFileSystem : public NullFileSystem {
 public:
  // Only allow for a single root directory.
  Status IsDirectory(const string& dirname) override {
    if (dirname == "." || dirname.empty()) {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
  }

  // Simulating a FS with a root dir and a single file underneath it.
  Status GetChildren(const string& dir, std::vector<string>* result) override {
    if (dir == "." || dir.empty()) {
      result->push_back("test");
    }
    return Status::OK();
  }
};

// Making sure that ./<pattern> and <pattern> have the same result.
TEST(TestFileSystemTest, RootDirectory) {
  TestFileSystem fs;
  std::vector<string> results;
  auto ret = fs.GetMatchingPaths("./te*", &results);
  EXPECT_EQ(1, results.size());
  EXPECT_EQ("./test", results[0]);
  ret = fs.GetMatchingPaths("te*", &results);
  EXPECT_EQ(1, results.size());
  EXPECT_EQ("./test", results[0]);
}

}  // namespace tensorflow
