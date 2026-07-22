/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"

// Raw-bytes fuzzer for tensorflow::checkpoint::CheckpointReader -- the
// C++ entry point exercised by Python's tf.train.load_checkpoint().
//
// A V2 checkpoint consists of two files that share a prefix:
//   <prefix>.index                     -- SSTable index
//   <prefix>.data-00000-of-00001       -- tensor data payload
//
// The fuzzer splits its byte input in half: the first half becomes the
// .index file, the second becomes the .data-00000-of-00001 file. The
// CheckpointReader is then constructed on the shared prefix, which
// exercises both SSTable parsing and BundleEntryProto deserialization
// in BuildV2VarMaps().
//
// The existing //tensorflow/security/fuzzing/cc:checkpoint_reader_fuzz
// takes a structured CheckpointReaderFuzzInput proto, pre-serializes
// both files via TableBuilder, and has known build issues that remove
// it from OSS-Fuzz target set. This raw-bytes harness is complementary:
// it covers the bytes-on-disk path directly (crashes found here are
// reproducible with the crashing .index/.data pair).

namespace {

void WriteFile(const std::string& path, std::string_view data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return;
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void FuzzLoadCheckpoint(std::string_view data) {
  if (data.size() < 2) return;

  // Split input: first half -> .index, second half -> .data-00000-of-00001.
  const size_t split = data.size() / 2;
  const std::string_view index_bytes = data.substr(0, split);
  const std::string_view data_bytes = data.substr(split);

  // Build a fresh temp dir per iteration so earlier iterations' files
  // don't influence this one. tmpnam would trigger -Wdeprecated; use
  // mkdtemp.
  char tmpl[] = "/tmp/tf_ckpt_fuzz_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  if (dir == nullptr) return;

  const std::string prefix = std::string(dir) + "/ckpt";
  WriteFile(prefix + ".index", index_bytes);
  WriteFile(prefix + ".data-00000-of-00001", data_bytes);

  {
    TF_Status* status = TF_NewStatus();
    auto reader = std::make_unique<
        tensorflow::checkpoint::CheckpointReader>(prefix, status);
    // CheckpointReader construction runs BuildV2VarMaps, which is where
    // the bugs tend to live; we don't need any further calls.
    TF_DeleteStatus(status);
  }

  // Best-effort cleanup. Ignore errors.
  std::remove((prefix + ".index").c_str());
  std::remove((prefix + ".data-00000-of-00001").c_str());
  std::remove(dir);
}
FUZZ_TEST(CC_FUZZING, FuzzLoadCheckpoint);

}  // namespace
