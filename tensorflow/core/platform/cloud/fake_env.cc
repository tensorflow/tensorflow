/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/fake_env.h"

namespace tensorflow {
namespace test {

Status FakeEnv::FakeRandomAccessFile::Read(uint64 offset, size_t n,
                                           StringPiece* result,
                                           char* scratch) const {
  CHECK_EQ(offset, 0);
  CHECK_EQ(n, 256);
  Status s;
  string platform;
  switch (env_type_) {
    case kGoogle: {
      platform = "Google\n  ";
      s = errors::OutOfRange("");
      break;
    }
    case kGce: {
      platform = "  Google Compute Engine\n  ";
      s = errors::OutOfRange("");
      break;
    }
    case kLocal: {
      platform = "HP Linux Workstation";
      s = Status::OK();
      break;
    }
    case kBad: {
      platform = "";
      s = errors::Internal("Expected");
      break;
    }
  }
  strncpy(scratch, platform.data(), strlen(platform.data()));
  *result = StringPiece(scratch, platform.length());
  return s;
}

Status FakeEnv::NewRandomAccessFile(const string& fname,
                                    std::unique_ptr<RandomAccessFile>* result) {
  result->reset(new FakeRandomAccessFile(env_type_));
  return Status::OK();
}

}  // namespace test
}  // namespace tensorflow
