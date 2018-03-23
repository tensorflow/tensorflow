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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_FAKE_ENV_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_FAKE_ENV_H_

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace test {

/// Env implementation that stubs out the calls to read a file and time.
class FakeEnv : public EnvWrapper {
 public:
  enum EnvType {
    kGoogle,
    kGce,
    kLocal,
    kBad,
  };

  FakeEnv(EnvType env_type) : EnvWrapper(Env::Default()), env_type_(env_type) {}

  class FakeRandomAccessFile : public RandomAccessFile {
   public:
    FakeRandomAccessFile(EnvType env_type) : env_type_(env_type) {}

    Status Read(uint64 offset, size_t n, StringPiece* result,
                char* scratch) const override;

   private:
    EnvType env_type_;
  };

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override;

  uint64 NowSeconds() override { return now; }
  uint64 now = 10000;

 private:
  EnvType env_type_;
};

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FAKE_ENV_H_
