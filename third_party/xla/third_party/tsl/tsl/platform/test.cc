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

#include "tsl/platform/test.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "tsl/platform/logging.h"
#include "tsl/platform/net.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace testing {
namespace {

std::string GetEnvVarOrDie(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (!value) {
    LOG(FATAL) << "Failed to find environment variable:" << env_var;
  }
  return value;
}

}  // namespace

std::string TmpDir() { return GetEnvVarOrDie("TEST_TMPDIR"); }

int PickUnusedPortOrDie() { return internal::PickUnusedPortOrDie(); }

int RandomSeed() {
  const char* random_seed_str = std::getenv("TEST_RANDOM_SEED");
  int seed;
  if (random_seed_str && std::sscanf(random_seed_str, "%d", &seed) == 1) {
    return seed;
  }
  return 301;
}

std::string TensorFlowSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");

  return kIsOpenSource
             ? io::JoinPath(srcdir, workspace, "tensorflow")
             : io::JoinPath(srcdir, workspace, "third_party/tensorflow");
}

std::string XlaSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  const char* xla_path = "tensorflow/compiler/xla";

  return kIsOpenSource
             ? io::JoinPath(srcdir, workspace, xla_path)
             : io::JoinPath(srcdir, workspace, "third_party", xla_path);
}

std::string TslSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  const char* tsl_path = "tsl";

  return kIsOpenSource
             ? io::JoinPath(srcdir, workspace, tsl_path)
             : io::JoinPath(srcdir, workspace, "third_party", tsl_path);
}

}  // namespace testing
}  // namespace tsl
