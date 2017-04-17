/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"

#include <string>

#include "tensorflow/core/lib/io/path.h"

#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

const char kSaxpyIRFile[] =
    "compiler/xla/service/gpu/llvm_gpu_backend/tests_data/saxpy.ll";

TEST(UtilsTest, TestLoadIRModule) {
  llvm::LLVMContext llvm_context;
  string test_srcdir = tensorflow::testing::TensorFlowSrcRoot();
  std::unique_ptr<llvm::Module> module = LoadIRModule(
      tensorflow::io::JoinPath(test_srcdir, kSaxpyIRFile), &llvm_context);
  // Sanity check that the module was loaded properly.
  ASSERT_NE(nullptr, module);
  ASSERT_NE(std::string::npos, module->getModuleIdentifier().find("saxpy.ll"));
  ASSERT_NE(nullptr, module->getFunction("cuda_saxpy"));
}

TEST(UtilsTest, TestReplaceFilenameExtension) {
  ASSERT_EQ(ReplaceFilenameExtension("baz.tx", "cc"), "baz.cc");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz.txt", "cc"), "/foo/baz.cc");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz.", "-nvptx.dummy"),
            "/foo/baz.-nvptx.dummy");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz", "cc"), "/foo/baz.cc");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
