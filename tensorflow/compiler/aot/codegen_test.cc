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

#include "tensorflow/compiler/aot/codegen.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/TargetSelect.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace {

using ::xla::cpu_function_runtime::BufferInfo;

void ExpectErrorContains(const Status& status, absl::string_view str) {
  EXPECT_NE(OkStatus(), status);
  EXPECT_TRUE(absl::StrContains(status.error_message(), str))
      << "expected error: " << status.error_message() << " to contain: " << str;
}

TEST(ValidateCppIdent, Simple) {
  TF_EXPECT_OK(ValidateCppIdent("a", ""));
  TF_EXPECT_OK(ValidateCppIdent("abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc123", ""));
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  TF_EXPECT_OK(ValidateCppIdent(ident, ""));

  ExpectErrorContains(ValidateCppIdent("", ""), "empty identifier");
  ExpectErrorContains(ValidateCppIdent(" ", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("0", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(".", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(":", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("a.", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
}

class ParseCppClassTest : public ::testing::Test {
 protected:
  void ExpectOK(const string& cpp_class, const string& want_class_name,
                const std::vector<string>& want_namespaces) {
    string class_name;
    std::vector<string> namespaces;
    TF_EXPECT_OK(ParseCppClass(cpp_class, &class_name, &namespaces));
    EXPECT_EQ(class_name, want_class_name);
    EXPECT_EQ(namespaces, want_namespaces);
  }

  void ExpectFail(const string& cpp_class) {
    string class_name;
    std::vector<string> namespaces;
    EXPECT_NE(ParseCppClass(cpp_class, &class_name, &namespaces), OkStatus())
        << cpp_class;
  }
};

TEST_F(ParseCppClassTest, ParseOK) {
  ExpectOK("MyClass", "MyClass", {});
  ExpectOK("_MyClass", "_MyClass", {});
  ExpectOK("a::MyClass", "MyClass", {"a"});
  ExpectOK("a::foo::MyClass", "MyClass", {"a", "foo"});
  ExpectOK("a::foo::b::MyClass", "MyClass", {"a", "foo", "b"});
  ExpectOK("a::foo::b::bar::MyClass", "MyClass", {"a", "foo", "b", "bar"});
  ExpectOK("foo::MyClass", "MyClass", {"foo"});
  ExpectOK("_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("_foo::_MyClass", "_MyClass", {"_foo"});
  ExpectOK("::foo::bar::MyClass", "MyClass", {"foo", "bar"});
  ExpectOK("::_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("::_foo::_MyClass", "_MyClass", {"_foo"});
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  ExpectOK(ident, ident, {});
  ExpectOK(ident + "::" + ident, ident, {ident});
  ExpectOK(ident + "::" + ident + "::" + ident, ident, {ident, ident});
}

TEST_F(ParseCppClassTest, ParseFail) {
  ExpectFail("");
  ExpectFail("::");
  ExpectFail("0");
  ExpectFail("a.b");
  ExpectFail("a:b");
  ExpectFail(":foo::bar");
  ExpectFail("good::.bad");
  ExpectFail("good:::bad");
  ExpectFail("good::bad::");
  ExpectFail("good::::bad");
  ExpectFail("::::bad");
  ExpectFail("good:: bad");
  ExpectFail("good::0bad");
}

static void CompareWithGoldenFile(
    const string& tensorflow_relative_golden_file_name,
    const string& expected_contents, bool ignore_cr) {
  // Get rid of all CR characters, we may be running under windows.
  string sanitized_expected_contents(expected_contents);
  if (ignore_cr) {
    sanitized_expected_contents.erase(
        std::remove(sanitized_expected_contents.begin(),
                    sanitized_expected_contents.end(), '\r'),
        sanitized_expected_contents.end());
  }

  // To update the golden file, flip update_golden to true and run the
  // following:
  // bazel test --test_strategy=local \
  //   "third_party/tensorflow/compiler/aot:codegen_test"
  const bool update_golden = false;
  string golden_file_name =
      GetDataDependencyFilepath(tensorflow_relative_golden_file_name);

  if (update_golden) {
    TF_EXPECT_OK(
        WriteStringToFile(Env::Default(), golden_file_name, expected_contents));
  }

  string golden_file_contents;
  TF_ASSERT_OK(ReadFileToString(Env::Default(), golden_file_name,
                                &golden_file_contents));
  if (ignore_cr) {
    golden_file_contents.erase(std::remove(golden_file_contents.begin(),
                                           golden_file_contents.end(), '\r'),
                               golden_file_contents.end());
  }
  EXPECT_EQ(golden_file_contents, expected_contents);
}

TEST(CodegenTest, Golden) {
  // Normally CpuCompiler::CpuCompiler does this, but in this test we've
  // bypassed the Cpu compiler so we have to do this manually.
#if defined(TF_LLVM_AARCH64_AVAILABLE)
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
#elif defined(TF_LLVM_AARCH32_AVAILABLE)
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmPrinter();
#elif defined(TF_LLVM_POWERPC_AVAILABLE)
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmPrinter();
#elif defined(TF_LLVM_S390X_AVAILABLE)
  LLVMInitializeSystemZTarget();
  LLVMInitializeSystemZTargetInfo();
  LLVMInitializeSystemZTargetMC();
  LLVMInitializeSystemZAsmPrinter();
#elif defined(TF_LLVM_X86_AVAILABLE)
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
#endif

  CodegenOpts opts;
  opts.class_name = "MyClass";
#if defined(TF_LLVM_AARCH64_AVAILABLE)
  opts.target_triple = "aarch64-linux-gnu";
#elif defined(TF_LLVM_ARM_AVAILABLE)
  opts.target_triple = "arm-linux-gnueabihf";
#elif defined(TF_LLVM_POWERPC_AVAILABLE)
  opts.target_triple = "powerpc64-linux-gnu";
#elif defined(TF_LLVM_S390X_AVAILABLE)
  opts.target_triple = "s390x-linux-gnu";
#elif defined(TF_LLVM_X86_AVAILABLE)
  opts.target_triple = "x86_64-pc-linux";
#endif
  opts.namespaces = {"foo", "bar"};
  opts.gen_name_to_index = true;
  opts.gen_program_shape = true;
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("feed0");
  feed->set_name("myfeed");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("feed1");
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("fetch0");
  fetch->set_name("myfetch");
  tf2xla::Variable* variable = config.add_variable();
  variable->set_node_name("myvar_readonly");
  variable->mutable_shape()->add_dim()->set_size(1);
  variable->set_type(DT_FLOAT);
  variable->set_readonly(true);
  tf2xla::Variable* variable2 = config.add_variable();
  variable2->set_node_name("myvar");
  variable2->mutable_shape()->add_dim()->set_size(1);
  variable2->set_type(DT_FLOAT);
  tf2xla::Variable* variable3 = config.add_variable();
  variable3->set_node_name("my/var");
  variable3->set_name("myvar2");
  variable3->mutable_shape()->add_dim()->set_size(5);
  variable3->set_type(DT_INT32);
  CompileResult compile_result;
  compile_result.aot.reset(new xla::cpu::CpuAotCompilationResult(
      {},
      {BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/8, /*param_number=*/0),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/1),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/2),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/3),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/4),
       BufferInfo::MakeTempBuffer(1), BufferInfo::MakeTempBuffer(120)},
      11, {}));
  compile_result.program_shape =
      xla::ShapeUtil::MakeProgramShape(
          {
              xla::ShapeUtil::MakeShape(xla::F32, {1, 2}),
              xla::ShapeUtil::MakeShape(xla::S64, {3, 4}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::S32, {5}),
          },
          xla::ShapeUtil::MakeTupleShape({
              xla::ShapeUtil::MakeShape(xla::U32, {5, 6}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::S32, {5}),
          }))
          .ToProto();
  compile_result.entry_point = "entry_point";
  compile_result.pointer_size = 8;

  MetadataResult metadata_result;
  TF_ASSERT_OK(GenerateMetadata(opts, compile_result, &metadata_result));

  // The other fields in metadata_result are tested as part of the generated
  // header test.

  // This specific golden test checks a binary file. It can potentially run into
  // issues due to ABIs not being stable, but has not so far.
  // If we see any ABI issues, we should reconsider this specific test case.
  CompareWithGoldenFile("tensorflow/compiler/aot/codegen_test_o.golden",
                        metadata_result.object_file_data, false);

  string header;
  TF_ASSERT_OK(
      GenerateHeader(opts, config, compile_result, metadata_result, &header));

  CompareWithGoldenFile("tensorflow/compiler/aot/codegen_test_h.golden", header,
                        true);
}
}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
