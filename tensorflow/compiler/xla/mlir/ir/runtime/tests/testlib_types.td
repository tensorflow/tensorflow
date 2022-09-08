// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_TESTLIB_TYPES
#define XLA_TESTLIB_TYPES

class TESTLIB_Type<string name, string typeMnemonic>
    : TypeDef<TestlibDialect, name> {
  let mnemonic = typeMnemonic;
}

// TODO(ezhulenev): Remove this type once all tests are migrated to the type
// defined below. This requires moving tests from `jitrt/cpp_tests` to XLA.
def TESTLIB_CustomArgType : TESTLIB_Type<"CustomArg", "custom_arg"> {
  let summary = "custom argument type";
  let description = [{
    Type for testing passing custom user-defined arguments to XLA executables.
  }];
}

def TESTLIB_ValueType : TESTLIB_Type<"Value", "value"> {
  let summary = "custom value type";
  let description = [{
    Custom type for testing passing custom user-defined types as XLA executable
    and XLA runtime custom calls arguments and results.
  }];
}

#endif // XLA_TESTLIB_TYPES
