/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/SourceMgr.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(filecheck_wrapper, m) {
  m.def("check", [](std::string input, std::string check) {
    llvm::FileCheckRequest fcr;
    llvm::FileCheck fc(fcr);
    llvm::SourceMgr SM = llvm::SourceMgr();
    SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(input),
                          llvm::SMLoc());
    SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(check),
                          llvm::SMLoc());
    fc.readCheckFile(SM, llvm::StringRef(check));
    return fc.checkInput(SM, llvm::StringRef(input));
  });
}
