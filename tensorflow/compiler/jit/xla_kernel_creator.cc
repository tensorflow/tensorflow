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
#include "tensorflow/compiler/jit/xla_kernel_creator.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_kernel_creator_util.h"
#include "tensorflow/core/common_runtime/function.h"

namespace tensorflow {

bool XlaKernelCreator::CanCreateKernel(const FunctionLibraryRuntime& flr,
                                       const NodeDef& node_def) const {
  return CanCreateXlaKernel(node_def);
}

Status XlaKernelCreator::CreateKernel(FunctionLibraryRuntime* flr,
                                      const NodeDef& node_def,
                                      std::unique_ptr<OpKernel>* kernel) const {
  return CreateXlaKernel(flr, node_def, kernel);
}

namespace {

bool RegisterLaunchOpCreator() {
  XlaKernelCreator* xla_kernel_creator = new XlaKernelCreator();
  RegisterDefaultCustomKernelCreator(xla_kernel_creator);
  return true;
}

static bool register_me = RegisterLaunchOpCreator();
static bool register_xla = [] {
  SetXlaIsEnabled();
  return true;
}();

}  // end namespace
}  // namespace tensorflow
