/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/eager/process_function_library_runtime.h"

#include <iterator>
#include <memory>
#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace eager {

#if !defined(IS_MOBILE_PLATFORM)
void EagerProcessFunctionLibraryRuntime::RunRemoteDevice(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle local_handle, const InternalArgsView& args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  if (!rets->empty()) {
    done(
        errors::Unimplemented("Remote outputs are not supported by "
                              "EagerClusterFunctionLibraryRuntime yet."));
    return;
  }
  if (!args.local_args.empty()) {
    done(
        errors::Unimplemented("Local inputs are not by supported by "
                              "EagerClusterFunctionLibraryRuntime."));
    return;
  }
  if (args.remote_args == nullptr) {
    done(
        errors::Internal("EagerClusterFunctionLibraryRuntime: remote_args "
                         "should never be null."));
    return;
  }
  parent_->Run(opts, local_handle, args.remote_args, std::move(done));
}

void EagerProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, const FunctionArgsInterface& args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  if (!args.HasRemoteInputs()) {
    return ProcessFunctionLibraryRuntime::Run(opts, handle, args, rets,
                                              std::move(done));
  }
  auto* cleanup_items = new std::vector<std::unique_ptr<CleanUpItem>>;
  done = ApplyCleanUpToDoneCallback(cleanup_items, done);

  auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                    InternalArgs* comp_args) -> Status {
    for (int i = 0; i < comp_data.arg_indices_.size(); ++i) {
      const int index = comp_data.arg_indices_.at(i);
      Tensor tensor;
      if (args.GetLocalArg(index, &tensor).ok()) {
        comp_args->local_args.push_back(std::move(tensor));
      } else {
        RemoteTensorHandle remote_handle;
        TF_RETURN_IF_ERROR(args.GetRemoteArg(index, &remote_handle));
        comp_args->remote_args.push_back(std::move(remote_handle));
      }
    }
    return Status::OK();
  };
  return RunMultiDevice(opts, handle, rets, cleanup_items, std::move(done),
                        std::move(get_component_args));
}
#endif  // IS_MOBILE_PLATFORM

}  // namespace eager
}  // namespace tensorflow
