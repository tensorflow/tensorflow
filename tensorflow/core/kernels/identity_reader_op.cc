/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc.

#include <memory>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_base.pb.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

class IdentityReader : public ReaderBase {
 public:
  explicit IdentityReader(const string& node_name)
      : ReaderBase(strings::StrCat("IdentityReader '", node_name, "'")) {}

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
    *key = current_work();
    *value = current_work();
    *produced = true;
    *at_end = true;
    return Status::OK();
  }

  // Stores state in a ReaderBaseState proto, since IdentityReader has
  // no additional state beyond ReaderBase.
  Status SerializeStateLocked(tstring* state) override {
    ReaderBaseState base_state;
    SaveBaseState(&base_state);
    SerializeToTString(base_state, state);
    return Status::OK();
  }

  Status RestoreStateLocked(const tstring& state) override {
    ReaderBaseState base_state;
    if (!ParseProtoUnlimited(&base_state, state)) {
      return errors::InvalidArgument("Could not parse state for ", name(), ": ",
                                     absl::CEscape(state));
    }
    TF_RETURN_IF_ERROR(RestoreBaseState(base_state));
    return Status::OK();
  }
};

class IdentityReaderOp : public ReaderOpKernel {
 public:
  explicit IdentityReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    SetReaderFactory([this]() { return new IdentityReader(name()); });
  }
};

REGISTER_KERNEL_BUILDER(Name("IdentityReader").Device(DEVICE_CPU),
                        IdentityReaderOp);
REGISTER_KERNEL_BUILDER(Name("IdentityReaderV2").Device(DEVICE_CPU),
                        IdentityReaderOp);

}  // namespace tensorflow
