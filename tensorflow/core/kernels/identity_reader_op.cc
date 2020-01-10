// See docs in ../ops/io_ops.cc.

#include <memory>
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

class IdentityReader : public ReaderBase {
 public:
  explicit IdentityReader(const string& node_name)
      : ReaderBase(strings::StrCat("IdentityReader '", node_name, "'")) {}

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    *key = current_work();
    *value = current_work();
    *produced = true;
    *at_end = true;
    return Status::OK();
  }

  // Stores state in a ReaderBaseState proto, since IdentityReader has
  // no additional state beyond ReaderBase.
  Status SerializeStateLocked(string* state) override {
    ReaderBaseState base_state;
    SaveBaseState(&base_state);
    base_state.SerializeToString(state);
    return Status::OK();
  }

  Status RestoreStateLocked(const string& state) override {
    ReaderBaseState base_state;
    if (!ParseProtoUnlimited(&base_state, state)) {
      return errors::InvalidArgument("Could not parse state for ", name(), ": ",
                                     str_util::CEscape(state));
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

}  // namespace tensorflow
