#include "tensorflow/core/framework/lookup_interface.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

Status LookupInterface::CheckKeyAndValueTensors(const Tensor& key,
                                                const Tensor& value) {
  if (key.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", key.dtype());
  }
  if (value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Value must be type ", value_dtype(),
                                   " but got ", value.dtype());
  }
  if (key.NumElements() != value.NumElements()) {
    return errors::InvalidArgument("Number of elements of key(",
                                   key.NumElements(), ") and value(",
                                   value.NumElements(), ") are different.");
  }
  if (!key.shape().IsSameSize(value.shape())) {
    return errors::InvalidArgument("key and value have different shapes.");
  }
  return Status::OK();
}

Status LookupInterface::CheckFindArguments(const Tensor& key,
                                           const Tensor& value,
                                           const Tensor& default_value) {
  TF_RETURN_IF_ERROR(CheckKeyAndValueTensors(key, value));

  if (default_value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Default value must be type ", value_dtype(),
                                   " but got ", default_value.dtype());
  }
  if (!TensorShapeUtils::IsScalar(default_value.shape())) {
    return errors::InvalidArgument("Default values must be scalar.");
  }
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
