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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_H_

#include <memory>
#include <string>

#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"

namespace tensorflow {

// RestoredResource represents a TF2 "Resource" object loaded from a savedmodel,
// analogous to the Python _RestoredResource object:
// https://github.com/tensorflow/tensorflow/blob/fda326e542ca67534e8411edb180e8760a4828b7/tensorflow/python/saved_model/load.py#L481
// TF2 resource objects typically extend TrackableResource:
// https://github.com/tensorflow/tensorflow/blob/fda326e542ca67534e8411edb180e8760a4828b7/tensorflow/python/training/tracking/tracking.py#L285
// and are expected to implement "_create_resource", "_initialize", and
// "_destroy_resource" functions:
// https://github.com/tensorflow/tensorflow/blob/139ba9c5284799beafdd1d7f895127cf00e7c48f/tensorflow/python/training/tracking/tracking.py#L262-L281
class RestoredResource : TensorHandleConvertible {
 public:
  // Note(bmzhao): RestoredResource stores non-owning pointers to its associated
  // functions because SavedModel internally owns all functions and objects in
  // the RevivedObjects struct (which owns all functions). One alternative would
  // be to have RevivedObjects store shared_ptr<TFConcreteFunction> instead, and
  // change RestoredResource's constructor take shared_ptr<TFConcreteFunction>.
  // To keep things simple, I've stuck to raw pointers for now.
  //
  // Params:
  //  device - The device string associated with the SavedResource
  //           https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_object_graph.proto#L182
  //           Conceptually, this is the same device used in CapturableResource:
  //           https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/python/training/tracking/tracking.py#L222-L225
  //           Implementation-wise, it is device used when invoking the
  //           create_resource function to produce the resource_handle
  //           associated with the object:
  //           https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/python/training/tracking/tracking.py#L246-L247
  //  create_resource - Non owning pointer to the create_resource function
  //                    associated with this object. Must be NON-NULL.
  //  initialize - Non owning pointer to the initialize function associated with
  //               this object. Must be NON-NULL.
  //  destroy_resource - Non owning pointer to the destroy_resource function
  //                     associated with this object. Ideally this should be
  //                     NON-NULL, but in order to support models saved prior to
  //                     https://github.com/tensorflow/tensorflow/commit/3c806101f57768e479f8646e7518bbdff1632ca3
  //                     we allow null here. This will, however, leak resources.
  RestoredResource(const std::string& device,
                   TFConcreteFunction* create_resource,
                   TFConcreteFunction* initialize,
                   TFConcreteFunction* destroy_resource,
                   ImmediateTensorHandlePtr resource_handle);

  Status Initialize() const;

  // RestoredResource is movable, but not copyable.
  RestoredResource(RestoredResource&& other) = default;
  RestoredResource& operator=(RestoredResource&& other) = default;

  ~RestoredResource() override;

 private:
  std::string device_;
  TFConcreteFunction* create_resource_;
  TFConcreteFunction* initialize_;
  TFConcreteFunction* destroy_resource_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_H_
