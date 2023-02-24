/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

// This file implements a serialization utility that TFLite delegates can use to
// read/write initialization data.
//
// Example code:
//
// Initialization
// ==============
// SerializationParams params;
// // Acts as a namespace for all data entries for a given model.
// // See StrFingerprint().
// params.model_token = options->model_token;
// // Location where data is stored, should be private to the app using this.
// params.serialization_dir = options->serialization_dir;
// Serialization serialization(params);
//
// Writing data
// ============
// TfLiteContext* context = ...;
// TfLiteDelegateParams* params = ...;
// SerializationEntry kernels_entry = serialization->GetEntryForKernel(
//     "gpuv2_kernels", context, delegate_params);
//
// TfLiteStatus kernels_save_status = kernels_entry.SetData(
//     reinterpret_cast<char*>(data_ptr),
//     data_size);
// if (kernels_save_status == kTfLiteOk) {
//   //...serialization successful...
// } else if (kernels_save_status == kTfLiteDelegateDataWriteError) {
//   //...error in serializing data to disk...
// } else {
//   //...unexpected error...
// }
//
// Reading data
// ============
// std::string kernels_data;
// TfLiteStatus kernels_data_status = kernels_entry.GetData(&kernels_data);
// if (kernels_data_status == kTfLiteOk) {
//   //...serialized data found...
// } else if (kernels_data_status == kTfLiteDelegateDataNotFound) {
//   //...serialized data missing...
// } else {
//   //...unexpected error...
// }
namespace tflite {
namespace delegates {

// Helper to generate a unique string (converted from 64-bit farmhash) given
// some data. Intended for use by:
//
// 1. Delegates, to 'fingerprint' some custom data (like options),
//    and provide it as custom_key to Serialization::GetEntryForDelegate or
//    GetEntryForKernel.
// 2. TFLite clients, to fingerprint a model flatbuffer & get a unique
//    model_token.
std::string StrFingerprint(const void* data, const size_t num_bytes);

// Encapsulates a unique blob of data serialized by a delegate.
// Needs to be initialized with a Serialization instance.
// Any data set with this entry is 'keyed' by a 64-bit fingerprint unique to the
// parameters used during initialization via
// Serialization::GetEntryForDelegate/GetEntryForKernel.
//
// NOTE: TFLite cannot guarantee that the read data is always fully valid,
// especially if the directory is accessible to other applications/processes.
// It is the delegate's responsibility to validate the retrieved data.
class SerializationEntry {
 public:
  friend class Serialization;

  // Returns a 64-bit fingerprint unique to the parameters provided during the
  // generation of this SerializationEntry.
  // Produces same value on every run.
  uint64_t GetFingerprint() const { return fingerprint_; }

  // Stores `data` into a file that is unique to this SerializationKey.
  // Overwrites any existing data if present.
  //
  // Returns:
  //   kTfLiteOk if data is successfully stored
  //   kTfLiteDelegateDataWriteError for data writing issues
  //   kTfLiteError for unexpected error.
  //
  // NOTE: We use a temp file & rename it as file renaming is an atomic
  // operation in most systems.
  TfLiteStatus SetData(TfLiteContext* context, const char* data,
                       const size_t size) const;

  // Get `data` corresponding to this key, if available.
  //
  // Returns:
  //   kTfLiteOk if data is successfully stored
  //   kTfLiteDataError for data writing issues
  //   kTfLiteError for unexpected error.
  TfLiteStatus GetData(TfLiteContext* context, std::string* data) const;

  // Non-copyable.
  SerializationEntry(const SerializationEntry&) = delete;
  SerializationEntry& operator=(const SerializationEntry&) = delete;
  SerializationEntry(SerializationEntry&& src) = default;

 protected:
  SerializationEntry(const std::string& cache_dir,
                     const std::string& model_token,
                     const uint64_t fingerprint_64);

  // Caching directory.
  const std::string cache_dir_;
  // Model Token.
  const std::string model_token_;
  // For most applications, 64-bit fingerprints are enough.
  const uint64_t fingerprint_ = 0;
};

// Encapsulates all the data that clients can use to parametrize a Serialization
// interface.
typedef struct SerializationParams {
  // Acts as a 'namespace' for all SerializationEntry instances.
  // Clients should ensure that the token is unique to the model graph & data.
  // StrFingerprint() can be used with the flatbuffer data to generate a unique
  // 64-bit token.
  // TODO(b/190055017): Add 64-bit fingerprints to TFLite flatbuffers to ensure
  // different model constants automatically lead to different fingerprints.
  // Required.
  const char* model_token;
  // Denotes the directory to be used to store data.
  // It is the client's responsibility to ensure this location is valid and
  // application-specific to avoid unintended data access issues.
  // On Android, `getCodeCacheDir()` is recommended.
  // Required.
  const char* cache_dir;
} SerializationParams;

// Utility to enable caching abilities for delegates.
// See documentation at the top of the file for usage details.
//
// WARNING: Experimental interface, subject to change.
class Serialization {
 public:
  // Initialize a Serialization interface for applicable delegates.
  explicit Serialization(const SerializationParams& params)
      : cache_dir_(params.cache_dir), model_token_(params.model_token) {}

  // Generate a SerializationEntry that incorporates both `custom_key` &
  // `context` into its unique fingerprint.
  //  Should be used to handle data common to all delegate kernels.
  // Delegates can incorporate versions & init arguments in custom_key using
  // StrFingerprint().
  SerializationEntry GetEntryForDelegate(const std::string& custom_key,
                                         TfLiteContext* context) {
    return GetEntryImpl(custom_key, context);
  }

  // Generate a SerializationEntry that incorporates `custom_key`, `context`,
  // and `delegate_params` into its unique fingerprint.
  // Should be used to handle data specific to a delegate kernel, since
  // the context+delegate_params combination is node-specific.
  // Delegates can incorporate versions & init arguments in custom_key using
  // StrFingerprint().
  SerializationEntry GetEntryForKernel(
      const std::string& custom_key, TfLiteContext* context,
      const TfLiteDelegateParams* partition_params) {
    return GetEntryImpl(custom_key, context, partition_params);
  }

  // Non-copyable.
  Serialization(const Serialization&) = delete;
  Serialization& operator=(const Serialization&) = delete;

 protected:
  SerializationEntry GetEntryImpl(
      const std::string& custom_key, TfLiteContext* context = nullptr,
      const TfLiteDelegateParams* delegate_params = nullptr);

  const std::string cache_dir_;
  const std::string model_token_;
};

// Helper for delegates to save their delegation decisions (which nodes to
// delegate) in TfLiteDelegate::Prepare().
// Internally, this uses a unique SerializationEntry based on the `context` &
// `delegate_id` to save the `node_ids`. It is recommended that `delegate_id` be
// unique to a backend/version to avoid reading back stale delegation decisions.
//
// NOTE: This implementation is platform-specific, so this method & the
// subsequent call to GetDelegatedNodes should happen on the same device.
TfLiteStatus SaveDelegatedNodes(TfLiteContext* context,
                                Serialization* serialization,
                                const std::string& delegate_id,
                                const TfLiteIntArray* node_ids);

// Retrieves list of delegated nodes that were saved earlier with
// SaveDelegatedNodes.
// Caller assumes ownership of data pointed by *nodes_ids.
//
// NOTE: This implementation is platform-specific, so SaveDelegatedNodes &
// corresponding GetDelegatedNodes should be called on the same device.
TfLiteStatus GetDelegatedNodes(TfLiteContext* context,
                               Serialization* serialization,
                               const std::string& delegate_id,
                               TfLiteIntArray** node_ids);

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_
