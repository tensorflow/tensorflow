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

// Some tests of the XNNPack Plugin using the FlatBuffers C API.
// This source file is C, not C++, to ensure that we're not accidentally
// depending on the FlatBuffers C++ API here.

#include "tensorflow/lite/core/acceleration/configuration/c/xnnpack_plugin_c_test_lib.h"

#include <stdio.h>
#include <stdlib.h>

#include "third_party/flatcc/include/flatcc/flatcc_builder.h"
#include "tensorflow/lite/core/acceleration/configuration/c/configuration_builder.h"
#include "tensorflow/lite/core/acceleration/configuration/c/configuration_reader.h"

struct SettingsStorage {
  // The builder object that allocates the storage.
  flatcc_builder_t builder;

  // The raw buffer.
  void* buffer;
  size_t size;

  // The parsed TFLiteSettings object.
  const struct tflite_TFLiteSettings_table* tflite_settings;
};
typedef struct SettingsStorage SettingsStorage;

SettingsStorage* SettingsStorageCreateWithXnnpackThreads(int num_threads) {
  struct SettingsStorage* storage =
      (struct SettingsStorage*) malloc(sizeof(SettingsStorage));

  flatcc_builder_t* builder = &storage->builder;
  flatcc_builder_init(builder);

  /* Construct a buffer specific to the schema. */
  tflite_TFLiteSettings_start_as_root(builder);
  tflite_TFLiteSettings_xnnpack_settings_start(builder);
  tflite_XNNPackSettings_num_threads_add(builder, num_threads);
  tflite_TFLiteSettings_xnnpack_settings_end(builder);
  tflite_TFLiteSettings_end_as_root(builder);

  /* Retrieve buffer - see also `flatcc_builder_get_direct_buffer`. */
  storage->buffer = flatcc_builder_finalize_buffer(builder, &storage->size);

  /* 'Parse' the buffer. (This is actually just an offset lookup.) */
  storage->tflite_settings = tflite_TFLiteSettings_as_root(storage->buffer);

  return storage;
}

SettingsStorage* SettingsStorageCreateWithXnnpackFlags(
    tflite_XNNPackFlags_enum_t flags) {
  struct SettingsStorage* storage =
      (struct SettingsStorage*) malloc(sizeof(SettingsStorage));

  flatcc_builder_t* builder = &storage->builder;
  flatcc_builder_init(builder);

  /* Construct a buffer specific to the schema. */
  tflite_TFLiteSettings_start_as_root(builder);
  tflite_TFLiteSettings_xnnpack_settings_start(builder);
  tflite_XNNPackSettings_flags_add(builder, flags);
  tflite_TFLiteSettings_xnnpack_settings_end(builder);
  tflite_TFLiteSettings_end_as_root(builder);

  /* Retrieve buffer - see also `flatcc_builder_get_direct_buffer`. */
  storage->buffer = flatcc_builder_finalize_buffer(builder, &storage->size);

  /* 'Parse' the buffer. (This is actually just an offset lookup.) */
  storage->tflite_settings = tflite_TFLiteSettings_as_root(storage->buffer);

  return storage;
}

tflite_TFLiteSettings_table_t SettingsStorageGetSettings(
    const SettingsStorage* storage) {
  return storage->tflite_settings;
}

void SettingsStorageDestroy(SettingsStorage* storage) {
    free(storage->buffer);
    flatcc_builder_clear(&storage->builder);
    storage->tflite_settings = NULL;
    storage->buffer = NULL;
    storage->size = 0;
    free(storage);
}
