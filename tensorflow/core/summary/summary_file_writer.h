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
#ifndef TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
#define TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_

#include "absl/status/status.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief Creates SummaryWriterInterface which writes to a file.
///
/// The file is an append-only records file of tf.Event protos. That
/// makes this summary writer suitable for file systems like GCS.
///
/// It will enqueue up to max_queue summaries, and flush at least every
/// flush_millis milliseconds. The summaries will be written to the
/// directory specified by logdir and with the filename suffixed by
/// filename_suffix. The caller owns a reference to result if the
/// returned status is ok. The Env object must not be destroyed until
/// after the returned writer.
absl::Status CreateSummaryFileWriter(int max_queue, int flush_millis,
                                     const string& logdir,
                                     const string& filename_suffix, Env* env,
                                     SummaryWriterInterface** result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
