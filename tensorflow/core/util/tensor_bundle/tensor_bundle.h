/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// A tensor bundle is a set of immutable persistent files storing a set of named
// tensors.  It is designed for checkpointing TensorFlow tensors.
//
// The paths of the managed files share a common prefix; e.g., with the prefix:
//   /fs/model/train/ckpt-step/ckpt
//
// the bundle may contain a metadata file, and sharded data files:
//   /fs/model/train/ckpt-step/
//       ckpt.index
//       ckpt.data-00000-of-00020
//       ckpt.data-00001-of-00020
//       ...
//       ckpt.data-00019-of-00020
//
// The ".index" file is a string-string immutable table
// (tensorflow::table::Table).  Each key is a name of a tensor and its value is
// a serialized BundleEntryProto.  Each BundleEntryProto describes the metadata
// of a tensor: which of the "data" files contains the content of a tensor, the
// offset into that file, checksum, some auxiliary data, etc.
//
// A tensor bundle can be accessed randomly using a BundleReader.  Usage:
//
//   BundleReader reader(env, "/fs/model/train/ckpt-step/ckpt");
//   reader.Lookup("name", &tensor);
//
// A tensor bundle can be built using BundleWriter.  Each BundleWriter builds a
// single data file bundle.  Multiple bundles can then be merged by
// MergeBundles() without reading and writing large chunk of data: it reads the
// metadata files and outputs a single merged metadata.  Typical usage:
//
//   worker 0:
//     BundleWriter writer(env, "/fs/model/train/ckpt-step/tmp/worker0-step");
//     writer.Add(...);  // Adds the tensors on this worker.
//     writer.Finish();  // Flushes.
//   worker 1:
//     BundleWriter writer(env, "/fs/model/train/ckpt-step/tmp/worker1-step");
//     writer.Add(...);
//     writer.Finish();
//   worker 2:
//     MergeBundles(env,
//       {"/fs/model/train/ckpt-step/tmp/worker0-step",
//        "/fs/model/train/ckpt-step/tmp/worker1-step"},
//       "/fs/model/train/ckpt-step/ckpt" /* merged prefix */);
//

#ifndef TENSORFLOW_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
#define TENSORFLOW_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_

#include "tensorflow/core/protobuf/tensor_bundle.pb.h"

#include <map>
#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_slice_set.h"

namespace tensorflow {

class FileOutputBuffer;

// Versioning of the tensor bundle format.
// Follows the same rules as 3p/tf/core/public/version.h.
//
// History:
// 0. Any tensor bundles produced before this field was added.
// 1. Added this field (2016-09-14).
extern const int kTensorBundleMinProducer;
extern const int kTensorBundleMinConsumer;
extern const int kTensorBundleVersion;

// The empty string, hence always the first key in the metadata table.  Its
// corresponding value is a BundleHeaderProto.
extern const char* const kHeaderEntryKey;

// Builds a string-string table of tensor names to BundleEntryProto (metadata).
//
// On construction, attempts to create a directory given by the dirname of
// "prefix", so "status()" must be checked before calling any member functions.
//
// All threads accessing the same BundleWriter must synchronize.
class BundleWriter {
 public:
  BundleWriter(Env* env, StringPiece prefix);

  // Adds the tensor "val" under key "key".
  // Across calls "key" must be unique but can be added in any order.
  Status Add(StringPiece key, const Tensor& val);

  // Partitioned variables support.
  // A slice of a full tensor is stored in two entries in the metadata table:
  //
  //   full_tensor_key   -> BundleEntryProto, describing all stored slices
  //                        of this full tensor.  Does not append to the data
  //                        file.
  //   encoded slice key -> BundleEntryProto, describing one particular slice.
  //                        Appends values of this slice to the data file.
  //
  // Slices of a full tensor can be added in any order.
  //
  // If a full tensor has slices placed on N devices and N BundleWriter's are
  // concurrently used, the caller must use MergeBundles() to ensure that a
  // consistent entry for "full_tensor_key" is produced.
  //
  // Returns an error if the same slice is added the second time.
  Status AddSlice(StringPiece full_tensor_key,
                  const TensorShape& full_tensor_shape,
                  const TensorSlice& slice_spec, const Tensor& slice_tensor);

  // Finishes the writer and flushes.
  Status Finish() TF_MUST_USE_RESULT;

  Status status() const { return status_; }

 private:
  Env* const env_;  // Not owned.
  const string prefix_;
  const string tmp_metadata_path_;
  const string tmp_data_path_;
  std::unique_ptr<FileOutputBuffer> out_;
  int64 size_;  // Number of bytes written into out_.
  std::map<string, BundleEntryProto> entries_;
  Status status_;

  TF_DISALLOW_COPY_AND_ASSIGN(BundleWriter);
};

// Merges a set of bundles (given their prefixes) into a single bundle with the
// given "merged_prefix".  The merged metadata is guaranteed to be consistent.
//
// If there are N bundles in "prefixes", during the merge the data files will be
// renamed to contain a proper sharded file spec, with num_shards set to the sum
// of num_shards across the N input bundles.
//
// The caller should only rely on the metadata file of the merged bundle to
// query information about a tensor.  In particular, this function does not
// guarantee not to re-order the input data files.
//
// Once merged, makes a best effort to delete the old metadata files.
// Returns OK iff all bundles are successfully merged.
Status MergeBundles(Env* env, gtl::ArraySlice<string> prefixes,
                    StringPiece merged_prefix);

// On construction, silently attempts to read the metadata associated with
// "prefix".  If caller intends to call any function afterwards, "status()"
// must be checked.
// All threads accessing the same BundleReader must synchronize.
class BundleReader {
 public:
  BundleReader(Env* const env, StringPiece prefix);
  ~BundleReader();

  // Is ok() iff the reader construction is successful (completed the read of
  // the metadata).
  Status status() const { return status_; }

  // Queries whether the bundle contains an entry keyed by "key".  Calls Seek()
  // internally, so this call invalidates the reader's current position.
  // REQUIRES: status().ok()
  bool Contains(StringPiece key);

  // Looks up the dtype and the shape of the tensor keyed by "key".
  // REQUIRES: status().ok()
  Status LookupDtypeAndShape(StringPiece key, DataType* dtype,
                             TensorShape* shape) TF_MUST_USE_RESULT;

  // Looks up the shape of the tensor keyed by "key".
  // Clears "shape" if not found.
  // REQUIRES: status().ok()
  Status LookupTensorShape(StringPiece key,
                           TensorShape* shape) TF_MUST_USE_RESULT;

  // Looks up the tensor keyed by "key".  If "key" refers to a partitioned
  // tensor, attempts to look up the full contents using all stored slices.
  //
  // Caller must make sure "val" has the same shape and dtype as the
  // corresponding contents, so that its buffer can be filled without needing
  // extra allocation.  These can be queried via "LookupDtypeAndShape()".
  //
  // On error, "val" may contain nonsense data.  Returns a NotFound error if
  // tensor keyed by "key" does not exist in this bundle.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok()
  Status Lookup(StringPiece key, Tensor* val) TF_MUST_USE_RESULT;

  // Looks up the tensor pointed to by the internal iterator.
  //
  // On error, "val" may contain nonsense data.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok() && Valid()
  Status ReadCurrent(Tensor* val) TF_MUST_USE_RESULT;

  // Looks up the slices of the tensor keyed by "key".  On OK, "slices"
  // is non-empty if and only if the tensor is a partitioned tensor.
  //
  // Warning - there is no guaranteed ordering for the returned slices, so
  // a slice with a larger start index in some dimension could come before
  // another slice with a smaller start index in the same dimension.
  // REQUIRES: status().ok()
  Status LookupTensorSlices(StringPiece key, std::vector<TensorSlice>* slices)
      TF_MUST_USE_RESULT;

  // Looks up a specific slice of a partitioned tensor.
  // It is only required that the stored slices cover the requested slice,
  // namely "slice_spec" is a subset of the union of the stored slices.
  // REQUIRES: status().ok()
  Status LookupSlice(StringPiece full_tensor_key, const TensorSlice& slice_spec,
                     Tensor* val) TF_MUST_USE_RESULT;

  // Seeks to the first position in the bundle whose key is no less than "key".
  // REQUIRES: status().ok()
  void Seek(StringPiece key) { return iter_->Seek(key); }
  // Moves to the next position in the bundle.
  // REQUIRES: status().ok()
  void Next() const { iter_->Next(); }
  // Returns true iff the reader is positioned to a key/val pair.
  // REQUIRES: status().ok()
  bool Valid() const { return iter_->Valid(); }

  // Returns the key at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece key() const { return iter_->key(); }
  // Returns the raw value at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece value() const { return iter_->value(); }

  string DebugString();

 private:
  // Seeks for "key" and reads the metadata proto.
  // On non-OK return, clears "entry" for the caller.
  // REQUIRES: status().ok()
  Status GetBundleEntryProto(StringPiece key,
                             BundleEntryProto* entry) TF_MUST_USE_RESULT;

  // Reads the tensor value described by the metadata proto "entry".
  // Usage for "val" follows the comment of "Lookup()".
  Status GetValue(const BundleEntryProto& entry,
                  Tensor* val) TF_MUST_USE_RESULT;

  // Reads the slice described by "slice_spec".  The corresponding full tensor
  // has key "ful_tensor_key" and metadata proto "full_tensor_entry".
  // REQUIRES: full_tensor_entry.slices_size() > 0
  Status GetSliceValue(StringPiece full_tensor_key,
                       const BundleEntryProto& full_tensor_entry,
                       const TensorSlice& slice_spec,
                       Tensor* val) TF_MUST_USE_RESULT;

  Env* env_;  // Not owned.
  const string prefix_;

  Status status_;
  RandomAccessFile* metadata_;  // Owned.
  table::Table* table_;
  table::Iterator* iter_;
  // Owned the InputBuffer objects and their underlying RandomAccessFile's.
  std::unordered_map<int32, io::InputBuffer*> data_;

  // Maps each partitioned tensor's key to its stored slices (represented in a
  // TensorSliceSet).  Populated on-demand.
  std::unordered_map<string, checkpoint::TensorSliceSet*> tensor_slices_;

  // Expected number of data file shards in the bundle.  Extracted by reading
  // the header entry in the metadata table.
  int num_shards_;

  // If set to true, try reading key + ":0" whenever key is not found in the
  // bundle. This is a temporary measure that will be removed on Jan 1st 2018.
  // TODO(b/64763924): Remove after Jan 1st 2018.
  bool lenient_names_;

  TF_DISALLOW_COPY_AND_ASSIGN(BundleReader);
};

// A buffering wrapper for a WritableFile.  Useful if the caller wishes to issue
// small writes to a file (e.g. writing out a list of small varints).
// External synchronization must be used in the presence of concurrent callers.
class FileOutputBuffer {
 public:
  FileOutputBuffer(WritableFile* file, size_t buffer_size)
      : file_(file), position_(0), buffer_size_(buffer_size) {
    DCHECK_GT(buffer_size, 0);
    buffer_.resize(buffer_size);
  }
  ~FileOutputBuffer();

  // Buffered append.
  Status Append(StringPiece data);

  // Returns the running crc32c checksum of all currently appended bytes.
  uint32 crc32c() { return crc32c_; }
  // Clears the running crc32c checksum.
  void clear_crc32c() { crc32c_ = 0; }

  // Appends the buffered data, then closes the underlying file.
  Status Close();

 private:
  // Appends the buffered data to the underlying file. Does NOT flush the file.
  Status FlushBuffer();

  WritableFile* file_;  // Owned.

  // buffer_[0, position_) holds the buffered data not yet appended to the
  // underlying file.
  size_t position_;
  const size_t buffer_size_;
  std::vector<char> buffer_;

  // Checksum of all appended bytes since construction or last clear_crc32c().
  uint32 crc32c_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
