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

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/buffered_file.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/cache.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/tensor_bundle.pb.h"
#include "tensorflow/core/util/tensor_slice_set.h"

namespace tensorflow {

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
  struct Options {
    Options() {}
    // Alignment, in bytes, for tensor data.
    // Must be >= 1. The default size of 1 densely packs tensors.
    int data_alignment{1};
  };
  BundleWriter(Env* env, absl::string_view prefix,
               const Options& options = Options());

  // Adds the tensor "val" under key "key".
  // Across calls "key" must be unique but can be added in any order.
  absl::Status Add(absl::string_view key, const Tensor& val);

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
  absl::Status AddSlice(absl::string_view full_tensor_key,
                        const TensorShape& full_tensor_shape,
                        const TensorSlice& slice_spec,
                        const Tensor& slice_tensor);

  // Finishes the writer and flushes.
  absl::Status Finish();

  absl::Status status() const { return status_; }

 private:
  Env* const env_;  // Not owned.
  const Options options_;
  const std::string prefix_;
  std::string metadata_path_;
  std::string data_path_;
  bool use_temp_file_;
  std::unique_ptr<tsl::BufferedWritableFile> out_;
  int64_t size_;  // Number of bytes written into out_.
  std::map<std::string, BundleEntryProto> entries_;
  absl::Status status_;

  BundleWriter(const BundleWriter&) = delete;
  void operator=(const BundleWriter&) = delete;
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
//
// "allow_missing_files": If set to true, merges "prefixes" as long as
// at least one file exists. (Defaults to false.)
//
// Returns an InvalidArgumentError when "allow_missing_files" is set to true
// and all data files named in "prefixes" do not exist.
//
// Returns a NotFoundError when "allow_missing_files" is set to false and
// any data file named in "prefixes" does not exist.
absl::Status MergeBundles(Env* env, absl::Span<const tstring> prefixes,
                          absl::string_view merged_prefix,
                          bool allow_missing_files = false);

class BundleCache;

// On construction, silently attempts to read the metadata associated with
// "prefix".  If caller intends to call any function afterwards, "status()"
// must be checked.
// All threads accessing the same BundleReader must synchronize.
class BundleReader {
 public:
  BundleReader(Env* env, absl::string_view prefix,
               bool enable_multi_threading_for_testing = false);

  struct Options {
    // If supplied, a shared cache that is used to read tensor data. If not
    // supplied, a BundleCache private to the BundleReader is used.
    BundleCache* cache = nullptr;

    // For tests only.
    bool enable_multi_threading_for_testing = false;
  };
  BundleReader(Env* env, absl::string_view prefix, Options options);

  ~BundleReader();

  // Is ok() iff the reader construction is successful (completed the read of
  // the metadata).
  absl::Status status() const { return status_; }

  // Queries whether the bundle contains an entry keyed by "key".  Calls Seek()
  // internally, so this call invalidates the reader's current position.
  // REQUIRES: status().ok()
  bool Contains(absl::string_view key);

  // Sorts a `container` of tensors to read such that when `Seek(key)` is called
  // on the elements of the sorted container, the underlying file access is
  // sequential. Sorting can greatly improve overall read speed.
  //
  // `get_key` should be a function that when passed an element in `container`,
  // returns the `key` of the tensor.
  //
  // REQUIRES: status().ok()
  template <class T>
  absl::Status SortForSequentialAccess(
      std::vector<T>& container,
      absl::FunctionRef<std::string(const T&)> get_key);

  // Looks up the dtype and the shape of the tensor keyed by "key".
  // REQUIRES: status().ok()
  absl::Status LookupDtypeAndShape(absl::string_view key, DataType* dtype,
                                   TensorShape* shape);

  // Looks up the shape of the tensor keyed by "key".
  // Clears "shape" if not found.
  // REQUIRES: status().ok()
  absl::Status LookupTensorShape(absl::string_view key, TensorShape* shape);

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
  absl::Status Lookup(absl::string_view key, Tensor* val);

  // Looks up the tensor pointed to by the internal iterator.
  //
  // On error, "val" may contain nonsense data.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok() && Valid()
  absl::Status ReadCurrent(Tensor* val);

  // Looks up the slices of the tensor keyed by "key".  On OK, "slices"
  // is non-empty if and only if the tensor is a partitioned tensor.
  //
  // Warning - there is no guaranteed ordering for the returned slices, so
  // a slice with a larger start index in some dimension could come before
  // another slice with a smaller start index in the same dimension.
  // REQUIRES: status().ok()
  absl::Status LookupTensorSlices(absl::string_view key,
                                  std::vector<TensorSlice>* slices);

  // Looks up a specific slice of a partitioned tensor.
  // It is only required that the stored slices cover the requested slice,
  // namely "slice_spec" is a subset of the union of the stored slices.
  // REQUIRES: status().ok()
  absl::Status LookupSlice(absl::string_view full_tensor_key,
                           const TensorSlice& slice_spec, Tensor* val);

  // Seeks to the first position in the bundle whose key is no less than "key".
  // REQUIRES: status().ok()
  void Seek(absl::string_view key) { return iter_->Seek(key); }
  // Moves to the next position in the bundle.
  // REQUIRES: status().ok()
  void Next() const { iter_->Next(); }
  // Returns true iff the reader is positioned to a key/val pair.
  // REQUIRES: status().ok()
  bool Valid() const { return iter_->Valid(); }

  // Returns the key at the current position.
  // REQUIRES: status().ok() && Valid()
  absl::string_view key() const { return iter_->key(); }
  // Returns the raw value at the current position.
  // REQUIRES: status().ok() && Valid()
  absl::string_view value() const { return iter_->value(); }

  std::string DebugString();

 private:
  // Seeks for "key" and reads the metadata proto.
  // On non-OK return, clears "entry" for the caller.
  // REQUIRES: status().ok()
  absl::Status GetBundleEntryProto(absl::string_view key,
                                   BundleEntryProto* entry);

  // Reads the tensor value described by the metadata proto "entry".
  // Usage for "val" follows the comment of "Lookup()".
  absl::Status GetValue(const BundleEntryProto& entry, Tensor* val);

  // Reads the slice described by "slice_spec".  The corresponding full tensor
  // has key "ful_tensor_key" and metadata proto "full_tensor_entry".
  // REQUIRES: full_tensor_entry.slices_size() > 0
  absl::Status GetSliceValue(absl::string_view full_tensor_key,
                             const BundleEntryProto& full_tensor_entry,
                             const TensorSlice& slice_spec, Tensor* val);

  Env* env_;  // Not owned.
  const std::string prefix_;
  std::unique_ptr<BundleCache> owned_cache_;  // may be null
  BundleCache* cache_;  // Not owned, or owned_cache_.get()

  absl::Status status_;
  RandomAccessFile* metadata_;  // Owned.
  table::Table* table_;
  table::Cache* index_cache_;
  table::Iterator* iter_;

  // Owned InputBuffer objects. cache_ owns the underlying RandomAccessFiles.
  std::unordered_map<int32_t, io::InputBuffer*> data_;

  // Maps each partitioned tensor's key to its stored slices (represented in a
  // TensorSliceSet).  Populated on-demand.
  std::unordered_map<std::string, checkpoint::TensorSliceSet*> tensor_slices_;

  // Expected number of data file shards in the bundle.  Extracted by reading
  // the header entry in the metadata table.
  int num_shards_;

  // Flag that this class sets to true when the endianness of the target bundle
  // differs from that of the current system's processor architecture.
  bool need_to_swap_bytes_;

  friend class TensorBundleAlignmentTest;  // For testing data alignment.

  bool enable_multi_threading_for_testing_ = false;

  BundleReader(const BundleReader&) = delete;
  void operator=(const BundleReader&) = delete;
};

template <class T>
absl::Status BundleReader::SortForSequentialAccess(
    std::vector<T>& container,
    absl::FunctionRef<std::string(const T&)> get_key) {
  struct FileOffset {
    int32_t shard_id;
    int64_t offset;
  };
  absl::flat_hash_map<std::string, FileOffset> file_offsets;
  for (const T& element : container) {
    BundleEntryProto entry;
    TF_RETURN_IF_ERROR(GetBundleEntryProto(get_key(element), &entry));
    file_offsets[get_key(element)] = {entry.shard_id(), entry.offset()};
  }
  absl::c_sort(container, [&get_key, &file_offsets](const T& a, const T& b) {
    const FileOffset& file_offset_a = file_offsets[get_key(a)];
    const FileOffset& file_offset_b = file_offsets[get_key(b)];
    if (file_offset_a.shard_id == file_offset_b.shard_id) {
      return file_offset_a.offset < file_offset_b.offset;
    } else {
      return file_offset_a.shard_id < file_offset_b.shard_id;
    }
  });
  return absl::OkStatus();
}

// BundleCache provides cached opening of files.
// Used internally by BundleReader.
// Safe for concurrent uses by multiple threads and BundleReaders.
class BundleCache {
 public:
  explicit BundleCache(Env* env);

  // Get the underlying file object for fname. The result will remain valid
  // while the BundleCache lives.
  absl::Status GetFile(const std::string& fname, RandomAccessFile** file);

 private:
  // State for each opened file (opened on first read).
  struct FileState {
    absl::once_flag once;  // Ensures file is opened exactly once.

    std::unique_ptr<RandomAccessFile> file;
    absl::Status open_status;  // Records any error encountered on open
  };

  FileState* EnsureOpened(std::string name);

  Env* const env_;
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<FileState>> opened_files_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
