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

#ifndef TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_
#define TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {

class GraphDef;

namespace data {

namespace experimental {

class SnapshotMetadataRecord;
class SnapshotTensorMetadata;

}  // namespace experimental

namespace snapshot_util {

constexpr char kMetadataFilename[] = "snapshot.metadata";

constexpr char kModeAuto[] = "auto";
constexpr char kModeWrite[] = "write";
constexpr char kModeRead[] = "read";
constexpr char kModePassthrough[] = "passthrough";
constexpr char kShardDirectorySuffix[] = ".shard";

enum Mode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

// Returns the name of the "hash" directory for the given base path and hash ID.
std::string HashDirectory(const std::string& path, uint64 hash);

// Returns the name of the "run" directory for the given base path and run ID.
std::string RunDirectory(const std::string& hash_directory, uint64 run_id);
std::string RunDirectory(const std::string& hash_directory,
                         const std::string& run_id);

// Returns the name of the "shard" directory for the given base path and shard
// ID.
std::string ShardDirectory(const std::string& run_directory, int64_t shard_id);

// Returns the checkpoint file name for the given directory and checkpoint ID.
std::string GetCheckpointFileName(const std::string& shard_directory,
                                  uint64 checkpoint_id);

// This is a interface class that exposes snapshot writing functionality.
class Writer {
 public:
  // Creates a new writer object.
  static Status Create(Env* env, const std::string& filename,
                       const std::string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Writer>* out_writer);

  // Writes a vector of tensors to the snapshot writer file.
  virtual Status WriteTensors(const std::vector<Tensor>& tensors) = 0;

  // Flushes any in-memory buffers to disk.
  virtual Status Sync() = 0;

  // Closes and finalizes the snapshot file. All calls to any other method will
  // be invalid after this call.
  virtual Status Close() = 0;

  virtual ~Writer() = default;

 protected:
  virtual Status Initialize(tensorflow::Env* env) = 0;
};

// Writes snapshots with the standard TFRecord file format.
class TFRecordWriter : public Writer {
 public:
  TFRecordWriter(const std::string& filename,
                 const std::string& compression_type,
                 bool overwrite_existing = false);

  Status Initialize(tensorflow::Env* env) override;

  Status WriteTensors(const std::vector<Tensor>& tensors) override;

  Status Sync() override;

  Status Close() override;

  ~TFRecordWriter() override;

 private:
  const std::string filename_;
  const std::string compression_type_;
  const bool overwrite_existing_;

  std::unique_ptr<WritableFile> dest_;
  std::unique_ptr<io::RecordWriter> record_writer_;
};

// Writes snapshot with a custom (legacy) file format.
class CustomWriter : public Writer {
 public:
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotWriter";
  static constexpr const char* const kWriteStringPiece = "WriteStringPiece";
  static constexpr const char* const kWriteCord = "WriteCord";
  static constexpr const char* const kSeparator = "::";

  CustomWriter(const std::string& filename, const std::string& compression_type,
               const DataTypeVector& dtypes);

  Status WriteTensors(const std::vector<Tensor>& tensors) override;

  Status Sync() override;

  Status Close() override;

  ~CustomWriter() override;

 protected:
  Status Initialize(tensorflow::Env* env) override;

 private:
  Status WriteRecord(const StringPiece& data);

#if defined(TF_CORD_SUPPORT)
  Status WriteRecord(const absl::Cord& data);
#endif  // TF_CORD_SUPPORT

  std::unique_ptr<WritableFile> dest_;
  const std::string filename_;
  const std::string compression_type_;
  const DataTypeVector dtypes_;
  // We hold zlib_dest_ because we may create a ZlibOutputBuffer and put that
  // in dest_ if we want compression. ZlibOutputBuffer doesn't own the original
  // dest_ and so we need somewhere to store the original one.
  std::unique_ptr<WritableFile> zlib_underlying_dest_;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
  int num_simple_ = 0;
  int num_complex_ = 0;
};

// Interface class for reading snapshot files previous written with Writer.
class Reader {
 public:
  // Op kernel that creates an instance of `Reader::Dataset` needed to support
  // serialization and deserialization of `Reader::Dataset`.
  class DatasetOp : public DatasetOpKernel {
   public:
    explicit DatasetOp(OpKernelConstruction* ctx);

   protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

   private:
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
    std::string compression_;
    int64_t version_;
  };

  // Op kernel that creates an instance of `Reader::NestedDataset` needed to
  // support serialization and deserialization of `Reader::NestedDataset`.
  class NestedDatasetOp : public DatasetOpKernel {
   public:
    explicit NestedDatasetOp(OpKernelConstruction* ctx);

   protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

   private:
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
  };

  // Creates a new Reader object that reads data from `filename`. Note that
  // the `version`, `compression_type`, and `dtypes` arguments passed into
  // `Writer` and `Reader` must be the same for the reading to succeed.
  static Status Create(Env* env, const std::string& filename,
                       const string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Reader>* out_reader);

  // Returns a nested dataset for a set of given snapshot file names.
  //
  // This function takes a vector of snapshot files, and returns a nested
  // dataset. Each element within the nested dataset is itself a dataset, and
  // contains all the elements written out to each individual snapshot file.
  static Status MakeNestedDataset(Env* env,
                                  const std::vector<std::string>& shard_dirs,
                                  const string& compression_type, int version,
                                  const DataTypeVector& dtypes,
                                  const std::vector<PartialTensorShape>& shapes,
                                  int64_t start_index, DatasetBase** output);

  // Returns a nested dataset for the given datasets.
  static void MakeNestedDataset(const std::vector<DatasetBase*>& datasets,
                                DatasetBase** output);

  // Reads a vector of Tensors from the snapshot file.
  virtual Status ReadTensors(std::vector<Tensor>* read_tensors) = 0;

  // Skips `num_records`. Equivalent to calling `ReadTensors` `num_records`
  // times then discarding the results.
  virtual Status SkipRecords(int64_t num_records);

  virtual ~Reader() = default;

 protected:
  virtual Status Initialize(Env* env) = 0;

  class Dataset;
  class NestedDataset;
};

class TFRecordReaderImpl {
 public:
  // Constructs a `TFRecordReaderImpl`.
  // `filename` is the file to read from.
  // `compression_type` is the compression method, as defined in
  // tensorflow/tsl/lib/io/compression.h.
  // `output_buffer_size` specifies the buffer size required by Snappy/Zlib
  // compression algorithms. Ignored if compression is not enabled.
  TFRecordReaderImpl(const std::string& filename, const string& compression,
                     std::optional<int64_t> output_buffer_size = std::nullopt);

  // Initializes the reader. Callers must initialize the reader before calling
  // `GetNext` or `GetTensors`.
  Status Initialize(Env* env);

  // Reads the next Tensor in the input file.
  StatusOr<Tensor> GetNext();

  // Reads all Tensors in the input file.
  StatusOr<std::vector<Tensor>> GetTensors();

  // Returns the number of bytes read.
  uint64_t BytesRead() const { return bytes_read_; }

 private:
  // Parses `record` into a Tensor.
  StatusOr<Tensor> Parse(const tstring& record);

  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::RecordReader> record_reader_;
  uint64_t offset_ = 0;
  uint64_t bytes_read_ = 0;

  const string compression_;
  const std::optional<int64_t> output_buffer_size_;
};

// Reads snapshots previously written with `TFRecordWriter`.
class TFRecordReader : public Reader {
 public:
  TFRecordReader(const std::string& filename, const string& compression,
                 const DataTypeVector& dtypes,
                 std::optional<int64_t> output_buffer_size = std::nullopt)
      : reader_impl_(filename, compression, output_buffer_size),
        dtypes_(dtypes) {}

  // Initializes the reader. Callers must initialize the reader before calling
  // `ReadTensors`.
  Status Initialize(Env* env) override { return reader_impl_.Initialize(env); }

  // Reads Tensors into `read_tensors`. Returns OK on success, OutOfRange for
  // end of file, or an error status if there is an error.
  Status ReadTensors(std::vector<Tensor>* read_tensors) override;

  // Returns the number of bytes read.
  uint64_t BytesRead() const { return reader_impl_.BytesRead(); }

 private:
  TFRecordReaderImpl reader_impl_;
  const DataTypeVector dtypes_;
};

// Reads snapshots previously written with `CustomWriter`.
class CustomReader : public Reader {
 public:
  // The reader input buffer size is deliberately large because the input reader
  // will throw an error if the compressed block length cannot fit in the input
  // buffer.
  static constexpr const int64_t kSnappyReaderInputBufferSizeBytes =
      1 << 30;  // 1 GiB
  // TODO(b/148804377): Set this in a smarter fashion.
  static constexpr const int64_t kSnappyReaderOutputBufferSizeBytes =
      32 << 20;  // 32 MiB
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotReader";
  static constexpr const char* const kReadString = "ReadString";
  static constexpr const char* const kReadCord = "ReadCord";
  static constexpr const char* const kSeparator = "::";

  CustomReader(const std::string& filename, const string& compression_type,
               int version, const DataTypeVector& dtypes);

  Status ReadTensors(std::vector<Tensor>* read_tensors) override;

  ~CustomReader() override = default;

 protected:
  Status Initialize(Env* env) override;

 private:
  Status ReadTensorsV0(std::vector<Tensor>* read_tensors);

  Status SnappyUncompress(
      const experimental::SnapshotTensorMetadata* metadata,
      std::vector<Tensor>* simple_tensors,
      std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
          tensor_proto_strs);

  Status ReadRecord(tstring* record);

#if defined(TF_CORD_SUPPORT)
  Status ReadRecord(absl::Cord* record);
#endif

  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
  const int version_;
  const DataTypeVector dtypes_;
  int num_simple_ = 0;
  int num_complex_ = 0;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
};

// Writes snapshot metadata to the given directory.
Status WriteMetadataFile(Env* env, const string& dir,
                         const experimental::SnapshotMetadataRecord* metadata);

// Writes distributed snapshot metadata to the given directory. An error is
// returned if `dir` is unable to be created or if `metadata` is unable to be
// written.
Status WriteMetadataFile(
    Env* env, const string& dir,
    const experimental::DistributedSnapshotMetadata* metadata);

// Reads snapshot metadata from the given directory.
Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists);

// Reads distributed snapshot metadata from the given directory. If the file
// doesn't exist in `dir`, `file_exists` is set to true and an ok status is
// returned. If the file exists in `dir` but is unable to be opened, an error
// is returned.
Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::DistributedSnapshotMetadata* metadata,
                        bool* file_exists);

// Writes a dataset graph to the given directory.
Status DumpDatasetGraph(Env* env, const std::string& path, uint64 hash,
                        const GraphDef* graph);

Status DetermineOpState(const std::string& mode_string, bool file_exists,
                        const experimental::SnapshotMetadataRecord* metadata,
                        uint64 pending_snapshot_expiry_seconds, Mode* mode);

// Represents a dataset element or EOF.
struct ElementOrEOF {
  std::vector<Tensor> value;
  bool end_of_sequence = false;
};

// AsyncWriter provides API for asynchronously writing dataset elements
// (each represented as a vector of tensors) to a file.
//
// The expected use of this API is:
//
// std::unique_ptr<AsyncWriter> writer = absl_make_unique<AsyncWriter>(...);
//
// while (data_available()) {
//   std::vector<Tensor> data = read_data()
//   writer->Write(data);
// }
// writer->SignalEOF();
// writer = nullptr;  // This will block until writes are flushed.
class AsyncWriter {
 public:
  explicit AsyncWriter(Env* env, int64_t file_index,
                       const std::string& shard_directory, uint64 checkpoint_id,
                       const std::string& compression, int64_t version,
                       const DataTypeVector& output_types,
                       std::function<void(Status)> done);

  // Writes the given tensors. The method is non-blocking and returns without
  // waiting for the element to be written.
  void Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_);

  // Signals the end of input. The method is non-blocking and returns without
  // waiting for the writer to be closed.
  void SignalEOF() TF_LOCKS_EXCLUDED(mu_);

 private:
  void Consume(ElementOrEOF* be) TF_LOCKS_EXCLUDED(mu_);
  bool ElementAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status WriterThread(Env* env, const std::string& shard_directory,
                      uint64 checkpoint_id, const std::string& compression,
                      int64_t version, DataTypeVector output_types);

  mutex mu_;
  std::deque<ElementOrEOF> deque_ TF_GUARDED_BY(mu_);

  // This has to be last. During destruction, we need to make sure that the
  // Thread object is destroyed first as its destructor blocks on thread
  // completion. If there are other member variables after this, they may get
  // destroyed first before the thread finishes, potentially causing the
  // thread to access invalid memory.
  std::unique_ptr<Thread> thread_;
};

}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_
