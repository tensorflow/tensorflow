// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the License_LevelDB.txt file. See the Authors_LevelDB.txt file for names of contributors.

// Modifications:
// Copyright 2015 The TensorFlow Authors. All Rights Reserved.
// ===========================================================================

// TableBuilder provides the interface used to build a Table
// (an immutable and sorted map from keys to values).
//
// Multiple threads can invoke const methods on a TableBuilder without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same TableBuilder must use
// external synchronization.

#ifndef TENSORFLOW_CORE_LIB_IO_TABLE_BUILDER_H_
#define TENSORFLOW_CORE_LIB_IO_TABLE_BUILDER_H_

#include <stdint.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/table_options.h"

namespace tensorflow {
class WritableFile;
namespace table {

class BlockBuilder;
class BlockHandle;

class TableBuilder {
 public:
  // Create a builder that will store the contents of the table it is
  // building in *file.  Does not close the file.  It is up to the
  // caller to close the file after calling Finish().
  TableBuilder(const Options& options, WritableFile* file);

  // REQUIRES: Either Finish() or Abandon() has been called.
  ~TableBuilder();

  // Add key,value to the table being constructed.
  // REQUIRES: key is after any previously added key in lexicographic order.
  // REQUIRES: Finish(), Abandon() have not been called
  void Add(const StringPiece& key, const StringPiece& value);

  // Advanced operation: writes any buffered key/value pairs to file.
  // Can be used to ensure that two adjacent entries never live in
  // the same data block.  Most clients should not need to use this method.
  // Does not flush the file itself.
  // REQUIRES: Finish(), Abandon() have not been called
  void Flush();

  // Return non-ok iff some error has been detected.
  Status status() const;

  // Finish building the table.  Stops using the file passed to the
  // constructor after this function returns.
  // REQUIRES: Finish(), Abandon() have not been called
  Status Finish();

  // Indicate that the contents of this builder should be abandoned.  Stops
  // using the file passed to the constructor after this function returns.
  // If the caller is not going to call Finish(), it must call Abandon()
  // before destroying this builder.
  // REQUIRES: Finish(), Abandon() have not been called
  void Abandon();

  // Number of calls to Add() so far.
  uint64 NumEntries() const;

  // Size of the file generated so far.  If invoked after a successful
  // Finish() call, returns the size of the final generated file.
  uint64 FileSize() const;

 private:
  bool ok() const { return status().ok(); }
  void WriteBlock(BlockBuilder* block, BlockHandle* handle);
  void WriteRawBlock(const StringPiece& data, CompressionType,
                     BlockHandle* handle);

  struct Rep;
  Rep* rep_;

  // No copying allowed
  TableBuilder(const TableBuilder&);
  void operator=(const TableBuilder&);
};

}  // namespace table
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_TABLE_BUILDER_H_
