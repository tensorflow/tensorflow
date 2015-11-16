// The utility to read checkpoints for google brain tensor ops and v3
// checkpoints for dist_belief.
//

#ifndef TENSORFLOW_UTIL_TENSOR_SLICE_READER_CACHE_H_
#define TENSORFLOW_UTIL_TENSOR_SLICE_READER_CACHE_H_

#include <unordered_map>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReaderCache;

// Wrapper to a lazily allocated TensorSliceReaderCache.
class TensorSliceReaderCacheWrapper {
 public:
  TensorSliceReaderCacheWrapper();
  ~TensorSliceReaderCacheWrapper();

  // Same as TensorSliceReaderCache::GetReader().
  const TensorSliceReader* GetReader(
      const string& filepattern,
      TensorSliceReader::OpenTableFunction open_function,
      int preferred_shard) const;

 private:
  mutable mutex mu_;
  mutable TensorSliceReaderCache* cache_ = nullptr;
};

// A cache of TensorSliceReaders.
class TensorSliceReaderCache {
 public:
  TensorSliceReaderCache();
  ~TensorSliceReaderCache();

  // Returns the TensorSliceReader corresponding to 'filepattern' and the
  // open_function.  May return nullptr if we can not create a new
  // TensorSliceReader for the filepattern/open_function combination.
  const TensorSliceReader* GetReader(
      const string& filepattern,
      TensorSliceReader::OpenTableFunction open_function, int preferred_shard);

 private:
  // Need to use a regular function type in the key map as std::function does
  // not support ==.
  typedef Status (*OpenFuncType)(const string&, TensorSliceReader::Table**);

  // Protects attributes below.
  mutex mu_;

  // Maps of opened readers.
  std::unordered_map<string, std::pair<OpenFuncType, TensorSliceReader*>>
      readers_;

  // Set of keys that a previous GetReader() call is still trying to populate.
  std::set<string> still_opening_;

  // Condition variable to notify when a reader has been created.
  condition_variable cv_;
};

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_SLICE_READER_CACHE_H_
