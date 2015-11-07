#ifndef TENSORFLOW_FRAMEWORK_READER_INTERFACE_H_
#define TENSORFLOW_FRAMEWORK_READER_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class QueueInterface;
class ReaderInterface;

// Readers are the mechanism for reading records from files in
// TensorFlow graphs.  Each supported file format has a corresponding
// ReaderInterface descendant and a corresponding Op & OpKernel
// (implemented using ReaderOpKernel from reader_op_kernel.h).
//
// To use a Reader, you first encode "work" (some string, typically a
// filename) in the Reader's "work queue".  It then processes the
// "work" (reading records from the file), to produce key/value
// strings.  The methods of this class are called by ReaderFoo ops,
// so see ../ops/io_ops.cc for detailed descriptions.
//
// All descendants of this class must be thread-safe.
//
// See the design document here:
// https://docs.google.com/document/d/1UAgZOoeehYr20TdzW2CoZ30V-aqQphU4SwKXsW7eJv4/edit#

// TODO(josh11b): Switch this to Async.
class ReaderInterface : public ResourceBase {
 public:
  // Read a single record into *key / *value.  May get more work from
  // *queue if the current work is complete.  Sets the status on
  // *context with an OutOfRange Status if the the current work is
  // complete and the queue is done (closed and empty).
  // This method may block.
  virtual void Read(QueueInterface* queue, string* key, string* value,
                    OpKernelContext* context) = 0;

  // Restore this reader to its newly-constructed state.
  virtual Status Reset() = 0;

  // Accessors
  virtual int64 NumRecordsProduced() = 0;
  virtual int64 NumWorkUnitsCompleted() = 0;

  // -- Serialization/Restoration support --
  // Not all readers will support saving and restoring state.
  virtual Status SerializeState(string* state) = 0;
  // Note: Must Reset on error.
  virtual Status RestoreState(const string& state) = 0;

  string DebugString() override { return "a reader"; }

 protected:
  virtual ~ReaderInterface() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_READER_INTERFACE_H_
