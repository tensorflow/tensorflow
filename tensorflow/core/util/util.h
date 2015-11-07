#ifndef TENSORFLOW_UTIL_UTIL_H_
#define TENSORFLOW_UTIL_UTIL_H_

#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// If op_name has '/' in it, then return everything before the first '/'.
// Otherwise return empty string.
StringPiece NodeNamePrefix(const StringPiece& op_name);

// If op_name has '/' in it, then return everything before the last '/'.
// Otherwise return empty string.
StringPiece NodeNameFullPrefix(const StringPiece& op_name);

class MovingAverage {
 public:
  explicit MovingAverage(int window);
  ~MovingAverage();

  void Clear();

  double GetAverage() const;
  void AddValue(double v);

 private:
  const int window_;  // Max size of interval
  double sum_;        // Sum over interval
  double* data_;      // Actual data values
  int head_;          // Offset of the newest statistic in data_
  int count_;         // # of valid data elements in window
};

// Returns a string printing bytes in ptr[0..n).  The output looks
// like "00 01 ef cd cd ef".
string PrintMemory(const char* ptr, int n);

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_UTIL_H_
