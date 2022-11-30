/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file should not have any dependencies apart from the standard library,
// as it will be used in OSS outside of this repository.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

static constexpr int kSeed = 42;
static constexpr int kUpperBound = 100;
static constexpr int kLowerBound = -100;
static constexpr double kLowerBoundFP = -0.1;
static constexpr double kUpperBoundFP = 0.1;
static const char* const kUsageString = R"(
Driver for executing an HLO reproducer in object form in order to let OSS
users reproduce the miscompiles.

Expected workflow:

1) In the .hlo file, rename the root computation to `EntryModule`.
2) Run the .hlo file with XLA_FLAGS=--xla_dump_to set, to obtain the .ll file.
3) Compile and link this file with the object file from step (2).
4) Run the resulting file with the buffer assignment table as an argument,
taken from step 2. The driver will print the output to stderr.
5) Compare the output with optimized and non-optimized .ll file from step (2).
If the outputs differ, there is a miscompile.

Run with an environment variable VERBOSE set to see logging.
)";

// Must be kept ABI-compatible with the real definition of XlaCustomCallStatus.
struct XlaCustomCallStatus {
  // If 'failed' is true then 'message' is present; otherwise it is absent.
  // (The 'bool' followed by 'std::string' is ABI-compatible with
  // 'std::optional<std::string>').
  bool failed;
  std::string message;
  // To account for extra struct padding at the end.
  std::string padding;
};

extern "C" {
// Function to be linked with.
extern void EntryModule(char* result_buffer, char* run_opts, char** params,
                        char** buffer_table, void* status,
                        int64_t* prof_counters);

// Must be kept in sync with the real definition of this runtime function.
bool __xla_cpu_runtime_StatusIsSuccess(  // NOLINT: This doesn't need a
                                         // prototype.
    const XlaCustomCallStatus* status) {
  return !(status->failed);
}
}

namespace {

[[noreturn]] void ExitWithMsg(const std::string& msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

void Check(bool cond, const std::string& msg = "Precondition failed") {
  if (!cond) {
    ExitWithMsg(msg);
  }
}

bool IsVerbose() { return getenv("VERBOSE") != nullptr; }

void Log(const std::string& msg) {
  if (IsVerbose()) {
    std::cerr << msg << std::endl;
  }
}

// Needs to be kept in sync with PrimitiveType in xla_data.proto.
enum PrimitiveType {
  S16 = 0,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  F16,
  BF16,
  F32,
  F64,
  C64,
  C128
};

const std::vector<std::string>& primitive_strings() {
  static auto vec = new std::vector<std::string>(
      {"s16", "s32", "s64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32",
       "f64", "c64", "c128"});
  return *vec;
}

std::string ToString(PrimitiveType type) { return primitive_strings()[type]; }

PrimitiveType PrimitiveTypeFromString(const std::string& s) {
  const auto& vec = primitive_strings();
  return static_cast<PrimitiveType>(
      std::distance(vec.begin(), std::find(vec.begin(), vec.end(), s)));
}

int ByteSize(PrimitiveType type) {
  std::string s = ToString(type);
  s = s.substr(1, s.size());
  return std::stoi(s) / 8;
}

struct ArrayShape {
  PrimitiveType type;
  std::vector<int> dimensions;
};

// We support tuples only for output, and we do not support nested tuples.
struct TupleShape {
  std::vector<ArrayShape> elements;
};

std::string ArrayShapeToString(ArrayShape shape) {
  std::ostringstream out;
  out << ToString(shape.type) << "[";
  for (int i = 0; i < shape.dimensions.size(); i++) {
    out << std::to_string(shape.dimensions[i]);
    if (i != shape.dimensions.size() - 1) {
      out << ",";
    }
  }
  out << "]";
  return out.str();
}

// Input: TYPE[D1,D2,...DN]
ArrayShape ArrayShapeFromString(const std::string& s) {
  Log("Array shape from string: " + s);
  Check(s.find('(') == std::string::npos, "Tuple shape is not supported");
  std::regex shape_r("([^\\[]+)\\[(.*)\\]");
  std::smatch match;
  Check(std::regex_match(s, match, shape_r), "Shape not found");
  std::string type = match[1];
  std::string dims = match[2];
  PrimitiveType ptype = PrimitiveTypeFromString(type);
  std::istringstream dims_stream(dims);
  std::string dim;
  std::vector<int> dimensions;
  while (std::getline(dims_stream, dim, ',')) {
    dimensions.push_back(std::stoi(dim));
  }
  return {ptype, dimensions};
}

// E.g. (f32[10,20], u32[])
TupleShape TupleShapeFromString(std::string s) {
  Log("Tuple shape from string: " + s);
  if (s[0] != '(') {
    return {{ArrayShapeFromString(s)}};
  }
  s = s.substr(1, s.size() - 2);
  std::istringstream sstream(s);
  std::string subshape;
  std::vector<ArrayShape> out;
  while (std::getline(sstream, subshape, ' ')) {
    if (subshape[subshape.size() - 1] == ',') {
      subshape = subshape.substr(0, subshape.size() - 1);
    }
    out.push_back(ArrayShapeFromString(subshape));
  }
  return {out};
}

std::string TupleShapeToString(TupleShape shape) {
  std::ostringstream out;
  if (shape.elements.size() == 1) {
    return ArrayShapeToString(shape.elements[0]);
  }
  out << "(";
  for (int idx = 0; idx < shape.elements.size(); idx++) {
    out << ArrayShapeToString(shape.elements[idx]);
    if (idx != shape.elements.size() - 1) {
      out << ", ";
    }
  }
  out << ")";
  return out.str();
}

// Information about the buffer assignment.
struct BufferAssignment {
  // Mapping from allocation index to buffer size (in bytes).
  std::vector<int> buffers_size;

  // Mapping from allocation index to its shape.
  std::map<int, TupleShape> buffers_shape;

  // Mapping from param index to allocation index.
  std::map<int, int> param_to_alloc_idx;

  // Index of the output parameter.
  int output_idx = -1;
};

std::string BufferAssignmentToString(const BufferAssignment& assignment) {
  std::ostringstream out;
  for (const auto& p : assignment.param_to_alloc_idx) {
    int param_idx = p.first;
    int allocation_idx = p.second;
    out << "Param: " << param_idx << " (allocation " << allocation_idx << "): ";
    auto p2 = assignment.buffers_shape.find(allocation_idx);
    Check(p2 != assignment.buffers_shape.end(),
          "Shape not found for parameter: " + std::to_string(param_idx));
    out << TupleShapeToString(p2->second)
        << ", size = " << assignment.buffers_size[allocation_idx] << "\n";
  }
  return out.str();
}

// RAII table for the given assignment: mapping from a allocation idx to the
// actual allocation.
class BufferTable {
 public:
  explicit BufferTable(BufferAssignment assignment) : assignment_(assignment) {
    int num_buffers = assignment.buffers_size.size();
    ptr_ = new char*[num_buffers];
    for (int buffer_idx = 0; buffer_idx < num_buffers; buffer_idx++) {
      // Call malloc to ensure alignment up to std::max_align_t.
      ptr_[buffer_idx] =
          static_cast<char*>(malloc(assignment.buffers_size[buffer_idx]));
    }
  }

  char** AsPtr() { return ptr_; }

  ~BufferTable() {
    int num_buffers = assignment_.buffers_size.size();
    for (int buffer_idx = 0; buffer_idx < num_buffers; buffer_idx++) {
      free(ptr_[buffer_idx]);
    }
    delete[] ptr_;
  }

 private:
  BufferAssignment assignment_;
  char** ptr_;
};

// Parse and populate the buffer table;
//
// Example of input:
//
// BufferAssignment:
// allocation 0: size 32768, parameter 0, shape f32[256,32] at
// ShapeIndex {}:
//  value: <3 parameter @0> (size=32768,offset=0): f32[256,32]{1,0}
// allocation 1: size 128, output shape is f32[32],
// maybe-live-out:
//  value: <5 reduce @0> (size=128,offset=0): f32[32]{0}
// allocation 2: size 4, constant:
//  value: <4 init_value @0> (size=4,offset=0): f32[]
// allocation 3: size 4, thread-local:
//  value: <0 x.1 @0> (size=4,offset=0): f32[]
// allocation 4: size 4, thread-local:
//  value: <1 y.1 @0> (size=4,offset=0): f32[]
// allocation 5: size 4, output shape is f32[], thread-local:
//  value: <2 add.1 @0> (size=4,offset=0): f32[]
BufferAssignment ParseBufferAssignment(const std::string& fname) {
  BufferAssignment assignment;
  std::ifstream infile(fname);
  std::string line;
  while (std::getline(infile, line)) {
    std::regex allocation_line_r("allocation ([0-9]+): size ([0-9]+), (.+)");
    std::smatch match;
    if (std::regex_search(line, match, allocation_line_r)) {
      Log("Matched allocation description: " + line);
      int allocation_idx = std::stoi(match[1]);
      int size = std::stoi(match[2]);
      Log("Allocation size = " + std::to_string(size));
      const std::string& postfix = match[3];
      Check(allocation_idx == assignment.buffers_size.size(),
            "Unordered allocations in input");
      assignment.buffers_size.push_back(size);

      std::regex output_r("output shape is \\|([^\\|]+)\\|,");
      std::smatch output_match;
      if (std::regex_search(postfix, output_match, output_r)) {
        Log("Matched out parameter: " + postfix);
        Check(assignment.output_idx == -1, "Multiple out-parameters");
        assignment.output_idx = allocation_idx;
        std::string output_shape = output_match[1];
        Log("output shape = " + output_shape);
        TupleShape shape = TupleShapeFromString(output_shape);
        assignment.buffers_shape[allocation_idx] = shape;
        Log("parsed output shape = " + TupleShapeToString(shape));
      }

      std::regex parameter_r("parameter ([0-9]+), shape \\|([^\\|]+)\\|");
      std::smatch param_match;
      if (std::regex_search(postfix, param_match, parameter_r)) {
        Log("Matched parameter description: " + postfix);
        int param_idx = std::stoi(param_match[1]);
        assignment.param_to_alloc_idx[param_idx] = allocation_idx;
        std::string param_shape = param_match[2];
        TupleShape shape = TupleShapeFromString(param_shape);
        assignment.buffers_shape[allocation_idx] = shape;
        Log("parsed parameter shape for param " + std::to_string(param_idx) +
            " = " + TupleShapeToString(shape));
      }
    }
  }
  Check(assignment.output_idx != -1, "Output not set");
  return assignment;
}

int GetNumElements(const ArrayShape& shape) {
  int num_elements = 1;
  for (int dim : shape.dimensions) {
    num_elements *= dim;
  }
  return num_elements;
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
void FillIntT(void* buffer, int num_elements) {
  std::mt19937 generator(kSeed);
  T* casted = static_cast<T*>(buffer);
  std::uniform_int_distribution<> distr(kLowerBound, kUpperBound);
  for (int i = 0; i < num_elements; i++) {
    casted[i] = static_cast<T>(distr(generator));
  }
}

template <typename T,
          typename = std::enable_if_t<std::is_floating_point<T>::value>>
void FillFloatT(void* buffer, int num_elements) {
  std::mt19937 generator(kSeed);
  T* casted = static_cast<T*>(buffer);
  std::uniform_real_distribution<T> distr(kLowerBoundFP, kUpperBoundFP);
  for (int i = 0; i < num_elements; i++) {
    casted[i] = distr(generator);
  }
}

void Fill(void* buffer, const ArrayShape& shape) {
  int num_elements = GetNumElements(shape);
  Log("Number of elements = " + std::to_string(num_elements));
  Log("Shape type = " + ToString(shape.type) +
      ", shape = " + ArrayShapeToString(shape));
  switch (shape.type) {
    case S16:
      return FillIntT<short>(buffer, num_elements);  // NOLINT
    case S32:
      return FillIntT<int>(buffer, num_elements);
    case S64:
      return FillIntT<long long>(buffer, num_elements);  // NOLINT
    case U8:
      return FillIntT<unsigned char>(buffer, num_elements);
    case U16:
      return FillIntT<unsigned short>(buffer, num_elements);  // NOLINT
    case U32:
      return FillIntT<unsigned int>(buffer, num_elements);
    case U64:
      return FillIntT<unsigned long long>(buffer, num_elements);  // NOLINT
    case F32:
      return FillFloatT<float>(buffer, num_elements);
    case F64:
      return FillFloatT<double>(buffer, num_elements);

    case F16:
    case BF16:
    case C64:
    case C128:
      ExitWithMsg("Unsupported type: " + ToString(shape.type));
  }
}

template <typename T>
#if defined(MEMORY_SANITIZER)
__attribute__((no_sanitize_memory))
#endif
void DisplayT(const void* buffer, int num_elements) {
  const T* casted = static_cast<const T*>(buffer);
  for (int i = 0; i < num_elements; i++) {
    std::cout << casted[i];
    if (i != num_elements - 1) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;
}

void Display(const void* buffer, const ArrayShape& shape) {
  int num_elements = GetNumElements(shape);
  switch (shape.type) {
    case S16:
      return DisplayT<short>(buffer, num_elements);  // NOLINT
    case S32:
      return DisplayT<int>(buffer, num_elements);
    case S64:
      return DisplayT<long long>(buffer, num_elements);  // NOLINT
    case U8:
      return DisplayT<unsigned char>(buffer, num_elements);
    case U16:
      return DisplayT<unsigned short>(buffer, num_elements);  // NOLINT
    case U32:
      return DisplayT<unsigned int>(buffer, num_elements);
    case U64:
      return DisplayT<unsigned long long>(buffer, num_elements);  // NOLINT
    case F32:
      return DisplayT<float>(buffer, num_elements);
    case F64:
      return DisplayT<double>(buffer, num_elements);

    case F16:
    case BF16:
    case C64:
    case C128:
      ExitWithMsg("Unsupported type: " + ToString(shape.type));
  }
}

void Display(const void* buffer, const TupleShape& shape) {
  if (shape.elements.size() == 1) {
    return Display(buffer, shape.elements[0]);
  }
  std::cout << "(" << std::endl;
  auto casted = static_cast<const void* const*>(buffer);
  for (int tuple_idx = 0; tuple_idx < shape.elements.size(); tuple_idx++) {
    ArrayShape array_shape = shape.elements[tuple_idx];
    Display(casted[tuple_idx], array_shape);
    if (tuple_idx != shape.elements.size() - 1) {
      std::cout << ", " << std::endl;
    }
  }
  std::cout << ")" << std::endl;
}

}  // end namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    ExitWithMsg(
        "Please provide buffer table filename as an argument, "
        "or invoke with --help for usage instructions.");
  }
  std::string arg = argv[1];
  if (arg == "--help") {
    std::cout << kUsageString << std::endl;
    return 0;
  }

  BufferAssignment assignment = ParseBufferAssignment(arg);
  Log("Buffer assignment: \n" + BufferAssignmentToString(assignment));
  BufferTable table(assignment);

  // Fill out input parameters.
  for (const auto& p : assignment.param_to_alloc_idx) {
    int param_idx = p.first;
    int allocation_idx = p.second;
    TupleShape tuple_shape = assignment.buffers_shape[allocation_idx];
    Check(tuple_shape.elements.size() == 1,
          "Parameters can not be tuples, got shape: " +
              TupleShapeToString(tuple_shape));
    ArrayShape shape = tuple_shape.elements[0];
    Check(GetNumElements(shape) ==
              assignment.buffers_size[allocation_idx] / ByteSize(shape.type),
          "Unexpected number of elements");
    Fill(table.AsPtr()[allocation_idx], shape);

    if (IsVerbose()) {
      std::cout << "Filled parameter buffer for param " << param_idx << ": "
                << std::endl;
      Display(table.AsPtr()[allocation_idx], shape);
    }
  }

  XlaCustomCallStatus status;

  Log("Launching module");
  EntryModule(/*result_buffer=*/nullptr,
              /*run_opts=*/nullptr,
              /*params=*/nullptr, table.AsPtr(),
              /*status=*/&status,
              /*prof_counters=*/nullptr);

  std::cout << "Output:" << std::endl;
  Log("Output shape: " +
      TupleShapeToString(assignment.buffers_shape[assignment.output_idx]));
  Display(table.AsPtr()[assignment.output_idx],
          assignment.buffers_shape[assignment.output_idx]);
}
