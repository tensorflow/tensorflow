//===- Location.h - MLIR Location Classes -----------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// These classes provide the ability to relate MLIR objects back to source
// location position information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_LOCATION_H
#define MLIR_IR_LOCATION_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;

/// Location objects represent source locations information in MLIR.
class alignas(8) Location {
public:
  enum class Kind {
    /// This represents an unknown location.
    Unknown,

    /// This represents a file/line/column location.
    FileLineCol,

    // Represents a location as a 'void*' pointer to a front-end's opaque
    // location information, which must live longer than the MLIR objects that
    // refer to it.  OpaqueLocation's are never serialized.
    //
    // TODO: OpaqueLocation,

    // Represents a value inlined through a function call.
    // TODO: InlinedLocation,

    // Represents a value composed of multiple source constructs.
    // TODO: FusedLocation,
  };

  /// Return the classification for this location.
  Kind getKind() const { return kind; }

  /// Print the location.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  explicit Location(Kind kind) : kind(kind) {}
  ~Location() {}

private:
  /// Classification of the subclass, used for type checking.
  Kind kind : 8;

  Location(const Location &) = delete;
  void operator=(const Location &) = delete;
};

inline raw_ostream &operator<<(raw_ostream &os, const Location &loc) {
  loc.print(os);
  return os;
}

/// Represents an unknown location.  This is always a singleton for a given
/// MLIRContext.
class UnknownLoc : public Location {
public:
  static UnknownLoc *get(MLIRContext *context);

private:
  explicit UnknownLoc() : Location(Kind::Unknown) {}
};

/// This class is used to represent a uniqued filename in an MLIRContext.  It is
/// a simple wrapper around a const char* to uniqued string memory.
class UniquedFilename {
public:
  /// Unique the specified filename and return a stable pointer owned by the
  /// specified context.  The filename is not allowed to contain embedded ASCII
  /// nul (\0) characters.
  static UniquedFilename get(StringRef filename, MLIRContext *context);

  StringRef getRef() const { return string; }
  const char *data() const { return string; }

private:
  explicit UniquedFilename(const char *string) : string(string) {}
  const char *string;
};

/// Represents a location derived from a file/line/column location.  The column
/// and line may be zero to represent unknown column and/or unknown line/column
/// information.
class FileLineColLoc : public Location {
public:
  /// Return a uniqued FileLineCol location object.
  static FileLineColLoc *get(UniquedFilename filename, unsigned line,
                             unsigned column, MLIRContext *context);

  StringRef getFilename() const { return filename.getRef(); }

  unsigned getLine() const { return line; }
  unsigned getColumn() const { return column; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Location *loc) {
    return loc->getKind() == Kind::FileLineCol;
  }

private:
  FileLineColLoc(UniquedFilename filename, unsigned line, unsigned column)
      : Location(Kind::FileLineCol), filename(filename), line(line),
        column(column) {}
  ~FileLineColLoc() = delete;

  const UniquedFilename filename;
  const unsigned line, column;
};

} // end namespace mlir

#endif
