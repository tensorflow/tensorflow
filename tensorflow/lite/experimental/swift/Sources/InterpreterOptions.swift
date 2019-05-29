// Copyright 2018 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Custom configuration options for a TensorFlow Lite `Interpreter`.
public struct InterpreterOptions: Equatable {

  /// Maximum number of CPU threads that the interpreter should run on. Default is `nil` which
  /// indicates that the `Interpreter` will decide the number of threads to use.
  public var threadCount: Int? = nil

  /// Creates a new instance of interpreter options.
  public init() {}
}
