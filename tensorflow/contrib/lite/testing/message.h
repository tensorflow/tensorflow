/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_MESSAGE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_MESSAGE_H_

#include <memory>
#include <string>
#include <vector>

namespace tflite {
namespace testing {

// A Message is a textual protobuf-like structure that looks like:
//    tag {
//      f : "values"
//      child {
//        a : 1
//       }
//    }
// This class provides the framework for processing message but does not
// associate any particular behavior to fields and submessage. In order
// to properly parse a stream this class must be derived.
class Message {
 public:
  // Reads a stream, tokenizes it and create a new message under the given
  // top-level message. Returns true if the parsing succeeded.
  static bool Read(std::istream* input, Message* message);

  Message() {}
  virtual ~Message() {}

  // Called when a new field is found. For example, when:
  //   f : "values"
  // is found, it triggers:
  //   SetField("f", "values");
  virtual void SetField(const std::string& name, const std::string& value) {}

  // Called when a submessage is started. For example, when:
  //   child {
  // is found, it triggers
  //   AddChild("child");
  // If nullptr is returned, the contents of the submessage will be ignored.
  // Otherwise, the returned Message will be used to handle new fields and new
  // submessages. The caller should not take ownership of the returned pointer.
  virtual Message* AddChild(const std::string& name) { return nullptr; }

  // Called when a submessage is completed, that is, whenever a '}' is found.
  virtual void Finish() {}

 protected:
  // Takes ownership of the given pointer. Subclasses can use this method if
  // they don't want to implement their own ownership semantics.
  Message* Store(Message* n) {
    children_.emplace_back(n);
    return n;
  }

  // Returns a list of all owned submessages.
  const std::vector<std::unique_ptr<Message>>& Children() const {
    return children_;
  }

 private:
  std::vector<std::unique_ptr<Message>> children_;
};

}  // namespace testing
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_MESSAGE_H_
