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

#include "tensor.h"
#include <map>
#include <vector>

namespace ignite {

struct BinaryField {
  std::string field_name;
  int type_id;
  int field_id;
};

struct BinaryType {
  int type_id;
  std::string type_name;
  int field_cnt;
  BinaryField** fields;
};

class BinaryObjectParser {
 public:
  char* Parse(char *ptr, std::map<int, BinaryType*>* types, std::vector<tensorflow::Tensor>* out_tensors);
 private:
  char ReadByte(char*& ptr);
  short ReadShort(char*& ptr);
  int ReadInt(char*& ptr);
  long ReadLong(char*& ptr);
};

} // namespace ignite