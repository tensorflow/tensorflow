/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_TABLE_ACCESSOR_TEST_DATA_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_TABLE_ACCESSOR_TEST_DATA_H_

#include <string>

namespace tensorflow {
namespace {

const string kSampleSchema = R"({
  "kind": "bigquery#table",
  "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
  "id": "test-project:test-dataset.test-table",
  "schema": {
    "fields": [
    {
      "name": "int_field",
      "type": "INTEGER",
      "mode": "REQUIRED"
    },{
      "name": "str_field",
      "type": "STRING",
      "mode": "NULLABLE"
    },{
     "name": "rec_field",
     "type": "RECORD",
     "fields": [
     {
       "name": "float_field",
       "type": "FLOAT",
       "mode": "NULLABLE"
      }]
    },{
      "name": "bool_field",
      "type": "BOOLEAN",
      "mode": "NULLABLE"
    },{
      "name": "bytes_field",
      "type": "BYTES",
      "mode": "NULLABLE"
    },{
      "name": "timestamp_field",
      "type": "TIMESTAMP",
      "mode": "NULLABLE"
    },{
      "name": "date_field",
      "type": "DATE",
      "mode": "NULLABLE"
    },{
      "name": "time_field",
      "type": "TIME",
      "mode": "NULLABLE"
    },{
      "name": "datetime_field",
      "type": "DATETIME",
      "mode": "NULLABLE"
    }]
  },
  "numRows": "4"
})";

const string kTestRow = R"({
  "kind": "bigquery#table",
  "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
  "id": "test-project:test-dataset.test-table",
  "rows": [
  {
    "f": [
    {
      "v": "1234"
    },{
      "v": ""
    },{
      "v": {
        "f": [
        {
          "v": "1.23456"
        }]
      }
    },{
      "v": "true"
    },{
      "v": "01010100101"
    },{
      "v": "timestamp"
    },{
      "v": "date"
    },{
      "v": "time"
    },{
      "v": "datetime"
    }]}]})";

const string kBrokenTestRow = R"({
  "kind": "bigquery#table",
  "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
  "id": "test-project:test-dataset.test-table",
  "rows": [
  {
    "f": [
    {
      "v": "1-234"   // This does not parse as integer.
    },{
      "v": ""
    },{
    },{
      "v": "true"
    },{
      "v": "01010100101"
    },{
      "v": "timestamp"
    },{
      "v": "date"
    },{
      "v": "time"
    },{
      "v": "datetime"
    }]}]})";

const string kTestRowWithNulls = R"({
  "kind": "bigquery#table",
  "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
  "id": "test-project:test-dataset.test-table",
  "rows": [
  {
    "f": [
    {
      "v": "1234"
    },{
      "v": "string"
    },{
      "v": null
    },{
      "v": "true"
    },{
      "v": "01010100101"
    },{
      "v": ""
    },{
      "v": null
    },{
      "v": null
    },{
      "v": "datetime"
    }]}]})";

// Example proto corresponding to kTestRow.
const string kTestExampleProto = R"(features {
  feature {
    key: "bool_field"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "bytes_field"
    value {
      bytes_list {
        value: "01010100101"
      }
    }
  }
  feature {
    key: "date_field"
    value {
      bytes_list {
        value: "date"
      }
    }
  }
  feature {
    key: "datetime_field"
    value {
      bytes_list {
        value: "datetime"
      }
    }
  }
  feature {
    key: "int_field"
    value {
      int64_list {
        value: 1234
      }
    }
  }
  feature {
    key: "rec_field.float_field"
    value {
      float_list {
        value: 1.23456
      }
    }
  }
  feature {
    key: "str_field"
    value {
      bytes_list {
        value: ""
      }
    }
  }
  feature {
    key: "time_field"
    value {
      bytes_list {
        value: "time"
      }
    }
  }
  feature {
    key: "timestamp_field"
    value {
      bytes_list {
        value: "timestamp"
      }
    }
  }
}
)";

// Example proto corresponding to kTestRowWithNulls.
const string kTestExampleProtoWithNulls = R"(features {
  feature {
    key: "bool_field"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "bytes_field"
    value {
      bytes_list {
        value: "01010100101"
      }
    }
  }
  feature {
    key: "datetime_field"
    value {
      bytes_list {
        value: "datetime"
      }
    }
  }
  feature {
    key: "int_field"
    value {
      int64_list {
        value: 1234
      }
    }
  }
  feature {
    key: "timestamp_field"
    value {
      bytes_list {
        value: ""
      }
    }
  }
  feature {
    key: "str_field"
    value {
      bytes_list {
        value: "string"
      }
    }
  }
}
)";

// Example proto corresponding to part of kTestRow.
const string kTestPartialExampleProto = R"(features {
  feature {
    key: "bool_field"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "rec_field.float_field"
    value {
      float_list {
        value: 1.23456
      }
    }
  }
}
)";

const string kTestTwoRows = R"({
  "kind": "bigquery#table",
  "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
  "pageToken": "next_page",
  "id": "test-project:test-dataset.test-table",
  "rows": [
    {"f": [{"v": "1111"},{},{},{},{},{},{},{},{}]},
    {"f": [{"v": "2222"},{},{},{},{},{},{},{},{}]}
  ]})";

}  // namespace
}  // namepsace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_TABLE_ACCESSOR_TEST_DATA_H_
