/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_helper.h"

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>

#include <string>
#include <string_view>

#include "tensorflow/c/tf_status.h"

namespace tf_s3_filesystem {

  namespace {
    char ascii_tolower(char in) {
      if (in <= 'Z' && in >= 'A')
        return in - ('Z' - 'z');
      return in;
    }
  }  // namespace

  std::string AsciiStrToLower(const std::string_view& s) {
    std::string result(s);
    for (auto& ch : result) {
      ch = ascii_tolower(ch);
    }
    return result;
  }

  void TF_SetStatusFromAWSError(TF_Status* status, const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
    if (error.GetResponseCode() == Aws::Http::HttpResponseCode::FORBIDDEN) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "AWS Credentials have not been set properly.\nUnable to access the specified S3 location");
    } else {
      TF_SetStatus(status, TF_UNKNOWN, std::string(error.GetExceptionName() + ": " + error.GetMessage()).c_str());
    }
  }

  void GetParentDir(const char* name, char** parent) {
    size_t idx = 0;
    for(int i = strlen(name) - 2; i >= 0; --i) {
        if(name[i] == '/') {
            idx = i;
            break;
        }
    }
    *parent = (char*) malloc(idx + 2);
    sprintf(*parent, "%.*s", idx + 1, name);
    (*parent)[idx + 1] = '\0';
  }

  void GetParentFile(const char* name, char** parent) {
        size_t idx = 0;
    for(int i = strlen(name) - 2; i >= 0; --i) {
        if(name[i] == '/') {
            idx = i;
            break;
        }
    }
    *parent = (char*) malloc(idx + 1);
    sprintf(*parent, "%.*s", idx, name);
    (*parent)[idx] = '\0';
  }

void ParseS3Test(const char* fname, bool object_empty_ok, char** bucket, char** object, TF_Status* status) {
  size_t scheme_index = strcspn(fname, "://");
  char* scheme = (char*) malloc(scheme_index + 1);
  sprintf(scheme, "%.*s", (int)scheme_index, fname);
  scheme[scheme_index] = '\0';
  if(strcmp(scheme, "s3")) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "S3 path doesn't start with 's3://': ");
    return;
  }

  size_t bucket_index = strcspn(fname + scheme_index + 3, "/");
  if(!bucket_index) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "S3 path doesn't contain a bucket name: ");
    return;
  }
  *bucket = (char*) malloc(bucket_index + 1);
  sprintf(*bucket, "%.*s", (int) bucket_index, fname + scheme_index + 3);
  (*bucket)[bucket_index] = '\0';

  size_t object_index = strlen(fname + scheme_index + 3 + bucket_index + 1);
  if(object_index == 0) {
    if(object_empty_ok) {
      TF_SetStatus(status, TF_OK, "");
      *object = nullptr;
      return;
    }
    else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "S3 path doesn't contain an object name: ");
      return;
    }
  }
  *object = (char*) malloc(object_index + 1);
  sprintf(*object, "%.*s", (int) object_index, fname + scheme_index + 3 + bucket_index + 1);
  (*object)[object_index] = '\0';

  free(scheme);
  TF_SetStatus(status, TF_OK, "");
}
  
}  // namespace tf_s3_filesystem
