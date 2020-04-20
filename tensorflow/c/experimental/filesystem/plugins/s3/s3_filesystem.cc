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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_filesystem.h"

#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/CopyObjectRequest.h>

#include <stdlib.h>
#include <sys/mman.h>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>

#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_shared.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_helper.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_copy.h"
#include "tensorflow/c/tf_status.h"

#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_crypto.h"
#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/s3/S3Client.h>
#include <aws/transfer/TransferManager.h>
#include <aws/core/utils/threading/Executor.h>

// Implementation of a filesystem for S3 environments.
// This filesystem will support `s3://` URI scheme.

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct S3File {
	const char* bucket;
	const char* object;
	const std::shared_ptr<Aws::S3::S3Client>& s3_client;
} S3File;

static void Cleanup(TF_RandomAccessFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  // This would be safe to free using `free` directly as it is only opaque.
  // However, it is better to be consistent everywhere.
	// s3_client is not owned by S3File and therefore we do not free it here.
  plugin_memory_free(const_cast<char*>(s3_file->bucket));
	plugin_memory_free(const_cast<char*>(s3_file->object));
  delete s3_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
	auto s3_file = static_cast<S3File*>(file->plugin_file);
	int64_t read = 0;

	Aws::S3::Model::GetObjectRequest getObjectRequest;
	getObjectRequest.WithBucket(s3_file->bucket).WithKey(s3_file->object);
	std::string bytes = "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + n - 1);
	getObjectRequest.SetRange(bytes.c_str());
	getObjectRequest.SetResponseStreamFactory([]() {
    return Aws::New<Aws::StringStream>(tf_s3_filesystem::kS3FileSystemAllocationTag);
  });

	auto getObjectOutcome = s3_file->s3_client->GetObject(getObjectRequest);
	if (!getObjectOutcome.IsSuccess()) {
		auto error = getObjectOutcome.GetError();
    if (error.GetResponseCode() ==
      	Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
			return 0;
    }
		else {
			tf_s3_filesystem::TF_SetStatusFromAWSError(status, error);
			return -1;
		}
  }
  read = getObjectOutcome.GetResult().GetContentLength();
  if(read < n) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
  }

	getObjectOutcome.GetResult().GetBody().read(buffer, read);

	return read;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

static void Sync(const TF_WritableFile* file, TF_Status* status);

typedef struct S3File {
  const char* bucket;
  const char* object;
	const std::shared_ptr<Aws::S3::S3Client>& s3_client;
	const std::shared_ptr<Aws::Transfer::TransferManager>& transfer_manager;
	bool sync_needed;
  std::shared_ptr<Aws::Utils::TempFile> outfile;
} S3File;

static void Cleanup(TF_WritableFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  // This would be safe to free using `free` directly as it is only opaque.
  // However, it is better to be consistent everywhere.
	// s3_client is not owned by S3File and therefore we do not free it here.
  plugin_memory_free(const_cast<char*>(s3_file->bucket));
	plugin_memory_free(const_cast<char*>(s3_file->object));
  delete s3_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
	auto s3_file = static_cast<S3File*>(file->plugin_file);

	if(!s3_file->outfile) {
		TF_SetStatus(status, TF_FAILED_PRECONDITION, "The internal temporary file is not writable.");
		return;
	}
	s3_file->sync_needed = true;
	s3_file->outfile->write(buffer, n);
	if(!s3_file->outfile->good()) {
		TF_SetStatus(status, TF_INTERNAL, "Could not append to the internal temporary file.");
		return;
	}
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);

  Sync(file, status);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);

	if(!s3_file->outfile) {
		TF_SetStatus(status, TF_FAILED_PRECONDITION, "The internal temporary file is not writable.");
		return;
	}
	if(!s3_file->sync_needed) {
		return;
	}

	long offset = s3_file->outfile->tellp();

	std::shared_ptr<Aws::Transfer::TransferHandle> handle =
        s3_file->transfer_manager->UploadFile(
            s3_file->outfile, s3_file->bucket, s3_file->object,
            "application/octet-stream", Aws::Map<Aws::String, Aws::String>());
	handle->WaitUntilFinished();

	int retries = 0;
	while (handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
      	 retries++ < tf_s3_filesystem::kUploadRetries) {
  	// if multipart upload was used, only the failed parts will be re-sent
    s3_file->transfer_manager->RetryUpload(s3_file->outfile, handle);
    handle->WaitUntilFinished();
  }

	if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED) {
    auto error = handle->GetLastError();
    tf_s3_filesystem::TF_SetStatusFromAWSError(status, error);
		return;
  }

	s3_file->outfile->clear();
  s3_file->outfile->seekp(offset);
  s3_file->sync_needed = false;
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);

  if(s3_file->outfile) {
		Sync(file, status);
		if(TF_GetCode(status) == TF_OK) {
			s3_file->outfile.reset();
		}
	}
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct S3MemoryRegion {
  std::unique_ptr<char[]> address;
  const uint64_t length;
} S3MemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return reinterpret_cast<const void*>(r->address.get());
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_s3_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status);

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status);

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status);

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status);

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status);

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status);

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status);
                      
static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status);

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = plugin_memory_allocate(sizeof(S3Shared));
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, false, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, path, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }
  else if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);
	file->plugin_file = new tf_random_access_file::S3File({bucket, object, s3_shared->s3_client});
	TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, false, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, path, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);

	GetS3Client(s3_shared);
	GetTransferManager(s3_shared);
	file->plugin_file = new tf_writable_file::S3File({bucket, object, s3_shared->s3_client, 
																										s3_shared->transfer_manager, true,
																										Aws::MakeShared<Aws::Utils::TempFile>(
																											kS3FileSystemAllocationTag, kS3SuffixFileSystem,
																											std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
																											std::ios_base::out)
																										});

  TF_SetStatus(status, TF_OK, "");
  tf_writable_file::Sync(file, status);
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  
  char* bucket;
  char* object;
	ParseS3Test(path, false, &bucket, &object, status);
  if(TF_GetCode(status) != TF_OK) return;

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);
	GetTransferManager(s3_shared);

	TF_RandomAccessFile reader;
  bool have_reader = false;
	NewRandomAccessFile(filesystem, path, &reader, status);
	if(TF_GetCode(status) == TF_NOT_FOUND) {
    have_reader = false;
  }
  else if(TF_GetCode(status) == TF_OK) {
    have_reader = true;
  }
  else {
    return;
  }


	std::unique_ptr<char[]> buffer(new char[kS3ReadAppendableFileBufferSize]);
	uint64_t offset = 0;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentDir(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

	file->plugin_file = new tf_writable_file::S3File({bucket, object, s3_shared->s3_client,
																										s3_shared->transfer_manager, true,
																										Aws::MakeShared<Aws::Utils::TempFile>(
																											kS3FileSystemAllocationTag, kS3SuffixFileSystem,
																											std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
																											std::ios_base::out)
																										});

	while(have_reader) {
		int64_t  read = tf_random_access_file::Read(&reader, offset, kS3ReadAppendableFileBufferSize, buffer.get(), status);
		if(TF_GetCode(status) == TF_OK) {
			tf_writable_file::Append(file, buffer.get(), read, status);
			offset += kS3ReadAppendableFileBufferSize;
		}
		else if(TF_GetCode(status) == TF_OUT_OF_RANGE) {
			tf_writable_file::Append(file, buffer.get(), read, status);
			break;
		}
		else {
			delete file->plugin_file;
			return;
		}
	}
  if(have_reader) tf_random_access_file::Cleanup(&reader);

  TF_SetStatus(status, TF_OK, "");
  tf_writable_file::Sync(file, status);
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
	auto size = GetFileSize(filesystem, path, status);
	if(TF_GetCode(status) != TF_OK) return;

	std::unique_ptr<char[]> data(new char[size]);
	TF_RandomAccessFile* file;
	NewRandomAccessFile(filesystem, path, file, status);
	if(TF_GetCode(status) != TF_OK) return;

	tf_random_access_file::Read(file, 0, size, data.get(), status);
	if(TF_GetCode(status) != TF_OK) return;

	region->plugin_memory_region = new tf_read_only_memory_region::S3MemoryRegion({std::move(data), static_cast<uint64_t>(size)});
	delete file->plugin_file;
	TF_SetStatus(status, TF_OK, "");
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, path, &stats, status);
  if(TF_GetCode(status) != TF_NOT_FOUND) {
    TF_SetStatus(status, TF_ALREADY_EXISTS, "Path already exists");
    return;
  }

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

	if(!object) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket);
    auto headBucketOutcome = s3_shared->s3_client->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      if(headBucketOutcome.GetError().GetResponseCode() ==
					Aws::Http::HttpResponseCode::FORBIDDEN) {
				TF_SetStatus(status, TF_FAILED_PRECONDITION, "AWS Credentials have not been set properly.\nUnable to access the specified S3 location");
			}
			else {
				TF_SetStatus(status, TF_NOT_FOUND, std::string("The bucket " + std::string(bucket) + " was not found.").c_str());
			}
    }
	}

	std::string filename = path;
  if (filename.back() != '/') {
    filename.push_back('/');
	}
  
	PathExists(filesystem, filename.c_str(), status);
  if(TF_GetCode(status) != TF_OK) {
		TF_WritableFile file;
		NewWritableFile(filesystem, filename.c_str(), &file, status);
		if(TF_GetCode(status) != TF_OK) return;

		tf_writable_file::Close(&file, status);
		if(TF_GetCode(status) != TF_OK) return;
	}
	TF_SetStatus(status, TF_OK, "");
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, path, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Path is a directory");
    return;
  }
  else if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

	Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket).WithKey(object);

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

	auto deleteObjectOutcome =
      s3_shared->s3_client->DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, deleteObjectOutcome.GetError());
		return;
  }
	TF_SetStatus(status, TF_OK, "");
}

static void DeleteDirImpl(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

	Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket).WithKey(object);

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

	auto deleteObjectOutcome =
      s3_shared->s3_client->DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, deleteObjectOutcome.GetError());
		return;
  }
	TF_SetStatus(status, TF_OK, "");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = true;
  Stat(filesystem, path, &stats, status);
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }
  else if(!stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Path is not a directory");
    return;
  }

  std::string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket)
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(2);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome =
      s3_shared->s3_client->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto contents = listObjectsOutcome.GetResult().GetContents();
    if (contents.size() > 1 ||
        (contents.size() == 1 && contents[0].GetKey() != prefix.c_str())) {
      TF_SetStatus(status, TF_FAILED_PRECONDITION, "Cannot delete a non-empty directory.\nThis operation will be retried in case this\nis due to S3's eventual consistency.");
      return;
    }
    if (contents.size() == 1 && contents[0].GetKey() == prefix.c_str()) {
      std::string filename = path;
      if (filename.back() != '/') {
        filename.push_back('/');
      }
      return DeleteDirImpl(filesystem, filename.c_str(), status);
    }
  } else {
		if(listObjectsOutcome.GetError().GetResponseCode() ==
				Aws::Http::HttpResponseCode::FORBIDDEN) {
			TF_SetStatus(status, TF_FAILED_PRECONDITION, "AWS Credentials have not been set properly.\nUnable to access the specified S3 location");
			return;
		}
  }
  TF_SetStatus(status, TF_OK, "");
}

static void RenameFileImpl(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
	char* src_bucket;
  char* src_object;
	ParseS3Test(src, false, &src_bucket, &src_object, status);
	if(TF_GetCode(status) != TF_OK) return;

	char* dst_bucket;
  char* dst_object;
	ParseS3Test(dst, false, &dst_bucket, &dst_object, status);
	if(TF_GetCode(status) != TF_OK) return;

  int64_t file_length = GetFileSize(filesystem, src, status);
  if(TF_GetCode(status) != TF_OK) return;

  int num_parts;
  if (file_length <= multi_part_copy_part_size_) {
    num_parts = 1;
  } else {
    num_parts = ceil((float)file_length / multi_part_copy_part_size_);
  }

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

  if (num_parts == 1) {
    // Source does not contain `s3://`
    SimpleCopy(src + strlen("s3://"), dst_bucket, dst_object, s3_shared->s3_client, status);
    return;
  } else if (num_parts > 10000) {
    std::string message = std::string(
        "MultiPartCopy with number of parts more than 10000 is not supported.\n Your object" + \
        std::string(src) + " required " + std::to_string(num_parts) + \
        " as multi_part_copy_part_size is set to " + std::to_string(multi_part_copy_part_size_) + \
        ". You can control this part size using the environment variable " + \
        "S3_MULTI_PART_COPY_PART_SIZE to increase it.");
    TF_SetStatus(status, TF_UNIMPLEMENTED, message.c_str());
    return;
  } else {
    MultiPartCopy(src + strlen("s3://"), dst_bucket, dst_object, num_parts, file_length, s3_shared->s3_client, status);
    return;
  }
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
	char* src_bucket;
  char* src_object_char;
	ParseS3Test(src, false, &src_bucket, &src_object_char, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(src, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(src, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, src, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }
  else if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats = {0, 0, false};
  GetParentFile(dst, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(dst, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, dst, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }

	char* dst_bucket;
  char* dst_object_char;
	ParseS3Test(dst, false, &dst_bucket, &dst_object_char, status);
	if(TF_GetCode(status) != TF_OK) return;

  std::string src_object = src_object_char;
  std::string target_object = dst_object_char;
  if (src_object.back() == '/') {
    if (target_object.back() != '/') {
      target_object.push_back('/');
    }
  } else {
    if (target_object.back() == '/') {
      target_object.pop_back();
    }
  }

  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;

	Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(src_bucket)
      .WithPrefix(src_object_char)
      .WithMaxKeys(kS3GetChildrenMaxKeys);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome =
        s3_shared->s3_client->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      TF_SetStatusFromAWSError(status, listObjectsOutcome.GetError());
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String src_key = object.GetKey();
      Aws::String target_key = src_key;
      target_key.replace(0, src_object.length(), target_object.c_str());

      RenameFileImpl(filesystem, src, dst, status);
      if(TF_GetCode(status) != TF_OK) return;

      deleteObjectRequest.SetBucket(src_bucket);
      deleteObjectRequest.SetKey(src_key.c_str());

      auto deleteObjectOutcome =
          s3_shared->s3_client->DeleteObject(deleteObjectRequest);
      if (!deleteObjectOutcome.IsSuccess()) {
        TF_SetStatusFromAWSError(status, deleteObjectOutcome.GetError());
        return;
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  TF_SetStatus(status, TF_OK, "");
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
	char* src_bucket;
  char* src_object;
	ParseS3Test(src, false, &src_bucket, &src_object, status);
	if(TF_GetCode(status) != TF_OK) return;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(src, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(src, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, src, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }
  else if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

	char* dst_bucket;
  char* dst_object;
	ParseS3Test(dst, false, &dst_bucket, &dst_object, status);
	if(TF_GetCode(status) != TF_OK) return;

  stats = {0, 0, false};
  GetParentFile(dst, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
  GetParentDir(dst, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return;
  }

  stats.is_directory = false;
  Stat(filesystem, dst, &stats, status);
  if(stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    return;
  }

  int64_t file_length = GetFileSize(filesystem, src, status);
  if(TF_GetCode(status) != TF_OK) return;

  int num_parts;
  if (file_length <= multi_part_copy_part_size_) {
    num_parts = 1;
  } else {
    num_parts = ceil((float)file_length / multi_part_copy_part_size_);
  }

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

  if (num_parts == 1) {
    // Source does not contain `s3://`
    SimpleCopy(src + strlen("s3://"), dst_bucket, dst_object, s3_shared->s3_client, status);
    return;
  } else if (num_parts > 10000) {
    std::string message = std::string(
        "MultiPartCopy with number of parts more than 10000 is not supported.\n Your object" + \
        std::string(src) + " required " + std::to_string(num_parts) + \
        " as multi_part_copy_part_size is set to " + std::to_string(multi_part_copy_part_size_) + \
        ". You can control this part size using the environment variable " + \
        "S3_MULTI_PART_COPY_PART_SIZE to increase it.");
    TF_SetStatus(status, TF_UNIMPLEMENTED, message.c_str());
    return;
  } else {
    MultiPartCopy(src + strlen("s3://"), dst_bucket, dst_object, num_parts, file_length, s3_shared->s3_client, status);
    return;
  }
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return;
  }

  stats.is_directory = false;
	Stat(filesystem, path, &stats, status);
	if(TF_GetCode(status) != TF_OK) return;
	TF_SetStatus(status, TF_OK, "");
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return;

	auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);

  if (!object) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket);
    auto headBucketOutcome = s3_shared->s3_client->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      TF_SetStatusFromAWSError(status, headBucketOutcome.GetError());
			return;
    }
    stats->length = 0;
    stats->is_directory = 1;
  }

	bool found = false;

  Aws::S3::Model::HeadObjectRequest headObjectRequest;
  headObjectRequest.WithBucket(bucket).WithKey(object);
  headObjectRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto headObjectOutcome = s3_shared->s3_client->HeadObject(headObjectRequest);
  if (headObjectOutcome.IsSuccess()) {
    stats->length = headObjectOutcome.GetResult().GetContentLength();
    stats->is_directory = 0;
    stats->mtime_nsec =
        headObjectOutcome.GetResult().GetLastModified().Millis() * 1e6;
    found = true;
  } else {
    if(headObjectOutcome.GetError().GetResponseCode() ==
				Aws::Http::HttpResponseCode::FORBIDDEN) {
			TF_SetStatus(status, TF_FAILED_PRECONDITION, "AWS Credentials have not been set properly.\nUnable to access the specified S3 location");
			return;
		}
  }

  std::string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }

	Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket)
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(1);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome =
      s3_shared->s3_client->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto listObjects = listObjectsOutcome.GetResult().GetContents();
    if (listObjects.size() > 0) {
      stats->length = 0;
      stats->is_directory = 1;
      stats->mtime_nsec = listObjects[0].GetLastModified().Millis() * 1e6;
      found = true;
    }
  } else {
    if(listObjectsOutcome.GetError().GetResponseCode() ==
				Aws::Http::HttpResponseCode::FORBIDDEN) {
			TF_SetStatus(status, TF_FAILED_PRECONDITION, "AWS Credentials have not been set properly.\nUnable to access the specified S3 location");
			return;
		}
  }

	if (!found) {
    TF_SetStatus(status, TF_NOT_FOUND, std::string("Object " + std::string(path) + " does not exist").c_str());
		return;
  }
	TF_SetStatus(status, TF_OK, "");
}

static int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status) {

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return -1;
  }

  stats = {0, 0, false};
	Stat(filesystem, path, &stats, status);
	if(TF_GetCode(status) != TF_OK) return -1;
  if(stats.is_directory){
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Path is a directory");
    return -1;
  }
	TF_SetStatus(status, TF_OK, "");
	return stats.length;
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  std::vector<std::string> result;
	char* bucket;
  char* object;
	ParseS3Test(path, true, &bucket, &object, status);
	if(TF_GetCode(status) != TF_OK) return -1;

  TF_FileStatistics stats = {0, 0, false};
  char* parent;
  GetParentFile(path, &parent);
  Stat(filesystem, parent, &stats, status);
  free(parent);
  parent = NULL;
  if(TF_GetCode(status) == TF_OK && !stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is invaild");
    return -1;
  }

  stats.is_directory = true;
  Stat(filesystem, path, &stats, status);
  if(TF_GetCode(status) == TF_NOT_FOUND) {
    return -1;
  }
  else if(!stats.is_directory) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "Path is not a directory");
    return -1;
  }

  std::string prefix = object;

  if (!prefix.empty() && prefix.back() != '/') {
    prefix.push_back('/');
  }

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket)
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys)
      .WithDelimiter("/");
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
	GetS3Client(s3_shared);
  do {
    auto listObjectsOutcome =
        s3_shared->s3_client->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      TF_SetStatusFromAWSError(status, listObjectsOutcome.GetError());
			return -1;
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetCommonPrefixes()) {
      Aws::String s = object.GetPrefix();
      s.erase(s.length() - 1);
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result.push_back(entry.c_str());
      }
    }
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String s = object.GetKey();
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result.push_back(entry.c_str());
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  *entries = static_cast<char**>(
      plugin_memory_allocate(result.size() * sizeof((*entries)[0])));
  // TODO(vnvo2409): Optimize
  for(int i = 0; i < result.size(); ++i) {
      (*entries)[i] = strdup(result.at(i).c_str());
  }
  return result.size();
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  char* name = (char*) malloc(strlen(uri) + 1);
  strcpy(name, uri);
  return name;
}

}  // namespace tf_s3_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      plugin_memory_allocate(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;

  ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
      plugin_memory_allocate(TF_WRITABLE_FILE_OPS_SIZE));
  ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
  ops->writable_file_ops->append = tf_writable_file::Append;
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_s3_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_s3_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_s3_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_s3_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_s3_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_s3_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_s3_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_s3_filesystem::DeleteFile;
  ops->filesystem_ops->delete_dir = tf_s3_filesystem::DeleteDir;
  ops->filesystem_ops->rename_file = tf_s3_filesystem::RenameFile;
  ops->filesystem_ops->copy_file = tf_s3_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_s3_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_s3_filesystem::Stat;
  ops->filesystem_ops->get_file_size = tf_s3_filesystem::GetFileSize;
  ops->filesystem_ops->get_children = tf_s3_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_s3_filesystem::TranslateName;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "s3");
}
