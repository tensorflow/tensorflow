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
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"

#include "google/cloud/storage/client.h"

#include <string>
#include <memory>
#include <fstream>
#include <iostream>

#include <stdio.h>

#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for GCS environments.
// This filesystem will support `gs://` URI scheme.
#define TF_ReturnIfError(status) if(TF_GetCode(status) != TF_OK) return

// We can cast `google::cloud::StatusCode` to `TF_Code` because they have the same integer values.
// See https://github.com/googleapis/google-cloud-cpp/blob/6c09cbfa0160bc046e5509b4dd2ab4b872648b4a/google/cloud/status.h#L32-L52
static inline void TF_SetStatusFromGCSStatus(const google::cloud::Status& gcs_status, TF_Status* status) {
	TF_SetStatus(status, static_cast<TF_Code>(gcs_status.code()), gcs_status.message().c_str());
}

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

static void ParseGCSPath(const char* fname, bool object_empty_ok, char** bucket, char** object, TF_Status* status) {
  size_t scheme_index = strcspn(fname, "://");
  char* scheme = (char*) malloc(scheme_index + 1);
  sprintf(scheme, "%.*s", (int)scheme_index, fname);
  scheme[scheme_index] = '\0';
  if(strcmp(scheme, "gs")) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "GCS path doesn't start with 'gs://'.");
    return;
  }

  size_t bucket_index = strcspn(fname + scheme_index + 3, "/");
  if(!bucket_index) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "GCS path doesn't contain a bucket name.");
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
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "GCS path doesn't contain an object name.");
      return;
    }
  }
  *object = (char*) malloc(object_index + 1);
  sprintf(*object, "%.*s", (int) object_index, fname + scheme_index + 3 + bucket_index + 1);
  (*object)[object_index] = '\0';

  free(scheme);
  TF_SetStatus(status, TF_OK, "");
}

static std::shared_ptr<std::fstream> CreateTempFile(char** temp_path_) {
	*temp_path_ = (char*) malloc(L_tmpnam);
	*temp_path_ = tmpnam(*temp_path_);
	std::shared_ptr<std::fstream> temp_file_ = std::make_shared<std::fstream>(*temp_path_, std::fstream::binary | std::fstream::in | std::fstream::out | std::fstream::trunc);
	//printf("TempFile: %s isopen: %d\n", *temp_path_, temp_file_->is_open());
	return temp_file_;
}

static int64_t GetBuffer(const std::shared_ptr<std::fstream>& file, char** buffer) {
	file->seekg(0, file->end);
	int64_t read = file->tellg();
	file->seekg(0, file->beg);
	*buffer = (char*)malloc(read);
	file->read(*buffer, read);
	return read;
} 

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
namespace gcs = google::cloud::storage;

typedef struct GCSFile {
	const char* bucket;
	const char* object;
	gcs::Client* gcs_client;
} GCSFile;

static void Cleanup(TF_RandomAccessFile* file) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  // This would be safe to free using `free` directly as it is only opaque.
  // However, it is better to be consistent everywhere.
	// gcs_client is not owned by GCSFile and therefore we do not free it here.
  plugin_memory_free(const_cast<char*>(gcs_file->bucket));
	plugin_memory_free(const_cast<char*>(gcs_file->object));
  delete gcs_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
	int64_t read = 0;
	auto reader = gcs_file->gcs_client->ReadObject(gcs_file->bucket, gcs_file->object, gcs::ReadRange(offset, offset + n));

	TF_SetStatusFromGCSStatus(reader.status(), status);
	if(TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_OUT_OF_RANGE) return -1;

	std::string contents{std::istreambuf_iterator<char>{reader}, {}};
	read = contents.size();
	//printf("Read: file: %s, size: %d, read: %d\n", gcs_file->object, n, read);
	if(read < n) TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");

	strncpy(buffer, contents.c_str(), read);
	reader.Close();
	return read;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
namespace gcs = google::cloud::storage;

typedef struct GCSFile {
	const char* bucket;
	const char* object;
	gcs::Client* gcs_client;
	// std::fstream can not be shared by normal pointer
	std::shared_ptr<std::fstream> temp_file_;
	const char* temp_path_;
	bool sync_need_;
} GCSFile;

static void Cleanup(TF_WritableFile* file) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  // This would be safe to free using `free` directly as it is only opaque.
  // However, it is better to be consistent everywhere.
	// gcs_client is not owned by GCSFile and therefore we do not free it here.
	// remove(gcs_file->temp_path_);
  plugin_memory_free(const_cast<char*>(gcs_file->bucket));
	plugin_memory_free(const_cast<char*>(gcs_file->object));
	plugin_memory_free(const_cast<char*>(gcs_file->temp_path_));
	delete gcs_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
	//printf("Append: file: %s, buffer: %s, size: %d, temp_path: %s\n", gcs_file->object, buffer, n, gcs_file->temp_path_);
	if(gcs_file->temp_file_->fail()) {
		TF_SetStatus(status, TF_FAILED_PRECONDITION, "The internal temporary file is not writable");
		return;
	}
	//printf("Append: buffer: %s, size: %d, bucket: %s, object: %s\n", buffer, n, gcs_file->bucket, gcs_file->object);

	gcs_file->temp_file_->write(buffer, n);
	if(!gcs_file->temp_file_->good()){
		TF_SetStatus(status, TF_INTERNAL, "Could not append to the internal temporary file");
	}
	gcs_file->sync_need_ = true;
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
	int64_t position = int64_t(gcs_file->temp_file_->tellp());
	if(position == -1) {
		TF_SetStatus(status, TF_INTERNAL, "Could not tellp on the internal temporary file");
	}
	return position;
}

// In `gcs_cloud_cpp`, `Close()` is the only way to upload internal file to cloud
static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if(gcs_file->sync_need_) {
		auto writer = gcs_file->gcs_client->WriteObject(gcs_file->bucket, gcs_file->object);
		char* buffer;
		int64_t read = GetBuffer(gcs_file->temp_file_, &buffer);
		writer.write(buffer, read);
		//printf("GetBuffer success: %s, length: %d\n", buffer, read);
		writer.Close();
		free(buffer);
		if(!writer.metadata()) {
			TF_SetStatusFromGCSStatus(writer.metadata().status(), status);
		}
	}
	gcs_file->sync_need_ = false;
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
	gcs_file->sync_need_ = true;
  Flush(file, status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
	auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
	if(gcs_file->sync_need_) Sync(file, status);
	TF_ReturnIfError(status);
	gcs_file->temp_file_->close();
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct GCSMemoryRegion {
  const void* const address;
  const uint64_t length;
} GCSMemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  plugin_memory_free(const_cast<void*>(r->address));
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------

namespace tf_gcs_filesystem {
namespace gcs = google::cloud::storage;

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  google::cloud::StatusOr<gcs::Client> client =
      gcs::Client::CreateDefaultClient();
  if (!client) {
		TF_SetStatusFromGCSStatus(client.status(), status);
		return;
  }
	filesystem->plugin_filesystem = plugin_memory_allocate(sizeof(gcs::Client));
	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	(*gcs_client) = client.value();
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {
	plugin_memory_free(filesystem->plugin_filesystem);
}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	file->plugin_file = new tf_random_access_file::GCSFile({bucket, object, gcs_client});
	TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);
	//printf("NewWritableFile bucket: %s, object: %s\n", bucket, object);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	char* temp_path_ = nullptr;
	std::shared_ptr<std::fstream> temp_file_ = CreateTempFile(&temp_path_);
	//printf("TempFile: &s isopen: %d\n", temp_path_, temp_file_->is_open());
	file->plugin_file = new tf_writable_file::GCSFile({bucket, object, gcs_client, temp_file_, temp_path_, false});
	TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto reader = gcs_client->ReadObject(bucket, object);
	char* buffer = nullptr;
	int64_t read = 0;
	if(reader.status().code() == google::cloud::StatusCode::kOk || reader.status().code() == google::cloud::StatusCode::kOutOfRange) {
		std::string contents{std::istreambuf_iterator<char>{reader}, {}};
		read = contents.size();
		if(read > 0) {
			buffer = (char*)plugin_memory_allocate(read);
			strncpy(buffer, contents.c_str(), read);
		}
	}
	else if(reader.status().code() != google::cloud::StatusCode::kNotFound) {
		TF_SetStatusFromGCSStatus(reader.status(), status);
		return;
	}

	char* temp_path_ = nullptr;
	std::shared_ptr<std::fstream> temp_file_ = CreateTempFile(&temp_path_);
	temp_file_->write(buffer, read);
	//printf("NewAppenableFile: buffer: %s, read: %d\n", buffer, read);
	file->plugin_file = new tf_writable_file::GCSFile({bucket, object, gcs_client, temp_file_, temp_path_, false});
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto reader = gcs_client->ReadObject(bucket, object);
	char* buffer = nullptr;
	int64_t read = 0;
	if(reader.status().code() == google::cloud::StatusCode::kOk) {
		std::string contents{std::istreambuf_iterator<char>{reader}, {}};
		read = contents.size();
		//printf("Reader ReadOnly: read: %d\n", read);
		if(read > 0) {
			buffer = (char*)plugin_memory_allocate(read);
			strncpy(buffer, contents.c_str(), read);
			region->plugin_memory_region = new tf_read_only_memory_region::GCSMemoryRegion({buffer, static_cast<uint64_t>(read)});
			return;
		}
		else {
			TF_SetStatus(status, TF_INVALID_ARGUMENT, "File is empty");
			return;
		}
	}
	else {
		TF_SetStatusFromGCSStatus(reader.status(), status);
		return;
	}
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
	char* bucket;
	char* object_temporary;
	ParseGCSPath(path, false, &bucket, &object_temporary, status);
	TF_ReturnIfError(status);
	char* object = nullptr;
	if(object_temporary[strlen(object_temporary) - 1] != '/') {
		object = (char*)malloc(strlen(object_temporary) + 2);
		strcpy(object, object_temporary);
		object[strlen(object_temporary)] = '/';
		object[strlen(object_temporary) + 1] = '\0';
		free(object_temporary);
	}
	else {
		object = object_temporary;
	}
	//printf("CreateDir: bucket: %s, object: %s, object_temp: %s\n", bucket, object, object_temporary);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto inserter = gcs_client->InsertObject(bucket, object, "");
	if(!inserter) {
		TF_SetStatusFromGCSStatus(inserter.status(), status);
		return;
	}
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto gcs_status = gcs_client->DeleteObject(bucket, object);
	TF_SetStatusFromGCSStatus(gcs_status, status);
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
	char* bucket;
	char* object_temporary;
	ParseGCSPath(path, false, &bucket, &object_temporary, status);
	TF_ReturnIfError(status);
	char* object = nullptr;
	if(object_temporary[strlen(object_temporary) - 1] != '/') {
		object = (char*)malloc(strlen(object_temporary) + 2);
		strcpy(object, object_temporary);
		object[strlen(object_temporary)] = '/';
		object[strlen(object_temporary) + 1] = '\0';
		free(object_temporary);
	}
	else {
		object = object_temporary;
	}
	//printf("DeleteDir: bucket: %s, object: %s, object_temp: %s\n", bucket, object, object_temporary);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto gcs_status = gcs_client->DeleteObject(bucket, object);
	TF_SetStatusFromGCSStatus(gcs_status, status);
}

static void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                             uint64_t* undeleted_files,
                             uint64_t* undeleted_dirs, TF_Status* status) {
	TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteRecursively is not implemented");
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
	char* bucket_src;
	char* object_src;
	ParseGCSPath(src, false, &bucket_src, &object_src, status);
	TF_ReturnIfError(status);

	char* bucket_dst;
	char* object_dst;
	ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto metadata = gcs_client->RewriteObjectBlocking(bucket_src, object_src, bucket_dst, object_dst);
	if(!metadata) {
		TF_SetStatusFromGCSStatus(metadata.status(), status);
		return;
	}
	auto gcs_status = gcs_client->DeleteObject(bucket_src, object_src);
	TF_SetStatusFromGCSStatus(metadata.status(), status);
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
	char* bucket_src;
	char* object_src;
	ParseGCSPath(src, false, &bucket_src, &object_src, status);
	TF_ReturnIfError(status);

	char* bucket_dst;
	char* object_dst;
	ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto metadata = gcs_client->RewriteObjectBlocking(bucket_src, object_src, bucket_dst, object_dst);
	if(!metadata) {
		TF_SetStatusFromGCSStatus(metadata.status(), status);
		return;
	}
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
	char* bucket;
	char* object;
	ParseGCSPath(path, false, &bucket, &object, status);
	TF_ReturnIfError(status);

	auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
	auto metadata = gcs_client->GetObjectMetadata(bucket, object);
	if(!metadata) {
		TF_SetStatusFromGCSStatus(metadata.status(), status);
		return;
	}
	stats->length = metadata.value().size();
	stats->mtime_nsec = metadata.value().time_storage_class_updated().time_since_epoch().count();
	if(path[strlen(path) - 1] == '/') {
		stats->is_directory = true;
	}
	else {
		stats->is_directory = false;
	}
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
	TF_FileStatistics stats;
	Stat(filesystem, path, &stats, status);
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  char* name = (char*) malloc(strlen(uri) + 1);
  strcpy(name, uri);
  return name;
}

}  // namespace tf_gcs_filesystem

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
  ops->writable_file_ops->tell = tf_writable_file::Tell;
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
  ops->filesystem_ops->init = tf_gcs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_gcs_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_gcs_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_gcs_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_gcs_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_gcs_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_gcs_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_gcs_filesystem::DeleteFile;
  ops->filesystem_ops->delete_dir = nullptr;
  ops->filesystem_ops->rename_file = tf_gcs_filesystem::RenameFile;
  ops->filesystem_ops->copy_file = tf_gcs_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_gcs_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_gcs_filesystem::Stat;
	ops->filesystem_ops->translate_name = tf_gcs_filesystem::TranslateName;
	ops->filesystem_ops->recursively_create_dir = nullptr;
	ops->filesystem_ops->delete_recursively = nullptr;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gs");
}