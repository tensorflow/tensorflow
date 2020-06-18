#include "tensorflow/c/experimental/filesystem/plugins/gcs/gcs_helper.h"

#include <stdio.h>

#include <fstream>
#include <string>

TempFile::TempFile(const char* temp_file_name, std::ios::openmode mode)
    : std::fstream(temp_file_name, mode), name(temp_file_name) {}

TempFile::TempFile(TempFile&& rhs)
    : std::fstream(std::move(rhs)), name(std::move(rhs.name)) {}

TempFile::~TempFile() {
  std::fstream::close();
  std::remove(name.c_str());
}

const std::string TempFile::getName() const { return name; }