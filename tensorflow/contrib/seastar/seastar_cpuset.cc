#include "tensorflow/contrib/seastar/seastar_cpuset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/cpu_info.h"

#include <dirent.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace tensorflow {
namespace {
const char* ROOT_PATH = "/tmp_tf";
const char* DEFAULT_ROOT_PATH = "/tmp";
const char* CPUSET_FILE_PATH = "/cpuset";
const size_t CORES_PER_FILE = 1;
const size_t INIT_CPU_ID = 0;
}

class FileLocker {
public:
  FileLocker(const std::string& rd) : root_dir_(rd) {}
  virtual ~FileLocker() {}

  bool Lock(const std::string& file_name) {
    return LockerOpImpl(file_name, LOCK_EX | LOCK_NB);
  }

  void Unlock(const std::string& file_name) {
    LockerOpImpl(file_name, LOCK_UN | LOCK_NB);
  }

private:
  bool LockerOpImpl(const std::string& file_name, int lock_type) {
    std::string file_path;
    file_path += root_dir_ + std::string("/") + file_name;
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      VLOG(2) << "can't open file:" << file_path;
      return false;
    }

    int stat = flock(fd, lock_type);
    return (stat == 0);
  }

private:
  const std::string root_dir_;
};

std::string CpusetAllocator::GetCpuset(size_t core_number) {
  // critical section: semphore to lock this function
  if (!ExistDir()) {
    CreateDir();
  }
  CreateFiles();
  auto locked_files = LockFiles(core_number);
  return ToCpuset(locked_files);
}

bool CpusetAllocator::ExistDir() {
  if (opendir(ROOT_PATH) != nullptr) {
    root_dir_ = ROOT_PATH;
  } else if (opendir(DEFAULT_ROOT_PATH) != nullptr) {
    root_dir_ = DEFAULT_ROOT_PATH;
  } else {
    return false;
  }
  root_dir_ += CPUSET_FILE_PATH;

  return opendir(root_dir_.c_str()) != nullptr;
}

void CpusetAllocator::CreateDir() {
  int flag=mkdir(root_dir_.c_str(), 0777);
  if (flag != 0) {
    LOG(FATAL) << "Seastar: create cpuset dir failure";
  }
}

void CpusetAllocator::CreateFiles() {
  // todo: port::NumAllCPUs(), all phsical core should be available in docker
  // or this would bug here, k8s could be a candidate to allocator cpu cores.
  for (auto i = INIT_CPU_ID; i < port::NumTotalCPUs(); ++i) {
    auto file_name = std::to_string(i);

    std::string file_path;
    file_path += root_dir_ + std::string("/") + file_name;
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      LOG(FATAL) << "Seastar error: can't create lock files for cpuset,"
                 << ", please try other protocol, filepath:"
                 << file_path;
    }
    close(fd);

    files_.emplace_back(file_name);
  }
}

std::vector<std::string> CpusetAllocator::LockFiles(size_t core_number) {
  std::vector<std::string> locked_files;
  FileLocker locker(root_dir_);
  for (auto file : files_) {
    if (core_number <= 0) 
      break;
    if (locker.Lock(file)) {
      core_number -= CORES_PER_FILE;
      locked_files.emplace_back(file);
    }
  }
  if (core_number > 0) {
    LOG(WARNING) << "Seastar: allocate cpuset by file lock failure,"
                 << "please try other protocol";
    for (auto file : locked_files) {
      locker.Unlock(file);
    }
    return std::vector<std::string>();
  }
  return locked_files;
}

std::string CpusetAllocator::ToCpuset(
    const std::vector<std::string>& locked_files) {
  if (locked_files.empty())
    return std::string();
  const std::string& cpuset =
    strings::StrCat("--cpuset=", str_util::Join(locked_files, ","));
  return cpuset.substr(0, cpuset.size() - 1);
}
}
