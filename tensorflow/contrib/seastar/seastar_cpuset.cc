#include "tensorflow/contrib/seastar/seastar_cpuset.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/cpu_info.h"

#include <sys/file.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

namespace tensorflow {
namespace {
const char* ROOT_PATH = "/tmp_tf";
const char* DEFAULT_ROOT_PATH = "/tmp";
const char* CPUSET_FILE_PATH = "/cpuset";
const size_t CORES_PER_FILE = 1;
const size_t INIT_CPU_ID = 16;
}

class FileLocker {
public:
  FileLocker(const std::string& rd) : _root_dir(rd) {}
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
    file_path += _root_dir + std::string("/") + file_name;
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      LOG(ERROR) << "can't open file:" << file_path;
      return false;
    }

    int stat = flock(fd, lock_type);
    return (stat == 0);
  }

private:
  const std::string _root_dir;
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
    _root_dir = ROOT_PATH;
  } else if (opendir(DEFAULT_ROOT_PATH) != nullptr) {
    _root_dir = DEFAULT_ROOT_PATH;
  } else {
    LOG(ERROR) << "create cpuset dir failure," 
               << "both /tmp & /tmp_tf not exist in the machine, "
               << "please try other protocol";
    return false;
  }
  _root_dir += CPUSET_FILE_PATH;

  return opendir(_root_dir.c_str()) != nullptr;
}

void CpusetAllocator::CreateDir() {
  int flag=mkdir(_root_dir.c_str(), 0777);
  if (flag != 0) {
    LOG(ERROR) << "create cpuset dir failure";
  }
}

void CpusetAllocator::CreateFiles() {
  // todo: port::NumAllCPUs(), all phsical core should be available in docker, or this would bug here
  // fuxi set value is better.
  for (auto i = INIT_CPU_ID; i < port::NumAllCPUs(); ++i) {
    auto file_name = std::to_string(i);

    std::string file_path;
    file_path += _root_dir + std::string("/") + file_name;
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      LOG(ERROR) << "can't create cpuset lock files" << file_path;
      return;
    }
    close(fd);

    _files.emplace_back(file_name); 
  }
}

std::vector<std::string> CpusetAllocator::LockFiles(size_t core_number) {
  std::vector<std::string> locked_files;
  FileLocker locker(_root_dir);
  for (auto file : _files) {
    if (core_number <= 0) 
      break;
    if (locker.Lock(file)) {
      core_number -= CORES_PER_FILE;
      locked_files.emplace_back(file);
    }
  }
  if (core_number > 0) {
    LOG(ERROR) << "allocate cpuset failure";
    for (auto file : locked_files) {
      locker.Unlock(file);
    }
    return std::vector<std::string>();
  }
  return locked_files;
}

std::string CpusetAllocator::ToCpuset(const std::vector<std::string>& locked_files) {
  if (locked_files.empty())
    return std::string();
  std::string cpuset("--cpuset=");
  for (auto file : locked_files) {
    cpuset += file + ",";
  }
  return cpuset.substr(0, cpuset.size() - 1);
}
}
