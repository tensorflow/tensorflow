
#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CPUSET_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CPUSET_H_

#include <vector>
#include <string>

namespace tensorflow {
class CpusetAllocator {
public:
  virtual ~CpusetAllocator(){}
  std::string GetCpuset(size_t core_number);

private:
  bool ExistDir();
  void CreateDir();
  void CreateFiles();

  std::vector<std::string> LockFiles(size_t core_number);
  std::string ToCpuset(const std::vector<std::string>& locked_files);

private:
  std::string _root_dir;
  std::vector<std::string> _files;
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CPUSET_H_
