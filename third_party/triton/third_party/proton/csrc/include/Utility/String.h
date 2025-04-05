#ifndef PROTON_UTILITY_STRING_H_
#define PROTON_UTILITY_STRING_H_

#include <string>

namespace proton {

inline std::string toLower(const std::string &str) {
  std::string lower;
  for (auto c : str) {
    lower += tolower(c);
  }
  return lower;
}

inline std::string replace(const std::string &str, const std::string &src,
                           const std::string &dst) {
  std::string replaced = str;
  size_t pos = replaced.find(src);
  while (pos != std::string::npos) {
    replaced.replace(pos, src.length(), dst);
    pos += dst.length();
    pos = replaced.find(src, pos);
  }
  return replaced;
}

inline bool endWith(const std::string &str, const std::string &sub) {
  if (str.length() < sub.length()) {
    return false;
  }
  return str.compare(str.length() - sub.length(), sub.length(), sub) == 0;
}

inline std::string trim(const std::string &str) {
  size_t start = 0;
  size_t end = str.length();
  while (start < end && isspace(str[start])) {
    start++;
  }
  while (end > start && isspace(str[end - 1])) {
    end--;
  }
  return str.substr(start, end - start);
}

} // namespace proton

#endif // PROTON_UTILITY_STRING_H_
