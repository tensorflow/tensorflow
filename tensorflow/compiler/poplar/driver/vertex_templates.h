#ifndef __vertex_templates_hpp__
#define __vertex_templates_hpp__
#include <string>

inline std::string templateVertexParams(bool first) {
  if (first)
    return "<>";
  else
    return ">";
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const std::string &val,
                                        Args... args);

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const char *val,
                                        Args... args);

template <typename T, typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const T&val, Args... args) {
  std::string p = first ? "<" : ",";
  p += std::to_string(val) + templateVertexParams(false, args...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const std::string &val,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, args...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const char *val,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, args...);
  return p;
}

template <typename ...Args>
inline std::string templateVertex(const std::string &name,
                                  Args... args) {
  return name + templateVertexParams(true, args...);
}

#endif // __vertex_templates_hpp__
