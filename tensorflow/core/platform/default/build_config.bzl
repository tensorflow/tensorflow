# Platform-specific build configurations.

load("/google/protobuf/protobuf", "cc_proto_library")
load("/google/protobuf/protobuf", "py_proto_library")

# Appends a suffix to a list of deps.
def tf_deps(deps, suffix):
  tf_deps = []

  # If the package name is in shorthand form (ie: does not contain a ':'),
  # expand it to the full name.
  for dep in deps:
    tf_dep = dep

    if not ":" in dep:
      dep_pieces = dep.split("/")
      tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

    tf_deps += [tf_dep + suffix]

  return tf_deps

def tf_proto_library(name, srcs = [], has_services = False,
                     deps = [], visibility = [], testonly = 0,
                     cc_api_version = 2, go_api_version = 2,
                     java_api_version = 2,
                     py_api_version = 2):
  native.filegroup(name=name + "_proto_srcs",
                   srcs=srcs + tf_deps(deps, "_proto_srcs"),
                   testonly=testonly,)

  cc_proto_library(name=name + "_cc",
                   srcs=srcs + tf_deps(deps, "_proto_srcs"),
                   deps=deps,
                   cc_libs = ["//google/protobuf:protobuf"],
                   testonly=testonly,
                   visibility=visibility,)

  py_proto_library(name=name + "_py",
                   srcs=srcs + tf_deps(deps, "_proto_srcs"),
                   deps=deps,
                   py_libs = ["//google/protobuf:protobuf_python"],
                   testonly=testonly,
                   visibility=visibility,)

def tf_proto_library_py(name, srcs=[], deps=[], visibility=[], testonly=0):
  py_proto_library(name = name + "_py",
                   srcs = srcs,
                   deps = deps,
                   visibility = visibility,
                   testonly = testonly)

def tf_additional_lib_srcs():
  return [
      "platform/default/*.h",
      "platform/default/*.cc",
      "platform/posix/*.h",
      "platform/posix/*.cc",
  ]

def tf_additional_test_srcs():
  return ["platform/default/test_benchmark.cc"]

def tf_kernel_tests_linkstatic():
  return 0
