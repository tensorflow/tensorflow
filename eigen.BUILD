package(default_visibility = ["//visibility:public"])

archive_dir = "benoitsteiner-opencl-9d4a08d57d0d"

cc_library(
    name = "eigen",
    hdrs = glob([archive_dir+"/**/*.h", archive_dir+"/unsupported/Eigen/CXX11/*", archive_dir+"/Eigen/*"]),
    includes = [ archive_dir ],
    visibility = ["//visibility:public"],
)
