# Tensorflow external dependencies that can be loaded in WORKSPACE files.

def tf_workspace(path_prefix = ""):
  native.new_http_archive(
    name = "gmock_archive",
    url = "https://googlemock.googlecode.com/files/gmock-1.7.0.zip",
    sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
    build_file = path_prefix + "google/protobuf/gmock.BUILD",
  )

  native.new_http_archive(
    name = "eigen_archive",
    url = "https://bitbucket.org/eigen/eigen/get/db7b61411772.tar.gz",
    sha256 = "832e1e082b91d40ad909a079b98630ce52bd904d1ec0c3cb4cdcd2e24bcf95e6",
    build_file = path_prefix + "eigen.BUILD",
  )

  native.bind(
    name = "gtest",
    actual = "@gmock_archive//:gtest",
  )

  native.bind(
    name = "gtest_main",
    actual = "@gmock_archive//:gtest_main",
  )

  native.git_repository(
    name = "re2",
    remote = "https://github.com/google/re2.git",
    commit = "791beff",
  )

  native.new_http_archive(
    name = "jpeg_archive",
    url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
    sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
    build_file = path_prefix + "jpeg.BUILD",
  )

  native.new_http_archive(
    name = "png_archive",
    url = "https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz",
    sha256 = "e05c9056d7f323088fd7824d8c6acc03a4a758c4b4916715924edc5dd3223a72",
    build_file = path_prefix + "png.BUILD",
  )

  native.new_http_archive(
    name = "six_archive",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    build_file = path_prefix + "six.BUILD",
  )

  native.bind(
    name = "six",
    actual = "@six_archive//:six",
  )
