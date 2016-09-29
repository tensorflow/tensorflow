# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")

# If TensorFlow is linked as a submodule.
# path_prefix and tf_repo_name are no longer used.
def tf_workspace(path_prefix = "", tf_repo_name = ""):
  cuda_configure(name = "local_config_cuda")
  if path_prefix:
    print("path_prefix was specified to tf_workspace but is no longer used and will be removed in the future.")
  if tf_repo_name:
    print("tf_repo_name was specified to tf_workspace but is no longer used and will be removed in the future.")

  # These lines need to be changed when updating Eigen. They are parsed from
  # this file by the cmake and make builds to determine the eigen version and
  # hash.
  eigen_version = "c78d757b69d3"
  eigen_sha256 = "dfb650e20a0dee6172dcc99796210a07e40af61348497503b42dc12935b4e6f5"

  native.new_http_archive(
    name = "eigen_archive",
    url = "http://bitbucket.org/eigen/eigen/get/" + eigen_version + ".tar.gz",
    sha256 = eigen_sha256,
    strip_prefix = "eigen-eigen-" + eigen_version,
    build_file = str(Label("//:eigen.BUILD")),
  )

  native.http_archive(
    name = "com_googlesource_code_re2",
    url = "http://github.com/google/re2/archive/7bab3dc83df6a838cc004cc7a7f51d5fe1a427d5.tar.gz",
    sha256 = "ef91af8850f734c8be65f2774747f4c2d8d81e556ba009faa79b4dd8b2759555",
    strip_prefix = "re2-7bab3dc83df6a838cc004cc7a7f51d5fe1a427d5",
  )

  native.http_archive(
    name = "gemmlowp",
    url = "http://github.com/google/gemmlowp/archive/8b20dd2ce142115857220bd6a35e8a081b3e0829.tar.gz",
    sha256 = "9cf5f1e3d64b3632dbae5c65efb79f4374ca9ac362d788fc61e086af937ff6d7",
    strip_prefix = "gemmlowp-8b20dd2ce142115857220bd6a35e8a081b3e0829",
  )

  native.new_http_archive(
    name = "farmhash_archive",
    url = "http://github.com/google/farmhash/archive/71a777924015693c69bc3c8c6492fb8d5372c636.zip",
    sha256 = "99190108fb96a5e38e183f6a23fb7742948214fc96a746a50c79eb09a255a298",
    strip_prefix = "farmhash-71a777924015693c69bc3c8c6492fb8d5372c636/src",
    build_file = str(Label("//:farmhash.BUILD")),
  )

  native.bind(
    name = "farmhash",
    actual = "@farmhash//:farmhash",
  )

  native.http_archive(
    name = "highwayhash",
    url = "http://github.com/google/highwayhash/archive/4bce8fc6a9ca454d9d377dbc4c4d33488bbab78f.tar.gz",
    sha256 = "b159a62fb05e5f6a6be20aa0df6a951ebf44a7bb96ed2e819e4e35e17f56854d",
    strip_prefix = "highwayhash-4bce8fc6a9ca454d9d377dbc4c4d33488bbab78f",
  )

  native.new_http_archive(
    name = "jpeg_archive",
    url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
    sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
    strip_prefix = "jpeg-9a",
    build_file = str(Label("//:jpeg.BUILD")),
  )

  native.new_http_archive(
    name = "png_archive",
    url = "http://github.com/glennrp/libpng/archive/v1.2.53.zip",
    sha256 = "c35bcc6387495ee6e757507a68ba036d38ad05b415c2553b3debe2a57647a692",
    strip_prefix = "libpng-1.2.53",
    build_file = str(Label("//:png.BUILD")),
  )

  native.new_http_archive(
    name = "gif_archive",
    url = "http://ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
    sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
    strip_prefix = "giflib-5.1.4/lib",
    build_file = str(Label("//:gif.BUILD")),
  )

  native.new_http_archive(
    name = "six_archive",
    url = "http://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = str(Label("//:six.BUILD")),
  )

  native.bind(
    name = "six",
    actual = "@six_archive//:six",
  )

  native.http_archive(
    name = "protobuf",
    url = "http://github.com/google/protobuf/archive/v3.1.0.tar.gz",
    sha256 = "0a0ae63cbffc274efb573bdde9a253e3f32e458c41261df51c5dbc5ad541e8f7",
    strip_prefix = "protobuf-3.1.0",
  )

  native.new_http_archive(
    name = "gmock_archive",
    url = "http://pkgs.fedoraproject.org/repo/pkgs/gmock/gmock-1.7.0.zip/073b984d8798ea1594f5e44d85b20d66/gmock-1.7.0.zip",
    sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
    strip_prefix = "gmock-1.7.0",
    build_file = str(Label("//:gmock.BUILD")),
  )

  native.bind(
    name = "gtest",
    actual = "@gmock_archive//:gtest",
  )

  native.bind(
    name = "gtest_main",
    actual = "@gmock_archive//:gtest_main",
  )

  native.bind(
    name = "python_headers",
    actual = str(Label("//util/python:python_headers")),
  )

  # grpc expects //external:protobuf_clib and //external:protobuf_compiler
  # to point to the protobuf's compiler library.
  native.bind(
    name = "protobuf_clib",
    actual = "@protobuf//:protoc_lib",
  )

  native.bind(
    name = "protobuf_compiler",
    actual = "@protobuf//:protoc_lib",
  )

  native.new_http_archive(
    name = "grpc",
    url = "http://github.com/grpc/grpc/archive/d7ff4ff40071d2b486a052183e3e9f9382afb745.tar.gz",
    sha256 = "a15f352436ab92c521b1ac11e729e155ace38d0856380cf25048c5d1d9ba8e31",
    strip_prefix = "grpc-d7ff4ff40071d2b486a052183e3e9f9382afb745",
    build_file = str(Label("//:grpc.BUILD")),
  )

  # protobuf expects //external:grpc_cpp_plugin to point to grpc's
  # C++ plugin code generator.
  native.bind(
    name = "grpc_cpp_plugin",
    actual = "@grpc//:grpc_cpp_plugin",
  )

  native.bind(
    name = "grpc_lib",
    actual = "@grpc//:grpc++_unsecure",
  )

  native.new_git_repository(
    name = "linenoise",
    commit = "c894b9e59f02203dbe4e2be657572cf88c4230c3",
    init_submodules = True,
    remote = "https://github.com/antirez/linenoise.git",
    build_file = str(Label("//:linenoise.BUILD")),
  )

  native.new_http_archive(
    name = "jsoncpp_git",
    url = "http://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
    sha256 = "07d34db40593d257324ec5fb9debc4dc33f29f8fb44e33a2eeb35503e61d0fe2",
    strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
    build_file = str(Label("//:jsoncpp.BUILD")),
  )

  native.bind(
    name = "jsoncpp",
    actual = "@jsoncpp_git//:jsoncpp",
  )

  native.http_archive(
    name = "boringssl",
    url = "http://github.com/google/boringssl/archive/bbcaa15b0647816b9a1a9b9e0d209cd6712f0105.tar.gz",  # 2016-07-11
    sha256 = "025264d6e9a7ad371f2f66d17a28b6627de0c9592dc2eb54afd062f68f1f9aa3",
    strip_prefix = "boringssl-bbcaa15b0647816b9a1a9b9e0d209cd6712f0105",
  )

  native.new_http_archive(
    name = "nanopb_git",
    url = "http://github.com/nanopb/nanopb/archive/1251fa1065afc0d62f635e0f63fec8276e14e13c.tar.gz",
    sha256 = "ab1455c8edff855f4f55b68480991559e51c11e7dab060bbab7cffb12dd3af33",
    strip_prefix = "nanopb-1251fa1065afc0d62f635e0f63fec8276e14e13c",
    build_file = str(Label("//:nanopb.BUILD")),
  )

  native.bind(
    name = "nanopb",
    actual = "@nanopb_git//:nanopb",
  )

  native.new_http_archive(
    name = "zlib_archive",
    url = "http://zlib.net/zlib-1.2.8.tar.gz",
    sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
    strip_prefix = "zlib-1.2.8",
    build_file = str(Label("//:zlib.BUILD")),
  )

  native.bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
  )
