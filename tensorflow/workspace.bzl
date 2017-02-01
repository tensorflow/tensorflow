# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")


# Parse the bazel version string from `native.bazel_version`.
def _parse_bazel_version(bazel_version):
  # Remove commit from version.
  version = bazel_version.split(" ", 1)[0]

  # Split into (release, date) parts and only return the release
  # as a tuple of integers.
  parts = version.split('-', 1)

  # Turn "release" into a tuple of strings
  version_tuple = ()
  for number in parts[0].split('.'):
    version_tuple += (str(number),)
  return version_tuple

# Check that a specific bazel version is being used.
def check_version(bazel_version):
  if "bazel_version" not in dir(native):
    fail("\nCurrent Bazel version is lower than 0.2.1, expected at least %s\n" % bazel_version)
  elif not native.bazel_version:
    print("\nCurrent Bazel is not a release version, cannot check for compatibility.")
    print("Make sure that you are running at least Bazel %s.\n" % bazel_version)
  else:
    current_bazel_version = _parse_bazel_version(native.bazel_version)
    minimum_bazel_version = _parse_bazel_version(bazel_version)
    if minimum_bazel_version > current_bazel_version:
      fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
          native.bazel_version, bazel_version))
  pass

# Temporary workaround to support including TensorFlow as a submodule until this
# use-case is supported in the next Bazel release.
def _temp_workaround_http_archive_impl(repo_ctx):
   repo_ctx.template("BUILD", repo_ctx.attr.build_file,
                     {"%ws%": repo_ctx.attr.repository}, False)
   repo_ctx.download_and_extract(repo_ctx.attr.urls, "", repo_ctx.attr.sha256,
                                 "", repo_ctx.attr.strip_prefix)

temp_workaround_http_archive = repository_rule(
   implementation=_temp_workaround_http_archive_impl,
   attrs = {
      "build_file": attr.label(),
      "repository": attr.string(),
      "urls": attr.string_list(default = []),
      "sha256": attr.string(default = ""),
      "strip_prefix": attr.string(default = ""),
   })

# If TensorFlow is linked as a submodule.
# path_prefix and tf_repo_name are no longer used.
def tf_workspace(path_prefix = "", tf_repo_name = ""):
  cuda_configure(name = "local_config_cuda")
  sycl_configure(name = "local_config_sycl")
  if path_prefix:
    print("path_prefix was specified to tf_workspace but is no longer used and will be removed in the future.")
  if tf_repo_name:
    print("tf_repo_name was specified to tf_workspace but is no longer used and will be removed in the future.")

  native.new_http_archive(
      name = "eigen_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/bitbucket.org/eigen/eigen/get/60578b474802.tar.gz",
          "https://bitbucket.org/eigen/eigen/get/60578b474802.tar.gz",
      ],
      sha256 = "7527cda827aff351981ebd910012e16be4d899c28a9ae7f143ae60e7f3f7b83d",
      strip_prefix = "eigen-eigen-60578b474802",
      build_file = str(Label("//third_party:eigen.BUILD")),
  )

  native.new_http_archive(
      name = "libxsmm_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/hfp/libxsmm/archive/1.7.tar.gz",
          "https://github.com/hfp/libxsmm/archive/1.7.tar.gz",
      ],
      sha256 = "2eea65624a697e74b939511cd2a686b4c957e90c99be168fe134d96771e811ad",
      strip_prefix = "libxsmm-1.7",
      build_file = str(Label("//third_party:libxsmm.BUILD")),
  )

  native.bind(
      name = "xsmm_avx",
      actual = "@libxsmm_archive//third_party:xsmm_avx",
  )

  native.http_archive(
      name = "com_googlesource_code_re2",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/re2/archive/b94b7cd42e9f02673cd748c1ac1d16db4052514c.tar.gz",
          "https://github.com/google/re2/archive/b94b7cd42e9f02673cd748c1ac1d16db4052514c.tar.gz",
      ],
      sha256 = "bd63550101e056427c9e7ff12a408c1c8b74e9803f393ca916b2926fc2c4906f",
      strip_prefix = "re2-b94b7cd42e9f02673cd748c1ac1d16db4052514c",
  )

  native.http_archive(
      name = "gemmlowp",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/gemmlowp/archive/a6f29d8ac48d63293f845f2253eccbf86bc28321.tar.gz",
          "https://github.com/google/gemmlowp/archive/a6f29d8ac48d63293f845f2253eccbf86bc28321.tar.gz",
      ],
      sha256 = "75d40ea8e68b0d1644f052fffe8f14a410b2a73d40ccb859a95c0578d194ec26",
      strip_prefix = "gemmlowp-a6f29d8ac48d63293f845f2253eccbf86bc28321",
  )

  native.new_http_archive(
      name = "farmhash_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/farmhash/archive/92e897b282426729f4724d91a637596c7e2fe28f.zip",
          "https://github.com/google/farmhash/archive/92e897b282426729f4724d91a637596c7e2fe28f.zip",
      ],
      sha256 = "4c626d1f306bda2c6804ab955892f803f5245f4dcaecb4979dc08b091256da54",
      strip_prefix = "farmhash-92e897b282426729f4724d91a637596c7e2fe28f",
      build_file = str(Label("//third_party:farmhash.BUILD")),
  )

  native.bind(
      name = "farmhash",
      actual = "@farmhash//:farmhash",
  )

  native.new_http_archive(
      name = "highwayhash",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
          "https://github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
      ],
      sha256 = "0f30a15b1566d93f146c8d149878a06e91d9bb7ec2cfd76906df62a82be4aac9",
      strip_prefix = "highwayhash-dfcb97ca4fe9277bf9dc1802dd979b071896453b",
      build_file = str(Label("//third_party:highwayhash.BUILD")),
  )

  native.new_http_archive(
      name = "nasm",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2",
          "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.12.02.tar.bz2/d15843c3fb7db39af80571ee27ec6fad/nasm-2.12.02.tar.bz2",
      ],
      sha256 = "00b0891c678c065446ca59bcee64719d0096d54d6886e6e472aeee2e170ae324",
      strip_prefix = "nasm-2.12.02",
      build_file = str(Label("//third_party:nasm.BUILD")),
  )

  temp_workaround_http_archive(
      name = "jpeg",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/libjpeg-turbo/libjpeg-turbo/archive/1.5.1.tar.gz",
          "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/1.5.1.tar.gz",
      ],
      sha256 = "c15a9607892113946379ccea3ca8b85018301b200754f209453ab21674268e77",
      strip_prefix = "libjpeg-turbo-1.5.1",
      build_file = str(Label("//third_party/jpeg:jpeg.BUILD")),
      repository = tf_repo_name,
  )

  native.new_http_archive(
      name = "png_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/glennrp/libpng/archive/v1.2.53.zip",
          "https://github.com/glennrp/libpng/archive/v1.2.53.zip",
      ],
      sha256 = "c35bcc6387495ee6e757507a68ba036d38ad05b415c2553b3debe2a57647a692",
      strip_prefix = "libpng-1.2.53",
      build_file = str(Label("//third_party:png.BUILD")),
  )

  native.new_http_archive(
      name = "gif_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
          "http://ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
      ],
      sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
      strip_prefix = "giflib-5.1.4",
      build_file = str(Label("//third_party:gif.BUILD")),
  )

  native.new_http_archive(
      name = "six_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
          "http://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
      ],
      sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
      strip_prefix = "six-1.10.0",
      build_file = str(Label("//third_party:six.BUILD")),
  )

  native.new_http_archive(
      name = "org_pocoo_werkzeug",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/pypi.python.org/packages/b7/7f/44d3cfe5a12ba002b253f6985a4477edfa66da53787a2a838a40f6415263/Werkzeug-0.11.10.tar.gz",
          "https://pypi.python.org/packages/b7/7f/44d3cfe5a12ba002b253f6985a4477edfa66da53787a2a838a40f6415263/Werkzeug-0.11.10.tar.gz",
      ],
      strip_prefix = "Werkzeug-0.11.10",
      sha256 = "cc64dafbacc716cdd42503cf6c44cb5a35576443d82f29f6829e5c49264aeeee",
      build_file = str(Label("//third_party:werkzeug.BUILD")),
  )

  native.bind(
      name = "six",
      actual = "@six_archive//:six",
  )

  native.http_archive(
      name = "protobuf",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/protobuf/archive/008b5a228b37c054f46ba478ccafa5e855cb16db.tar.gz",
          "https://github.com/google/protobuf/archive/008b5a228b37c054f46ba478ccafa5e855cb16db.tar.gz",
      ],
      sha256 = "2737ad055eb8a9bc63ed068e32c4ea280b62d8236578cb4d4120eb5543f759ab",
      strip_prefix = "protobuf-008b5a228b37c054f46ba478ccafa5e855cb16db",
  )

  native.new_http_archive(
      name = "gmock_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/pkgs.fedoraproject.org/repo/pkgs/gmock/gmock-1.7.0.zip/073b984d8798ea1594f5e44d85b20d66/gmock-1.7.0.zip",
          "http://pkgs.fedoraproject.org/repo/pkgs/gmock/gmock-1.7.0.zip/073b984d8798ea1594f5e44d85b20d66/gmock-1.7.0.zip",
      ],
      sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
      strip_prefix = "gmock-1.7.0",
      build_file = str(Label("//third_party:gmock.BUILD")),
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

  native.new_http_archive(
      name = "pcre",
      sha256 = "ccdf7e788769838f8285b3ee672ed573358202305ee361cfec7a4a4fb005bbc7",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
          "http://ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
      ],
      strip_prefix = "pcre-8.39",
      build_file = str(Label("//third_party:pcre.BUILD")),
  )

  native.new_http_archive(
      name = "swig",
      sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
      ],
      strip_prefix = "swig-3.0.8",
      build_file = str(Label("//third_party:swig.BUILD")),
  )

  native.new_http_archive(
      name = "curl",
      sha256 = "ff3e80c1ca6a068428726cd7dd19037a47cc538ce58ef61c59587191039b2ca6",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/curl.haxx.se/download/curl-7.49.1.tar.gz",
          "https://curl.haxx.se/download/curl-7.49.1.tar.gz",
      ],
      strip_prefix = "curl-7.49.1",
      build_file = str(Label("//third_party:curl.BUILD")),
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
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/grpc/grpc/archive/d7ff4ff40071d2b486a052183e3e9f9382afb745.tar.gz",
          "https://github.com/grpc/grpc/archive/d7ff4ff40071d2b486a052183e3e9f9382afb745.tar.gz",
      ],
      sha256 = "a15f352436ab92c521b1ac11e729e155ace38d0856380cf25048c5d1d9ba8e31",
      strip_prefix = "grpc-d7ff4ff40071d2b486a052183e3e9f9382afb745",
      build_file = str(Label("//third_party:grpc.BUILD")),
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

  native.new_http_archive(
      name = "linenoise",
      sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
          "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
      ],
      strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
      build_file = str(Label("//third_party:linenoise.BUILD")),
  )

  # TODO(phawkins): currently, this rule uses an unofficial LLVM mirror.
  # Switch to an official source of snapshots if/when possible.
  temp_workaround_http_archive(
      name = "llvm",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/llvm-mirror/llvm/archive/2276fd31f36aa58f39397c435a8be6632d8c8505.tar.gz",
          "https://github.com/llvm-mirror/llvm/archive/2276fd31f36aa58f39397c435a8be6632d8c8505.tar.gz",
      ],
      sha256 = "0e08c91752732227280466d12f330a5854569deddf28ff4a6c3898334dbb0d16",
      strip_prefix = "llvm-2276fd31f36aa58f39397c435a8be6632d8c8505",
      build_file = str(Label("//third_party/llvm:llvm.BUILD")),
      repository = tf_repo_name,
  )

  native.new_http_archive(
      name = "jsoncpp_git",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
          "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
      ],
      sha256 = "07d34db40593d257324ec5fb9debc4dc33f29f8fb44e33a2eeb35503e61d0fe2",
      strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
      build_file = str(Label("//third_party:jsoncpp.BUILD")),
  )

  native.bind(
      name = "jsoncpp",
      actual = "@jsoncpp_git//:jsoncpp",
  )

  native.http_archive(
      name = "boringssl",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/google/boringssl/archive/bbcaa15b0647816b9a1a9b9e0d209cd6712f0105.tar.gz",
          "https://github.com/google/boringssl/archive/bbcaa15b0647816b9a1a9b9e0d209cd6712f0105.tar.gz",  # 2016-07-11
      ],
      sha256 = "025264d6e9a7ad371f2f66d17a28b6627de0c9592dc2eb54afd062f68f1f9aa3",
      strip_prefix = "boringssl-bbcaa15b0647816b9a1a9b9e0d209cd6712f0105",
  )

  native.new_http_archive(
      name = "nanopb_git",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/nanopb/nanopb/archive/1251fa1065afc0d62f635e0f63fec8276e14e13c.tar.gz",
          "https://github.com/nanopb/nanopb/archive/1251fa1065afc0d62f635e0f63fec8276e14e13c.tar.gz",
      ],
      sha256 = "ab1455c8edff855f4f55b68480991559e51c11e7dab060bbab7cffb12dd3af33",
      strip_prefix = "nanopb-1251fa1065afc0d62f635e0f63fec8276e14e13c",
      build_file = str(Label("//third_party:nanopb.BUILD")),
  )

  native.bind(
      name = "nanopb",
      actual = "@nanopb_git//:nanopb",
  )

  native.new_http_archive(
      name = "zlib_archive",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/zlib.net/zlib-1.2.8.tar.gz",
          "http://zlib.net/fossils/zlib-1.2.8.tar.gz",
      ],
      sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
      strip_prefix = "zlib-1.2.8",
      build_file = str(Label("//third_party:zlib.BUILD")),
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )

  native.new_http_archive(
      name = "nccl_archive",
      url = "https://github.com/NVIDIA/nccl/archive/2a974f5ca2aa12b178046b2206b43f1fd69d9fae.tar.gz",
      sha256 = "d6aa1a3f20ae85358890d9a96f49c51a75baa1d3af3598501f29ff9ef8a3107d",
      strip_prefix = "nccl-2a974f5ca2aa12b178046b2206b43f1fd69d9fae",
      build_file = str(Label("//third_party:nccl.BUILD")),
  )

  # Make junit-4.12 available as //external:junit
  native.http_jar(
      name = "junit_jar",
      url = "https://github.com/junit-team/junit4/releases/download/r4.12/junit-4.12.jar",
      sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
  )

  native.bind(
      name = "junit",
      actual = "@junit_jar//jar",
  )

  temp_workaround_http_archive(
      name = "jemalloc",
      urls = [
          "http://bazel-mirror.storage.googleapis.com/github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
          "https://github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
      ],
      sha256 = "3c8f25c02e806c3ce0ab5fb7da1817f89fc9732709024e2a81b6b82f7cc792a8",
      strip_prefix = "jemalloc-4.4.0",
      build_file = str(Label("//third_party:jemalloc.BUILD")),
      repository = tf_repo_name,
  )
