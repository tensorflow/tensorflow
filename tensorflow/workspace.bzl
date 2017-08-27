# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")
load("//third_party/mkl:build_defs.bzl", "mkl_repository")
load("@io_bazel_rules_closure//closure/private:java_import_external.bzl",
     "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl",
     "arm_compiler_configure")


def _is_windows(repository_ctx):
  """Returns true if the host operating system is windows."""
  return repository_ctx.os.name.lower().find("windows") != -1


def _get_env_var(repository_ctx, name):
  """Find an environment variable."""
  if name in repository_ctx.os.environ:
    return repository_ctx.os.environ[name]
  else:
    return None


# Parse the bazel version string from `native.bazel_version`.
def _parse_bazel_version(bazel_version):
  # Remove commit from version.
  version = bazel_version.split(" ", 1)[0]

  # Split into (release, date) parts and only return the release
  # as a tuple of integers.
  parts = version.split("-", 1)

  # Turn "release" into a tuple of strings
  version_tuple = ()
  for number in parts[0].split("."):
    version_tuple += (str(number),)
  return version_tuple


# Check that a specific bazel version is being used.
def check_version(bazel_version):
  if "bazel_version" not in dir(native):
    fail("\nCurrent Bazel version is lower than 0.2.1, expected at least %s\n" %
         bazel_version)
  elif not native.bazel_version:
    print("\nCurrent Bazel is not a release version, cannot check for " +
          "compatibility.")
    print("Make sure that you are running at least Bazel %s.\n" % bazel_version)
  else:
    current_bazel_version = _parse_bazel_version(native.bazel_version)
    minimum_bazel_version = _parse_bazel_version(bazel_version)
    if minimum_bazel_version > current_bazel_version:
      fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
          native.bazel_version, bazel_version))


def _repos_are_siblings():
  return Label("@foo//bar").workspace_root.startswith("../")


# Temporary workaround to support including TensorFlow as a submodule until this
# use-case is supported in the next Bazel release.
def _temp_workaround_http_archive_impl(repo_ctx):
  repo_ctx.template("BUILD", repo_ctx.attr.build_file, {
      "%prefix%": ".." if _repos_are_siblings() else "external",
      "%ws%": repo_ctx.attr.repository
  }, False)
  repo_ctx.download_and_extract(repo_ctx.attr.urls, "", repo_ctx.attr.sha256,
                                "", repo_ctx.attr.strip_prefix)
  if repo_ctx.attr.patch_file != None:
    _apply_patch(repo_ctx, repo_ctx.attr.patch_file)


temp_workaround_http_archive = repository_rule(
    implementation = _temp_workaround_http_archive_impl,
    attrs = {
        "build_file": attr.label(),
        "repository": attr.string(),
        "patch_file": attr.label(default = None),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
  result = repo_ctx.execute(cmd_and_args, timeout=10)
  if result.return_code != 0:
    fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n"
          + "Stderr: {3}").format(" ".join(cmd_and_args), result.return_code,
                                  result.stdout, result.stderr))


# Apply a patch_file to the repository root directory
# Runs 'patch -p1'
def _apply_patch(repo_ctx, patch_file):
  cmd = [
      "patch", "-p1", "-d", repo_ctx.path("."), "-i", repo_ctx.path(patch_file)
  ]
  if _is_windows(repo_ctx):
    bazel_sh = _get_env_var(repo_ctx, "BAZEL_SH")
    if not bazel_sh:
      fail("BAZEL_SH environment variable is not set")
    cmd = [bazel_sh, "-c", " ".join(cmd)]
  _execute_and_check_ret_code(repo_ctx, cmd)


# Download the repository and apply a patch to its root
def _patched_http_archive_impl(repo_ctx):
  repo_ctx.download_and_extract(
      repo_ctx.attr.urls,
      sha256=repo_ctx.attr.sha256,
      stripPrefix=repo_ctx.attr.strip_prefix)
  _apply_patch(repo_ctx, repo_ctx.attr.patch_file)


patched_http_archive = repository_rule(
    implementation = _patched_http_archive_impl,
    attrs = {
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "repository": attr.string(),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)


# If TensorFlow is linked as a submodule.
# path_prefix is no longer used.
# tf_repo_name is thought to be under consideration.
def tf_workspace(path_prefix="", tf_repo_name=""):
  # We must check the bazel version before trying to parse any other BUILD
  # files, in case the parsing of those build files depends on the bazel
  # version we require here.
  check_version("0.4.5")
  cuda_configure(name="local_config_cuda")
  sycl_configure(name="local_config_sycl")
  python_configure(name="local_config_python")

  # Point //external/local_config_arm_compiler to //external/arm_compiler
  arm_compiler_configure(
      name="local_config_arm_compiler",
      remote_config_repo="../arm_compiler",
      build_file = str(Label("//third_party/toolchains/cpus/arm:BUILD")))

  mkl_repository(
      name = "mkl",
      urls = [
          "http://mirror.bazel.build/github.com/01org/mkl-dnn/releases/download/v0.9/mklml_lnx_2018.0.20170720.tgz",
          "https://github.com/01org/mkl-dnn/releases/download/v0.9/mklml_lnx_2018.0.20170720.tgz",
      ],
      sha256 = "57ba56c4c243f403ff78f417ff854ef50b9eddf4a610a917b7c95e7fa8553a4b",
      strip_prefix = "mklml_lnx_2018.0.20170720",
      build_file = str(Label("//third_party/mkl:mkl.BUILD")),
      repository = tf_repo_name,
  )

  if path_prefix:
    print("path_prefix was specified to tf_workspace but is no longer used " +
          "and will be removed in the future.")

  native.new_http_archive(
      name = "eigen_archive",
      urls = [
          "http://mirror.bazel.build/bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
          "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
      ],
      sha256 = "ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4",
      strip_prefix = "eigen-eigen-f3a22f35b044",
      build_file = str(Label("//third_party:eigen.BUILD")),
  )

  native.new_http_archive(
      name = "arm_compiler",
      build_file = str(Label("//:arm_compiler.BUILD")),
      sha256 = "970285762565c7890c6c087d262b0a18286e7d0384f13a37786d8521773bc969",
      strip_prefix = "tools-0e906ebc527eab1cdbf7adabff5b474da9562e9f/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf",
      urls = [
          "http://mirror.bazel.build/github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
          "https://github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
      ],
  )

  native.new_http_archive(
      name = "libxsmm_archive",
      urls = [
          "http://mirror.bazel.build/github.com/hfp/libxsmm/archive/1.8.1.tar.gz",
          "https://github.com/hfp/libxsmm/archive/1.8.1.tar.gz",
      ],
      sha256 = "2ade869c3f42f23b5263c7d594aa3c7e5e61ac6a3afcaf5d6e42899d2a7986ce",
      strip_prefix = "libxsmm-1.8.1",
      build_file = str(Label("//third_party:libxsmm.BUILD")),
  )

  native.bind(
      name = "xsmm_avx",
      actual = "@libxsmm_archive//third_party:xsmm_avx",
  )

  native.new_http_archive(
      name = "ortools_archive",
      urls = [
          "http://mirror.bazel.build/github.com/google/or-tools/archive/253f7955c6a1fd805408fba2e42ac6d45b312d15.tar.gz",
          "https://github.com/google/or-tools/archive/253f7955c6a1fd805408fba2e42ac6d45b312d15.tar.gz",
      ],
      sha256 = "932075525642b04ac6f1b50589f1df5cd72ec2f448b721fd32234cf183f0e755",
      strip_prefix = "or-tools-253f7955c6a1fd805408fba2e42ac6d45b312d15/src",
      build_file = str(Label("//third_party:ortools.BUILD")),
  )

  native.http_archive(
      name = "com_googlesource_code_re2",
      urls = [
          "http://mirror.bazel.build/github.com/google/re2/archive/b94b7cd42e9f02673cd748c1ac1d16db4052514c.tar.gz",
          "https://github.com/google/re2/archive/b94b7cd42e9f02673cd748c1ac1d16db4052514c.tar.gz",
      ],
      sha256 = "bd63550101e056427c9e7ff12a408c1c8b74e9803f393ca916b2926fc2c4906f",
      strip_prefix = "re2-b94b7cd42e9f02673cd748c1ac1d16db4052514c",
  )

  native.http_archive(
      name = "gemmlowp",
      urls = [
          "http://mirror.bazel.build/github.com/google/gemmlowp/archive/010bb3e71a26ca1d0884a167081d092b43563996.tar.gz",
          "https://github.com/google/gemmlowp/archive/010bb3e71a26ca1d0884a167081d092b43563996.tar.gz",
      ],
      sha256 = "0d7a44327e26b622ee08faaea10f8d10b439bcfda622f9c98be1c036bc645cad",
      strip_prefix = "gemmlowp-010bb3e71a26ca1d0884a167081d092b43563996",
  )

  native.new_http_archive(
      name = "farmhash_archive",
      urls = [
          "http://mirror.bazel.build/github.com/google/farmhash/archive/23eecfbe7e84ebf2e229bd02248f431c36e12f1a.tar.gz",
          "https://github.com/google/farmhash/archive/23eecfbe7e84ebf2e229bd02248f431c36e12f1a.tar.gz",
      ],
      sha256 = "e5c86a2e32e7cb1d027d713cbf338be68ebbea76dbb2b2fdaae918864d3f8f3d",
      strip_prefix = "farmhash-23eecfbe7e84ebf2e229bd02248f431c36e12f1a",
      build_file = str(Label("//third_party:farmhash.BUILD")),
  )

  native.bind(
      name = "farmhash",
      actual = "@farmhash//:farmhash",
  )

  native.new_http_archive(
      name = "highwayhash",
      urls = [
          "http://mirror.bazel.build/github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
          "https://github.com/google/highwayhash/archive/dfcb97ca4fe9277bf9dc1802dd979b071896453b.tar.gz",
      ],
      sha256 = "0f30a15b1566d93f146c8d149878a06e91d9bb7ec2cfd76906df62a82be4aac9",
      strip_prefix = "highwayhash-dfcb97ca4fe9277bf9dc1802dd979b071896453b",
      build_file = str(Label("//third_party:highwayhash.BUILD")),
  )

  native.new_http_archive(
      name = "nasm",
      urls = [
          "http://mirror.bazel.build/www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2",
          "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.12.02.tar.bz2/d15843c3fb7db39af80571ee27ec6fad/nasm-2.12.02.tar.bz2",
      ],
      sha256 = "00b0891c678c065446ca59bcee64719d0096d54d6886e6e472aeee2e170ae324",
      strip_prefix = "nasm-2.12.02",
      build_file = str(Label("//third_party:nasm.BUILD")),
  )

  temp_workaround_http_archive(
      name = "jpeg",
      urls = [
          "http://mirror.bazel.build/github.com/libjpeg-turbo/libjpeg-turbo/archive/1.5.1.tar.gz",
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
          "http://mirror.bazel.build/github.com/glennrp/libpng/archive/v1.2.53.tar.gz",
          "https://github.com/glennrp/libpng/archive/v1.2.53.tar.gz",
      ],
      sha256 = "716c59c7dfc808a4c368f8ada526932be72b2fcea11dd85dc9d88b1df1dfe9c2",
      strip_prefix = "libpng-1.2.53",
      build_file = str(Label("//third_party:png.BUILD")),
  )

  native.new_http_archive(
      name = "gif_archive",
      urls = [
          "http://mirror.bazel.build/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
      ],
      sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
      strip_prefix = "giflib-5.1.4",
      build_file = str(Label("//third_party:gif.BUILD")),
  )

  native.new_http_archive(
      name = "six_archive",
      urls = [
          "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
          "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
      ],
      sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
      strip_prefix = "six-1.10.0",
      build_file = str(Label("//third_party:six.BUILD")),
  )

  native.new_http_archive(
      name = "org_python_pypi_backports_weakref",
      urls = [
          "http://mirror.bazel.build/pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
          "https://pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
      ],
      sha256 = "8813bf712a66b3d8b85dc289e1104ed220f1878cf981e2fe756dfaabe9a82892",
      strip_prefix = "backports.weakref-1.0rc1/src",
      build_file = str(Label("//third_party:backports_weakref.BUILD")),
  )

  native.new_http_archive(
      name = "com_github_andreif_codegen",
      urls = [
          "http://mirror.bazel.build/github.com/andreif/codegen/archive/1.0.tar.gz",
          "https://github.com/andreif/codegen/archive/1.0.tar.gz",
      ],
      sha256 = "2dadd04a2802de27e0fe5a19b76538f6da9d39ff244036afa00c1bba754de5ee",
      strip_prefix = "codegen-1.0",
      build_file = str(Label("//third_party:codegen.BUILD")),
  )

  filegroup_external(
      name = "org_python_license",
      licenses = ["notice"],  # Python 2.0
      sha256_urls = {
          "b5556e921715ddb9242c076cae3963f483aa47266c5e37ea4c187f77cc79501c": [
              "http://mirror.bazel.build/docs.python.org/2.7/_sources/license.txt",
              "https://docs.python.org/2.7/_sources/license.txt",
          ],
      },
  )

  native.bind(
      name = "six",
      actual = "@six_archive//:six",
  )

  patched_http_archive(
      name = "protobuf_archive",
      urls = [
          "http://mirror.bazel.build/github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
          "https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
      ],
      sha256 = "6d43b9d223ce09e5d4ce8b0060cb8a7513577a35a64c7e3dad10f0703bf3ad93",
      strip_prefix = "protobuf-0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66",
      # TODO: remove patching when tensorflow stops linking same protos into
      #       multiple shared libraries loaded in runtime by python.
      #       This patch fixes a runtime crash when tensorflow is compiled
      #       with clang -O2 on Linux (see https://github.com/tensorflow/tensorflow/issues/8394)
      patch_file = str(Label("//third_party/protobuf:add_noinlines.patch")),
  )

  native.bind(
      name = "protobuf",
      actual = "@protobuf_archive//:protobuf",
  )

  # We need to import the protobuf library under the names com_google_protobuf
  # and com_google_protobuf_cc to enable proto_library support in bazel.
  # Unfortunately there is no way to alias http_archives at the moment.
  native.http_archive(
      name = "com_google_protobuf",
      urls = [
          "http://mirror.bazel.build/github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
          "https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
      ],
      sha256 = "6d43b9d223ce09e5d4ce8b0060cb8a7513577a35a64c7e3dad10f0703bf3ad93",
      strip_prefix = "protobuf-0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66",
  )

  native.http_archive(
      name = "com_google_protobuf_cc",
      urls = [
          "http://mirror.bazel.build/github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
          "https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz",
      ],
      sha256 = "6d43b9d223ce09e5d4ce8b0060cb8a7513577a35a64c7e3dad10f0703bf3ad93",
      strip_prefix = "protobuf-0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66",
  )

  native.http_archive(
      name = "nsync",
      urls = [
          "https://github.com/google/nsync/archive/ad722c76c6e6653f66be2e1f69521b7f7517da55.tar.gz",
      ],
      sha256 = "7dd8ca49319f77e8226cd020a9210a525f88ac26e7041c59c95418223a1cdf55",
      strip_prefix = "nsync-ad722c76c6e6653f66be2e1f69521b7f7517da55",
  )

  native.http_archive(
      name = "com_google_googletest",
      urls = [
          "http://mirror.bazel.build/github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
          "https://github.com/google/googletest/archive/9816b96a6ddc0430671693df90192bbee57108b6.zip",
      ],
      sha256 = "9cbca84c4256bed17df2c8f4d00c912c19d247c11c9ba6647cd6dd5b5c996b8d",
      strip_prefix = "googletest-9816b96a6ddc0430671693df90192bbee57108b6",
  )

  native.http_archive(
      name = "com_github_gflags_gflags",
      urls = [
          "http://mirror.bazel.build/github.com/gflags/gflags/archive/f8a0efe03aa69b3336d8e228b37d4ccb17324b88.tar.gz",
          "https://github.com/gflags/gflags/archive/f8a0efe03aa69b3336d8e228b37d4ccb17324b88.tar.gz",
      ],
      sha256 = "4d222fab8f1ede4709cdff417d15a1336f862d7334a81abf76d09c15ecf9acd1",
      strip_prefix = "gflags-f8a0efe03aa69b3336d8e228b37d4ccb17324b88",
  )

  native.bind(
      name = "python_headers",
      actual = str(Label("//util/python:python_headers")),
  )

  native.new_http_archive(
      name = "pcre",
      sha256 = "ccdf7e788769838f8285b3ee672ed573358202305ee361cfec7a4a4fb005bbc7",
      urls = [
          "http://mirror.bazel.build/ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
          "http://ftp.exim.org/pub/pcre/pcre-8.39.tar.gz",
      ],
      strip_prefix = "pcre-8.39",
      build_file = str(Label("//third_party:pcre.BUILD")),
  )

  native.new_http_archive(
      name = "swig",
      sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
      urls = [
          "http://mirror.bazel.build/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
          "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
      ],
      strip_prefix = "swig-3.0.8",
      build_file = str(Label("//third_party:swig.BUILD")),
  )

  temp_workaround_http_archive(
      name = "curl",
      sha256 = "ff3e80c1ca6a068428726cd7dd19037a47cc538ce58ef61c59587191039b2ca6",
      urls = [
          "http://mirror.bazel.build/curl.haxx.se/download/curl-7.49.1.tar.gz",
          "https://curl.haxx.se/download/curl-7.49.1.tar.gz",
      ],
      strip_prefix = "curl-7.49.1",
      build_file = str(Label("//third_party:curl.BUILD")),
      repository = tf_repo_name
  )

  # grpc expects //external:protobuf_clib and //external:protobuf_compiler
  # to point to the protobuf's compiler library.
  native.bind(
      name = "protobuf_clib",
      actual = "@protobuf_archive//:protoc_lib",
  )

  native.bind(
      name = "libssl",
      actual = "@boringssl//:ssl",
  )

  # gRPC has includes directly from their third_party path for nanopb, so we
  # must depend on their version of it.
  native.bind(
      name = "nanopb",
      actual = "@grpc//third_party/nanopb:nanopb",
  )

  patched_http_archive(
      name = "grpc",
      urls = [
          "http://mirror.bazel.build/github.com/grpc/grpc/archive/781fd6f6ea03645a520cd5c675da67ab61f87e4b.tar.gz",
          "https://github.com/grpc/grpc/archive/781fd6f6ea03645a520cd5c675da67ab61f87e4b.tar.gz",
      ],
      sha256 = "2004635e6a078acfac8ffa71738397796be4f8fb72f572cc44ecee5d99511d9f",
      strip_prefix = "grpc-781fd6f6ea03645a520cd5c675da67ab61f87e4b",
      patch_file = str(Label("//third_party/grpc:grpc.patch")),
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
          "http://mirror.bazel.build/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
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
          "http://mirror.bazel.build/github.com/llvm-mirror/llvm/archive/9aafb854cc7cb8df8338c50cb411a54ce1e09796.tar.gz",
          "https://github.com/llvm-mirror/llvm/archive/9aafb854cc7cb8df8338c50cb411a54ce1e09796.tar.gz",
      ],
      sha256 = "2a6d4c23f6660d9130d8d5f16267db53a87f8d0104f9618b558c033570f110af",
      strip_prefix = "llvm-9aafb854cc7cb8df8338c50cb411a54ce1e09796",
      build_file = str(Label("//third_party/llvm:llvm.BUILD")),
      repository = tf_repo_name,
  )

  native.new_http_archive(
      name = "lmdb",
      urls = [
          "http://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.19.tar.gz",
          "https://github.com/LMDB/lmdb/archive/LMDB_0.9.19.tar.gz",
      ],
      sha256 = "108532fb94c6f227558d45be3f3347b52539f0f58290a7bb31ec06c462d05326",
      strip_prefix = "lmdb-LMDB_0.9.19/libraries/liblmdb",
      build_file = str(Label("//third_party:lmdb.BUILD")),
  )

  native.new_http_archive(
      name = "jsoncpp_git",
      urls = [
          "http://mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
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

  patched_http_archive(
      name = "boringssl",
      urls = [
          "http://mirror.bazel.build/github.com/google/boringssl/archive/bbcaa15b0647816b9a1a9b9e0d209cd6712f0105.tar.gz",
          "https://github.com/google/boringssl/archive/bbcaa15b0647816b9a1a9b9e0d209cd6712f0105.tar.gz",  # 2016-07-11
      ],
      sha256 = "025264d6e9a7ad371f2f66d17a28b6627de0c9592dc2eb54afd062f68f1f9aa3",
      strip_prefix = "boringssl-bbcaa15b0647816b9a1a9b9e0d209cd6712f0105",
      # Add patch to boringssl code to support s390x
      patch_file = str(Label("//third_party/boringssl:add_boringssl_s390x.patch")),
  )

  native.new_http_archive(
      name = "zlib_archive",
      urls = [
          "http://mirror.bazel.build/zlib.net/zlib-1.2.8.tar.gz",
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
      name = "fft2d",
      urls = [
          "http://mirror.bazel.build/www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
          "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
      ],
      sha256 = "52bb637c70b971958ec79c9c8752b1df5ff0218a4db4510e60826e0cb79b5296",
      build_file = str(Label("//third_party/fft2d:fft2d.BUILD")),
  )

  temp_workaround_http_archive(
      name = "snappy",
      urls = [
          "http://mirror.bazel.build/github.com/google/snappy/archive/1.1.4.tar.gz",
          "https://github.com/google/snappy/archive/1.1.4.tar.gz",
      ],
      sha256 = "2f7504c73d85bac842e893340333be8cb8561710642fc9562fccdd9d2c3fcc94",
      strip_prefix = "snappy-1.1.4",
      build_file = str(Label("//third_party:snappy.BUILD")),
      repository = tf_repo_name,
  )

  temp_workaround_http_archive(
      name = "nccl_archive",
      urls = [
          "http://mirror.bazel.build/github.com/nvidia/nccl/archive/ccfc4567dc3e2a37fb42cfbc64d10eb526e7da7b.tar.gz",
          "https://github.com/nvidia/nccl/archive/ccfc4567dc3e2a37fb42cfbc64d10eb526e7da7b.tar.gz",
      ],
      sha256 = "6c34a0862d9f8ed4ad5984c6a8206b351957bb14cf6ad7822720f285f4aada04",
      strip_prefix = "nccl-ccfc4567dc3e2a37fb42cfbc64d10eb526e7da7b",
      build_file = str(Label("//third_party:nccl.BUILD")),
      repository = tf_repo_name,
  )

  java_import_external(
      name = "junit",
      jar_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
      jar_urls = [
          "http://mirror.bazel.build/repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
          "http://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
          "http://maven.ibiblio.org/maven2/junit/junit/4.12/junit-4.12.jar",
      ],
      licenses = ["reciprocal"],  # Common Public License Version 1.0
      testonly_ = True,
      deps = ["@org_hamcrest_core"],
  )

  java_import_external(
      name = "org_hamcrest_core",
      jar_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
      jar_urls = [
          "http://mirror.bazel.build/repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
          "http://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
          "http://maven.ibiblio.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
      ],
      licenses = ["notice"],  # New BSD License
      testonly_ = True,
  )

  temp_workaround_http_archive(
      name = "jemalloc",
      urls = [
          "http://mirror.bazel.build/github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
          "https://github.com/jemalloc/jemalloc/archive/4.4.0.tar.gz",
      ],
      sha256 = "3c8f25c02e806c3ce0ab5fb7da1817f89fc9732709024e2a81b6b82f7cc792a8",
      strip_prefix = "jemalloc-4.4.0",
      build_file = str(Label("//third_party:jemalloc.BUILD")),
      repository = tf_repo_name,
  )

  native.new_http_archive(
      name = "com_google_pprof",
      urls = [
          "http://mirror.bazel.build/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
          "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
      ],
      sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
      strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
      build_file = str(Label("//third_party:pprof.BUILD")),
  )

  native.new_http_archive(
      name = "cub_archive",
      urls = [
          "http://mirror.bazel.build/github.com/NVlabs/cub/archive/69ceda618313df8e9cac6659d607b08949455d14.tar.gz",
          "https://github.com/NVlabs/cub/archive/69ceda618313df8e9cac6659d607b08949455d14.tar.gz",
      ],
      sha256 = "87e856522c283b8ea887c3b61d7d5b252d2dd74abac4f1d756d776e721223e82",
      strip_prefix = "cub-69ceda618313df8e9cac6659d607b08949455d14",
      build_file = str(Label("//third_party:cub.BUILD")),
  )

  native.bind(
      name = "cub",
      actual = "@cub_archive//:cub",
  )

  native.http_archive(
      name = "bazel_toolchains",
      urls = [
          "http://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/bccee4855c049d34bac481083b4c68e2fab8cc50.tar.gz",
          "https://github.com/bazelbuild/bazel-toolchains/archive/bccee4855c049d34bac481083b4c68e2fab8cc50.tar.gz",
      ],
      sha256 = "3903fd93b96b42067e00b7973a2c16c34e761ad7a0b55e1557d408f352849e41",
      strip_prefix = "bazel-toolchains-bccee4855c049d34bac481083b4c68e2fab8cc50",
  )
