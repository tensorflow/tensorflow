import sys
import os
import subprocess
from pathlib import Path


def generate_proto(source, require = True, protoc = None):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  if not require and not os.path.exists(source):
    return

  output = source.replace(".proto", "_pb2.py").replace("../src/", "")

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print("Generating %s..." % output)

    if not os.path.exists(source):
      sys.stderr.write("Can't find required file: %s\n" % source)
      sys.exit(-1)

    if protoc is None:
      sys.stderr.write(
          "protoc is not installed nor found in ../src.  Please compile it "
          "or install the binary package.\n")
      sys.exit(-1)

    protoc_command = [ protoc, "-I../src", "-I.", "--python_out=.", source ]
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Wrong arguments. Usage: build_pb [proto_python_path] [proto_exe]')

    protobuf_dir = Path(sys.argv[1]).resolve()
    protobuf_exe = Path(sys.argv[2]).resolve()
    if not Path(protobuf_dir).exists():
        raise ValueError('Protobuf python directory doesn\'t exist!')
    if not Path(protobuf_exe).exists():
        raise ValueError('Protobuf seems hasn\'t been compiled')
    
    protobuf_dir = str(protobuf_dir)
    protobuf_exe = str(protobuf_exe)
    
    os.chdir(protobuf_dir)
    
    generate_proto("../src/google/protobuf/descriptor.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/compiler/plugin.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/any.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/api.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/duration.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/empty.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/field_mask.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/source_context.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/struct.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/timestamp.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/type.proto", protoc=protobuf_exe)
    generate_proto("../src/google/protobuf/wrappers.proto", protoc=protobuf_exe)
