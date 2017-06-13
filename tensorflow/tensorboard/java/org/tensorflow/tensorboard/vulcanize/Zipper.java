// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.tensorflow.tensorboard.vulcanize;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.protobuf.TextFormat;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.Webfiles;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfilesSource;
import io.bazel.rules.closure.webfiles.WebfilesWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.Deflater;

/**
 * Simple one-off solution for TensorBoard zipping of web_library rules.
 *
 * <p>This is intended to collect static assets for production web server deployment. The paths of
 * files inside the zip will be web paths, with the prefix slash removed. These files will be
 * topologically ordered, i.e. web files higher up in the build tree come first.
 */
public final class Zipper {

  public static void main(String[] args) throws IOException {
    Set<String> alreadyZipped = new HashSet<>();
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(
                Paths.get(args[0]),
                StandardOpenOption.WRITE,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING),
            Deflater.BEST_SPEED)) {
      for (int i = 1; i < args.length; i++) {
        Webfiles manifest = loadWebfilesPbtxt(Paths.get(args[i]));
        for (WebfilesSource src : manifest.getSrcList()) {
          if (!alreadyZipped.add(src.getWebpath())) {
            continue;
          }
          try (InputStream input = Files.newInputStream(Paths.get(src.getPath()))) {
            writer.writeWebfile(
                WebfileInfo.newBuilder().setWebpath(src.getWebpath()).build(), input);
          }
        }
      }
    }
  }

  private static Webfiles loadWebfilesPbtxt(Path path) throws IOException {
    Webfiles.Builder build = Webfiles.newBuilder();
    TextFormat.getParser().merge(new String(Files.readAllBytes(path), UTF_8), build);
    return build.build();
  }
}
