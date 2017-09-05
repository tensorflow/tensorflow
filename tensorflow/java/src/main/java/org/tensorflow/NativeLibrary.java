/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Helper class for loading the TensorFlow Java native library.
 *
 * <p>The Java TensorFlow bindings require a native (JNI) library. This library
 * (libtensorflow_jni.so on Linux, libtensorflow_jni.dylib on OS X, tensorflow_jni.dll on Windows)
 * can be made available to the JVM using the java.library.path System property (e.g., using
 * -Djava.library.path command-line argument). However, doing so requires an additional step of
 * configuration.
 *
 * <p>Alternatively, the native libraries can be packaed in a .jar, making them easily usable from
 * build systems like Maven. However, in such cases, the native library has to be extracted from the
 * .jar archive.
 *
 * <p>NativeLibrary.load() takes care of this. First looking for the library in java.library.path
 * and failing that, it tries to find the OS and architecture specific version of the library in the
 * set of ClassLoader resources (under org/tensorflow/native/OS-ARCH). The resources paths used for
 * lookup must be consistent with any packaging (such as on Maven Central) of the TensorFlow Java
 * native libraries.
 */
final class NativeLibrary {
  private static final boolean DEBUG =
      System.getProperty("org.tensorflow.NativeLibrary.DEBUG") != null;
  private static final String LIBNAME = "tensorflow_jni";

  public static void load() {
    if (isLoaded() || tryLoadLibrary()) {
      // Either:
      // (1) The native library has already been statically loaded, OR
      // (2) The required native code has been statically linked (through a custom launcher), OR
      // (3) The native code is part of another library (such as an application-level library)
      // that has already been loaded. For example, tensorflow/examples/android and
      // tensorflow/contrib/android include the required native code in differently named libraries.
      //
      // Doesn't matter how, but it seems the native code is loaded, so nothing else to do.
      return;
    }
    // Native code is not present, perhaps it has been packaged into the .jar file containing this.
    final String resourceName = makeResourceName();
    log("resourceName: " + resourceName);
    final InputStream resource =
        NativeLibrary.class.getClassLoader().getResourceAsStream(resourceName);
    if (resource == null) {
      throw new UnsatisfiedLinkError(
          String.format(
              "Cannot find TensorFlow native library for OS: %s, architecture: %s. See "
                  + "https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java/README.md"
                  + " for possible solutions (such as building the library from source). Additional"
                  + " information on attempts to find the native library can be obtained by adding"
                  + " org.tensorflow.NativeLibrary.DEBUG=1 to the system properties of the JVM.",
              os(), architecture()));
    }
    try {
      System.load(extractResource(resource));
    } catch (IOException e) {
      throw new UnsatisfiedLinkError(
          String.format(
              "Unable to extract native library into a temporary file (%s)", e.toString()));
    }
  }

  private static boolean tryLoadLibrary() {
    try {
      System.loadLibrary(LIBNAME);
      return true;
    } catch (UnsatisfiedLinkError e) {
      log("tryLoadLibraryFailed: " + e.getMessage());
      return false;
    }
  }

  private static boolean isLoaded() {
    try {
      TensorFlow.version();
      log("isLoaded: true");
      return true;
    } catch (UnsatisfiedLinkError e) {
      return false;
    }
  }

  private static String extractResource(InputStream resource) throws IOException {
    final String sampleFilename = System.mapLibraryName(LIBNAME);
    final int dot = sampleFilename.indexOf(".");
    final String prefix = (dot < 0) ? sampleFilename : sampleFilename.substring(0, dot);
    final String suffix = (dot < 0) ? null : sampleFilename.substring(dot);

    final File dst = File.createTempFile(prefix, suffix);
    final String dstPath = dst.getAbsolutePath();
    dst.deleteOnExit();
    log("extracting native library to: " + dstPath);
    final long nbytes = copy(resource, dst);
    log(String.format("copied %d bytes to %s", nbytes, dstPath));
    return dstPath;
  }

  private static String os() {
    final String p = System.getProperty("os.name").toLowerCase();
    if (p.contains("linux")) {
      return "linux";
    } else if (p.contains("os x") || p.contains("darwin")) {
      return "darwin";
    } else if (p.contains("windows")) {
      return "windows";
    } else {
      return p.replaceAll("\\s", "");
    }
  }

  private static String architecture() {
    final String arch = System.getProperty("os.arch").toLowerCase();
    return (arch.equals("amd64")) ? "x86_64" : arch;
  }

  private static void log(String msg) {
    if (DEBUG) {
      System.err.println("org.tensorflow.NativeLibrary: " + msg);
    }
  }

  private static String makeResourceName() {
    return "org/tensorflow/native/"
        + String.format("%s-%s/", os(), architecture())
        + System.mapLibraryName(LIBNAME);
  }

  private static long copy(InputStream src, File dstFile) throws IOException {
    FileOutputStream dst = new FileOutputStream(dstFile);
    try {
      byte[] buffer = new byte[1 << 20]; // 1MB
      long ret = 0;
      int n = 0;
      while ((n = src.read(buffer)) >= 0) {
        dst.write(buffer, 0, n);
        ret += n;
      }
      return ret;
    } finally {
      dst.close();
      src.close();
    }
  }

  private NativeLibrary() {}
}
