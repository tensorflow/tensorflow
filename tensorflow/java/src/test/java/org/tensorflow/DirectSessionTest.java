package org.tensorflow;

//import static java.nio.charset.StandardCharsets.UTF_8;

//import static org.junit.Assert.assertTrue;
//import static org.junit.Assert.assertFalse;
//import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.nio.FloatBuffer;
import java.util.Arrays;

//import org.tensorflow.framework.CallableOptions;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DirectSessionTest {
  private static final float EPSILON_F = 1e-7f;

  @Test
  public void DirectSessionTest() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {

      Operation input1 = g.opBuilder("Placeholder","input1")
                          .setAttr("dtype",DataType.FLOAT)
                          .build();

      Operation input2 = g.opBuilder("Const", "input2")
                          .setAttr("dtype", DataType.FLOAT)
                          .setAttr("value", Tensor.<Float>create((float) 2.0, Float.class))
                          .build();

      Operation output = g.opBuilder("Mul", "output1")
                          .addInput(input1.output(0))
                          .addInput(input2.output(0))
                          .build();

      int size = 256;

      long[] shape = {size};
      float[] floats = new float[size];
      float[] expected = new float[size];

      for (int i = 0; i < size; ++i) {
        floats[i] = i+1;
        expected[i] = 2*floats[i];
      }

      Tensor<Float> t_cpu = Tensor.create(shape,FloatBuffer.wrap(floats));

      // dummy run
      Tensor<?> res_cpu = s.runner().fetch("output1").feed("input1", t_cpu).run().get(0);

      // don't have to test regular run
      /*
      float [] res1 = new float[floats.length];
      res_cpu.copyTo(res1);

      for (int i = 0; i < size; ++i) {
        assertEquals(expected[i], res1[i], EPSILON_F);
      }
      */

      // now we need to teleport data to GPU;

      //String gpuDeviceName = s.GPUDeviceName();
      // Expecting but not insisting on "/job:localhost/replica:0/task:0/device:GPU:0"
      //byte[] gpuName = gpuDeviceName.getBytes(UTF_8);

      /*
       * The following message is found in SavedModelBundleTest.java, enjoy:
       */

      // Ideally this would use the generated Java sources for protocol buffers
      // and end up with something like the snippet below. However, generating
      // the Java files for the .proto files in tensorflow/core:protos_all is
      // a bit cumbersome in bazel until the proto_library rule is setup.
      //
      // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
      // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
      // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
      //
      // For this test, for now, the use of specific bytes suffices.

      /*
       * In other words, "package org.tensorflow.framework does not exist".
       * The following will work in a standalone program, but will not work in the test:
       */

      /*
       *  CallableOptions callableOpts1 = CallableOptions.newBuilder()
       *                                                .addFetch("output1:0")
       *                                                .addFeed("input1:0")
       *                                                .putFetchDevices("output1:0", gpuDeviceName)
       *                                                .build();
       */

      /*
       * "For this test, for now, the use of specific bytes suffices."
       */
      byte[] opts = {
        0x0a, 0x08, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x31, 0x3a, 0x30, 0x12, 0x09, 0x6f, 0x75, 0x74, 0x70,
        0x75, 0x74, 0x31, 0x3a, 0x30, 0x32, 0x38, 0x0a, 0x08, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x31, 0x3a,
        0x30, 0x12, 0x2c, 0x2f, 0x6a, 0x6f, 0x62, 0x3a, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x68, 0x6f, 0x73,
        0x74, 0x2f, 0x72, 0x65, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x3a, 0x30, 0x2f, 0x74, 0x61, 0x73, 0x6b,
        0x3a, 0x30, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x3a, 0x47, 0x50, 0x55, 0x3a, 0x30
      };

      /*
      feed from GPU options:
      {
        0x0a, 0x08, "input1:0",
        0x12, 0x09, "output1:0",
        0x32, 0x38,
          0x0a, 0x08, "input1:0",
          0x12, 0x2c, "/job:localhost/replica:0/task:0/device:GPU:0"
      }
      */

      Tensor<?> t_gpu = Tensor.createGPU(new long[]{256},DataType.FLOAT);

      long handle = s.makeCallable(opts);
      Tensor<?> res_gpu = s.runner().fetch("output1").feed("input1",t_gpu).runCallable(handle).get(0);

      float [] res2 = new float[floats.length];
      res_gpu.copyTo(res2);

      // print-that-crap!
      System.out.println(Arrays.toString(res2));

    }

  }

}