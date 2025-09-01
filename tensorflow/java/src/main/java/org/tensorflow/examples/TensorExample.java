package org.tensorflow.examples;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

/**
 * This class demonstrates a simple example of creating and manipulating a TensorFlow tensor in Java.
 */
public class TensorExample {

    /**
     * Main method to create a tensor, populate it with data, and print its contents.
     * @param args Not used.
     */
    public static void main(String[] args) {

        // Create a 2x3 tensor of type float and populate it with values.
        Tensor tensor = Tensor.of(TFloat32.class, Shape.of(2, 3), data -> {
            data.setFloat(1.0f, 0, 0);
            data.setFloat(2.0f, 0, 1);
            data.setFloat(3.0f, 0, 2);
            data.setFloat(4.0f, 1, 0);
            data.setFloat(5.0f, 1, 1);
            data.setFloat(6.0f, 1, 2);
        });

        // Print the size and contents of the tensor. Tensor size is 6 (2*3).
        System.out.println("Tensor size: " + tensor.size());

        // All the tensor elements are printed in a single loop.
        for (int i = 0; i < tensor.size(); i ++) {
            System.out.println("Tensor: " + tensor.asRawTensor().data().asFloats().getFloat(i));
        }
    }
}