import tensorflow.compat.v2 as tf

from tensorflow.python.keras import backend

class BackendTest(tf.test.TestCase):
    def test_categorical_crossentropy_zero_outputs(self):
        # Create a true label tensor and an all-zero prediction tensor
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        # Calculate categorical crossentropy
        loss = backend.categorical_crossentropy(y_true, y_pred)
        
        # Verify that the loss is not NaN
        self.assertFalse(tf.reduce_any(tf.math.is_nan(loss)))
        
        # Optionally verify the output matches expectation
        # With zero predictions, they sum to 0. 
        # The code adds epsilon and clips to avoid division by zero and log(0).
        # The result should be a valid number, not NaN.
        loss_val = self.evaluate(loss)
        self.assertFalse(tf.math.is_nan(loss_val[0]))
        self.assertFalse(tf.math.is_nan(loss_val[1]))

if __name__ == '__main__':
    tf.test.main()
