import unittest
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class TestSentimentAnalysisModel(unittest.TestCase):
    def setUp(self):
        # Initialize the model for testing
        self.model = keras.Sequential([
            layers.Embedding(input_dim=1000, output_dim=64),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def test_model_output_shape(self):
        # Simulate input data
        input_data = np.random.randint(1000, size=(10, 50))  # 10 samples, 50 tokens each
        output = self.model.predict(input_data)
        self.assertEqual(output.shape, (10, 1), "Output shape should be (10, 1)")

    def test_model_training(self):
        # Simulate training data
        training_sentences = np.random.randint(1000, size=(20, 50))  # 20 samples, 50 tokens each
        training_labels = np.random.randint(2, size=(20, 1))  # Binary labels
        history = self.model.fit(training_sentences, training_labels, epochs=1)
        self.assertIn('accuracy', history.history, "Accuracy metric should be present after training")

if __name__ == '__main__':
    unittest.main()
