import tensorflow as tf
from gpu_memory_tracker import GPUMemoryTracker, track_memory

# Create a tracker instance
tracker = GPUMemoryTracker(log_file="memory_log.txt")

@track_memory("model_training")
def train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(10)
    ])
    
    x = tf.random.normal((1000, 100))
    y = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)
    
    tracker.log_memory("Before compilation")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    with tracker.track("model_fit"):
        model.fit(x, y, epochs=2, batch_size=32)

if __name__ == "__main__":
    train_model()
