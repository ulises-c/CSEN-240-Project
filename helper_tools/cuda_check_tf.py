import tensorflow as tf
import time

# Ensure GPU is detected
print("---")
print(tf.__version__)
print(tf.sysconfig.get_build_info())
print("---")
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available:", gpus)
print("---")

if gpus:
    try:
        # Enable memory growth (optional, but can help prevent OOM errors)
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Define a large matrix multiplication task
size = 20000  # You can increase this for a bigger test
A = tf.random.uniform((size, size), dtype=tf.float32)
B = tf.random.uniform((size, size), dtype=tf.float32)

# Run on GPU
with tf.device("/GPU:0"):
    start = time.time()
    C = tf.matmul(A, B)
    tf.keras.backend.eval(C)  # Ensure computation is executed
    end = time.time()
    print(f"GPU computation time: {end - start:.4f} seconds")

# Run on CPU for comparison
with tf.device("/CPU:0"):
    start = time.time()
    C_cpu = tf.matmul(A, B)
    tf.keras.backend.eval(C_cpu)
    end = time.time()
    print(f"CPU computation time: {end - start:.4f} seconds")
