import multiprocessing
from functools import lru_cache
import traceback


def pre_process_network(input_tensor):
    """Pre-process the input tensor."""
    w1 = tf.Variable(tf.random.normal([10000, 10000]))
    w2 = tf.Variable(tf.random.normal([10000, 10000]))
    w3 = tf.Variable(tf.random.normal([10000, 10000]))
    x = tf.matmul(input_tensor, w1)
    x = tf.matmul(x, w2)
    x = tf.matmul(x, w3)
    return x


def mid_process_network_async(input_tensor):
    """Process the input tensor asynchronously."""
    w1 = tf.Variable(tf.random.normal([10000, 100]))
    w2 = tf.Variable(tf.random.normal([100, 100]))
    w3 = tf.Variable(tf.random.normal([100, 10000]))
    input_tensor = tf.compat.v1.convert_to_tensor(input_tensor)
    x = tf.matmul(input_tensor, w1)
    x = tf.matmul(x, w2)
    x = tf.matmul(x, w3)
    return x


@lru_cache(maxsize=None)  # Cache all results
async def cached_mid_process_network_async(input_tensor):
    """Asynchronously process the input tensor and cache the result."""
    return await mid_process_network_async(input_tensor)


def post_process_network(input_tensor):
    """Post-process the input tensor."""
    w1 = tf.Variable(tf.random.normal([10000, 10000]))
    w2 = tf.Variable(tf.random.normal([10000, 10000]))
    w3 = tf.Variable(tf.random.normal([10000, 10000]))
    x = tf.matmul(input_tensor, w1)
    x = tf.matmul(x, w2)
    x = tf.matmul(x, w3)
    return x


def process(input_tensor):
    """Process a single input tensor."""
    pre_output = pre_process_network(input_tensor)
    try:
        mid_output = cached_mid_process_network_async(pre_output)
        post_output = post_process_network(mid_output)
    except Exception:
        print("An error occurred:")
        print(traceback.format_exc())
        return None

    return post_output


def main(number_of_networks, number_of_cpus):
    """Main function."""
    input_tensors = [tf.ones([1, 10000]) for _ in range(number_of_networks)]

    pool = multiprocessing.Pool(processes=number_of_cpus)

    results = pool.map(process, input_tensors)

    pool.close()
    pool.join()

    for result in results:
        if result is not None:
            print(result)


if __name__ == "__main__":
    """Entry point."""
    number_of_networks = int(input("Enter the number of asynchronous networks: "))
    number_of_cpus = int(input("Enter the number of CPUs: "))
    main(number_of_networks, number_of_cpus)
