TensorFlow has many kernels for doing (deep) learning and data manipulation.
There are typically assembled into computational graphs which can run
efficiently in a variety of environments.

We are exploring an alternative interaction, where kernels are invoked
immediately and call this "eager execution". We are hoping to retain the
benefits of graphs while improving usability with benefits like:

- Immediate error messages and easier debugging
- Flexibility to use Python datastructures and control flow
- Reduced boilerplate

Eager execution is under active development.
There are not many developer-facing materials yet, but stay tuned for updates
in this directory.
