# Integrate (contrib)
[TOC]

Integration and ODE solvers for TensorFlow.

## Example: Lorenz attractor

We can use `odeint` to solve the
[Lorentz system](https://en.wikipedia.org/wiki/Lorenz_system) of ordinary
differential equations, a prototypical example of chaotic dynamics:

```python
rho = 28.0
sigma = 10.0
beta = 8.0/3.0

def lorenz_equation(state, t):
  x, y, z = tf.unstack(state)
  dx = sigma * (y - x)
  dy = x * (rho - z) - y
  dz = x * y - beta * z
  return tf.stack([dx, dy, dz])

init_state = tf.constant([0, 2, 20], dtype=tf.float64)
t = np.linspace(0, 50, num=5000)
tensor_state, tensor_info = tf.contrib.integrate.odeint(
    lorenz_equation, init_state, t, full_output=True)

sess = tf.Session()
state, info = sess.run([tensor_state, tensor_info])
x, y, z = state.T
plt.plot(x, z)
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/lorenz_attractor.png" alt>
</div>

## Ops

*   @{tf.contrib.integrate.odeint}
