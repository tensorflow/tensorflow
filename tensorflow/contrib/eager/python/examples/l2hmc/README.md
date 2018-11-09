# L2HMC with TensorFlow eager execution

This folder contains an implementation of [L2HMC](https://arxiv.org/pdf/1711.09268.pdf)  adapted from the released implementation by the authors. The presented implementation runs in both eager and graph mode.
With eager execution enabled, longer sample chains can be handled compared to graph mode, since no graph is explicitly stored. Moreover, with eager execution enabled, there is no need to use a `tf.while_loop`.

## What is L2HMC?
L2HMC is an adaptive Markov Chain Monte Carlo (MCMC) algorithm that learns a non-volume preserving transformation
for a Hamiltonian Monte Carlo (HMC) sampling algorithm. More specifically, the non-volume preserving
transformation is learned with neural nets instantiated within Normalizing Flows
(real-NVPs).

##  Content

- `l2hmc.py`: Dynamics definitions and example energy functions,
including the 2D strongly correlated Gaussian and the rough well energy function,
- `l2hmc_test.py`: Unit tests and benchmarks for training a sampler on the energy functions in both eager and graph mode.
- `neural_nets.py`: The neural net for learning the kernel on the 2D strongly correlated example.
- `main.py`: Run to train a samplers on 2D energy landscapes.

## To run
- Make sure you have installed TensorFlow 1.9+ or the latest `tf-nightly` or `tf-nightly-gpu` pip package.
- Execute the command

```bash
python main.py --train_dir ${PWD}/dump --use_defun
```

Specifying the optional argument `train_dir` will store event files for
tensorboard and a plot of sampled chain from the trained sampler.

Specifying the optional argument `use_defun` will let the program use compiled
graphs when running specific sections and improve the overall speed.

## Boosting Performance with `tfe.defun`
Currently, some models may experience increased overhead with eager execution enabled.
To improve performance, we could wrap certain functions with the decorator `@tfe.defun`.
For example, we could wrap the function that does the sampling step:

```python
@tfe.defun
def apply_transition(old_sample):
  new_sample = ...
  return new_sample
```

We could also explicitly wrap the desired function with `tfe.defun`:

```python
apply_transition = tfe.defun(apply_transition)
```

## Reference
Generalizing Hamiltonian Monte Carlo with Neural Networks. Levy, Daniel, Hoffman, Matthew D, and Sohl-Dickstein, Jascha. International Conference on Learning Representations (ICLR), 2018.
