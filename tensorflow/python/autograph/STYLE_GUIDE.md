# AutoGraph Style Guide

This page contains style decisions that developers should follow when
contributing code to AutoGraph.

## TensorFlow Style

Follow the [TensorFlow style
guide](https://www.tensorflow.org/community/contribute/code_style), the [documentation
guide](https://www.tensorflow.org/community/contribute/docs) and the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).

Naming conventions:

1.  The name is TensorFlow, not Tensorflow.
2.  The name is AutoGraph, not Autograph.

## AutoGraph Style

Below are AutoGraph-specific conventions. In the event of conflict, it
supersedes all previous conventions.

1. __Types in docstrings.__ Use [PEP 484][https://www.python.org/dev/peps/pep-0484/]
    notation to describe the type for args, return values and attributes.

    Example:

    ```
    Args:
      foo: Dict[str, List[int]], a dictionary of sorts
    ```

2.  __Citations in Docstrings.__ Write a `#### References` subsection at the
    bottom of any docstring with citations. Use ICLR’s bibliography style to
    write references; for example, order entries by the first author's last
    name. Add a link to the paper if the publication is open source (ideally,
    arXiv).

    Write in-paragraph citations in general, e.g., [(Tran and Blei, 2018)][1].
    Write in-text citations when the citation is a noun, e.g., [Tran and Blei
    (2018)][1]. Write citations with more than two authors using et al., e.g.,
    [(Tran et al., 2018)][1]. Separate multiple citations with semicolon, e.g.,
    ([Tran and Blei, 2018][1]; [Gelman and Rubin, 1992][2]).

    Examples:

    ```none
    #### References

    # technical report
    [1]: Tony Finch. Incremental calculation of weighted mean and variance.
         _Technical Report_, 2009.
         http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

    # journal
    [2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
         Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.

    # arXiv preprint
    # use "et al." for papers with too many authors to maintain
    [3]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
         Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
         https://arxiv.org/abs/1711.10433

    # conference
    [4]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse.
         Flipout: Efficient Pseudo-Independent Weight Perturbations on
         Mini-Batches. In _International Conference on Learning
         Representations_, 2018.
         https://arxiv.org/abs/1803.04386
    ```

3.  Avoid LaTeX in docstrings.

    *   It is not rendered in many (if not most) editors and can be hard to read
        for both LaTeX experts and non-experts.

4. Write docstring and comment math using ASCII friendly notation; python using
    operators. E.g., `x**2` better than `x^2`, `x[i, j]` better than `x_{i,j}`,
    `sum{ f(x[i]) : i=1...n }` better than `\sum_{i=1}^n f(x_i)` `int{sin(x) dx:
    x in [0, 2 pi]}` better than `\int_0^{2\pi} sin(x) dx`.

    *   The more we stick to python style, the more someone can
        copy/paste/execute.
    *   Python style is usually easier to read as ASCII.
