# TensorFlow AutoGraph Style Guide

This page contains style decisions that both developers and users of TensorFlow
AutoGraph should follow to increase the readability of their code, reduce the
number of errors, and promote consistency. We borrow many style principles from the TensorFlow Probability style guide.

## TensorFlow Style

Follow the [TensorFlow style
guide](https://www.tensorflow.org/community/style_guide) and [documentation
guide](https://www.tensorflow.org/community/documentation). Below are additional
TensorFlow conventions not noted in those guides. In the future, these noted
conventions may be moved upstream.

1.  The name is TensorFlow, not Tensorflow.
2.  The name is AutoGraph, not Autograph.

## TensorFlow Code of Conduct
Please review and follow the [TensorFlow Code of Conduct](../../CODE_OF_CONDUCT.md).

## TensorFlow AutoGraph Style

Below are TensorFlow AutoGraph-specific conventions. In the event of conflict,
it supercedes all previous conventions.

1.  __Importing submodule aliases.__ Use the Pythonic style 
`from tensorflow.contrib.autograph.converters import ifexp` and `from tensorflow.contrib import autograph as ag`.

2.  __Examples in Docstrings.__ Write a `#### Examples` subsection below `Args`,
    `Returns`, `Raises`, etc. to illustrate examples. If the docstring's last
    line is a fence bracket (\`\`\`) closing a code snippet, add an empty line
    before closing the docstring with \"\"\". This properly displays the code
    snippet.

    Justification: Users regularly need to remind themselves of args and
    semantics. But rarely look at examples more than the first time. But since
    examples are usually long (which is great!) it means they have to do a lot
    of annoying scrolling ...unless Examples follow Args/Returns/Raises.

3.  __Citations in Docstrings.__ Write a `#### References` subsection at the
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

4.  When doing float math over literals eg use `1.` instead of `1` or `1.0`.

    *   Using `1.` is another line of defense against an automatic casting
        mistake. (Using `1.0` is also such a defense but is not minimal.)

5.  Prefer using named args for functions' 2nd args onward.

    *   Definitely use named args for 2nd args onward in docstrings.

9.  Avoid LaTeX in docstrings.

    *   It is not rendered in many (if not most) editors and can be hard to read
        for both LaTeX experts and non-experts.

10. Write docstring and comment math using ASCII friendly notation; python using
    operators. E.g., `x**2` better than `x^2`, `x[i, j]` better than `x_{i,j}`,
    `sum{ f(x[i]) : i=1...n }` better than `\sum_{i=1}^n f(x_i)` `int{sin(x) dx:
    x in [0, 2 pi]}` better than `\int_0^{2\pi} sin(x) dx`.

    *   The more we stick to python style, the more someone can
        copy/paste/execute.
    *   Python style is usually easier to read as ASCII.

11. All public functions require docstrings with: one line description, Args,
    Returns, Raises (if raises exceptions).

    *   Returns docstrings should be in the same format as Args, eg, of the form
        "name: Description." Part of the rationale is that we are suggesting a
        reasonable variable name for the returned object(s).

12. Regard `*args` and/or `**kwargs` as features of last resort.

    *   Keyword arguments make the intention of a function call more clear.
    *   [Possible exceptions for
        `kwargs`](https://stackoverflow.com/questions/1415812/why-use-kwargs-in-python-what-are-some-real-world-advantages-over-using-named).

18. The `__init__.py` file for modules should use TensorFlow's
    `remove_undocumented` feature, which seals the module's methods.

21. Use `"{}".format()` rather than `"" %` for string formatting.

    Justification: [PEP 3101](https://www.python.org/dev/peps/pep-3101/) and
    [Python official
    tutorials](https://docs.python.org/3.2/tutorial/inputoutput.html#old-string-formatting):
    "...this old style of formatting will eventually be removed from the
    language, str.format() should generally be used."
