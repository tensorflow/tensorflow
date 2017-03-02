# TensorFlow Python API Upgrade Utility

This tool allows you to upgrade your existing TensorFlow Python scripts.
This script can be run on a single Python file:

```
tf_upgrade.py --infile foo.py --outfile foo-upgraded.py
```

It will print a list of errors it finds that it can't fix. You can also run
it on a directory tree:

```
tf_upgrade.py --intree coolcode --outtree coolcode-upgraded
```

In either case, it will also dump out a report e.g. which will detail changes
e.g.:

```
third_party/tensorflow/tools/compatibility/test_file_v0.11.py Line 125

Renamed keyword argument from `dim` to `axis`
Renamed keyword argument from `squeeze_dims` to `axis`

    Old:                   [[1, 2, 3]], dim=1), squeeze_dims=[1]).eval(),
                                        ~~~~    ~~~~~~~~~~~~~
    New:                   [[1, 2, 3]], axis=1), axis=[1]).eval(),
                                        ~~~~~    ~~~~~
```

## Caveats

- Don't update parts of your code manually before running this script. In
particular, functions that have had reordered arguments like `tf.concat`
or `tf.split` will cause the script to incorrectly add keyword arguments that
mismap arguments.

- This script wouldn't actually reorder arguments. Instead, the script will add
keyword arguments to functions that had their arguments reordered.

- This script is not able to upgrade all functions. One notable example is
`tf.reverse()` which has been changed to take a list of indices rather than
a tensor of bools. If the script detects this, it will report this to stdout
(and in the report), and you can fix it manually. For example if you have
`tf.reverse(a, [False, True, True])` you will need to manually change it to
`tf.reverse(a, [1, 2])`.

- There are some syntaxes that are not handleable with this script as this
script was designed to use only standard python packages. If the script fails
with "A necessary keyword argument failed to be inserted." or
"Failed to find keyword lexicographically. Fix manually.", you can try
[@machrisaa's fork of this script](https://github.com/machrisaa/tf0to1).
[@machrisaa](https://github.com/machrisaa) has used the
[RedBaron Python refactoring engine](https://redbaron.readthedocs.io/en/latest/)
which is able to localize syntactic elements more reliably than the built-in
`ast` module this script is based upon.
