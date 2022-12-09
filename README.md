# A collection of Python scripts related to typing

### `typograph.py`

Import the file and use the only function from it to make a correct typing hint from runtime data.

It supports basic types, `re.Match`, and `re.Pattern`.
For the rest, the behavior is a shot in the dark, depending on the imports in the file.

###### Sample usage:

```python
from typograph import name_type

a_list = [1, 2, None]
print(f'a_list: {name_type(a_list)}')
a_tuple = (1, 2, [3, {1: 1, 2: 'a', 3: [3]}])
print(f'a_tuple: {name_type(a_tuple)}')
a_dict = {1: 1, False: 'a', 3: {1: 1, 2: 'a', 3: [3]}}
print(f'a_dict: {name_type(a_dict)}')
```

### `py34.py`

Provide the Python code filenames as its command line arguments to get new files with typing annotations removed.
The result promises to be more compatible to Python 3.4.

The new files are of the same names, but stored in “py34” directory near the original files.

The script mostly works, but not everything is implemented:

 - It leaves `@` and `:=` operators unchanged, although the former was introduced in Python 3.5 and the latter in 3.8.
 - It leaves `async` statements and `await` operator unchanged, although they were introduced in Python 3.5.
 - It leaves the subclasses of `typing` classes unresolved, unless they can be substituted by classes from `collections`.
 - It doesn't turn a `match` statement into a sequence of `if`, `elif`, and `else`, but only removes type annotations in the cases.
 - It omits nested formatting options in f-strings.

To see the result, specify `-p` or `--print` parameter.
Beware, though, that all the output comes with not separator between the content of different files. 

###### Sample usage:

```commandline
python py34.py /usr/lib64/python3.11/tomllib/*.py
```

### `type_qt.py`

It requires `QtPy` to work.

PyCharm provides awful stubs for Qt for Python (PySide6, PyQt6, PyQt5, PySide2, PyQt4, and PySide).
There, the arguments count and the defaults may be wrong, static functions have an inappropriate `self` argument,
and there are no type hints.
It looks bizarre, for the correct definitions of the functions are in the docstrings.
Finally, the stubs consider Qt Signals to be functions with arbitrary arguments.

The script looks through the stubs located in `f'~/.cache/JetBrains/PyCharm*/python_stubs/*/{qtpy.API_NAME}'`
and in `f'~/AppData/Local/JetBrains/PyCharm*/python_stubs/*/{qtpy.API_NAME}'`.
I don't know the location of the stubs used in macOS.

`QtPy` has a beautiful way of specifying the version of the Qt bindings in use,
while regularizing the differences between the bindings.
If you need other bindings than what `QtPy` defaults to, there are environment variables to alter the behavior.

The script does everything on its own, takes to command line arguments.

###### Sample usage:

```commandline
python type_qt.py
```
