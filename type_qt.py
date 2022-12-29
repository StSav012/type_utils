# coding: utf-8
import builtins
import functools
import keyword
import logging
from pathlib import Path

import qtpy  # to get the currently used Qt bindings
from qtpy.QtCore import Signal

logging.basicConfig()
log: logging.Logger = logging.getLogger()
log.setLevel(logging.INFO)

BUILTIN_NAMES: list[str] = dir(builtins)


def indent(line: str) -> int:
    """ Count the spaces at the beginning of the string """
    i: int
    for i in range(len(line)):
        if not line[i].isspace():
            return i
    return 0


@functools.lru_cache(maxsize=1, typed=True)
def func_name(func: str) -> str:
    """ Find the function name from its definition """
    if 'def ' not in func:
        raise ValueError(f'Not a function:\n{func}')

    def_pos: int = func.find('def ')
    return func[(def_pos + 4):(func.find('(', def_pos + 4))].strip()


def ensure_arg_names(line: str) -> str:
    """ Add dummy arg names if only arg types specified """

    # should we type `self`? (see https://peps.python.org/pep-0673/)

    def split_args(args_str: str) -> list[str]:
        in_quotes: bool = False
        in_double_quotes: bool = False
        in_triple_quotes: bool = False
        in_triple_double_quotes: bool = False
        parentheses_level: int = 0
        brackets_level: int = 0
        braces_level: int = 0
        args_list: list[str] = []
        while args_str:
            i: int
            for i in range(len(args_str)):
                if i == len(args_str) - 1:
                    args_list.append(args_str.strip())
                    args_str = ''
                    break
                if (args_str[i] == ','
                        and not in_quotes
                        and not in_double_quotes
                        and not in_triple_quotes
                        and not in_triple_double_quotes
                        and parentheses_level == 0
                        and brackets_level == 0
                        and braces_level == 0):
                    args_list.append(args_str[:i].strip())
                    args_str = args_str[(i + 1):].strip()
                    break
                if not in_quotes and not in_double_quotes and not in_triple_quotes and not in_triple_double_quotes:
                    match args_str[i]:
                        case '(':
                            parentheses_level += 1
                        case ')':
                            parentheses_level -= 1
                        case '[':
                            brackets_level += 1
                        case ']':
                            brackets_level -= 1
                        case '{':
                            braces_level += 1
                        case '}':
                            braces_level -= 1
                if not in_double_quotes and not in_triple_double_quotes and args_str.startswith("'''", i):
                    if in_quotes:
                        in_quotes = False
                    else:
                        in_triple_quotes = not in_triple_quotes
                if not in_quotes and not in_triple_quotes and args_str.startswith('"""', i):
                    if in_double_quotes:
                        in_double_quotes = False
                    else:
                        in_triple_double_quotes = not in_triple_double_quotes
                if not in_triple_quotes and not in_triple_double_quotes:
                    match args_str[i]:
                        case '\'':
                            in_quotes = not in_quotes
                        case '"':
                            in_double_quotes = not in_double_quotes

        return args_list

    def looks_bad(a: str) -> bool:
        if ':' in a:
            return False
        if '=' in a:
            a = a[:a.find('=')].strip()
        return not a.isidentifier() or keyword.iskeyword(a) or a in BUILTIN_NAMES

    def looks_like_qt_type(a: str) -> bool:
        if not a.startswith('Q'):
            return False
        return all(map(str.isidentifier, a.split('.')))

    def contains_bad_arg() -> bool:
        if len(set(args)) != len(args):  # identical args found
            return True

        a: str
        for a in args:
            if a == 'self':
                continue
            if looks_bad(a):
                return True
            if looks_like_qt_type(a):
                return True
        return False

    before_args: str = line[:line.find('(') + 1]
    after_args: str = line[line.rfind(')'):]
    arg: str
    args: list[str] = split_args(line[len(before_args):-len(after_args)])
    if not contains_bad_arg():
        return line

    index: int
    for index in range(len(args)):
        arg = args[index]
        if arg == 'self':
            continue
        if arg.startswith('*'):
            continue
        if looks_bad(arg) or looks_like_qt_type(arg):
            args[index] = arg = f'arg_{index}: {arg}'

        # fix duplicate args
        arg_name: str
        arg_type: str
        if ':' in arg:
            arg_name, arg_type = arg.split(':', maxsplit=1)
        else:
            arg_name = arg
            arg_type = ''
        if arg_name in [arg.split(':', maxsplit=1)[0] for arg in args[:index]]:
            if arg_type:
                args[index] = f'{arg_name}_{arg_type}: {arg_type}'
            else:
                args[index] = f'{arg_name}_'
    return before_args + ', '.join(args) + after_args


def add_def_types(lines: list[str]) -> list[str]:
    """
    Get the typed function definitions from the docstring, if any

    Returns the list of the lines of the function, possibly with the definitions typed.
    If there are multiple typed definitions, each one gets '@overload' decorator.
    The function erases the inline comments after the line starting with 'def ', for they're useless anyway.
    """

    base_indent: int = indent(lines[0])

    decorators_count: int = 0
    while not lines[decorators_count].startswith('def ', base_indent):
        decorators_count += 1

    func: str = '\n'.join(lines)
    if func.count('"""') < 2 and not func_name(func).startswith('_'):
        log.warning(f'Function {func_name(func)} has no docstring')
        if not decorators_count:
            # the item is likely not a function but a signal
            return [
                f'{" " * base_indent}{func_name(func)}: {qtpy.API_NAME}.QtCore.{Signal.__name__} = ...',
                ''
            ]
    docstring_start: int = func.find('"""') + 3
    docstring_end: int = func.find('"""', docstring_start)
    docstring_lines: list[str] = [line.strip() for line in func[docstring_start:docstring_end].splitlines()]
    typed_lines: list[str] = [line for line in docstring_lines if ') -> ' in line]

    if not typed_lines:  # no typed definitions found
        log.warning(f'Function {func_name(func)} has no types in docstring')
        if (len(docstring_lines) == 1
                and docstring_lines[0].startswith(func_name(func) + '(')
                and docstring_lines[0].endswith(')')):
            # likely to be the only definition of a function that returns `None`
            return (lines[:decorators_count]
                    + ([' ' * base_indent + '@staticmethod']
                       * ('(self' not in docstring_lines[0] and
                          not any(line.startswith('@staticmethod', base_indent)
                                  for line in lines[:decorators_count])))
                    + [' ' * base_indent + 'def ' + ensure_arg_names(docstring_lines[0]) + ' -> None:']
                    + lines[(decorators_count + 1):])
        else:
            # return the initial lines
            return lines
    if len(typed_lines) == 1:  # only one typed definition found
        return (lines[:decorators_count]
                + ([' ' * base_indent + '@staticmethod']
                   * ('(self' not in typed_lines[0] and
                      not any(line.startswith('@staticmethod', base_indent)
                              for line in lines[:decorators_count])))
                + [' ' * base_indent + 'def ' + ensure_arg_names(typed_lines[0]) + ':']
                + lines[(decorators_count + 1):])
    else:  # multiple typed definitions found: prepend the definitions with '@overload' decorations
        overload_lines: list[str] = []
        line: str
        for line in typed_lines:
            overload_lines.extend(
                ([' ' * base_indent + '@overload']
                 * (not any(line.startswith('@overload', base_indent)
                            for line in lines[:decorators_count])))
                + lines[:decorators_count]
                + ([' ' * base_indent + '@staticmethod']
                   * ('(self' not in line and
                      not any(line.startswith('@staticmethod', base_indent)
                              for line in lines[:decorators_count])))
                + [' ' * base_indent + 'def ' + ensure_arg_names(line) + ':']
                + lines[(decorators_count + 1):])
        return overload_lines


def add_types(filename: Path) -> None:
    """ Replace the definitions of the functions in the file with the typed ones specified in their docstrings """

    log.name = str(filename)

    lines: list[str] = filename.read_text(encoding='utf-8').splitlines()
    if 'from typing import *' in lines:
        log.info('Already typed')
        return
    new_lines: list[str] = []

    imports_started: bool = False
    def_line: int
    def_indent: int

    index: int = 0
    line: str
    while index < len(lines):
        line = lines[index]

        # find the imports section and prepend it with importing everything from `typing`
        if line.lstrip().startswith('import'):
            if not imports_started:
                new_lines.append('from typing import *')
                new_lines.append('')
            imports_started = True

        # fortunately, the stubs have very consistent formatting, so the function might have decorators or just start
        if line.lstrip().startswith('@') or line.lstrip().startswith('def '):
            def_line = index
            def_indent = indent(line)
            # skip the decorators, if any
            while not lines[index].startswith('def ', def_indent):
                index += 1
            # skip the function heading, the one starting with 'def'
            index += 1
            while index < len(lines) and (not lines[index].strip() or indent(lines[index]) > def_indent):
                index += 1
            new_lines.extend(add_def_types(lines[def_line:index]))
        else:
            # nothing to do with the line, just add it
            new_lines.append(line)
            index += 1

    filename.write_text('\n'.join(new_lines), encoding='utf-8')


def main() -> None:
    """ Walk through the stubs for the currently used Qt for Python bindings and rewrite the function definitions """

    jb_path: Path
    # FIXME: add the cache locations for other platforms (maybe, from PyCharm settings)
    for jb_path in (Path('~/.cache/JetBrains'), Path('~/AppData/Local/JetBrains')):
        jb_path = jb_path.expanduser()
        stubs_path: Path
        for stubs_path in jb_path.glob(f'PyCharm*/python_stubs/*/{qtpy.API_NAME}'):
            stub_path: Path
            for stub_path in stubs_path.glob('*.py'):
                add_types(stub_path)
            for stub_path in stubs_path.glob('*/*.py'):
                add_types(stub_path)


if __name__ == '__main__':
    main()
