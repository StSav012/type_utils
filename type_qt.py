# coding: utf-8

import logging
from pathlib import Path

import qtpy  # to get the currently used Qt bindings
from qtpy.QtCore import Signal

logging.basicConfig()
log: logging.Logger = logging.getLogger()
log.setLevel(logging.INFO)


def indent(line: str) -> int:
    """ Count the spaces at the beginning of the string """
    i: int
    for i in range(len(line)):
        if not line[i].isspace():
            return i
    return 0


def func_name(func: str) -> str:
    """ Find the function name from its definition """
    if 'def ' not in func:
        raise ValueError(f'Not a function:\n{func}')

    def_pos: int = func.find('def ')
    return func[(def_pos + 4):(func.find('(', def_pos + 4))].strip()


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
    typed_lines: list[str] = [line for line in map(str.strip, func[docstring_start:docstring_end].splitlines())
                              if ') -> ' in line]

    if not typed_lines:  # no typed definitions found: return the initial lines
        log.warning(f'Function {func_name(func)} has no types in docstring')
        return lines
    if len(typed_lines) == 1:  # only one typed definition found
        return (lines[:decorators_count]
                + [' ' * base_indent + 'def ' + typed_lines[0] + ':']
                + lines[(decorators_count + 1):])
    else:  # multiple typed definitions found: prepend the definitions with '@overload' decorations
        overload_lines: list[str] = []
        line: str
        for line in typed_lines:
            overload_lines.extend(
                [' ' * base_indent + '@overload']
                + lines[:decorators_count]
                + ([' ' * base_indent + '@staticmethod']
                   * ('(self' not in line and
                      not any(line.startswith('@staticmethod', base_indent)
                              for line in lines[:decorators_count])))
                + [' ' * base_indent + 'def ' + line + ':']
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
