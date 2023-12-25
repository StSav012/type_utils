# coding: utf-8
import ast
import builtins
import enum
import importlib
import importlib.util
import inspect
import keyword
import logging
import re
import site
import sys
from importlib.machinery import ModuleSpec
from inspect import FullArgSpec
from itertools import zip_longest
from pathlib import Path
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    ModuleType,
)
from typing import Callable, Iterable, Iterator

from qtpy import API_NAME  # to get the currently used Qt bindings
from qtpy.QtCore import (
    Property,
    QByteArray,
    QMetaMethod,
    QMetaObject,
    QMetaProperty,
    QOperatingSystemVersion,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor

try:
    from qtpy.QtCore import QOperatingSystemVersionBase
except ImportError:
    QOperatingSystemVersionBase = QOperatingSystemVersion
try:
    DebuggerWidget = sys.modules[
        API_NAME
    ].QtScriptTools.QScriptEngineDebugger.DebuggerWidget
except AttributeError:
    DebuggerWidget = None

logging.basicConfig()
log: logging.Logger = logging.getLogger()
log.setLevel(logging.INFO)

BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

empty_line: str = ""


def indent(line: str) -> int:
    """Count the spaces at the beginning of the string"""
    i: int
    for i in range(len(line)):
        if not line[i].isspace():
            return i
    return 0


def func_name(func: str) -> str:
    """Find the function name from its definition"""
    if "def " not in func or "(" not in func:
        raise ValueError(f"Not a function:\n{func}")

    return func.strip().removeprefix("def ").split("(", maxsplit=1)[0].strip()


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
                args_str = ""
                break
            if (
                args_str[i] == ","
                and not in_quotes
                and not in_double_quotes
                and not in_triple_quotes
                and not in_triple_double_quotes
                and parentheses_level == 0
                and brackets_level == 0
                and braces_level == 0
            ):
                args_list.append(args_str[:i].strip())
                args_str = args_str[(i + 1) :].strip()
                break
            if (
                not in_quotes
                and not in_double_quotes
                and not in_triple_quotes
                and not in_triple_double_quotes
            ):
                match args_str[i]:
                    case "(":
                        parentheses_level += 1
                    case ")":
                        parentheses_level -= 1
                    case "[":
                        brackets_level += 1
                    case "]":
                        brackets_level -= 1
                    case "{":
                        braces_level += 1
                    case "}":
                        braces_level -= 1
            if (
                not in_double_quotes
                and not in_triple_double_quotes
                and args_str.startswith("'''", i)
            ):
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
                    case "'":
                        in_quotes = not in_quotes
                    case '"':
                        in_double_quotes = not in_double_quotes

    return args_list


def fix_type(s: str) -> str:
    s = s.removeprefix("const ")
    s = re.sub(r"QFlags<(?P<enum_type>[\w:]+)>", r"\g<enum_type>", s)
    s = re.sub(r"<function <lambda> at 0x\w+>", "Any", s)
    s = s.replace("::", ".")

    match: re.Match[str] | None = re.match(r"(?P<qt_type>Q\w+)<(?P<subtype>\w+)>", s)
    if match is not None:
        s = repr(
            f"{c_type_to_python(match.group('qt_type'))}[{c_type_to_python(match.group('subtype'))}]"
        )

    return s.removesuffix("*")


def ensure_arg_names(cls: str, line: str) -> str:
    """Make the function definition correct"""

    # should we type `self`? (see https://peps.python.org/pep-0673/)

    line = line.removesuffix(" # real signature unknown")

    hardcoded_corrections: dict[str, str | dict[str, str]] = {
        # common functions
        "__new__(*args, **kwargs):": "__new__(cls, *args, **kwargs) -> Self:",
        "__new__(type, *args, **kwargs):": "__new__(cls, *args, **kwargs) -> Self:",
        "__repr__(self):": "__repr__(self) -> str:",
        "__str__(self):": "__str__(self) -> str:",
        "__hash__(self):": "__hash__(self) -> int:",
        "__len__(self):": "__len__(self) -> int:",
        "__getattr__(self, *args, **kwargs):": "__getattr__(self, name: str) -> Any:",
        "__getattr__(self, name):": "__getattr__(self, name: str) -> Any:",
        "__setattr__(self, *args, **kwargs):": "__setattr__(self, name: str, value: Any) -> None:",
        "__setattr__(self, name, value):": "__setattr__(self, name: str, value: Any) -> None:",
        "__delattr__(self, *args, **kwargs):": "__delattr__(self, *args, **kwargs) -> None:",
        "__delattr__(self, name):": "__delattr__(self, name: str) -> None:",
        "raise(self): ...": "raise_(self) -> None: ...",
        # class-specific functions
        "QCoreApplication": {
            # PySide2
            "translate(context: bytes, key: bytes, disambiguation: typing.Optional[bytes] = None, n: int = -1) -> str": "translate(context: str, sourceText: str, disambiguation: typing.Optional[str] = None, n: int = -1) -> str",
            # PyQt5
            "translate(str, str, disambiguation: str = None, n: int = -1) -> str": "translate(context: str, sourceText: str, disambiguation: typing.Optional[str] = None, n: int = -1) -> str",
            # PySide6
            "translate(context: bytes, key: bytes, disambiguation: Optional[bytes] = None, n: int = -1) -> str": "translate(context: str, sourceText: str, disambiguation: Optional[str] = None, n: int = -1) -> str",
        },
        "QSettings": {
            # PySide2
            "value(self, arg__1: str, defaultValue: typing.Optional[Any] = None, type: typing.Optional[object] = None) -> object": "value(self, key: str, defaultValue: typing.Optional[_T] = None, type: typing.Optional[Type[_T]] = None) -> _T",
            # PyQt5
            "value(self, str, defaultValue: Any = None, type: type = None) -> object": "value(self, key: str, defaultValue: typing.Optional[_T] = None, type: typing.Optional[Type[_T]] = None) -> _T",
            # PySide6
            "value(self, arg__1: str, defaultValue: Optional[Any] = None, type: Optional[object] = None) -> object": "value(self, key: str, defaultValue: Optional[_T] = None, type: Optional[Type[_T]] = None) -> _T",
            # PyQt6
            "value(self, key: Union[QByteArray, str], defaultValue: Any = None, type: type = None) -> object": "value(self, key: str, defaultValue: Optional[_T] = None, type: Optional[Type[_T]] = None) -> _T",
        },
        "QLayout": {
            # PySide6
            "getContentsMargins(self) -> object": "getContentsMargins(self) -> Tuple[int, int, int, int]",
        },
        "QListWidgetItem": {
            # PySide6
            "data(self, role: int) -> Any": "data(self, role: PySide6.QtCore.Qt.ItemDataRole) -> Any",
            "setData(self, role: int, value: Any) -> None": "setData(self, role: PySide6.QtCore.Qt.ItemDataRole, value: Any) -> None",
        },
        "QMetaProperty": {
            # PySide6
            "typeName(self) -> bytes": "typeName(self) -> str",
        },
        "QNetworkAccessManager": {
            # PySide6
            "setTransferTimeout(self, timeout: int = <enum 'TransferTimeoutConstant'>) -> None": "setTransferTimeout(self, timeout: int = PySide6.QtNetwork.QNetworkRequest.TransferTimeoutConstant.DefaultTransferTimeoutConstant) -> None",
        },
        "QValidator": {
            # PySide6
            "validate(self, arg__1: str, arg__2: int) -> object": "validate(self, text: str, pos: int) -> PySide6.QtGui.QValidator.State",
        },
    }
    if cls in hardcoded_corrections and line in hardcoded_corrections[cls]:
        return hardcoded_corrections[cls][line]
    if line in hardcoded_corrections:
        return hardcoded_corrections[line]

    def looks_like_python_type(a: str) -> bool:
        if ":" in a:
            return False
        if "=" in a:
            a = a[: a.find("=")].strip()
        return not a.isidentifier() or keyword.iskeyword(a) or a in BUILTIN_NAMES

    def looks_like_qt_type(a: str) -> bool:
        if not a.startswith("Q"):
            return False
        return all(map(str.isidentifier, a.split(".")))

    def contains_bad_value(a: str) -> bool:
        if "=" not in a:
            return False
        if a.count("=") > 1:
            return True
        v: str = a[a.find("=") + 1 :].strip()
        try:
            ast.parse(v)
        except SyntaxError:
            return True
        else:
            return False

    def fix_bad_value(a: str) -> str:
        if "=" not in a:
            return a
        n: str
        v: str
        n, _, v = a.partition("=")
        n, v = n.strip(), v.strip()
        match: re.Match[str] | None

        match = re.match(r"<(?P<enum_member>\w+\.\w+): .+>", v)
        if match is not None:
            return " = ".join((n, cls + "." + match.group("enum_member")))

        match = re.match(r"Default\((?P<qt_type>Q\w+)<\s?([\w.]+(,\s?)?)+\s?>\)", v)
        if match is not None:
            return " = ".join((n, c_type_to_python(match.group("qt_type")) + "()"))

        match = re.match(r"<class '(?P<type>\w+)'>", v)
        if match is not None:
            return " = ".join((n, c_type_to_python(match.group("type")) + "()"))

        raise ValueError(f"Unknown argument: {a!r}")

    def fix_ret_type() -> str:
        if "->" not in after_args:
            return after_args
        ret_type: str
        res: str
        ret_type, _, res = fix_type(after_args.partition("->")[-1]).rpartition(":")
        if not ret_type and res:
            # no ‘:’ in the string
            ret_type = fix_type(res)
            return f") ->{ret_type}"
        else:
            ret_type = fix_type(ret_type.strip())
            return f") -> {ret_type}:{res}"

    before_args: str = line[: line.find("(") + 1]
    after_args: str = line[line.rfind(")") :]
    arg: str
    args: list[str] = split_args(line[len(before_args) : -len(after_args)])

    index: int
    for index in range(len(args)):
        arg = args[index]
        if index == 0 and arg == "self":
            continue
        if arg.startswith("*"):
            continue
        args[index] = arg = fix_type(arg)
        if looks_like_python_type(arg) or looks_like_qt_type(arg):
            args[index] = arg = f"arg_{index}: {arg}"
        if contains_bad_value(arg):
            args[index] = arg = fix_bad_value(arg)

        # fix duplicate args
        arg_name: str
        arg_type: str
        if ":" in arg:
            arg_name, arg_type = arg.split(":", maxsplit=1)
            arg_type = fix_type(arg_type.strip())
        else:
            arg_name = arg
            arg_type = ""
        if not arg_name or arg_name in [
            arg.split(":", maxsplit=1)[0] for arg in args[:index]
        ]:
            if arg_type:
                args[index] = f"{arg_name}_{arg_type.replace('.', '_')}: {arg_type}"
            else:
                args[index] = f"{arg_name}_"
        else:
            if arg_type:
                args[index] = f"{arg_name}: {arg_type}"
            else:
                args[index] = arg_name
    return before_args + ", ".join(args) + fix_ret_type()


def add_def_types(
    lines: list[str],
    assume_signals: bool = True,
    cls: str = "",
) -> list[str]:
    """
    Get the typed function definitions from the docstring, if any

    Returns the list of the lines of the function, possibly with the definitions typed.
    If there are multiple typed definitions, each one gets '@overload' decorator.
    The function erases the inline comments after the line starting with 'def ', for they're useless anyway.
    """

    def fix_decorator(d: str) -> str:
        if "(" not in d:
            return d
        before_args: str = d.partition("(")[0]
        after_args: str = d.rpartition(")")[-1]
        args: list[str] = split_args(d[len(before_args) + 1 : -len(after_args) - 1])
        return (
            before_args
            + "("
            + ", ".join(fix_type(arg) for arg in args)
            + ")"
            + after_args
        )

    base_indent: int = indent(lines[0])

    decorators_count: int = 0
    while decorators_count < len(lines) and not lines[decorators_count].startswith(
        "def ", base_indent
    ):
        decorators_count += 1
    decorators: list[str] = [fix_decorator(line) for line in lines[:decorators_count]]

    func: str = "\n".join(lines)
    _func_name: str = func_name("\n".join(lines[decorators_count:]))
    if _func_name.startswith("_q_"):
        return []

    if func.count('"""') < 2 and not _func_name.startswith("_"):
        log.warning(f"Function {_func_name} has no docstring")
        if assume_signals and not decorators_count:
            # the item is likely not a function but a signal
            return [
                f'{" " * base_indent}{_func_name}: {Signal.__module__}.{Signal.__name__} = ...',
                "",
            ]

    typed_lines: list[str] = []
    docstring_lines: list[str] = []
    if func.count('"""') >= 2:
        docstring_start: int = func.find('"""') + 3
        docstring_end: int = func.find('"""', docstring_start)
        docstring_lines = [
            line.strip() for line in func[docstring_start:docstring_end].splitlines()
        ]
        typed_lines = [line for line in docstring_lines if ") -> " in line]

    line: str

    if not typed_lines:  # no typed definitions found
        log.warning(f"Function {_func_name} has no types in docstring")
        if (
            len(docstring_lines) == 1
            and docstring_lines[0].startswith(_func_name + "(")
            and docstring_lines[0].endswith(")")
        ):
            # likely to be the only definition of a function that returns `None`
            return (
                decorators
                + (
                    [" " * base_indent + "@staticmethod"]
                    * (
                        "(self" not in docstring_lines[0]
                        and not any(
                            line.startswith("@staticmethod", base_indent)
                            for line in decorators
                        )
                    )
                )
                + [
                    " " * base_indent
                    + "def "
                    + ensure_arg_names(cls, docstring_lines[0])
                    + " -> None:"
                ]
                + lines[(decorators_count + 1) :]
            )
        else:
            return decorators + [
                (
                    (
                        " " * base_indent
                        + "def "
                        + ensure_arg_names(cls, line.strip().removeprefix("def "))
                    )
                    if line.strip().startswith("def ")
                    else line
                )
                for line in lines[decorators_count:]
            ]
    if len(typed_lines) == 1:  # only one typed definition found
        line = typed_lines[0].removeprefix("def ")
        return (
            decorators
            + (
                [" " * base_indent + "@staticmethod"]
                * (
                    "(self" not in typed_lines[0]
                    and not any(
                        line.startswith("@staticmethod", base_indent)
                        for line in decorators
                    )
                )
            )
            + [" " * base_indent + "def " + ensure_arg_names(cls, line) + ":"]
            + lines[(decorators_count + 1) :]
        )
    else:  # multiple typed definitions found: prepend the definitions with '@overload' decorations
        overload_lines: list[str] = []
        for line in typed_lines:
            line = line.removeprefix("def ")
            if not any(
                line.startswith("@overload", base_indent) for line in decorators
            ):
                overload_lines.append(" " * base_indent + "@overload")
            overload_lines.extend(decorators)
            if "(self" not in line and not any(
                line.startswith("@staticmethod", base_indent) for line in decorators
            ):
                overload_lines.append(" " * base_indent + "@staticmethod")
            overload_lines.extend(
                [" " * base_indent + "def " + ensure_arg_names(cls, line) + ":"]
            )
            overload_lines.extend(lines[(decorators_count + 1) :])
        return overload_lines


def _add_types(lines: list[str], assume_signals: bool = True) -> Iterable[str]:
    """Replace the definitions of the functions in the lines with the typed ones specified in their docstrings"""

    imports_started: bool = False
    def_line: int
    def_indent: int

    cls: str = ""

    index: int = 0
    line: str
    while index < len(lines):
        line = lines[index]

        # find the imports section and prepend it with importing everything from `typing`
        if line.lstrip().startswith("import"):
            if not imports_started:
                if "from typing import *" not in lines[:index]:
                    yield "from typing import *"
                    yield "_T = TypeVar('_T')"
                    yield empty_line
            imports_started = True

        if line.lstrip().startswith("class "):
            cls = (
                line.lstrip()
                .removeprefix("class ")
                .split("(", maxsplit=1)[0]
                .rstrip(":")
            )

        # fortunately, the stubs have very consistent formatting, so the function might have decorators or just start
        if line.lstrip().startswith("@") or line.lstrip().startswith("def "):
            def_line = index
            def_indent = indent(line)
            # skip the decorators, if any
            while index < len(lines) and not lines[index].startswith(
                "def ", def_indent
            ):
                index += 1
            # skip the function heading, the one starting with 'def'
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or indent(lines[index]) > def_indent
            ):
                index += 1
            yield from add_def_types(
                lines[def_line:index],
                assume_signals=assume_signals,
                cls=cls,
            )
        else:
            # nothing to do with the line, just add it
            yield line
            index += 1


def add_types(filename: Path, force: bool = False) -> None:
    """Replace the definitions of the functions in the file with the typed ones specified in their docstrings"""

    log.name = str(filename)

    lines: list[str] = filename.read_text(encoding="utf-8").splitlines()
    if not force and "from typing import *" in lines:
        log.info("Already typed")
        return

    filename.write_text("\n".join(_add_types(lines)), encoding="utf-8")


def import_module(
    file_path: Path,
    module_name: str | None = None,
) -> tuple[ModuleType, str]:
    if not module_name:
        module_name = file_path.stem
    module: ModuleType
    if file_path.is_absolute():
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        spec: ModuleSpec | None = importlib.util.spec_from_file_location(
            module_name, file_path
        )
        if spec is None:
            raise ImportError(f"Failed to import {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
    else:
        module = importlib.import_module(".".join(file_path.with_suffix("").parts))
    sys.modules[module_name] = module
    # spec.loader.exec_module(module)
    return module, module_name


def c_type_to_python(t: QByteArray | str | bytes) -> str:
    ts: str = (
        t.toStdString()
        if isinstance(t, QByteArray)
        else (t.decode() if isinstance(t, bytes) else t)
    )
    aliases: dict[str, str] = {
        "void*": "Iterable[Any]",
        "double": "float",
        "std::pair": "tuple",
        "QString": "str",
        "QList": "list",
        "QMap": "dict",
    }
    return aliases.get(ts, ts)


def format_docs(doc: str, offset: int = 0) -> Iterator[str]:
    def _o() -> str:
        return " " * offset

    if isinstance(doc, str) and doc:
        doc_lines: list[str] = doc.splitlines()
        if len(doc_lines) == 1:
            if ") -> " in doc and not doc.startswith("def "):
                doc = "def " + doc
            yield _o() + '"""' + doc + '"""'
        else:
            yield _o() + '"""'
            for doc_line in doc_lines:
                if doc_line:
                    if ") -> " in doc_line and not doc_line.startswith("def "):
                        doc_line = "def " + doc_line
                    yield _o() + doc_line
                else:
                    yield empty_line
            yield _o() + '"""'
        yield empty_line


def class_stubs(cls: type, offset: int = 0) -> Iterator[str]:
    def _o() -> str:
        return " " * offset

    bases: str = ", ".join(
        f"{t.__module__}.{t.__name__}"
        for t in (
            cls.__bases__ if hasattr(cls, "__bases__") else cls.__class__.__bases__
        )
        if t is not object
    )
    if bases:
        yield _o() + f"class {cls.__name__}({bases}):"
    else:
        yield _o() + f"class {cls.__name__}:"
    empty_class: bool = True
    offset += 4
    if isinstance(cls.__doc__, str) and cls.__doc__:
        yield from format_docs(cls.__doc__, offset=offset)
        empty_class = False
    if issubclass(cls, enum.Enum):
        for m, v in cls.__members__.items():
            if isinstance(v.value, int) and v.value >= 0:
                yield _o() + f"{m} = 0x{v.value:x}"
            else:
                yield _o() + f"{m} = {v.value!r}"
            empty_class = False
        yield empty_line
    else:
        for mn, m in cls.__dict__.items():
            if mn.startswith("_") and not callable(m):
                continue
            match m:
                case int():
                    m: int
                    if m >= 0:
                        yield (
                            _o()
                            + f"{mn}: ClassVar[{cls.__module__}.{m.__class__.__name__}] = 0x{m:x}"
                        )
                        yield empty_line
                    else:
                        yield (
                            _o()
                            + f"{mn}: ClassVar[{cls.__module__}.{m.__class__.__name__}] = {m!r}"
                        )
                        yield empty_line
                    empty_class = False
                case str():
                    yield (
                        _o()
                        + f"{mn}: ClassVar[{cls.__module__}.{m.__class__.__name__}] = {m!r}"
                    )
                    yield empty_line
                    empty_class = False
                case type():
                    m: type
                    yield from class_stubs(m, offset=offset)
                    empty_class = False
                case BuiltinFunctionType() | FunctionType():
                    yield from function_or_method_stubs(m, offset=offset)
                    empty_class = False
                case staticmethod():
                    m: "staticmethod"
                    if m.__isabstractmethod__:
                        yield _o() + "@abstractmethod"
                    yield _o() + "@staticmethod"
                    yield from function_or_method_stubs(m.__func__, offset=offset)
                    empty_class = False
                case enum.Enum():
                    m: enum.Enum
                    yield (
                        _o()
                        + (
                            f"{mn}: {m.__module__}.{m.__class__.__name__}"
                            f" = {m.__module__}.{m.__class__.__name__}.{m.name}"
                        )
                    )
                    empty_class = False
                case QMetaObject():
                    m: QMetaObject
                    yield from q_meta_object_stubs(m, offset=offset)
                    if m.methodCount() or m.propertyCount():
                        empty_class = False
                case Signal():
                    yield (
                        _o()
                        + f"{mn}: ClassVar[{m.__module__}.{m.__class__.__name__}] = ..."
                        + f"  # {m.__module__}.{m.__class__.__name__}{m!s}"
                    )
                    yield empty_line
                case QOperatingSystemVersion() | QOperatingSystemVersionBase():
                    m: QOperatingSystemVersion | QOperatingSystemVersionBase
                    yield (
                        _o()
                        + (
                            f"{mn}: ClassVar[{m.__module__}.{m.__class__.__name__}]"
                            f" = {m.__module__}.{m.__class__.__name__}({m.type()}, {m.majorVersion()},"
                            f" {m.minorVersion()}, {m.microVersion()})"
                        )
                    )
                    yield empty_line
                    empty_class = False
                case QColor():
                    yield _o() + f"{mn}: {m.__module__}.{m.__class__.__name__} = {m!r}"
                    empty_class = False
                case _:
                    if DebuggerWidget is not None and (
                        m is DebuggerWidget or isinstance(m, DebuggerWidget)
                    ):
                        raise NotImplementedError
                    match m.__class__.__name__:
                        case "wrapper_descriptor" | "method_descriptor" | "getset_descriptor":
                            yield from function_or_method_stubs(m, offset=offset)
                            empty_class = False
                        case "classmethod_descriptor":
                            yield _o() + "@classmethod"
                            yield from function_or_method_stubs(m, offset=offset)
                            empty_class = False
                        case _:
                            if cls is not Qt:
                                raise TypeError(f"Unknown class {cls!r}")
    if empty_class:
        yield _o() + "pass"
    yield empty_line


def func_args_str(func: Callable) -> str:
    try:
        spec: FullArgSpec = inspect.getfullargspec(func)
    except TypeError:
        return "(*args, **kwargs)"

    def annotated(arg_name: str) -> str:
        if arg_name not in spec.annotations:
            return arg_name
        t: type = spec.annotations[arg_name]
        if t.__name__ in BUILTIN_NAMES or t.__name__ in locals():
            return f"{arg_name}: {t.__name__}"
        else:
            return f"{arg_name}: '{t.__name__}'"

    items: list[str] = [
        (
            (f"{annotated(i)} = {d}" if i in spec.annotations else f"{i}={d}")
            if d is not None
            else annotated(i)
        )
        for i, d in zip_longest(spec.args[::-1], spec.defaults or [])
    ][::-1]
    if spec.varargs:
        items.append(f"*{spec.varargs}")
    elif spec.kwonlyargs:
        items.append("*")
    items.extend(
        [
            (
                (f"{annotated(i)} = {d}" if i in spec.annotations else f"{i}={d}")
                if d is not None
                else annotated(i)
            )
            for i, d in zip_longest(spec.kwonlyargs[::-1], spec.kwonlydefaults or [])
        ][::-1]
    )
    if spec.varkw:
        items.append(f"**{spec.varkw}")
    ret: str = ""
    if ret_type := spec.annotations.get("return", None):
        ret = " -> " + str(ret_type).removeprefix("typing.")
    return "(" + ", ".join(items) + ")" + ret


def function_or_method_stubs(func: Callable, offset: int = 0) -> Iterator[str]:
    def _o() -> str:
        return " " * offset

    if not func.__name__.startswith("_q_"):
        if isinstance(func.__doc__, str) and func.__doc__:
            yield _o() + f"def {func.__name__}{func_args_str(func)}:"
            offset += 4
            yield from format_docs(func.__doc__, offset=offset)
        else:
            yield _o() + f"def {func.__name__}{func_args_str(func)}: ..."
            yield empty_line


def q_meta_object_stubs(m: QMetaObject, offset: int = 0) -> Iterator[str]:
    def _o() -> str:
        return " " * offset

    for _m_i in range(m.methodCount()):
        _m: QMetaMethod = m.method(_m_i)
        if _m.enclosingMetaObject() is not m:
            continue
        if _m.methodType() == QMetaMethod.MethodType.Signal:
            # signals are skipped here to be defined another way
            pass
        elif _m.methodType() != QMetaMethod.MethodType.Slot:
            yield (
                _o()
                + f"{_m.name().toStdString()}: {_m.methodType().name} = ..."
                + f"  # {_m.methodSignature().toStdString()}"
            )
            yield empty_line
        else:
            slot_types: Iterator[str] = map(c_type_to_python, _m.parameterTypes())
            yield _o() + f"@{_m.methodType().name}({', '.join(slot_types)})"
            slot_arguments: list[str] = ["self"] + list(
                f"{_pn.toStdString() if _pn is not None else 'arg__' + str(_pi)}: {_pt}"
                for _pi, (_pn, _pt) in enumerate(
                    zip_longest(
                        _m.parameterNames(),
                        map(c_type_to_python, _m.parameterTypes()),
                    ),
                    start=1,
                )
            )
            yield _o() + f"def {_m.name().toStdString()}({', '.join(slot_arguments)}): ..."
            yield empty_line

    for _p_i in range(m.propertyCount()):
        _p: QMetaProperty = m.property(_p_i)
        yield _o() + f"@{m.__module__}.{Property.__name__}"
        yield _o() + f"def {_p.name()}(self) -> {c_type_to_python(_p.typeName())}: ..."
        yield empty_line

        if _p.isWritable():
            yield _o() + f"@{_p.name()}.setter"
            yield (
                _o()
                + f"def {_p.name()}(self, value: {c_type_to_python(_p.typeName())}) -> None: ..."
            )
            yield empty_line


def fix_pyi() -> None:
    for site_path in site.getsitepackages():
        pyi_path: Path = Path(site_path) / API_NAME
        for module_path in pyi_path.glob("*.pyi"):
            log.name = str(module_path)

            try:
                module, module_name = import_module(module_path.relative_to(site_path))
            except ImportError as ex:
                print(str(ex), file=sys.stderr)
                continue

            new_lines: list[str] = [
                "# coding: utf-8",
                f"import {module.__name__}",
                empty_line,
            ]
            for n, o in module.__dict__.items():
                if hasattr(o, "__module__"):
                    if not o.__module__.startswith(API_NAME):
                        continue
                    if not o.__module__.endswith(module_name):
                        # imported for compatibility
                        continue
                    match o:
                        case type():
                            new_lines.extend(class_stubs(o))
                        case BuiltinFunctionType() | BuiltinMethodType():
                            new_lines.extend(function_or_method_stubs(o))
                        case _:
                            raise TypeError(f"{type(o) = !r}")
                elif not n.startswith("_"):
                    new_lines.append(f"{n}: {o.__class__.__name__} = {o}")
            with open(module_path, "wt") as f_out:
                f_out.write(
                    "\n".join(
                        _add_types(
                            new_lines,
                            assume_signals=False,
                        )
                    )
                )


def fix_jb_stubs() -> None:
    """Walk through the stubs for the currently used Qt for Python bindings and rewrite the function definitions"""

    force: bool = "-f" in sys.argv or "--force" in sys.argv

    jb_path: Path
    # FIXME: add the cache locations for other platforms (maybe, from PyCharm settings)
    for jb_path in (Path("~/.cache/JetBrains"), Path("~/AppData/Local/JetBrains")):
        jb_path = jb_path.expanduser()
        stubs_path: Path
        for stubs_path in jb_path.glob(f"PyCharm*/python_stubs/*/{API_NAME}"):
            stub_path: Path
            for stub_path in stubs_path.glob("*.py"):
                add_types(stub_path, force=force)
            for stub_path in stubs_path.glob("*/*.py"):
                add_types(stub_path, force=force)


if __name__ == "__main__":
    # fix_pyi()
    fix_jb_stubs()
