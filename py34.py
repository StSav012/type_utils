# coding: utf-8
import ast
import collections.abc
import logging
import typing
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, cast

TYPING_CLASSES_VERSIONS: dict[str, tuple[int, ...]] = {
    # from https://docs.python.org/3.11/library/typing.html
    'NewType': (3, 5, 2),
    'LiteralString': (3, 11),
    'Never': (3, 11),
    'NoReturn': (3, 5, 4),
    'Self': (3, 11),
    'TypeAlias': (3, 10),
    'Concatenate': (3, 10),
    'Type': (3, 5, 2),
    'Literal': (3, 8),
    'ClassVar': (3, 5, 3),
    'Final': (3, 8),
    'Required': (3, 11),
    'NotRequired': (3, 11),
    'Annotated': (3, 9),
    'TypeGuard': (3, 10),
    'TypeVarTuple': (3, 11),
    'Unpack': (3, 11),
    'ParamSpec': (3, 10),
    'ParamSpecArgs': (3, 10),
    'ParamSpecKwargs': (3, 10),
    'Protocol': (3, 8),
    'runtime_checkable': (3, 8),
    'TypedDict': (3, 8),
    'DefaultDict': (3, 5, 2),
    'OrderedDict': (3, 7, 2),
    'ChainMap': (3, 5, 4),
    'Counter': (3, 5, 4),
    'Deque': (3, 5, 4),
    'Text': (3, 5, 2),
    'Collection': (3, 6, 0),
    'Coroutine': (3, 5, 3),
    'AsyncGenerator': (3, 6, 1),
    'AsyncIterable': (3, 5, 2),
    'AsyncIterator': (3, 5, 2),
    'Awaitable': (3, 5, 2),
    'ContextManager': (3, 5, 4),
    'AsyncContextManager': (3, 5, 4),
    'SupportsIndex': (3, 8),
    'assert_type': (3, 11),
    'assert_never': (3, 11),
    'reveal_type': (3, 11),
    'dataclass_transform': (3, 11),
    'get_overloads': (3, 11),
    'clear_overloads': (3, 11),
    'final': (3, 8),
    'get_origin': (3, 8),
    'is_typeddict': (3, 10),
    'ForwardRef': (3, 7, 4),
    'TYPE_CHECKING': (3, 5, 2),
}

logging.basicConfig()
log: logging.Logger = logging.getLogger(Path(__file__).name)
log.setLevel(logging.INFO)


class HasLinesInfo(Protocol):
    lineno: int
    end_lineno: int


def future_code_warning(item_description: str,
                        since_version: str | tuple[int, ...],
                        item: HasLinesInfo | None = None) -> None:
    """ Log a warning about a feature appeared after Python 3.4 """
    version_str: str = since_version if isinstance(since_version, str) else '.'.join(map(str, since_version))
    if item is None:
        log.warning(f'{item_description} ({version_str}+) found')
        return
    if item.lineno != item.end_lineno:
        log.warning(f'{item_description} ({version_str}+) found at lines {item.lineno} to {item.end_lineno}')
    else:
        log.warning(f'{item_description} ({version_str}+) found at line {item.lineno}')


def clean_annotations_from_code(original_filename: str | PathLike[str],
                                result_filename: str | PathLike[str] | None = None) -> str:
    """
    Clean type annotations from Python code

    :type original_filename: str | PathLike[str]
    :param original_filename: the file to read the code from
    :type result_filename: str | Path | NoneLike[str]
    :param result_filename: if set, save into the file
    :rtype: str
    :returns: the code from the original file, with type annotations removed,
              just like what's written into `result_filename`
    """
    if not isinstance(original_filename, Path):
        original_filename = Path(original_filename)
    if result_filename is not None and not isinstance(result_filename, Path):
        result_filename = Path(result_filename)

    log.name = f'{Path(__file__).name}:{original_filename}'

    m: ast.Module = ast.parse(source=original_filename.read_text(), filename=str(original_filename))

    import_typing_as: list[str] = []
    import_typing_cast_as: list[str] = []
    import_typing_overload_as: list[str] = []
    import_typing_named_tuple_as: list[str] = []
    replace_typing_named_tuple_as: list[str] = []
    import_typing_generic_as: list[str] = []
    from_typing_import: list[str] = []

    import_dataclasses_as: list[str] = []
    import_dataclasses_dataclass_as: list[str] = []

    import_os_as: list[str] = []
    import_os_path_like_as: list[str] = []

    import_math_as: list[str] = []
    import_math_nan_as: list[str] = []
    import_math_inf_as: list[str] = []

    def make_named_tuple_class(item: ast.ClassDef) -> ast.ClassDef:
        if not replace_typing_named_tuple_as:
            raise RuntimeError('Import NamedTuple first')
        name: str = item.name
        statement: ast.stmt
        members: list[ast.AnnAssign] = [statement for statement in item.body if isinstance(statement, ast.AnnAssign)]
        ann_assign: ast.AnnAssign
        ids: list[str] = [ann_assign.target.id for ann_assign in members]
        defaults: list[ast.expr] = [ann_assign.value for ann_assign in members if ann_assign.value is not None]
        named_tuple_call: ast.Call = ast.Call(func=ast.Name(id=replace_typing_named_tuple_as[0]),
                                              args=[ast.Constant(value=name),
                                                    ast.List(elts=[ast.Constant(value=_id) for _id in ids])],
                                              keywords=[])

        item.body = [statement for statement in item.body
                     if not (isinstance(statement, ast.AnnAssign) or isinstance(statement, ast.Assign))]

        if any(v is not None for v in defaults):
            # defaults = [v if v is not None else ast.Constant(value=v) for v in defaults]
            i: int = 0
            while i < len(item.body):
                if isinstance(item.body[i], ast.Expr):
                    i += 1
                else:
                    break

            new_function: ast.FunctionDef = ast.FunctionDef(
                name='__new__',
                decorator_list=[],
                args=ast.arguments(args=[ast.arg(arg='cls'),
                                         *[ast.arg(arg=_id) for _id in ids]],
                                   defaults=defaults,
                                   kw_defaults=[],
                                   kwonlyargs=[],
                                   posonlyargs=[]),
                body=[ast.Return(value=ast.Call(func=ast.Attribute(attr='__new__',
                                                                   value=ast.Call(func=ast.Name(id='super'),
                                                                                  args=[],
                                                                                  keywords=[])),
                                                args=[ast.Name(id='cls'),
                                                      *[ast.Name(id=_id) for _id in ids]],
                                                keywords=[]))],
                lineno=item.lineno
            )
            item.body.insert(i, new_function)

        new_bases: list[ast.expr] = []
        base: ast.expr
        for base in item.bases:
            if isinstance(base, ast.Name) and base.id == 'NamedTuple':
                new_bases.append(named_tuple_call)
            else:
                new_bases.append(base)
        item.bases = new_bases

        if not item.body:
            item.body = [ast.Pass()]

        return item

    def make_base_class_generic(item: ast.ClassDef) -> ast.ClassDef:
        new_bases: list[ast.expr] = []
        base: ast.expr
        for base in item.bases:
            if isinstance(base, ast.Subscript):
                base = cast(ast.expr, ast.Name(id=base.value.id))
            new_bases.append(base)
        item.bases = new_bases
        return item

    def fix_dataclass(item: ast.ClassDef) -> ast.ClassDef:
        decorator: ast.Attribute | ast.Name
        new_decorators: list[ast.Name, ast.Attribute] = []
        for decorator in item.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.value.id in import_dataclasses_as:
                continue
            if isinstance(decorator, ast.Name) and decorator.id in import_dataclasses_dataclass_as:
                continue
            new_decorators.append(decorator)
        item.decorator_list = new_decorators

        statement: ast.stmt
        if any(isinstance(statement, ast.FunctionDef) and statement.name == '__init__' for statement in item.body):
            raise ValueError(f'Class {item.name} already has __init__ function; this cannot be right')
        if any(isinstance(statement, ast.FunctionDef) and statement.name == '__setattr__' for statement in item.body):
            raise ValueError(f'Class {item.name} already has __setattr__ function; this cannot be right')

        _id: str
        members: list[ast.Assign] = [statement for statement in item.body if isinstance(statement, ast.Assign)]
        assign: ast.Assign
        ann_members: list[ast.AnnAssign] = [statement for statement in item.body
                                            if isinstance(statement, ast.AnnAssign)]
        ann_assign: ast.AnnAssign
        ids: list[str] = []
        target: ast.expr
        for assign in members:
            ids.extend(target.id for target in assign.targets)
        ids += [ann_assign.target.id for ann_assign in ann_members]

        defaults: list[ast.expr | None] = []
        target: ast.expr
        for assign in members:
            defaults.extend([assign.value] * len(assign.targets))
        defaults += [ann_assign.value for ann_assign in ann_members]
        i: int
        for i, v in enumerate(defaults):
            if (isinstance(v, ast.Attribute)
                    and v.value.id in import_dataclasses_as):
                v = None
            if (isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and v.func.value.id in import_dataclasses_as):
                v = None
            if v is None:
                defaults[i] = cast(ast.expr, ast.Constant(value=v))

        item.body = [statement for statement in item.body
                     if not (isinstance(statement, ast.AnnAssign) or isinstance(statement, ast.Assign))]

        init_function: ast.FunctionDef = ast.FunctionDef(
            name='__init__',
            decorator_list=[],
            args=ast.arguments(args=[ast.arg(arg='self'),
                                     *[ast.arg(arg=_id) for _id in ids]],
                               defaults=defaults,
                               kw_defaults=[],
                               kwonlyargs=[],
                               posonlyargs=[]),
            body=[ast.Assign(targets=[ast.Attribute(attr=_id,
                                                    value=ast.Name(id='self'))],
                             value=ast.Name(id=_id),
                             lineno=item.lineno) for _id in ids],
            lineno=item.lineno)

        i = 0
        while i < len(item.body):
            if isinstance(item.body[i], ast.Expr):
                i += 1
            else:
                break
        item.body.insert(i, cast(ast.stmt, init_function))
        return item

    def joined_str_to_format_call(operator: ast.JoinedStr) -> ast.Call:
        def duplicate_braces(text: str) -> str:
            return text.replace('{', '{{').replace('}', '}}')

        format_string: str = ''
        args: list[ast.expr] = []
        a: ast.expr
        for a in operator.values:
            if isinstance(a, ast.Constant):
                format_string += duplicate_braces(a.value)
            elif isinstance(a, ast.FormattedValue):
                if a.format_spec is None:
                    if a.conversion is not None and a.conversion > 0:
                        format_string += f'{{{len(args)}!{chr(a.conversion)}}}'
                    else:
                        format_string += f'{{{len(args)}}}'
                elif isinstance(a.format_spec, ast.JoinedStr):
                    if len(a.format_spec.values) == 1:
                        fs: ast.expr
                        for fs in a.format_spec.values:
                            if not isinstance(fs, ast.Constant):
                                format_string += f'{{{len(args)}}}'
                                future_code_warning(f'unknown formatting spec item: {fs}', (4, 0), operator)
                                continue
                            if '_' in fs.value:
                                future_code_warning('`_` f-string formatting option', (3, 6), operator)
                            if 'z' in fs.value:
                                future_code_warning('`z` f-string formatting option', (3, 11), operator)
                            if '_' not in fs.value and 'z' not in fs.value:
                                format_string += f'{{{len(args)}:{fs.value}}}'
                            else:
                                format_string += f'{{{len(args)}}}'
                    else:
                        format_string += f'{{{len(args)}}}'
                        future_code_warning('nested f-string formatting', (3, 6), operator)
                elif (isinstance(a.format_spec, ast.Call)
                      and isinstance(a.format_spec.func, ast.Attribute)
                      and isinstance(a.format_spec.func.value, ast.Constant)):
                    format_string += f'{{{len(args)}:{a.format_spec.func.value.value}}}'
                else:
                    format_string += f'{{{len(args)}}}'
                    future_code_warning(f'unknown formatting spec: {a.format_spec}', (4, 0), operator)
                args.append(a.value)
        func: ast.Attribute = ast.Attribute(value=ast.Constant(value=format_string),
                                            attr='format')
        return ast.Call(func=func, args=args, keywords=[])

    def unpack_starred_tuple(operator: ast.Tuple) -> ast.expr:
        def make_tuple(elements: list[ast.expr]) -> ast.Call:
            return ast.Call(func=ast.Name(id='tuple'),
                            args=[check_expression(cast(ast.Starred, e).value) for e in elements],
                            keywords=[])

        def make_left(elements: list[ast.expr]) -> ast.expr | ast.BinOp | ast.Tuple | ast.Call:
            if not any(isinstance(e, ast.Starred) for e in elements):
                return ast.Tuple(elts=list(check_expression(e) for e in elements))
            if len(elements) == 1 and isinstance(elements[0], ast.Starred):
                if isinstance(cast(ast.Starred, elements[0]).value, ast.List):
                    return ast.Tuple(elts=list(check_expression(e)
                                               for e in cast(ast.List, cast(ast.Starred, elements[0]).value).elts))
                else:
                    return make_tuple([elements[0]])
            if isinstance(elements[-1], ast.Starred):
                if isinstance(cast(ast.Starred, elements[-1]).value, ast.List):
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=ast.Tuple(elts=list(check_expression(e)
                                                               for e in cast(ast.List, cast(ast.Starred,
                                                                                            elements[-1]).value).elts)),
                                     op=ast.Add())
                else:
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=make_tuple([elements[-1]]),
                                     op=ast.Add())
            not_starred: list[ast.expr] = []
            i: int = 0
            for i in range(len(elements) - 1, -1, -1):
                if isinstance(elements[i], ast.Starred):
                    break
                else:
                    not_starred.append(check_expression(elements[i]))
            return ast.BinOp(left=make_left(elements[:(i + 1)]),
                             right=ast.Tuple(elts=not_starred),
                             op=ast.Add())

        return make_left(operator.elts)

    def unpack_starred_list(operator: ast.List) -> ast.expr:
        def make_list(elements: list[ast.expr]) -> ast.Call:
            return ast.Call(func=ast.Name(id='list'),
                            args=[check_expression(cast(ast.Starred, e).value) for e in elements],
                            keywords=[])

        def make_left(elements: list[ast.expr]) -> ast.expr | ast.BinOp | ast.List | ast.Call:
            if not any(isinstance(e, ast.Starred) for e in elements):
                return ast.List(elts=list(check_expression(e) for e in elements))
            if len(elements) == 1 and isinstance(elements[0], ast.Starred):
                if isinstance(cast(ast.Starred, elements[0]).value, ast.List):
                    return ast.List(elts=list(check_expression(e)
                                              for e in cast(ast.List, cast(ast.Starred, elements[0]).value).elts))
                else:
                    return make_list([elements[0]])
            if isinstance(elements[-1], ast.Starred):
                if isinstance(cast(ast.Starred, elements[-1]).value, ast.List):
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=ast.List(elts=list(check_expression(e)
                                                              for e in cast(ast.List, cast(ast.Starred,
                                                                                           elements[-1]).value).elts)),
                                     op=ast.Add())
                else:
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=make_list([elements[-1]]),
                                     op=ast.Add())
            not_starred: list[ast.expr] = []
            i: int = 0
            for i in range(len(elements) - 1, -1, -1):
                if isinstance(elements[i], ast.Starred):
                    break
                else:
                    not_starred.append(check_expression(elements[i]))
            return ast.BinOp(left=make_left(elements[:(i + 1)]),
                             right=ast.List(elts=not_starred),
                             op=ast.Add())

        return make_left(operator.elts)

    def unpack_starred_set(operator: ast.Set) -> ast.expr:
        def make_set(elements: list[ast.expr]) -> ast.Call:
            return ast.Call(func=ast.Name(id='set'),
                            args=[check_expression(cast(ast.Starred, e).value) for e in elements],
                            keywords=[])

        def make_left(elements: list[ast.expr]) -> ast.expr | ast.BinOp | ast.Set | ast.Call:
            if not any(isinstance(e, ast.Starred) for e in elements):
                return ast.Set(elts=list(check_expression(e) for e in elements))
            if len(elements) == 1 and isinstance(elements[0], ast.Starred):
                if isinstance(cast(ast.Starred, elements[0]).value, ast.List):
                    return ast.Set(elts=list(check_expression(e)
                                             for e in cast(ast.List, cast(ast.Starred, elements[0]).value).elts))
                else:
                    return make_set([elements[0]])
            if isinstance(elements[-1], ast.Starred):
                if isinstance(cast(ast.Starred, elements[-1]).value, ast.List):
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=ast.Set(elts=list(check_expression(e)
                                                             for e in cast(ast.List, cast(ast.Starred,
                                                                                          elements[-1]).value).elts)),
                                     op=ast.BitOr())
                else:
                    return ast.BinOp(left=make_left(elements[:-1]),
                                     right=make_set([elements[-1]]),
                                     op=ast.BitOr())
            not_starred: list[ast.expr] = []
            i: int = 0
            for i in range(len(elements) - 1, -1, -1):
                if isinstance(elements[i], ast.Starred):
                    break
                else:
                    not_starred.append(check_expression(elements[i]))
            return ast.BinOp(left=make_left(elements[:(i + 1)]),
                             right=ast.Set(elts=not_starred),
                             op=ast.BitOr())

        return make_left(operator.elts)

    def unpack_starred_dict(operator: ast.Dict) -> ast.expr | ast.Call:
        def get_items(d: ast.Dict) -> ast.Call:
            return ast.Call(func=ast.Attribute(attr='items',
                                               value=d),
                            args=[],
                            keywords=[])

        def make_tuple(keys: list[ast.expr | None], values: list[ast.expr | None]) -> ast.Call:
            d: ast.Dict = ast.Dict(keys=list(check_expression(key) for key in keys),
                                   values=list(check_expression(value) for value in values))
            return ast.Call(func=ast.Name(id='tuple'),
                            args=[get_items(d)],
                            keywords=[])

        def make_left(keys: list[ast.expr | None], values: list[ast.expr | None]) -> ast.expr | ast.BinOp:
            if not any(key is None for key in keys):
                return make_tuple(keys, values)
            if len(keys) == 1 and keys[-1] is None:
                return make_tuple(cast(ast.Dict, values[-1]).keys, cast(ast.Dict, values[-1]).values)
            if keys[-1] is None:
                return ast.BinOp(left=make_left(keys[:-1], values[:-1]),
                                 right=make_tuple(cast(ast.Dict, values[-1]).keys, cast(ast.Dict, values[-1]).values),
                                 op=ast.Add())

            i: int = 0
            for i in range(len(keys) - 1, -1, -1):
                if keys[i] is None:
                    break
            return ast.BinOp(left=make_left(keys[:(i + 1)], values[:(i + 1)]),
                             right=make_tuple(keys[(i + 1):], values[(i + 1):]),
                             op=ast.Add())

        if len(operator.keys) == 1 and operator.keys[0] is None:
            return operator.values[0]

        return ast.Call(func=ast.Name(id='dict'),
                        args=[make_left(operator.keys, operator.values)],
                        keywords=[])

    def check_expression(operator: ast.expr) -> ast.expr:
        field: str
        for field in getattr(operator, '_fields', ()):
            if not hasattr(operator, field):
                continue
            operator_field: Any = getattr(operator, field)
            if isinstance(operator_field, ast.expr):
                setattr(operator, field, check_expression(operator_field))
            elif isinstance(operator_field, list):
                field_item: Any
                if all(isinstance(field_item, ast.expr | None) for field_item in operator_field):
                    setattr(operator, field, list(check_expression(field_item) if field_item is not None else field_item
                                                  for field_item in operator_field))
                elif all(isinstance(field_item, ast.keyword) for field_item in operator_field):
                    pass  # see ast.Call below
                elif all(isinstance(field_item, ast.cmpop) for field_item in operator_field):
                    pass  # no changes since Python 3.4
                elif all(isinstance(field_item, ast.comprehension) for field_item in operator_field):
                    pass  # no changes since Python 3.4
                else:  # non-homogeneous list
                    log.error(f'do not know what to do with list {operator_field} of operator {operator}')
            else:
                pass  # all the troubling expressions are below

        match operator:
            case ast.BinOp():
                operator: ast.BinOp
                if isinstance(operator.op, ast.MatMult):
                    future_code_warning('`@` operator', (3, 5), operator)
                operator.left = check_expression(operator.left)
                operator.right = check_expression(operator.right)
            case ast.NamedExpr():
                operator: ast.NamedExpr
                future_code_warning('`:=` operator', (3, 8), operator)
                operator.value = check_expression(operator.value)
            case ast.Await():
                operator: ast.Await
                future_code_warning('`await` expression', (3, 5), operator)
                operator.value = check_expression(operator.value)
            case ast.Call():
                operator: ast.Call
                if isinstance(operator.func, ast.Name) and operator.func.id in import_typing_cast_as:
                    operator: ast.expr = check_expression(operator.args[1])
                elif (isinstance(operator.func, ast.Attribute)
                      and operator.func.attr in import_typing_cast_as
                      and isinstance(operator.func.value, ast.Name)
                      and operator.func.value.id in import_typing_as):
                    operator: ast.expr = check_expression(operator.args[1])
                else:
                    operator.func = check_expression(operator.func)
                    operator.args = list(map(check_expression, operator.args))
                    keyword: ast.keyword
                    for keyword in operator.keywords:
                        keyword.value = check_expression(keyword.value)
            case ast.JoinedStr():
                operator: ast.JoinedStr
                operator: ast.Call = joined_str_to_format_call(operator)
            case ast.Attribute():
                operator: ast.Attribute
                if isinstance(operator.value, ast.Name):
                    if operator.value.id in import_math_as:
                        if operator.attr in ('inf', 'nan'):
                            future_code_warning(f'`{operator.value.id}.{operator.attr}` expression', (3, 5), operator)
                            operator: ast.Call = ast.Call(func=ast.Name(id='float'),
                                                          args=[ast.Constant(value=operator.attr)],
                                                          keywords=[])
                    elif operator.value.id in import_typing_as:
                        if operator.attr == 'TYPE_CHECKING':
                            future_code_warning(f'`{operator.value.id}.{operator.attr}` expression',
                                                TYPING_CLASSES_VERSIONS.get(operator.attr, (3, 5)), operator)
                            operator: ast.Constant = ast.Constant(value=False)
                        else:
                            if operator.attr in from_typing_import:
                                future_code_warning(f'`{operator.value.id}.{operator.attr}` expression',
                                                    TYPING_CLASSES_VERSIONS.get(operator.attr, (3, 5)), operator)
                            else:
                                operator.value.id = 'collections'
            case ast.Name():
                operator: ast.Name
                if operator.id in import_math_nan_as:
                    future_code_warning(f'`math.nan` expression', (3, 5), operator)
                    operator: ast.Call = ast.Call(func=ast.Name(id='float'),
                                                  args=[ast.Constant(value='nan')],
                                                  keywords=[])
                if operator.id in import_math_inf_as:
                    future_code_warning(f'`math.inf` expression', (3, 5), operator)
                    operator: ast.Call = ast.Call(func=ast.Name(id='float'),
                                                  args=[ast.Constant(value='inf')],
                                                  keywords=[])
            case ast.Tuple():
                operator: ast.Tuple
                if any(isinstance(e, ast.Starred) for e in operator.elts):
                    future_code_warning('starred expression as not an assignment target', (3, 5), operator)
                    operator: ast.expr = unpack_starred_tuple(operator)
            case ast.List():
                operator: ast.List
                if any(isinstance(e, ast.Starred) for e in operator.elts):
                    future_code_warning('starred expression as not an assignment target', (3, 5), operator)
                    operator: ast.expr = unpack_starred_list(operator)
            case ast.Set():
                operator: ast.Set
                if any(isinstance(e, ast.Starred) for e in operator.elts):
                    future_code_warning('starred expression as not an assignment target', (3, 5), operator)
                    operator: ast.expr = unpack_starred_set(operator)
            case ast.Dict():
                operator: ast.Dict
                if any(key is None for key in operator.keys):
                    future_code_warning('starred expression as not an assignment target', (3, 5), operator)
                    operator: ast.Call = unpack_starred_dict(operator)
            case _:
                pass
        return operator

    def clean_body(body: list[ast.stmt] | list[ast.excepthandler]) -> list[ast.stmt] | list[ast.excepthandler]:
        if not body:
            return body

        cleaned_body: list[ast.stmt] | list[ast.excepthandler] = []
        item: ast.stmt | ast.excepthandler
        for item in body:
            match item:
                case ast.ImportFrom():
                    item: ast.ImportFrom
                    name: ast.alias
                    match item.module:
                        case '__future__':
                            no_annotations_names: list[ast.alias] = [name for name in item.names
                                                                     if name.name != 'annotations']
                            if len(item.names) != len(no_annotations_names):
                                future_code_warning('`annotations` import', (3, 7), item)

                            if no_annotations_names:
                                item.names = no_annotations_names
                            else:
                                continue
                        case 'typing':
                            future_code_warning('`typing` module', (3, 5), item)
                            import_typing_cast_as.extend([name.asname or name.name for name in item.names
                                                          if name.name == 'cast'])
                            import_typing_overload_as.extend([name.asname or name.name for name in item.names
                                                              if name.name == 'overload'])
                            import_typing_named_tuple_as.extend([name.asname or name.name for name in item.names
                                                                 if name.name == 'NamedTuple'])
                            import_typing_generic_as.extend([name.asname or name.name for name in item.names
                                                             if name.name == 'Generic'])
                            if any(name.name == 'NamedTuple' for name in item.names):
                                replace_typing_named_tuple_as.append('named''tuple')
                            if any(name.name == 'TYPE_CHECKING' for name in item.names):
                                cleaned_body.append(cast(ast.stmt,
                                                         ast.Assign(targets=[ast.Name(id='TYPE_CHECKING')],
                                                                    value=ast.Constant(value=False),
                                                                    lineno=item.lineno)))
                            collections_replacements: list[ast.alias] = []
                            for name in item.names:
                                if name.name in TYPING_CLASSES_VERSIONS:
                                    future_code_warning(f'`typing.{name.name}` import',
                                                        TYPING_CLASSES_VERSIONS[name.name], item)
                                    from_typing_import.append(name.asname or name.name)
                                else:
                                    if hasattr(collections, name.name) or hasattr(collections.abc, name.name):
                                        collections_replacements.append(name)
                                    elif (hasattr(collections, name.name.casefold())
                                          or hasattr(collections.abc, name.name.casefold())):
                                        name.name = name.name.casefold()
                                        collections_replacements.append(name)
                                    else:
                                        future_code_warning(f'`typing.{name.name}` import', (3, 5), item)
                                        from_typing_import.append(name.asname or name.name)
                            if collections_replacements:
                                item.module = 'collections'  # no `collections.abc` in Python 3.4
                                item.names = collections_replacements
                            else:
                                continue
                        case 'dataclasses':
                            if any(name.name == 'dataclass' for name in item.names):
                                no_dataclass_names: list[ast.alias] = [name for name in item.names
                                                                       if name.name != 'dataclass']
                                import_dataclasses_dataclass_as.extend([name.asname or name.name for name in item.names
                                                                        if name.name == 'dataclass'])
                                future_code_warning('`dataclasses` module', (3, 7), item)
                                if no_dataclass_names:
                                    item.names = no_dataclass_names
                                else:
                                    continue
                        case 'os':
                            if any(name.name == 'PathLike' for name in item.names):
                                no_path_like_names: list[ast.alias] = [name for name in item.names
                                                                       if name.name != 'PathLike']
                                import_os_path_like_as.extend([name.asname or name.name for name in item.names
                                                               if name.name == 'PathLike'])
                                future_code_warning('`os.PathLike` import', (3, 6), item)
                                if no_path_like_names:
                                    item.names = no_path_like_names
                                else:
                                    continue
                        case 'math':
                            no_nan_names: list[ast.alias]
                            if any(name.name == 'nan' for name in item.names):
                                no_nan_names = [name for name in item.names
                                                if name.name != 'nan']
                                import_math_nan_as.extend([name.asname or name.name for name in item.names
                                                           if name.name == 'nan'])
                                future_code_warning('`math.nan` import', (3, 5), item)
                            else:
                                no_nan_names = item.names
                            no_nan_inf_names: list[ast.alias]
                            if any(name.name == 'inf' for name in item.names):
                                no_nan_inf_names = [name for name in no_nan_names
                                                    if name.name != 'inf']
                                import_math_inf_as.extend([name.asname or name.name for name in item.names
                                                           if name.name == 'inf'])
                                future_code_warning('`math.inf` import', (3, 5), item)
                            else:
                                no_nan_inf_names = no_nan_names

                            if no_nan_inf_names:
                                item.names = no_nan_inf_names
                            else:
                                continue
                case ast.Import():
                    item: ast.Import
                    name: ast.alias
                    new_names: list[ast.alias]
                    if any(name.name == 'typing' or name.name.startswith('typing.') for name in item.names):
                        new_names = [name for name in item.names
                                     if name.name != 'typing' and not name.name.startswith('typing.')]
                        future_code_warning('`typing` module', (3, 5), item)
                        import_typing_as.extend([name.asname or name.name for name in item.names
                                                 if name.name == 'typing'])
                        import_typing_cast_as.extend([name.asname or name.name for name in item.names
                                                      if name.name == 'typing.cast'])
                        import_typing_overload_as.extend([name.asname or name.name for name in item.names
                                                          if name.name == 'typing.overload'])
                        import_typing_named_tuple_as.extend([name.asname or name.name for name in item.names
                                                             if name.name == 'typing.NamedTuple'])
                        import_typing_generic_as.extend([name.asname or name.name for name in item.names
                                                         if name.name == 'typing.Generic'])
                        if any(name.name == 'typing.NamedTuple' for name in item.names):
                            replace_typing_named_tuple_as.append('collections.''named''tuple')
                        if new_names:
                            item.names = new_names
                        else:
                            continue
                    if any(name.name.startswith('typing.') for name in item.names):
                        future_code_warning('`typing` module', (3, 5), item)
                        import_typing_as.append('typing')
                        for name in item.names:
                            if not name.name.startswith('typing.'):
                                continue
                            if name.name in TYPING_CLASSES_VERSIONS:
                                future_code_warning(f'`{name.name}` import',
                                                    TYPING_CLASSES_VERSIONS[name.name], item)
                                from_typing_import.append(name.asname or name.name)
                            else:
                                # NB: no `collections.abc` in Python 3.4
                                if hasattr(collections.abc, name.name.removeprefix('typing.')):
                                    name.name = 'collections.' + name.name.removeprefix('typing.')
                                elif hasattr(collections.abc, name.name.removeprefix('typing.').casefold()):
                                    name.name = 'collections.' + name.name.removeprefix('typing.').casefold()
                                else:
                                    future_code_warning(f'`typing.{name.name}` import', (3, 5), item)
                                    from_typing_import.append(name.asname or name.name)

                    if any(name.name == 'dataclasses' or name.name.startswith('dataclasses.') for name in item.names):
                        new_names = [name for name in item.names
                                     if name.name != 'dataclasses' and not name.name.startswith('dataclasses.')]
                        future_code_warning('`dataclasses` module', (3, 7), item)
                        import_dataclasses_as.extend([name.asname or name.name for name in item.names
                                                      if name.name == 'dataclasses'])
                        import_dataclasses_dataclass_as.extend([name.asname or name.name for name in item.names
                                                                if name.name == 'dataclasses.dataclass'])
                        if new_names:
                            item.names = new_names
                        else:
                            continue

                    if any(name.name == 'os' for name in item.names):
                        import_os_as.extend([name.asname or name.name for name in item.names
                                             if name.name == 'os'])
                    if any(name.name == 'os.PathLike' for name in item.names):
                        import_os_as.append('os')
                        import_os_path_like_as.extend([name.asname or name.name for name in item.names
                                                       if name.name == 'os.PathLike'])
                        new_names = [name for name in item.names if name.name != 'os.PathLike']
                        future_code_warning('`os.PathLike` import', (3, 6), item)
                        if new_names:
                            item.names = new_names
                        else:
                            continue

                    if any(name.name == 'math' for name in item.names):
                        import_math_as.extend([name.asname or name.name for name in item.names
                                               if name.name == 'math'])
                    if any(name.name == 'math.inf' for name in item.names):
                        import_math_as.append('math')
                        new_names = [name for name in item.names if name.name != 'math.inf']
                        future_code_warning('`math.inf` import', (3, 5), item)
                        if new_names:
                            item.names = new_names
                        else:
                            continue
                    if any(name.name == 'math.nan' for name in item.names):
                        import_math_as.append('math')
                        new_names = [name for name in item.names if name.name != 'math.nan']
                        future_code_warning('`math.nan` import', (3, 5), item)
                        if new_names:
                            item.names = new_names
                        else:
                            continue
                case ast.AnnAssign():
                    item: ast.AnnAssign
                    if item.value is None:
                        continue
                    item: ast.Assign = ast.Assign(targets=[item.target], value=item.value, lineno=item.lineno)
                case ast.Assign():
                    """
                    special case for `T = typing.TypeVar('T')`, `T = typing.NewType('T')` 
                    and other classes from `typing`
                    """
                    item: ast.Assign
                    if isinstance(item.value, ast.Call):
                        if isinstance(item.value.func, ast.Name):
                            if item.value.func.id in from_typing_import:
                                continue  # something that uses `typing`
                        if isinstance(item.value.func, ast.Attribute):
                            if hasattr(typing, item.value.func.attr) and isinstance(item.value.func.value, ast.Name):
                                if item.value.func.value.id in import_typing_as:
                                    continue  # something that uses `typing`
                case ast.AsyncFunctionDef():
                    item: ast.AsyncFunctionDef
                    future_code_warning('`async`', (3, 5), item)
                    arg: ast.arg
                    for arg in item.args.args:
                        arg.annotation = None
                    if item.args.kwarg is not None:
                        item.args.kwarg.annotation = None
                    item.returns = None
                case ast.FunctionDef():
                    item: ast.FunctionDef
                    base: ast.expr
                    decorator: ast.Attribute | ast.Name
                    if any((isinstance(decorator, ast.Attribute)
                            and decorator.value.id in import_typing_as
                            and decorator.attr == 'overload')
                           for decorator in item.decorator_list):
                        continue
                    if any((isinstance(decorator, ast.Name) and decorator.id in import_typing_overload_as)
                           for decorator in item.decorator_list):
                        continue
                    arg: ast.arg
                    for arg in item.args.args:
                        arg.annotation = None
                    for arg in item.args.kwonlyargs:
                        arg.annotation = None
                    for arg in item.args.posonlyargs:
                        future_code_warning('position-only arg', (3, 8), arg)
                        arg.annotation = None
                    if item.args.posonlyargs:
                        item.args.args = item.args.posonlyargs + item.args.args
                        item.args.posonlyargs = []
                    if item.args.vararg is not None:
                        item.args.vararg.annotation = None
                    if item.args.kwarg is not None:
                        item.args.kwarg.annotation = None
                    item.returns = None
                case ast.Match():
                    item: ast.Match
                    future_code_warning('`match` statement', (3, 10), item)
                    match_case: ast.match_case
                    for match_case in item.cases:
                        match_case.body = clean_body(match_case.body)
                case ast.AsyncFor():
                    item: ast.AsyncFor
                    future_code_warning('`async`', (3, 5), item)
                case ast.AsyncWith() | ast.With():
                    item: ast.AsyncWith | ast.With
                    if isinstance(item, ast.AsyncWith):
                        future_code_warning('`async`', (3, 5), item)
                    with_item: ast.withitem
                    for with_item in item.items:
                        with_item.context_expr = check_expression(with_item.context_expr)
                        if with_item.optional_vars is not None:
                            with_item.optional_vars = check_expression(with_item.optional_vars)
                case ast.ClassDef():
                    item: ast.ClassDef
                    base: ast.expr
                    decorator: ast.Attribute | ast.Name
                    if any(isinstance(base, ast.Name) and base.id in import_typing_named_tuple_as
                           for base in item.bases):
                        item = make_named_tuple_class(item)
                    if any(isinstance(base, ast.Subscript)
                           for base in item.bases):
                        item = make_base_class_generic(item)
                    if any(isinstance(base, ast.Name) and base.id in import_typing_generic_as
                           for base in item.bases):
                        item.bases = [base for base in item.bases
                                      if isinstance(base, ast.Name) and base.id not in import_typing_generic_as]
                    if any((isinstance(decorator, ast.Attribute)
                            and decorator.value.id in import_dataclasses_as
                            and decorator.attr == 'dataclass')
                           for decorator in item.decorator_list):
                        item = fix_dataclass(item)
                    if any((isinstance(decorator, ast.Name) and decorator.id in import_dataclasses_dataclass_as)
                           for decorator in item.decorator_list):
                        item = fix_dataclass(item)
                case _:
                    pass

            field: str
            for field in getattr(item, '_fields', ()):
                if not hasattr(item, field):
                    continue
                item_field: Any = getattr(item, field)
                if isinstance(item_field, ast.expr):
                    setattr(item, field, check_expression(item_field))
                elif isinstance(item_field, list):
                    field_item: Any
                    if all(isinstance(field_item, ast.stmt) for field_item in item_field):
                        setattr(item, field, clean_body(item_field))
                    elif all(isinstance(field_item, ast.excepthandler) for field_item in item_field):
                        setattr(item, field, clean_body(item_field))
                    elif all(isinstance(field_item, ast.expr) for field_item in item_field):
                        setattr(item, field, list(map(check_expression, item_field)))
                    elif all(isinstance(field_item, ast.keyword) for field_item in item_field):
                        setattr(item, field, list(map(check_expression, item_field)))
                    elif all(isinstance(field_item, ast.match_case) for field_item in item_field):
                        pass  # see ast.Match below
                    elif all(isinstance(field_item, ast.withitem) for field_item in item_field):
                        pass  # see ast.AsyncWith & ast.With below
                    elif all(isinstance(field_item, ast.alias) for field_item in item_field):
                        pass  # when importing
                    elif all(isinstance(field_item, str) for field_item in item_field):
                        pass  # `nonlocal` and `global`
                    else:  # non-homogeneous list
                        log.error(f'do not know what to do with {field} = {item_field} of statement {item}')
                else:
                    pass  # all the troubling statements are below

            cleaned_body.append(item)

        if not cleaned_body:
            cleaned_body = [ast.Pass()]
        return cleaned_body

    m.body = clean_body(m.body)

    result: str = ast.unparse(m)

    if result_filename is not None:
        result_filename.write_text(result)

    return result


if __name__ == '__main__':
    import argparse

    def _file_arg(s: str) -> Path:
        p: Path = Path(s)
        if not p.is_file():
            raise ValueError(f'`{s}` is not a file')
        return p

    def _directory_arg(s: str) -> Path:
        p: Path = Path(s)
        if p.exists() and not p.is_dir():
            raise ValueError(f'`{s}` is not a directory')
        return p

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-p', '--print', action='store_true', help='copy the result into stdout')
    parser.add_argument('-o', '--out', type=_directory_arg, default='py34',
                        help='a directory to place the result file(s) into')
    parser.add_argument('file', nargs='+', metavar='FILE', type=_file_arg, help='a Python file to process')
    cl_args: argparse.Namespace = parser.parse_args()

    filename: Path
    for filename in cl_args.file:
        (filename.parent / cl_args.out).mkdir(exist_ok=True, parents=True)
        if cl_args.print:
            print(clean_annotations_from_code(filename, filename.parent / cl_args.out / filename.name))
        else:
            clean_annotations_from_code(filename, filename.parent / cl_args.out / filename.name)
