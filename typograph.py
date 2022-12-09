# coding: utf-8
import re
from typing import *

__all__ = ['name_type']


def name_type(an_object: Any) -> str:
    """ Find the proper type hint for an object """

    none_type_str: Final[str] = 'None'

    def _union(types: Iterable[str]) -> str:
        types = list(types)
        if not types:
            return 'Any'
        if len(types) == 1:
            return types[0]

        types = (sorted(t for t in types if t != none_type_str and t.isalnum())  # first, simple types except None
                 + sorted(t for t in types if not t.isalnum())  # then, complex types
                 + ([none_type_str] * (none_type_str in types)))  # finally, None, if any
        if hasattr(type, '__or__'):  # and `'__ror__'`
            return ' | '.join(types)
        else:
            if none_type_str in types:
                return f'Optional[{_union(t for t in types if t != none_type_str)}]'
            else:
                return 'Union[' + ', '.join(types) + ']'

    def _list(types: Iterable[str]) -> str:
        if hasattr(list, '__class_getitem__'):
            return f'list[{_union(types)}]'
        else:
            return f'List[{_union(types)}]'

    def _set(types: Iterable[str]) -> str:
        if hasattr(set, '__class_getitem__'):
            return f'set[{_union(types)}]'
        else:
            return f'Set[{_union(types)}]'

    def _tuple(types: str) -> str:
        if hasattr(tuple, '__class_getitem__'):
            return f'tuple[{types}]'
        else:
            return f'Tuple[{types}]'

    def _dict(key_types: Iterable[str], value_types: Iterable[str]) -> str:
        if hasattr(dict, '__class_getitem__'):
            return f'dict[{_union(key_types)}, {_union(value_types)}]'
        else:
            return f'Dict[{_union(key_types)}, {_union(value_types)}]'

    def _pattern(types: str) -> str:
        if hasattr(re.Pattern, '__class_getitem__'):
            return f're.Pattern[{types}]'
        else:
            return f'Pattern[{types}]'

    def _match(types: str) -> str:
        if hasattr(re.Match, '__class_getitem__'):
            return f're.Match[{types}]'
        else:
            return f'Match[{types}]'

    def name_list(a: List[Any]) -> str:
        assert isinstance(a, list)
        types: Set[str] = set()
        item: Any
        for item in a:
            types.add(name_type(item))
        return _list(types)

    def name_set(a: Set[Any]) -> str:
        assert isinstance(a, set)
        types: Set[str] = set()
        item: Any
        for item in a:
            types.add(name_type(item))
        return _set(types)

    def name_tuple(a: Tuple[Any, ...]) -> str:
        assert isinstance(a, tuple)
        types: List[str] = []
        item: Any
        for item in a:
            types.append(name_type(item))
        if types:
            return _tuple(', '.join(types))
        else:
            return _tuple('Any')

    def name_dict(a: Dict[Any, Any]) -> str:
        assert isinstance(a, dict)
        key_types: Set[str] = set()
        value_types: Set[str] = set()
        key: Any
        value: Any
        for key, value in a.items():
            key_types.add(name_type(key))
            value_types.add(name_type(value))
        return _dict(key_types, value_types)

    if isinstance(an_object, list):
        return name_list(an_object)
    if isinstance(an_object, set):
        return name_set(an_object)
    if isinstance(an_object, tuple):
        return name_tuple(an_object)
    if isinstance(an_object, dict):
        return name_dict(an_object)
    if isinstance(an_object, re.Pattern):
        return _pattern(name_type(an_object.pattern))
    if isinstance(an_object, re.Match):
        return _union((_match(name_type(an_object.string)), none_type_str))
    if an_object is None:
        return none_type_str
    return an_object.__class__.__name__  # got nothing else to do


if __name__ == '__main__':
    print(name_type([1, 2, None]))
    print(name_type((re.compile(''), re.search('', ''))))
    print(name_type((1, 2, [3, {1: 1, 2: 'a', 3: [3]}])))
    print(name_type({1: 1, False: 'a', 3: {1: 1, 2: 'a', 3: [3]}}))
