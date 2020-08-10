import re
from pydoc import locate
from typing import List, Union, Any


def locate_class(base: Union[str, List[str]], name: str) -> Any:
    """ load the class given the base directory(ies) and the CamelCase name"""
    if isinstance(base, str):
        base = [base]

    file_and_name = camel_to_snake(name) + '.' + name  # example: my_class_name.MyClassName
    complete_names = [one_base + '.' + file_and_name for one_base in base]  # complete paths with name

    for complete_name in complete_names:
        loaded = locate(complete_name)
        if loaded is not None:
            return loaded

    names = "\n\n\t".join(complete_names)
    raise Exception(f'Could not find the class in any of these locations:\n\n\t{names}')


def camel_to_snake(name: str) -> str:
    # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

