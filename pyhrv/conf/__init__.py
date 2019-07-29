import os
import confuse


CONFIG_DEFAULT_FILENAME = os.path.join(
    os.path.dirname(__file__), 'config_default.yaml'
)

pyhrv_conf = confuse.LazyConfig('pyhrv', __name__)


def _get(key: str):
    view = pyhrv_conf
    for subkey in key.split("."):
        view = view[subkey]
    return view.get()


def get_val(key: str):
    """
    Returns the value of a configuration parameter.
    :param key: The full name of the parameter, e.g. foo.bar.baz
    :return: The value of that parameter.
    """
    return _get(key + '.value')


def get_desc(key: str):
    """
    Returns the description of a configuration parameter.
    :param key: The full name of the parameter, e.g. 'foo.bar.baz'
    :return: The description of that parameter.
    """
    return _get(key + '.description')


def get_override(common_prefix, **param_overrides):
    """
    Overrides default parameter values with given values.
    :param common_prefix: The prefix of all the given parameters,
    e.g. 'foo.bar'.
    :param param_overrides: Key-value pairs of parameter names and their
    override value. If the value is None, then the returned value will be
    fetched from the current configuration.
    :return: A list of values for the given parameters, either taken
    from the overrides or the current configuration.

    Note: This assumes python>=3.6, in which **kwargs order is preserved.
    """
    retvals = []
    for k, v in param_overrides.items():
        retvals.append(v if v else get_val(f'{common_prefix}.{k}'))
    return retvals


def load(filename: str):
    pyhrv_conf.set_file(filename)


def load_default():
    load(CONFIG_DEFAULT_FILENAME)


load_default()
