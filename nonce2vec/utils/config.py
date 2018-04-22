"""Config utilities.

Methods used to manipulate YAML-based configuration files.
"""

import logging
import yaml

from pyfn.exceptions.method import InvalidMethodError
from pyfn.exceptions.parameter import InvalidParameterError
from pyfn.utils.immutables import ImmutableConfig

__all__ = ('load')

logger = logging.getLogger(__name__)


def load(config_file):
    """Load an ImmutableConfig from a YAML configuration file."""
    logger.info('Loading config from file {}'.format(config_file))
    try:
        with open(config_file, 'r') as config_stream:
            config = yaml.safe_load(config_stream)
            return ImmutableConfig(config)
    except (InvalidMethodError, InvalidParameterError):
        raise
