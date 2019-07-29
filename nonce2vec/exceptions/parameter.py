"""Docstring.

Details
"""

__all__ = ('InvalidParameterError')


class InvalidParameterError(Exception):
    """A specific exception for invalid parameter."""

    def __init__(self, message):  # pylint:disable=W0235
        """Init function."""
        super().__init__(message)
