"""Docstring.

Details
"""

__all__ = ('InvalidMethodError')


class InvalidMethodError(Exception):
    """A specific exception for invalid method."""

    def __init__(self, message):  # pylint:disable=W0235
        """Init function."""
        super().__init__(message)
