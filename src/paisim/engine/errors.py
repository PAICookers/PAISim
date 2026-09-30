"""Exceptions for numerical behavior that lacks hardware evidence."""


class UndefinedHardwareBehavior(RuntimeError):
    """The chip behavior is unknown without design confirmation or measurement."""
