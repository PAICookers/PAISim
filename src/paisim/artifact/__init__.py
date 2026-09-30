"""Validated artifact readers and bounded tensor/frame mapping helpers.

``read_artifact`` is a pure file reader.  ``Artifact`` converts one declared
thread's integer tensors to WORK frames and decodes returned DATA/VOLTAGE words;
it does not execute a simulator or infer application semantics.
"""

from .adapter import Artifact, ArtifactView
from .io import read_artifact

__all__ = [
    "Artifact",
    "ArtifactView",
    "read_artifact",
]
