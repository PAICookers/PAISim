"""Reusable artifact fixtures for adapter and protobuf tests."""

import json

from paisim.artifact.adapter import Artifact


def document():
    return {
        "schemaVersion": 1,
        "configFrames": {"wordOrder": "HIGH_FIRST", "words": [0x01234567, 0x89ABCDEF]},
        "ioMapping": {
            "threads": [
                {
                    "rootCoreOffset": {"y": 2},
                    "runtime": {"decodeMode": "STREAM", "timesteps": 2, "syncSteps": 3},
                    "inputMappings": {
                        "items": [
                            {
                                "name": "x",
                                "shape": {"size": [1, 2]},
                                "bitWidth": 4,
                                "entries": [
                                    {
                                        "elemIdx": 0,
                                        "coreOffset": {"y": 2},
                                        "addrAxon": 4,
                                        "targetLcn": 1,
                                        "tickRelative": 1,
                                        "dtype": "INT4",
                                    },
                                    {
                                        "elemIdx": 1,
                                        "coreOffset": {"y": 3},
                                        "addrAxon": 8,
                                        "targetLcn": 0,
                                        "dtype": "INT4",
                                    },
                                ],
                            }
                        ]
                    },
                    "outputMappings": {
                        "targetLcn": 0,
                        "items": [
                            {
                                "name": "y",
                                "shape": {"size": [1, 2]},
                                "bitWidth": 4,
                                "kind": "DATA",
                                "entries": [
                                    {"elemIdx": 0, "axonBitIdx": 0, "dtype": "INT4"},
                                    {"elemIdx": 1, "axonBitIdx": 1, "dtype": "INT4"},
                                ],
                            }
                        ],
                    },
                }
            ]
        },
    }


def load(tmp_path, doc=None):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(document() if doc is None else doc))
    return Artifact(path)
