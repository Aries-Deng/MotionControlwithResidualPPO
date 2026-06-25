"""Local Unitree G1 helpers layered beside the upstream mjlab package.

Importing this package registers no tasks by itself. The path extension keeps
upstream submodules such as ``mjlab.envs`` importable after this local package
was renamed to ``mjlab``.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
