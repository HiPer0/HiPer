from .utils import (
    is_torch_available,
    is_transformers_available,
)


__version__ = "0.9.0"


if is_torch_available() and is_transformers_available():
    from .stable_diffusion import (
        StableDiffusionPipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
