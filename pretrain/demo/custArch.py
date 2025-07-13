from mmpretrain.models.backbones import VisionTransformer
from mmengine.registry import MODELS

# Extend the arch_zoo dictionary
VisionTransformer.arch_zoo.update({
    **dict.fromkeys(
        ['sapiens_0.3b', '0.3b'],
        {
            'embed_dims': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_channels': 1024 * 4
        }
    )
})

# Optional: Create a custom registration if needed
@MODELS.register_module()
class SapiensViT(VisionTransformer):
    def __init__(self, arch='sapiens_0.3b', **kwargs):
        super().__init__(arch=arch, **kwargs)