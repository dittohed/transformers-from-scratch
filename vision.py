import numpy as np
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet

from patchify import patchify


class Patchify():
    """
    Split an input image into square patches, then flatten them
    to patch_size**2 * c (number of channels) long vectors.

    To be applied at the very end of transformations pipeline, for training
    vision transformers.

    Args:
        input_shape (tuple): Tuple containing spatial dimensions of input, i.e.
            in a (height, width) format.
        patch_size (int): Specifies height and width of each patch.
    """

    # TODO: finish up typing
    def __init__(self, input_shape, patch_size):
        super(Patchify, self).__init__()
        self._assert_shapes(input_shape, patch_size)
        self.input_shape = input_shape
        self.patch_size = patch_size
    
    def _assert_shapes(self, input_shape, patch_size):
        for i in (0, 1):
            assert input_shape[i] % patch_size == 0, (
                f'Dim. {i} for input ({input_shape[i]}) not divisible by '
                f'patch size {patch_size}.'
            )

    def __call__(self, img):
        img = np.asarray(img)
        assert self.input_shape == img.shape[:2]

        height, width, c = img.shape
        patches = patchify(img, 
                           patch_size=(self.patch_size, self.patch_size, c),
                           step=self.patch_size)
        patches = patches.reshape((-1, self.patch_size**2 * c))
        
        return patches