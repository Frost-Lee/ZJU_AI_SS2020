import numpy as np
import imgaug as ia

def add_pepper(image, ratio=0.2):
    assert 0.0 <= ratio <= 1
    image_shape = image.shape
    row_pepper_count = int(image_shape[1] * ratio)
    pepper_mask = np.concatenate([
        np.full((image_shape[0], row_pepper_count), 0), 
        np.full((image_shape[0], image_shape[1] - row_pepper_count), 1)
    ], axis=-1)
    [*map(np.random.shuffle, pepper_mask)]
    if len(image_shape) > 2:
        pepper_mask = np.stack([pepper_mask] * image_shape[-1], axis=-1)
    return image * pepper_mask

def add_gaussian(image, scale=0.2):
    return np.clip(ia.augmenters.AdditiveGaussianNoise(
        scale=scale, 
        per_channel=True
    )(
        images=np.array([image])
    )[0], 0.0, 1.0)

def add_patch(image, count=2):
    return ia.augmenters.Cutout(
        nb_iterations=count,
        fill_mode="constant", 
        cval=(0.0, 1.0),
        fill_per_channel=0.5
    )(
        images=np.array([image])
    )[0]
