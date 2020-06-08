import numpy as np
import math
import cv2


class BM3DImage(object):
    def __init__(self, image):
        assert len(image.shape) == 3 and image.shape[-1] == 3
        self.image_array = image.astype(np.float64)
    
    def iterate(self, block_len, stride):
        height, width, _ = self.image_array.shape
        for row_index in range(math.ceil((height - block_len) / stride) + 1):
            for col_index in range(math.ceil((width - block_len) / stride) + 1):
                origin = (min(row_index, height - block_len), min(col_index, width - block_len))
                yield self.indice(origin, block_len)

    def indice(self, origin, block_len):
        height, width, _ = self.image_array.shape
        assert 0 <= origin[0] < height and 0 <= origin[1] < width
        assert 0 <= origin[0] + block_len < height and 0 <= origin[1] + block_len < width
        return BM3DBlock(self, origin, block_len)


class BM3DBlock(object):
    def __init__(self, parent, origin, block_len):
        self.parent = parent
        self.origin = origin
        self.block_len = block_len
        self._transformed_array = None
    
    @property
    def array(self):
        return self.parent.image_array[
            self.origin[0] : self.origin[0] + self.block_len,
            self.origin[1] : self.origin[1] + self.block_len,
            :
        ]
    
    @property
    def transformed_array(self):
        if self._transformed_array is None:
            self._transformed_array = np.stack([self.array[:, :, c] for c in range(self.array.shape[-1])], axis=-1)
        return self._transformed_array


