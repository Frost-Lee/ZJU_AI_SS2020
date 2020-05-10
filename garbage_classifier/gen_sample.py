import numpy as np
import os
from PIL import Image
import random

file_paths = []
sample_amount = 256
container = np.zeros((sample_amount, 384, 512, 3))
for root, dirs, files in os.walk('/Volumes/ccschunk2/project_chunks/artificial_intelligence/garbage_classifier'):
        for file_name in files:
            if file_name[0] == '.':
                continue
            file_paths.append(os.path.join(root, file_name))
file_paths = random.choices(file_paths, k=sample_amount)
for i in range(sample_amount):
    image = np.array(Image.open(file_paths[i]))
    container[i] = image
np.save('/Users/Frost/Desktop/samples.npy', container)
