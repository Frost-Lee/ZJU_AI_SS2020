import numpy as np
import skimage
import cv2
from tensorflow import keras

def clean_image_nlmeans(noisy_image):
    sigma_hat = skimage.restoration.estimate_sigma(noisy_image)
    denoised_image = skimage.restoration.denoise_nl_means(
        noisy_image,
        h=sigma_hat,
        sigma=sigma_hat,
        fast_mode=True,
        patch_size=9,
        patch_distance=12,
        multichannel=True
    )
    return denoised_image

def clean_image_median(noisy_image, kernel_size=5):
    return cv2.medianBlur((noisy_image * 255).astype('uint8'), kernel_size).astype(np.float32) / 255.0

def clean_image_tv(noisy_image, weight=0.15, addition=0.2):
    return np.clip(skimage.restoration.denoise_tv_chambolle(
        noisy_image, 
        weight=weight, 
        multichannel=True
    ) + addition, 0.0, 1.0)

def clean_image_resnet(noisy_image, denoising_model):
    return np.clip(denoising_model.predict(np.array([noisy_image]))[0], 0.0, 1.0)
