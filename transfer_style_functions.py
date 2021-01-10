import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from PIL import Image

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, new_shape, new_shape)
    return image

def config_image(img, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = img.astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def format_image(image_tensor):
    '''format the tensor to can be used in pil module'''
    ndarray = image_tensor.numpy()[0] *255
    ndarray = ndarray.astype(np.uint8)
    return ndarray

def save_image(image, file_name):
    '''save the image'''
    ndarray = format_image(image)
    Image.fromarray(ndarray).save(file_name)

def config_images(content_image, style_image_path, content_img_size, style_img_size):
    content_image = config_image(content_image, content_img_size)
    style_image_path = load_image(style_image_path, style_img_size)
    
    style_image_path = tf.nn.avg_pool(style_image_path, ksize=[3,3], strides=[1,1], padding='SAME')

    return (content_image, style_image_path)

def transfer_style(content_image, style_image_path, output_image_size, style_img_size, hub_module):
    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)

    content_image, style_image = config_images(content_image, style_image_path, content_img_size, style_img_size)
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    image = outputs[0]
    
    return image
