from settings import Settings
from get_webcam_images import get_frame
from transfer_style_functions import transfer_style, save_image

settings = Settings()

while True:
    img_ndarray_frame = get_frame()

    stylized_image = transfer_style(img_ndarray_frame,
                                    settings.style_image_path,
                                    settings.output_image_size,
                                    settings.style_img_size,
                                    settings.hub_module)

    save_image(stylized_image, settings.new_style_image_path)