import tensorflow_hub as hub

class Settings():
    def __init__(self):
        self.style_img_size = (256, 256)
        self.output_image_size = 384

        self.hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        self.hub_module = hub.load(self.hub_handle)

        # self.content_image_path = r'C:\Users\Paulo\Desktop\transfer_video_style_programs\images\img.jpg'
        self.style_image_path = r'C:\Users\Paulo\Desktop\transfer_video_style_programs\images\style_img.jpg'
        self.new_style_image_path = r'C:\Users\Paulo\Desktop\transfer_video_style_programs\images\image_new_style.jpg'

        self.num_camera = 1
