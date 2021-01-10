import cv2
from settings import Settings

settings = Settings()

video = cv2.VideoCapture(settings.num_camera)

def inverts_red_and_blue(ndarray):
    for num_line, line in enumerate(ndarray):
        for num_pixel, pixel in enumerate(line):
            ndarray[num_line][num_pixel] = list(reversed(pixel))
    return ndarray

def get_frame():
    conectado, frame = video.read()
    frame = inverts_red_and_blue(frame)
    return frame

# for frame in get_frame():
#     print(frame)
    #Image.fromarray(frame).save(r'C:\Users\Paulo\Desktop\transfer_video_style_programs\images\img.jpg')
    #time.sleep(5)