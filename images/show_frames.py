import cv2

while True:
    frame = cv2.imread('image_new_style.jpg')
    try:
        cv2.imshow('video', frame)
    except:
        continue

    if cv2.waitKey(1) == ord('q'):
        break