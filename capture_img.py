import cv2

camera=cv2.VideoCapture(2)
index=0
while True:
    ret,frame=camera.read()
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    elif key==ord('s'):
        cv2.imwrite(f'shot_{index}.png',frame)
        index+=1
        print('Image saved')
camera.release()