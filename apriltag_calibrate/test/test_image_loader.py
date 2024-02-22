from apriltag_calibrate.utils.ImageLoader import ImageLoader

def test_ImageLoader():
    img=ImageLoader('/home/zxw2600/Workspace_Disk/inertia_toolkit_ws/apriltag_calib_ws/img/camerax1')
    img.load()
    import cv2
    cv2.namedWindow('img', cv2.WINDOW_GUI_NORMAL)
    for i in img.images:
        cv2.imshow('img', i)
        cv2.waitKey(0)

test_ImageLoader()