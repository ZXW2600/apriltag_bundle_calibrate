from apriltag import DetectorOptions, Detector
import cv2

import argparse
import os

# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")

args = ap.parse_args()
image_files = args.image

img = cv2.imread(image_files)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", gray)

detect_option = DetectorOptions(families='tag25h9',
                                         border=1,
                                         nthreads=1,
                                         quad_decimate=4,
                                         quad_blur=0.0,
                                         refine_edges=True,
                                         refine_decode=False,
                                         refine_pose=False,
                                         debug=True,
                                         quad_contours=True)

detector = Detector(detect_option)


results = detector.detect(gray)

print(f"detected {len(results)} tags")
img_show = img.copy()

for result in results:
    i = 0
    for c in result.corners:
        c: tuple
        cv2.circle(img_show, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
        cv2.putText(img_show, f"{i}", (int(c[0]), int(
            c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        i += 1
    print(f"tag id:{result.tag_id}")

cv2.imshow("det", img_show)
cv2.waitKey(0)
