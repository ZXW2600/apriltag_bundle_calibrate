import os
import cv2
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


class BundleImageLoader:
    def __init__(self, path) -> None:
        self.path = path
        self.images = []

    def load_img(self, folder_path, file_path):
        if not file_path.endswith(".jpg"):
            return False, "", None
        image_path = os.path.join(folder_path, file_path)
        image = cv2.imread(image_path)
        return image is not None, file_path.split("/")[-1], image

    def load_bundle(self, folder, files):
        bundle = []
        print("reading images...")
        with ProcessPoolExecutor() as executor:
            images = list(
                tqdm(executor.map(partial(self.load_img, folder), files), total=len(files)))
            for img in images:
                if img[0]:
                    bundle.append((img[1], img[2]))
        return bundle

    def load(self):
        for folder, _, files in os.walk(self.path):
            bundle = self.load_bundle(folder, files)
            self.images.append(bundle)