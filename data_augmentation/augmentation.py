
"""Image augmentation and generator."""

# Standard imports
from random import randint, randrange, uniform
import logging

# Dependecy imports
import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

__all__ = [
    "RANDOM_NOISE",
    "RANDOM_ROTATIONS",
    "RANDOM_SHIFTS",
    "RANDOM_BRIGHTNESS",
    "RANDOM_ZOOMS",
    "RANDOM_BLUR",
]


class Augmentation(object):
    """Augmentation and generator class."""

    def __init__(self, degree=10, output_size=(28, 28), min_bright=-33, max_bright=33, amount=5):
        """Initialize class variables."""
        self.degree = degree
        self.output_h, self.output_w = output_size
        self.min_bright = min_bright
        self.max_bright = max_bright
        self.amount = amount
        self.zoom_in = 0.5
        self.zoom_out = 0.5
        self.blur_range = None  # Must be tuple min, max --> e.g. 0, 16

        # Generate distribution for blur
        self.distro = None

    def random_padding(self, image, output_size, override_random=None):
        """Add random horizontal/vertical shifts and increases size of image to output_size."""
        h_img, w_img, ch_img = image.shape
        h_output, w_output = output_size

        asser_msg = ("For Random padding input image Hight must be less or equal to "
                     "output_size hight")
        assert h_img <= h_output, asser_msg
        assert_msg = ("For Random padding input image Width must be less or equal to "
                      "output_size width")
        assert w_img <= w_output, assert_msg

        output_image = np.zeros((h_output, w_output, ch_img), dtype=np.float32)

        if override_random is None:
            pad_h_up = randint(0, h_output - h_img)
            pad_w_left = randint(0, w_output - w_img)
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left
        else:
            pad_h_up = override_random[0]
            pad_w_left = override_random[1]
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left

        output_image = np.pad(image, ((pad_h_up, pad_h_down), (pad_w_left, pad_w_right), (0, 0)),
                              'constant', constant_values=0)

        return output_image, (pad_h_up, pad_w_left)

    def random_rotations(self, image1, image2, image3, degree, label=None):
        """Rotate image randomly."""

        (h_img, w_img, ch_img) = image1.shape[:3]
        center = (w_img / 2, h_img / 2)

        # rotation = np.random.uniform(-1, 1, 1)[0] * degree
        rotation = uniform(-degree, degree)
        rot_mtrx = cv2.getRotationMatrix2D(center, rotation, 1.0)

        image1 = cv2.warpAffine(image1, rot_mtrx, (w_img, h_img)).reshape(h_img, w_img, ch_img)
        image2 = cv2.warpAffine(image2, rot_mtrx, (w_img, h_img)).reshape(h_img, w_img, ch_img)
        image3 = cv2.warpAffine(image3, rot_mtrx, (w_img, h_img)).reshape(h_img, w_img, ch_img)

        label = cv2.warpAffine(label, rot_mtrx, (w_img, h_img))

        return image1, image2, image3, label, rotation



    def random_zooms(self, image1, image2, image3, label, zoom_in=0.25, zoom_out=0):
        """Randomly zoom image."""
        output_size_h = image1.shape[0]
        output_size_w = image1.shape[1]

        rand_size = uniform(-1, 1)

        if rand_size < 0:
            max_zoom = output_size_h * zoom_out
            random_size_h = int(output_size_h + max_zoom * rand_size)
        else:
            max_zoom = output_size_h * zoom_in
            random_size_h = int(output_size_h + max_zoom * rand_size)

        random_size_w = output_size_w * random_size_h // output_size_h
        if random_size_w == output_size_w:
            return image1,image2,image3, label, rand_size

        # Image zooming
        image1 = cv2.resize(image1, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)
        image3 = cv2.resize(image3, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)


        if random_size_w < output_size_w:
            image1, _ = self.random_padding(image1, output_size=(output_size_h, output_size_w))
            image2, _ = self.random_padding(image2, output_size=(output_size_h, output_size_w))
            image3, _ = self.random_padding(image3, output_size=(output_size_h, output_size_w))

        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            image1 = image1[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
            image2 = image2[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
            image3 = image3[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]

        else:
            logging.info("Failed random_zooms ? %s", image.shape)

        # Label zooming
        label = cv2.resize(label, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)

        if random_size_w < output_size_w:
            label, _ = self.random_padding(label, output_size=(output_size_h, output_size_w))
        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            label = label[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
        else:
            logging.info("Failed random_zooms ? %s", label.shape)

        return image1, image2, image3, label, rand_size


    def random_flip(self, image1, image2, image3, label, horizontal=False):
        """Apply random flip to single image and label."""

        flip = 1
        rand_float = uniform(0, 1)

        if horizontal:
            # 1 == vertical flip
            # 0 == horizontal flip
            flip = randint(0, 1)

        if rand_float > 0.5:
            image1 = cv2.flip(image1, flip)
            image2 = cv2.flip(image2, flip)
            image3 = cv2.flip(image3, flip)

            label = cv2.flip(label, flip)

        return image1, image2, image3, label



_INIT = Augmentation()
RANDOM_ROTATIONS = _INIT.random_rotations
RANDOM_ZOOMS = _INIT.random_zooms
RANDOM_FLIP = _INIT.random_flip
