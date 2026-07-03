"""
Image / data-augmentation operations shared by dataset generation, training and
inference.  This is the single copy of the augmentation suite that used to be
duplicated across ``detection_module`` / ``orientation_module`` /
``identification_module`` in the original code base.
"""

import cv2
import numpy as np

from core.config import (
    WIDTH, HEIGHT, CHANNELS,
    LIGHT_AUGMENTATION_THRESHOLD, NUMBER_NEON_STRIP, NEON_STRIP_WIDTH, NEON_BRIGHTNESS,
    NUMBER_SPOT_STRIP, SPOT_STRIP_RADIUS, SPOT_BRIGHTNESS,
    SATURATION_AUGMENTATION_THRESHOLD, NOISE_AUGMENTATION_THRESHOLD, PEPPER_NOISE_OCCURANCE,
    PERSPECTIVE_DISTORTION_AMPLITUDE,
)


def data_augmentation(image):
    """Apply a random combination of light / saturation / noise effects to an image."""
    light_augmentation = np.random.rand()
    saturation_augmentation = np.random.rand()
    noise_augmentation = np.random.rand()

    # dsize is (width, height); image.shape is (height, width, ...).  Passing the raw
    # shape[:2] would transpose the mask and break broadcasting on non-square images.
    dsize = (image.shape[1], image.shape[0])
    if light_augmentation < LIGHT_AUGMENTATION_THRESHOLD:
        if light_augmentation < (LIGHT_AUGMENTATION_THRESHOLD / 6):
            neon_mask = random_neon_light_effect(NUMBER_NEON_STRIP, NEON_STRIP_WIDTH, dsize=dsize)
            image = modify_brightness(image, neon_mask, NEON_BRIGHTNESS)
        elif light_augmentation > (LIGHT_AUGMENTATION_THRESHOLD * 5 / 6):
            spot_mask = random_spot_light_effect(NUMBER_SPOT_STRIP, SPOT_STRIP_RADIUS, dsize=dsize)
            image = modify_brightness(image, spot_mask, SPOT_BRIGHTNESS)
        else:
            brightness_mask = np.full(tuple(image.shape[:2]), 255)
            brightness_value = np.random.randint(127)
            if np.random.rand() < 0.5:
                brightness_value = -brightness_value
            image = modify_brightness(image, brightness_mask, brightness_value)

    if saturation_augmentation < SATURATION_AUGMENTATION_THRESHOLD:
        saturation_mask = np.full(tuple(image.shape[:2]), 255)
        saturation_value = np.random.randint(256)
        if np.random.rand() < 0.5:
            saturation_value = -saturation_value
        image = modify_saturation(image, saturation_mask, saturation_value)

    if noise_augmentation < NOISE_AUGMENTATION_THRESHOLD:
        if noise_augmentation < (NOISE_AUGMENTATION_THRESHOLD / 2):
            image = add_white_noise(image)
        else:
            image = add_salt_and_pepper_noise(image, PEPPER_NOISE_OCCURANCE)

    return image


def affine_transformation(image, dx, dy, angle, dsize=(WIDTH, HEIGHT)):
    """Translate by (dx, dy) and rotate by ``angle`` degrees."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    rot_mat[0, 2] += dx
    rot_mat[1, 2] += dy
    return cv2.warpAffine(image, rot_mat, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)


def add_white_noise(image, amount=0.33):
    if not (0 <= amount <= 1):
        raise ValueError("Amount must be between 0 and 1")
    img_float = image.astype(np.float32)
    noise = np.random.normal(0, amount * 255, img_float.shape).astype(np.float32)
    return np.clip(img_float + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, amount=0.01):
    noisy_img = image.copy()
    h, w, _ = noisy_img.shape
    r = np.random.default_rng().random((h, w), dtype=np.float32)
    noisy_img[r < amount / 2] = [255, 255, 255]
    noisy_img[r > (1.0 - amount / 2)] = [0, 0, 0]
    return noisy_img


def modify_brightness(image, mask, brightness_value):
    mask_norm = mask.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + mask_norm * brightness_value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def modify_saturation(image, mask, saturation_value):
    mask_norm = mask.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] + mask_norm * saturation_value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_neon_light_effect(number, tickness, dsize=(WIDTH, HEIGHT)):
    h, w = dsize[1], dsize[0]
    neon_mask = np.zeros((2 * h, 2 * w, 1), dtype=np.uint8)
    color = 255
    neon_mask = cv2.rectangle(neon_mask, (0, h - tickness // 2), (2 * w, h + tickness // 2), color, -1)

    intensity_decrease = int(255 / number)
    number -= 1
    start_height1 = end_height_top = h - tickness // 2
    start_height2 = end_height_bottom = h - tickness // 2
    end_height1 = h + tickness // 2
    end_height2 = h + tickness // 2

    for _ in range(number):
        start_height1 -= tickness
        end_height1 -= tickness
        start_height2 += tickness
        end_height2 += tickness
        color -= intensity_decrease
        neon_mask = cv2.rectangle(neon_mask, (0, start_height1), (2 * w, end_height1), color, -1)
        neon_mask = cv2.rectangle(neon_mask, (0, start_height2), (2 * w, end_height2), color, -1)

    dx = np.random.randint(w) - w
    dy = np.random.randint(h) - h
    angle = np.random.randint(180)
    return affine_transformation(neon_mask, dx, dy, angle, dsize=dsize)


def random_spot_light_effect(number, radius, dsize=(WIDTH, HEIGHT)):
    h, w = dsize[1], dsize[0]
    spot_mask = np.zeros((2 * h, 2 * w, 1), dtype=np.uint8)
    intensity_decrease = int(255 / number)
    color = 0
    radius_decrease = radius
    radius = number * radius
    for _ in range(number):
        color += intensity_decrease
        spot_mask = cv2.circle(spot_mask, (h, w), radius, color, -1)
        radius -= radius_decrease
    dx = np.random.randint(w) - w
    dy = np.random.randint(h) - h
    return affine_transformation(spot_mask, dx, dy, 0, dsize=dsize)


def add_perspective_transform(image, data_annotations):
    distortion1 = np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * WIDTH
    distortion2 = np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * HEIGHT
    distortion3 = np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * WIDTH
    distortion4 = np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * HEIGHT
    distortion5 = np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * WIDTH
    distortion6 = np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * HEIGHT
    distortion7 = np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * WIDTH
    distortion8 = np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * HEIGHT

    pts1 = np.float32([[distortion1, distortion2], [distortion3, distortion4],
                       [distortion5, distortion6], [distortion7, distortion8]])
    pts2 = np.float32([[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    for annotation in data_annotations:
        center = np.array([int(annotation[1] + annotation[3] / 2), int(annotation[2] + annotation[4] / 2), 1], dtype=np.float32)
        center_dst = np.dot(M, center)
        center_dst = center_dst / center_dst[2]
        center = center[0:2]
        center_dst = center_dst[0:2]

        RM_temp = cv2.getRotationMatrix2D(center, -annotation[5], 1.0)
        RM = np.zeros((3, 3)); RM[0:2, :] = RM_temp; RM[2, 2] = 1
        reverse_RM_temp = cv2.getRotationMatrix2D(center_dst, annotation[5], 1.0)
        reverse_RM = np.zeros((3, 3)); reverse_RM[0:2, :] = reverse_RM_temp; reverse_RM[2, 2] = 1

        point1 = np.array([annotation[1], annotation[2], 1])
        point_dst1 = np.dot(RM, point1); point_dst1 = point_dst1 / point_dst1[2]
        point_dst1 = np.dot(M, point_dst1); point_dst1 = point_dst1 / point_dst1[2]
        point_dst1 = np.dot(reverse_RM, point_dst1); point_dst1 = point_dst1 / point_dst1[2]

        point2 = np.array([annotation[1] + annotation[3], annotation[2] + annotation[4], 1])
        point_dst2 = np.dot(RM, point2); point_dst2 = point_dst2 / point_dst2[2]
        point_dst2 = np.dot(M, point_dst2); point_dst2 = point_dst2 / point_dst2[2]
        point_dst2 = np.dot(reverse_RM, point_dst2); point_dst2 = point_dst2 / point_dst2[2]

        annotation[1] = int(point_dst1[0])
        annotation[2] = int(point_dst1[1])
        annotation[3] = int(point_dst2[0] - point_dst1[0])
        annotation[4] = int(point_dst2[1] - point_dst1[1])

    image = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))
    return image, data_annotations


def add_red_boxes(image, annotations):
    """Draw rotated red bounding boxes on the image following the annotations."""
    for annotation in annotations:
        rectangle = np.zeros((annotation[4], annotation[3], CHANNELS), dtype=np.uint8)
        rectangle_mask = np.zeros((annotation[4], annotation[3], CHANNELS), dtype=np.uint8)
        end_point = (annotation[3], annotation[4])
        rectangle = cv2.rectangle(rectangle, (0, 0), end_point, (0, 0, 255), 2)
        rectangle_mask = cv2.rectangle(rectangle_mask, (0, 0), end_point, (255, 255, 255), 2)
        rectangle = affine_transformation(rectangle, annotation[1], annotation[2], annotation[5])
        rectangle_mask = affine_transformation(rectangle_mask, annotation[1], annotation[2], annotation[5])

        reverse_rectangle_mask = cv2.bitwise_not(rectangle_mask)
        image = cv2.bitwise_and(image, reverse_rectangle_mask)
        image = cv2.bitwise_or(rectangle, image)
    return image


def transfer_rgb_distribution(image, means, stds):
    """Recolour a BGR image so each channel matches the given (mean, std)."""
    b, g, r = cv2.split(image)
    out = []
    for channel, target_mean, target_std in zip((b, g, r), means, stds):
        src_mean = np.mean(channel)
        src_std = np.std(channel)
        c = channel.astype(np.int64)
        c = (c - src_mean) * target_std / src_std + target_mean
        c[c > 255] = 255
        c[c < 0] = 0
        out.append(c.astype(np.uint8))
    return cv2.merge(tuple(out))


# --- Aliases for the original root naming (so ported call sites keep working) ---
increase_brightness = modify_brightness
increase_saturation = modify_saturation
white_noise = add_white_noise
pepper_noise = add_salt_and_pepper_noise
