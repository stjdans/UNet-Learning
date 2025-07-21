import tensorflow as tf
import cv2
import numpy as np


def load_image(img_path, mask_path, img_size=(256, 256)):
    # 이미지 불러오기
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0


    # 마스크 불러오기
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method='nearest')  # 정수형 보존
    mask = tf.cast(mask, tf.int32)

    return image, mask

def create_dataset(image_paths, mask_paths, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset


def apply_mask_overlay(original, mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # 마스크에 컬러맵 적용 (클래스 값이 0~N 사이의 정수일 때)
    mask_colored = cv2.applyColorMap((mask * (255 // mask.max())).astype(np.uint8), colormap)
    # 이미지 크기 맞추기
    mask_colored = cv2.resize(mask_colored, (original.shape[1], original.shape[0]))
    # 오버레이: alpha는 마스크 투명도
    overlay = cv2.addWeighted(original, 1 - alpha, mask_colored, alpha, 0)
    return overlay