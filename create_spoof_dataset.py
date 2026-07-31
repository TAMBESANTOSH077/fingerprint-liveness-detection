import os
import cv2
import random
import numpy as np

folders = [
    ("dataset_split/train/live", "dataset_split/train/spoof"),
    ("dataset_split/val/live", "dataset_split/val/spoof"),
    ("dataset_split/test/live", "dataset_split/test/spoof"),
]

for live_folder, spoof_folder in folders:

    os.makedirs(spoof_folder, exist_ok=True)

    for filename in os.listdir(live_folder):

        if not filename.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(live_folder, filename)

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # -----------------------------
        # Random Rotation
        # -----------------------------
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        # -----------------------------
        # Perspective Warp
        # -----------------------------
        shift = random.randint(3, 10)

        pts1 = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        pts2 = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w-random.randint(0, shift), random.randint(0, shift)],
            [random.randint(0, shift), h-random.randint(0, shift)],
            [w-random.randint(0, shift), h-random.randint(0, shift)]
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (w, h))

        # -----------------------------
        # Blur
        # -----------------------------
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img = cv2.medianBlur(img, 5)

        # -----------------------------
        # Brightness & Contrast
        # -----------------------------
        alpha = random.uniform(0.6, 1.5)
        beta = random.randint(-40, 40)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # -----------------------------
        # Gaussian Noise
        # -----------------------------
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)

        img = img.astype(np.int16)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        # -----------------------------
        # JPEG Compression
        # -----------------------------
        quality = random.randint(25, 70)

        _, enc = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )

        img = cv2.imdecode(enc, 1)

        # -----------------------------
        # Random Shadow
        # -----------------------------
        if random.random() < 0.5:

            shadow = np.zeros_like(img)

            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)

            cv2.line(
                shadow,
                (x1, y1),
                (x2, y2),
                (60, 60, 60),
                random.randint(30, 70)
            )

            img = cv2.addWeighted(
                img,
                1,
                shadow,
                0.25,
                0
            )

        # -----------------------------
        # Save
        # -----------------------------
        save_path = os.path.join(spoof_folder, filename)

        cv2.imwrite(save_path, img)

print("\n================================")
print("Spoof Dataset Created Successfully")
print("================================")