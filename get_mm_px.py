import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import cv2

from load_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    known_width_mm = cfg["vert_known_width"]
    base_image = cv2.imread(cfg["base_image_dir"])
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(base_image)
    plt.title("Click two points to measure distance (press Enter if done)")
    plt.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close()

    (x1, y1), (x2, y2) = pts
    pixel_dist = x2-x1

    mm_per_pixel = known_width_mm / pixel_dist

    plt.axis("off")
    plt.tight_layout()
    plt.close()

    print(mm_per_pixel, "mm/pixel")

if __name__ == "__main__":
    main()
