import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from load_config import load_config

def get_box_from_points(base_image):
    height, width, _ = base_image.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_image)
    ax.set_title("Click top and bottom of the box (press Enter if done)")
    pts = plt.ginput(2, timeout=0)  # waits until two clicks
    plt.close(fig)

    # Extract Y coordinates only
    y_coords = [int(p[1]) for p in pts]
    y0, y1 = min(y_coords), max(y_coords)

    # X bounds default to image edges
    x0, x1 = 0, width - 1

    input_box = np.array([x0, y0, x1, y1])

    return input_box

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               edgecolor='green',
                               facecolor=(0,0,0,0),
                               lw=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load the base image
    base_image = cv2.imread(cfg["base_image_dir"])
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)

    # Show image and let the user click top and bottom Y coords
    input_box = get_box_from_points(base_image)

    # Draw the selected box
    plt.figure(figsize=(10, 10))
    plt.imshow(base_image)
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()

    print("input_box =", input_box.tolist())

if __name__ == "__main__":
    main()