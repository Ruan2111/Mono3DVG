import argparse
import os
import sys

import cv2
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIEWER_ROOT = os.environ.get(
    "KITTI_OBJECT_VIS_ROOT",
    os.path.abspath(os.path.join(REPO_ROOT, "..", "kitti_object_vis")),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render KITTI-format predictions to PNG using kitti_object_vis.")
    parser.add_argument("--workspace", required=True, help="Path containing a KITTI split directory such as training/.")
    parser.add_argument("--split", default="training", help="KITTI split name inside the workspace.")
    parser.add_argument("--image-id", type=int, default=None, help="Image index to render. If omitted, infer when only one image exists.")
    parser.add_argument("--viewer-root", default=DEFAULT_VIEWER_ROOT, help="Path to the cloned kitti_object_vis repository.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to <workspace>/rendered/<image_id>.png.")
    return parser.parse_args()


def resolve_image_id(image_dir, image_id):
    if image_id is not None:
        return image_id

    image_files = sorted(
        file_name for file_name in os.listdir(image_dir)
        if file_name.endswith(".png")
    )
    if len(image_files) != 1:
        raise ValueError("image-id is required when the workspace contains multiple images")
    return int(os.path.splitext(image_files[0])[0])


def load_kitti_utils(viewer_root):
    if not os.path.isdir(viewer_root):
        raise FileNotFoundError(f"kitti_object_vis not found: {viewer_root}")
    sys.path.insert(0, viewer_root)
    import kitti_util as utils  # pylint: disable=import-error

    return utils


def draw_object_set(image_2d, image_3d, objects, calib, utils, color, prefix):
    for obj in objects:
        if obj.type == "DontCare":
            continue

        xmin = int(round(obj.xmin))
        ymin = int(round(obj.ymin))
        xmax = int(round(obj.xmax))
        ymax = int(round(obj.ymax))

        cv2.rectangle(image_2d, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image_2d,
            f"{prefix}:{obj.type}",
            (xmin, max(18, ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            continue
        image_3d[:] = utils.draw_projected_box3d(image_3d, box3d_pts_2d, color=color)
        cv2.putText(
            image_3d,
            f"{prefix}:{obj.type}",
            (xmin, max(18, ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def add_panel_title(image, title):
    cv2.putText(
        image,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    utils = load_kitti_utils(args.viewer_root)

    split_dir = os.path.join(args.workspace, args.split)
    image_dir = os.path.join(split_dir, "image_2")
    label_dir = os.path.join(split_dir, "label_2")
    calib_dir = os.path.join(split_dir, "calib")
    pred_dir = os.path.join(split_dir, "pred")

    image_id = resolve_image_id(image_dir, args.image_id)
    image_key = f"{image_id:06d}"

    image_path = os.path.join(image_dir, f"{image_key}.png")
    label_path = os.path.join(label_dir, f"{image_key}.txt")
    calib_path = os.path.join(calib_dir, f"{image_key}.txt")
    pred_path = os.path.join(pred_dir, f"{image_key}.txt")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration not found: {calib_path}")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    calib = utils.Calibration(calib_path)
    gt_objects = utils.read_label(label_path) if os.path.isfile(label_path) else []
    pred_objects = utils.read_label(pred_path)

    image_2d = image.copy()
    image_3d = image.copy()

    draw_object_set(image_2d, image_3d, gt_objects, calib, utils, (0, 255, 0), "GT")
    draw_object_set(image_2d, image_3d, pred_objects, calib, utils, (0, 0, 255), "Pred")

    add_panel_title(image_2d, "2D Boxes")
    add_panel_title(image_3d, "3D Projection")

    separator = np.full((image.shape[0], 24, 3), 24, dtype=np.uint8)
    canvas = np.concatenate([image_2d, separator, image_3d], axis=1)

    if args.output is None:
        render_dir = os.path.join(args.workspace, "rendered")
        os.makedirs(render_dir, exist_ok=True)
        output_path = os.path.join(render_dir, f"{image_key}.png")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not cv2.imwrite(output_path, canvas):
        raise RuntimeError(f"Failed to write render to: {output_path}")

    print(output_path)


if __name__ == "__main__":
    main()
