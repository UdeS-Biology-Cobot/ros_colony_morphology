#!/usr/bin/env python3
# Supress warnings from colony_morphology outputs
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before pyplot is imported

import matplotlib
matplotlib.use("Agg", force=True)


import rospy
import numpy as np
from cv_bridge import CvBridge

from ros_colony_morphology.msg import ColonyMetrics
from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyResponse

# --- use your library ---
from colony_morphology.config import Params, Weights, Thresholds, OutputOpts  # dataclasses & defaults
from colony_morphology.pipeline import run_from_array                         # array-based pipeline
# (run_from_array internally uses: dish.detect_dish_circle, segment.preprocess_and_segment,
#  props.compute_region_properties/compute_nn_metrics, score.score_and_filter, outputs.*)

# image readback for response (optional)
import imageio.v3 as iio

bridge = CvBridge()


def _to_rgb8(img_msg):
    """Return an HxWx3 RGB uint8 array from the incoming ROS Image."""
    # Ask CvBridge for RGB so we avoid cv2 conversions here.
    rgb = bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
    # Ensure contiguous uint8 for downstream code
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def callback_compute_morphology(req):
    rospy.loginfo("get_colony_morphology: received request")

    # 1) Convert incoming image → RGB array
    img_rgb = _to_rgb8(req.image)

    # 2) Map request → pipeline configuration (with safe fallbacks)
    params = Params(
        image_path="",                         # not used in run_from_array
        dish_diameter=float(req.dish_diameter),
        dish_offset=float(getattr(req, "dish_offset", 0.0)),
        cell_min_diameter=int(getattr(req, "cell_min_diameter", 0)),
        cell_max_diameter=int(getattr(req, "cell_max_diameter", 2**31 - 1)),
        max_cells=int(getattr(req, "max_cells", 15)),
        nn_query_size=int(getattr(req, "nn_query_size", 10)),
        scale_for_hough=float(getattr(req, "scale_for_hough", 0.25)),
        close_radius=int(getattr(req, "close_radius", 5)),
        open_radius=int(getattr(req, "open_radius", 1)),
    )

    weights = Weights(
        area=float(getattr(req, "weight_area", 2.0)),
        compactness=float(getattr(req, "weight_compactness", 1.0)),
        eccentricity=float(getattr(req, "weight_eccentricity", 0.25)),
        nn_collision_distance=float(getattr(req, "weight_nn_collision_distance", 2.0)),
        solidity=float(getattr(req, "weight_solidity", 1.0)),
    )

    thresholds = Thresholds(
        min_compactness=float(getattr(req, "cell_min_compactness", 0.6)),
        min_solidity=float(getattr(req, "cell_min_solidity", 0.85)),
        max_eccentricity=float(getattr(req, "cell_max_eccentricity", 0.8)),
        std_weight_diameter=int(getattr(req, "std_weight_diameter", 3)),
    )

    outputs = OutputOpts(
        save_circle_detection=bool(getattr(req, "save_circle_detection", True)),
        save_segmentation_process=bool(getattr(req, "save_segmentation_process", True)),
        save_cell_annotation=bool(getattr(req, "save_cell_annotation", True)),
        save_properties=bool(getattr(req, "save_properties", True)),
        plot_interactive_properties=bool(getattr(req, "plot_interactive_properties", False)),
        outdir=str(getattr(req, "save_path", "result")),
    )

    # 3) Run the array pipeline (returns top indices, props, crop offsets, and outdir if outputs enabled)
    top_idx, props, (x_min, y_min), outdir = run_from_array(
        img_rgb, params, weights, thresholds, outputs, verbose=bool(getattr(req, "verbose", False))
    )

    # 4) Build ROS response (ranked, same fields as your original)
    resp = GetColonyMorphologyResponse()
    for _, idx in enumerate(top_idx, start=1):
        p = props[idx]
        m = ColonyMetrics()
        m.area = float(p["area"])
        m.cell_quality = float(getattr(p, "cell_quality", 0.0))
        # local vs global centroids (global wrt original uncropped image)
        cy, cx = p.centroid
        m.centroid_local = [float(cy), float(cx)]
        m.centroid_global = [float(cy + x_min), float(cx + y_min)]
        m.compactness = float(p["compactness"])
        m.diameter = float(p["equivalent_diameter_area"])
        m.nn_centroid_distance = float(getattr(p, "nn_centroid_distance", 0.0))
        m.nn_collision_distance = float(getattr(p, "nn_collision_distance", 0.0))
        m.discarded = bool(getattr(p, "discarded", False))
        m.discarded_description = str(getattr(p, "discarded_description", ""))
        resp.cell_metrics.append(m)

    # 5) Optionally attach the annotated PNG produced by the pipeline
    if bool(getattr(req, "send_image_result", False)) and outdir:
        try:
            annotated = iio.imread(f"{outdir}/annotated_cell.png")
            if annotated.ndim == 3 and annotated.shape[2] == 4:
                annotated = annotated[:, :, :3]  # drop alpha
            resp.image_result = bridge.cv2_to_imgmsg(annotated, encoding="rgb8")
        except Exception as e:
            rospy.logwarn(f"Could not attach annotated image: {e}")

    rospy.loginfo("get_colony_morphology: sending response")
    return resp


def get_colony_morphology_server():
    rospy.init_node("colony_morphology")
    _srv = rospy.Service("get_colony_morphology", GetColonyMorphology, callback_compute_morphology)
    rospy.loginfo("Ready to compute colony morphology")
    rospy.spin()


if __name__ == "__main__":
    get_colony_morphology_server()
