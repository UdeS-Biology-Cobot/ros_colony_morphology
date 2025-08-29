#!/usr/bin/env python3
"""
ROS client for GetColonyMorphology.
- Loads an image (default: dataset/original.png)
- Sends it to the service with configurable params
- Prints the returned metrics
- Optionally visualizes the result with --show
"""

import argparse
from pathlib import Path

import cv2 as cv
import rospy
from cv_bridge import CvBridge

from ros_colony_morphology.srv import (
    GetColonyMorphology,
    GetColonyMorphologyRequest,
)

def make_request(img_msg: "sensor_msgs/Image", args: argparse.Namespace) -> GetColonyMorphologyRequest:
    req = GetColonyMorphologyRequest()
    req.image = img_msg

    # knobs
    req.max_cells = args.max_cells
    req.nn_query_size = args.nn_query_size

    # dish
    req.mask_petri_dish = True
    req.dish_diameter = float(args.dish_diameter)
    req.dish_offset   = float(args.dish_offset)

    # weights
    req.weight_area = args.weight_area
    req.weight_compactness = args.weight_compactness
    req.weight_eccentricity = args.weight_eccentricity
    req.weight_nn_collision_distance = args.weight_nn_collision_distance
    req.weight_solidity = args.weight_solidity

    # thresholds
    req.cell_min_diameter = args.cell_min_diameter
    req.cell_max_diameter = args.cell_max_diameter
    req.cell_min_compactness = args.cell_min_compactness
    req.cell_min_solidity = args.cell_min_solidity
    req.cell_max_eccentricity = args.cell_max_eccentricity
    req.std_weight_diameter = args.std_weight_diameter

    # outputs (server/lib will save artifacts to this folder)
    req.send_image_result = args.send_image_result
    req.save_path = str(Path(args.save_path).expanduser().resolve())
    req.save_circle_detection   = args.save_circle_detection
    req.save_segmentation_process = args.save_segmentation_process
    req.save_cell_annotation    = args.save_cell_annotation
    req.save_properties         = args.save_properties
    req.plot_interactive_properties = False  # keep off in service context

    return req


def call_service(req: GetColonyMorphologyRequest):
    rospy.wait_for_service("get_colony_morphology")
    proxy = rospy.ServiceProxy("get_colony_morphology", GetColonyMorphology)
    return proxy(req)


def visualize(original_rgb, response, bridge: CvBridge, title="Best colonies to pick"):
    # import pyplot only if needed (helps in headless environments)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    if response.image_result.data:
        result_img = bridge.imgmsg_to_cv2(response.image_result, desired_encoding="rgb8")
        ax.imshow(result_img)
        local_coords = True
    else:
        ax.imshow(original_rgb)
        local_coords = False

    ax.set_title(title)
    # Draw circles and labels
    for i, metric in enumerate(response.cell_metrics, start=1):
        if local_coords:
            y, x = metric.centroid_local  # stored as [y, x]
        else:
            y, x = metric.centroid_global
        r = metric.diameter / 2.0
        circle = plt.Circle((x, y), radius=r, fc="none", ec="red")
        ax.add_patch(circle)
        ax.annotate(i, xy=(x + r, y - r), color="red")

    ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Call GetColonyMorphology service.")
    here = Path(__file__).parent
    default_img = (here / "../ext/colony-morphology/dataset/intel_rs_d415_1.png").resolve()

    # I/O
    parser.add_argument("--image", type=Path, default=default_img, help="Path to input image")
    parser.add_argument("--show", action="store_true", help="Display annotated result")
    parser.add_argument("--save-path", default="~/colony_results", help="Folder where the server saves artifacts")
    parser.add_argument("--send-image-result", action="store_true", help="Ask server to return annotated PNG")

    # Dish params
    parser.add_argument("--dish-diameter", type=float, default=856.0)
    parser.add_argument("--dish-offset",   type=float, default=66.0)

    # Core knobs
    parser.add_argument("--max-cells", type=int, default=15)
    parser.add_argument("--nn-query-size", type=int, default=10)

    # Weights
    parser.add_argument("--weight-area", type=float, default=2.0)
    parser.add_argument("--weight-compactness", type=float, default=1.0)
    parser.add_argument("--weight-eccentricity", type=float, default=0.25)
    parser.add_argument("--weight-nn-collision-distance", type=float, default=2.0)
    parser.add_argument("--weight-solidity", type=float, default=1.0)

    # Thresholds
    parser.add_argument("--cell-min-diameter", type=int, default=0)
    parser.add_argument("--cell-max-diameter", type=int, default=0)
    parser.add_argument("--cell-min-compactness", type=float, default=0.6)
    parser.add_argument("--cell-min-solidity", type=float, default=0.85)
    parser.add_argument("--cell-max-eccentricity", type=float, default=0.8)
    parser.add_argument("--std-weight-diameter", type=float, default=2.5)

    # Server-side artifact toggles
    parser.add_argument("--save-circle-detection", action="store_true", default=True)
    parser.add_argument("--save-segmentation-process", action="store_true", default=True)
    parser.add_argument("--save-cell-annotation", action="store_true", default=True)
    parser.add_argument("--save-properties", action="store_true", default=True)

    args = parser.parse_args()

    rospy.init_node("get_colony_morphology_client", anonymous=True)
    bridge = CvBridge()

    # Load & convert image
    img_bgr = cv.imread(str(args.image))
    assert img_bgr is not None, f"File not found or unreadable: {args.image}"
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    # Pack as RGB8 for ROS
    img_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")

    # Build & call
    request = make_request(img_msg, args)
    resp = call_service(request)

    # Print results
    if not resp.cell_metrics:
        print("No cells returned.")
    else:
        for i, cell in enumerate(resp.cell_metrics, start=1):
            print(f"### Cell {i} ######################")
            print(f"area               = {cell.area:.1f}")
            print(f"cell_quality       = {cell.cell_quality:.3f}")
            print(f"compactness        = {cell.compactness:.3f}")
            print(f"diameter           = {cell.diameter:.2f}")
            # Remember: centroids are stored as [y, x]
            print(f"centroid_local_y   = {cell.centroid_local[0]:.2f}")
            print(f"centroid_local_x   = {cell.centroid_local[1]:.2f}")
            print(f"centroid_global_y  = {cell.centroid_global[0]:.2f}")
            print(f"centroid_global_x  = {cell.centroid_global[1]:.2f}")
            print(f"nn_centroid_dist   = {cell.nn_centroid_distance:.2f}")
            print(f"nn_collision_dist  = {cell.nn_collision_distance:.2f}")
            print(f"discarded          = {cell.discarded}")
            if cell.discarded and cell.discarded_description:
                print(f"discarded_reason   = {cell.discarded_description.strip()}")
            print()

    # Visualize only if requested
    if args.show and resp.cell_metrics:
        visualize(img_rgb, resp, bridge)

if __name__ == "__main__":
    main()
