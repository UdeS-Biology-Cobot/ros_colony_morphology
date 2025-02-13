#!/usr/bin/env python3
from ros_colony_morphology.msg import ColonyMetrics
from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyResponse
import rospy

from colony_morphology.plotting import plot_bboxes, plot_region_roperties
from colony_morphology.geometry import *
from colony_morphology.image_transform import *
# from colony_morphology.plotting import plot_bboxes, plot_region_roperties
from colony_morphology.metric import compactness as compute_compactness
from colony_morphology.skimage_util import compactness as compactness_cb
from colony_morphology.skimage_util import nn_centroid_distance as nn_centroid_distance_cb
from colony_morphology.skimage_util import nn_collision_distance as nn_collision_distance_cb
from colony_morphology.skimage_util import cell_quality as cell_quality_cb
from colony_morphology.skimage_util import axes_closness as axes_closness_cb
from colony_morphology.skimage_util import discarded as discarded_cb
from colony_morphology.skimage_util import discarded_description as discarded_description_cb
from colony_morphology.skimage_util import regionprops_to_dict
from colony_morphology.metric import axes_closness as compute_axes_closness

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
from scipy.spatial import cKDTree

import statistics
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage import data, color
import time

import pandas as pd

bridge = CvBridge()

def callback_compute_morphology(req):

    print("===")
    print("Received request")
    print("===")

    response = GetColonyMorphologyResponse()

    img = bridge.imgmsg_to_cv2(req.image, desired_encoding='passthrough')

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Mask image to contain only the petri dish
    # 1a- Resize image to speedup circle detection
    scale, img_resize = resize_image(img_gray, pixel_threshold=480*480)

    # 1b- Canny edge detector
    edges = canny(img_resize, sigma=3, low_threshold=10, high_threshold=50)

    # 1c- Find the most prominent circle
    radius = (req.dish_diameter/2.0)*scale
    low = radius - radius*0.1
    high = radius + radius*0.1
    hough_radii = np.arange(int(low), int(high), 2)
    hough_res = hough_circle(edges, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if(len(radii) == 0):
        print('No circle detected, check the requested dish diameter');
        return response

    # Save circle detection picture
    if req.save_circle_detection:
        # wrt. resized image
        fig, ax = plt.subplots(ncols=1, nrows=1)
        dummy, img_circle_detection = resize_image(img, pixel_threshold=480*480)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(int(center_y),
                                            int(center_x),
                                            int(radius),
                                            shape=img_circle_detection.shape)
            # Draw green perimeter
            img_circle_detection[circy, circx] = (0, 255, 51)

        ax.imshow(img_circle_detection)

        plt.tight_layout()
        plt.savefig(f'{req.save_path}/circle_detection_resize.png')

        # wrt. original image
        fig, ax = plt.subplots(ncols=1, nrows=1)
        img_circle_detection = img.copy()
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(int(center_y/scale),
                                            int(center_x/scale),
                                            int(radius/scale),
                                            shape=img_circle_detection.shape)
            # Draw green perimeter
            img_circle_detection[circy, circx] = (0, 255, 51)

        ax.imshow(img_circle_detection)

        plt.tight_layout()
        plt.savefig(f'{req.save_path}/circle_detection_original.png')

    # 1d- Scale centroid and radius back to original image
    centroid = (cy[0]/scale, cx[0]/scale)
    radius = radii[0]/scale
    radius -= req.dish_offset   # apply offset

    # 1e- Create circular masks
    circular_mask = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius)
    circular_mask_threshold = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius -8)

    # 1f- Mask original image
    idx = (circular_mask== False)
    img_masked = np.copy(img)
    img_masked[idx] = 0; # black

    # 1g- Crop image
    x_min = int(centroid[0]-radius)
    x_max = int(centroid[0]+radius)
    y_min = int(centroid[1]-radius)
    y_max = int(centroid[1]+radius)

    if(x_min < 0):
        x_min = 0
    if(x_max > img_masked.shape[0]):
        x_max = img_masked.shape[0]
    if(y_min < 0):
        y_min = 0
    if(y_max > img_masked.shape[1]):
        y_max = img_masked.shape[1]

    img_cropped = img_masked[x_min:x_max, y_min:y_max]
    img_original_cropped = img[x_min:x_max, y_min:y_max]
    circular_mask_threshold = circular_mask_threshold[x_min:x_max, y_min:y_max]

    # 2- Convert to grayscale
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_RGB2GRAY)

    # 3- Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)

    # 4- Adaptive threshold
    img_threshold = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, 2)

    # 5- Remove contour artifacts generated from adaptive threshold
    img_mask_artifacts = img_threshold.copy()
    idx = (circular_mask_threshold== False)
    img_mask_artifacts[idx] = 0; # black


    # 6- Closing - dilation followed by erosion
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((2,2),np.uint8)
    closing = cv.morphologyEx(img_mask_artifacts, cv.MORPH_CLOSE,kernel, iterations = 1)

    # 7- Opening - erosion followed by dilation
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN,kernel, iterations = 1)

    # Segmentation for separating different objects in an image
    # 8a- Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(opening.shape)
    ndi.distance_transform_edt(opening, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed...')
    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=opening)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]

    # 8b- Watershed
    img_labels = watershed(img_distance, markers, mask=opening, connectivity=1, compactness=0)

    # Generate metrics from labels
    # 9a- Add extra properties, some function must be populated afterwards
    print('Computing region properties...')
    extra_callbacks = (compactness_cb,
                       nn_collision_distance_cb,
                       nn_centroid_distance_cb,
                       cell_quality_cb,
                       discarded_cb,
                       discarded_description_cb,
                       axes_closness_cb)

    # 9b- Measure properties of labelled image regions
    properties = regionprops(img_labels, intensity_image=img_gray, extra_properties=extra_callbacks)
    print(f'Region properties = {len(properties)}')

    # 9c- Compute compactness
    for p in properties:
        # avoid division by zero
        if(p.perimeter == 0.0):
            p.compactness = 0.0
        else:
            p.compactness = compute_compactness(p.area, p.perimeter)

    # 9d- Remove every properties that have a perimeter of 0
    properties[:] = [p for p in properties if p["perimeter"] > 0.0]
    print(f'Region properties, after removing small objects = {len(properties)}')

    # 9e- Compute axes_closness
    for p in properties:
        # avoid division by zero
        if p.axis_major_length == 0.0 or p.axis_minor_length == 0:
            p.axes_closness = 0.0
        else:
            p.axes_closness = compute_axes_closness(p.axis_major_length, p.axis_minor_length)

    # 9f- Find the nearest neighbors with ckDTree
    print('Computing distance to nearest neighboring cells...')

    centroids = [p["centroid"] for p in properties]
    tree = cKDTree(centroids)
    k = req.nn_query_size
    if(k > len(centroids)):
        k = len(centroids)

    for i in range(0, len(centroids)):
        centroid = centroids[i]
        dd, ii = tree.query(centroid, k)

        p = properties[i]
        p.nn_centroid_distance = dd[1]
        radius = p.equivalent_diameter_area/2.0

        # compute collision distance
        prev_nn_diameter = float('-inf')
        prev_collision_distance = float('+inf')

        for index in range(1, len(ii)):
            pnn = properties[ii[index]]
            nn_diameter = pnn.equivalent_diameter_area

            # only compute if the radius of the cell is greater then the previous
            # cell, because the centroid distance will be greater, so a greater
            # radius must be achieved so that the collision distance may shrink
            if  nn_diameter > prev_nn_diameter:
                prev_nn_diameter = nn_diameter
                nn_radius = nn_diameter / 2.0

                collision_distance = dd[index] - (radius + nn_radius)

                if(collision_distance < prev_collision_distance):
                    prev_collision_distance = collision_distance
                    p.nn_collision_distance = collision_distance

    # 9g- Compute cell_quality metric an discard cell's based on user threshold
    max_nn_collision_distance = max(p["nn_collision_distance"] for p in properties if p["compactness"] >= 0.2 and p["nn_collision_distance"] >= 0)
    max_area = max(p["area"] for p in properties if p["compactness"] >= 0.2)

    quality_metrics = np.empty(len(properties), dtype=object)
    for i in range(0, len(properties)):
        p = properties[i]

        # normalize
        n_area = p.area / max_area
        n_nn_collision_distance = p.nn_collision_distance / max_nn_collision_distance

        # clamp compactness to 1
        n_compactness = p.compactness
        if(n_compactness > 1.0):
           n_compactness =1.0

        # invert eccentricity ratio
        n_eccentricity = 1.0 - p.eccentricity

        metrics_used = 0
        if(req.weight_area):
            metrics_used += 1
        if(req.weight_compactness):
            metrics_used += 1
        if(req.weight_eccentricity):
            metrics_used += 1
        if(req.weight_nn_collision_distance):
            metrics_used += 1
        if(req.weight_solidity):
            metrics_used += 1

        if( not metrics_used):
            print("not metrics used, weights are all zeroes")
            return

        # compute quality metric
        cell_quality = (req.weight_area  * n_area +
                        req.weight_compactness  * n_compactness +
                        req.weight_eccentricity * n_eccentricity +
                        req.weight_nn_collision_distance  * n_nn_collision_distance +
                        req.weight_solidity  * p.solidity) / metrics_used

        # discard cells
        if(req.cell_min_diameter and p.equivalent_diameter < req.cell_min_diameter):
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} equivalent diameter is lower then the requested threshold: {p.equivalent_diameter} < {req.cell_min_diameter}\n'
            cell_quality = 0.0
        if(req.cell_max_diameter and p.equivalent_diameter > req.cell_max_diameter):
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} equivalent diameter is higher then the requested threshold: {p.equivalent_diameter} < {req.cell_max_diameter}\n'
            cell_quality = 0.0
        if(req.cell_min_compactness and p.compactness < req.cell_min_compactness):
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} compactness is lower then the requested threshold: {p.compactness} < {req.cell_min_compactness}\n'
            cell_quality = 0.0
        if(req.cell_min_solidity and p.solidity < req.cell_min_solidity):
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} solidity is lower then the requested threshold: {p.solidity} < {req.cell_min_solidity}\n'
            cell_quality = 0.0
        if(req.cell_max_eccentricity and p.eccentricity > req.cell_max_eccentricity):
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} eccentricity is higher then the requested threshold: {p.eccentricity} < {req.cell_max_eccentricity}\n'
            cell_quality = 0.0
        if(p.nn_collision_distance < 0): # in collision
            p.discarded = True
            p.discarded_description += f'Cell #{p.label} is in collision. Distance: {p.nn_collision_distance}\n'
            cell_quality = 0.0


        quality_metrics[i] = (cell_quality, i)
        p.cell_quality = cell_quality


    # remove outliers wrt. diameter of non discarded cell's
    if(req.std_weight_diameter):
        diameter_std = statistics.pstdev(p["equivalent_diameter"] for p in properties if p.cell_quality > 0.0)
        diameter_mean = statistics.mean(p["equivalent_diameter"] for p in properties if p.cell_quality > 0.0)

        for i in range(0, len(properties)):
            p = properties[i]
            if (p.equivalent_diameter < diameter_mean - req.std_weight_diameter* diameter_std):
                p.discarded = True
                p.discarded_description += f'Cell #{p.label} diameter is outside the requested std distribution: {p.equivalent_diameter} < {diameter_mean - req.std_weight_diameter* diameter_std},\nwhere mean = {diameter_mean}, sigma = {req.std_weight_diameter}, std = {diameter_std}\n'
                # print(f'label {p.label} discarded due to being below std')
                p.cell_quality = 0.0
                quality_metrics[i] = (p.cell_quality, i)
            elif(p.equivalent_diameter > diameter_mean + req.std_weight_diameter* diameter_std):
                p.discarded = True
                p.discarded_description += f'Cell #{p.label} diameter is outside the requested std distribution: {p.equivalent_diameter} > {diameter_mean + req.std_weight_diameter* diameter_std},\nwhere mean = {diameter_mean}, sigma = {req.std_weight_diameter}, std = {diameter_std}\n'
                # print(f'label {p.label} discarded due to being below std')
                p.cell_quality = 0.0
                quality_metrics[i] = (p.cell_quality, i)



    # 10- sort by best cell_quality metrics (higher is better)
    # TODO itemgetter might be faster then a lambda
    # https://stackoverflow.com/a/10695158
    reverse_metrics = sorted(quality_metrics, key=lambda x: x[0], reverse=True)

    # 11- Reduce list to requested number of cells
    reverse_metrics_slice = reverse_metrics
    if(req.max_cells and len(reverse_metrics) > req.max_cells):
        reverse_metrics_slice = reverse_metrics[0:req.max_cells]

    # 12- Fill Response
    for metric in reverse_metrics_slice:
        p = properties[metric[1]]

        metric_msg = ColonyMetrics()

        metric_msg.area = p["area"]
        metric_msg.cell_quality = p["cell_quality"]
        # centroid wrt. the input image
        metric_msg.centroid_local = [p.centroid[0], p.centroid[1]]
        metric_msg.centroid_global = [p.centroid[0] + x_min, p.centroid[1] + y_min]
        metric_msg.compactness = p["compactness"]
        metric_msg.diameter = p["equivalent_diameter_area"]
        metric_msg.nn_centroid_distance = p["nn_centroid_distance"]
        metric_msg.nn_collision_distance = p["nn_collision_distance"]
        metric_msg.discarded = p["discarded"]
        metric_msg.discarded_description = p["discarded_description"]

        response.cell_metrics.append(metric_msg)

    if req.send_image_result:
        response.image_result = bridge.cv2_to_imgmsg(img_original_cropped, encoding="passthrough")

    # Save cell annotation
    ax_annotation = None
    if req.save_cell_annotation:
        if len(response.cell_metrics) != 0:
            fig, ax = plt.subplots()

            if(len(response.image_result.data) == 0):
                ax.imshow(cv_img)
            else:
                result_img = bridge.imgmsg_to_cv2(response.image_result, desired_encoding='passthrough')
                ax.imshow(result_img)

            ax.set_title(f'Best colonies to pick')

            # circle up best matches
            index  = 1
            for metric in response.cell_metrics:

                point = (0,0)
                if(len(response.image_result.data) == 0):
                    point = (metric.centroid_global[1], metric.centroid_global[0])
                else:
                    point = (metric.centroid_local[1], metric.centroid_local[0])

                radius = metric.diameter/2.0
                circle = plt.Circle(point, radius=radius, fc='none', color='red')
                ax.add_patch(circle)
                ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')
                index += 1

            plt.tight_layout()
            plt.savefig(f'{req.save_path}/cell_annotation.png')

    # Save segementation process
    if req.save_segmentation_process:
        layout = [
            ["A", "B", "C",],
            ["D", "E", "F",],
            ["G", "H", "I",],
        ]

        fig, axd = plt.subplot_mosaic(layout, constrained_layout=True, dpi=300)


        axd['A'].imshow(img_original_cropped)
        axd['A'].set_title('Crop')
        axd['A'].set_axis_off()

        axd['B'].imshow(img_gray, cmap=plt.cm.gray)
        axd['B'].set_title('Mask')
        axd['B'].set_axis_off()

        axd['C'].imshow(img_threshold, cmap=plt.cm.gray)
        axd['C'].set_title('Threshold')
        axd['C'].set_axis_off()

        axd['D'].imshow(img_mask_artifacts, cmap=plt.cm.gray)
        axd['D'].set_title('Filter')
        axd['D'].set_axis_off()

        axd['E'].imshow(closing, cmap=plt.cm.gray)
        axd['E'].set_title('Closing')
        axd['E'].set_axis_off()

        axd['F'].imshow(opening, cmap=plt.cm.gray)
        axd['F'].set_title('Opening')
        axd['F'].set_axis_off()

        axd['G'].imshow(img_distance, cmap=plt.cm.gray)
        axd['G'].set_title('Distance')
        axd['G'].set_axis_off()

        axd['H'].imshow(img_labels, cmap=plt.cm.nipy_spectral)
        axd['H'].set_title('Watershed')
        axd['H'].set_axis_off()

        if ax_annotation:
            print("OK")
            axd['I'] = ax_annotation
            axd['I'].set_title('Best Cell\'s')
            axd['I'].set_axis_off()

        plt.tight_layout()
        plt.savefig(f"{req.save_path}/segmentation.pdf")
        plt.savefig(f"{req.save_path}/segmentation.png")

    # Generate excel properties sheet
    if req.save_properties:
        # select properties included in the table
        prop_names = ('label',
                      'cell_quality',    # custom
                      'compactness',     # custom
                      'nn_centroid_distance', # custom
                      'nn_collision_distance', # custom
                      'discarded', # custom
                      'discarded_description', # custom
                      'axes_closness',   # custom
                      'area',
                      'area_bbox',
                      'area_convex',
                      'area_filled',
                      'axis_major_length',
                      'axis_minor_length',
                      'bbox',
                      'centroid',
                      'centroid_local',
                      'centroid_weighted',
                      'centroid_weighted_local',
                      'coords',
                      'coords_scaled',
                      'eccentricity',
                      'equivalent_diameter_area',
                      'euler_number',
                      'extent',
                      'feret_diameter_max',
                      'image',
                      'image_convex',
                      'image_filled',
                      'image_intensity',
                      'inertia_tensor',
                      'inertia_tensor_eigvals',
                      'intensity_max',
                      'intensity_mean',
                      'intensity_min',
                      # 'intensity_std', # requires scikit-image 0.24.0
                      'label',
                      'moments',
                      'moments_central',
                      'moments_hu',
                      'moments_normalized',
                      'moments_weighted',
                      'moments_weighted_central',
                      'moments_weighted_hu',
                      'moments_weighted_normalized',
                      'num_pixels',
                      'orientation',
                      'perimeter',
                      'perimeter_crofton',
                      'slice',
                      'solidity',)

        props_dict = regionprops_to_dict(properties, prop_names)

        df = pd.DataFrame(props_dict)
        df.to_excel(f'{req.save_path}/region_properties.xlsx', index=False)


    # Plot interactive region properties
    if req.plot_interactive_properties:
        property_names = ['area',
                          'eccentricity',
                          'perimeter',
                          'solidity',
                          'compactness',
                          'axes_closness',
                          'equivalent_diameter',
                          'nn_collision_distance',
                          'cell_quality',
                          'discarded',
                          'discarded_description']

        print(f"Generating region properties ({len(properties)}) interactively plot, this may take some time...")
        plot_region_roperties(img_original_cropped, img_labels, properties, property_names)


    print("===")
    print("Sending response")
    print("===")
    return response

def get_colony_morphology_server():
    rospy.init_node('colony_morphogy')
    s = rospy.Service('get_colony_morphology', GetColonyMorphology, callback_compute_morphology)
    print("Ready to compute colony morphology")
    rospy.spin()

if __name__ == "__main__":
    get_colony_morphology_server()
