#!/usr/bin/env python3
from ros_colony_morphology.msg import ColonyMetrics
from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyResponse
import rospy

from colony_morphology.geometry import *
from colony_morphology.image_transform import *
# from colony_morphology.plotting import plot_bboxes, plot_region_roperties
from colony_morphology.metric import compactness as compute_compactness
from colony_morphology.skimage_util import compactness_cb
from colony_morphology.skimage_util import nn_centroid_distance_cb
from colony_morphology.skimage_util import nn_collision_distance_cb
from colony_morphology.skimage_util import cell_quality_cb
from colony_morphology.skimage_util import axes_closness_cb
from colony_morphology.metric import axes_closness as compute_axes_closness

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
from scipy.spatial import cKDTree

import statistics
from cv_bridge import CvBridge


bridge = CvBridge()

def callback_compute_morphology(req):

    img = bridge.imgmsg_to_cv2(req.image, desired_encoding='passthrough')

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Mask image to contain only the petri dish
    # 1- resize image to speedup detection
    # scale, img_resize = resize_image(img_gray, pixel_threshold=1280*1280)

    # 2- detect circle radius + centroid
    dish_regions = detect_area_by_canny(img_gray, radius=(req.dish_diameter/2.0))
    if(len(dish_regions) == 0):
        print('No circle detected, check the requested dish diameter');
        return response

    region_prop = dish_regions[0]
    centroid = region_prop["centroid"]
    radius = region_prop["equivalent_diameter_area"]/2.0

    centroid = tuple(c for c in centroid)
    radius -= req.dish_offset

    # 3- create circular masks
    circular_mask = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius)
    circular_mask_artifacts = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius -8)

    # 4- mask orighinal image
    idx = (circular_mask== False)
    img_masked = np.copy(img)
    img_masked[idx] = 0; # black

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


    # 5- crop image
    img_cropped = img_masked[x_min:x_max, y_min:y_max]
    img_original_cropped = img[x_min:x_max, y_min:y_max]
    circular_mask_artifacts = circular_mask_artifacts[x_min:x_max, y_min:y_max]

    # Convert to grayscale
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)


    # Apply threshold
    img_bw = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, 2)


    # fill smal holes
    # https://stackoverflow.com/a/10317883
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
    img_fill = cv.morphologyEx(img_bw,cv.MORPH_CLOSE,kernel)


    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_fill,cv.MORPH_OPEN,kernel, iterations = 2)
    img_bw = opening


    # remove contour artifacts from mask + post processing
    idx = (circular_mask_artifacts== False)
    img_bw[idx] = 0; # make mask white to lower the size of fake region properties found


    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_bw,cv.MORPH_OPEN,kernel, iterations = 2)

    # Sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    distance = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(distance,0.1*distance.max(),255,0)
    sure_fg = np.uint8(sure_fg)     # Convert to int

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(sure_fg.shape)
    ndi.distance_transform_edt(sure_fg, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed...')

    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=sure_fg)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]
    img_labels = watershed(img_distance, markers, mask=sure_fg, connectivity=1, compactness=0)

    # Retrieve metric from labels
    print('Computing region properties...')

    # add extra properties, some function must be populated afterwards
    extra_callbacks = (compactness_cb, nn_centroid_distance_cb, nn_centroid_distance_cb, cell_quality_cb, axes_closness_cb)

    properties = regionprops(img_labels, intensity_image=img_gray, extra_properties=extra_callbacks)

    # Compute compactness
    for p in properties:
        # avoid division by zero
        if(p.perimeter == 0.0):
            p.compactness = 0.0
        else:
            p.compactness = compute_compactness(p.area, p.perimeter)


    # Remove every properties that have a perimeter of zero
    properties[:] = [p for p in properties if p["compactness"] > 0.0]
    print(f'Region properties, after removing small objects = {len(properties)}')


    # Compute axes_closness
    for p in properties:
        p.axes_closness = compute_axes_closness(p.axis_major_length, p.axis_minor_length)


    # Find the nearest neighbors with ckDTree
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

                    # print(f"centroid = {dd[index]}")
                    # print(f"radius = {radius}")
                    # print(f"nnradius = {nn_radius}")
                    # print(f"collision_distance = {collision_distance}")




    # Compute metric for best colonies
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
        if(req.cell_min_diameter and p.equivalent_diameter_area < req.cell_min_diameter):
            cell_quality = 0.0
        if(req.cell_max_diameter and p.equivalent_diameter_area > req.cell_max_diameter):
            cell_quality = 0.0
        if(req.cell_min_compactness and p.compactness < req.cell_min_compactness):
            cell_quality = 0.0
        if(req.cell_min_solidity and p.solidity < req.cell_min_solidity):
            cell_quality = 0.0
        if(req.cell_max_eccentricity and p.eccentricity > req.cell_max_eccentricity):
            cell_quality = 0.0
        if(p.nn_collision_distance < 0): # in collision
            cell_quality = 0.0


        quality_metrics[i] = (cell_quality, i)
        p.cell_quality = cell_quality


    # remove outliers wrt. area of non discarded cell's
    if(req.std_weight_area):
        area_std = statistics.pstdev(p["area"] for p in properties if p.cell_quality > 0.0)
        area_mean = statistics.mean(p["area"] for p in properties if p.cell_quality > 0.0)

        # print(area_std)
        # print(area_mean)
        # print(area_mean - 1.5* area_std)
        # print(area_mean + 1.5* area_std)
        for i in range(0, len(properties)):
            p = properties[i]
            if (p.area < area_mean - req.std_weight_area* area_std or
                p.area > area_mean + req.std_weight_area* area_std):
                # print(f'label {p.label} discarded due to being below std')
                p.cell_quality = 0.0
                quality_metrics[i] = (p.cell_quality, i)


    # TODO itemgetter might be faster then a lambda
    # https://stackoverflow.com/a/10695158
    reverse_metrics = sorted(quality_metrics, key=lambda x: x[0], reverse=True)


    # Fill Response
    response = GetColonyMorphologyResponse()

    # reduce list to requested number of cells
    reverse_metrics_slice = reverse_metrics
    if(req.max_cells and len(reverse_metrics) > req.max_cells):
        reverse_metrics_slice = reverse_metrics[0:req.max_cells]


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

        response.cell_metrics.append(metric_msg)

    if req.send_image_result:
        response.image_result = bridge.cv2_to_imgmsg(img_original_cropped, encoding="passthrough")


    return response

def get_colony_morphology_server():
    rospy.init_node('colony_morphogy')
    s = rospy.Service('get_colony_morphology', GetColonyMorphology, callback_compute_morphology)
    print("Ready to compute colony morphology")
    rospy.spin()

if __name__ == "__main__":
    get_colony_morphology_server()
