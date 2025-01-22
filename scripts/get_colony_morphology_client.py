#!/usr/bin/env python3
import cv2 as cv
import os
import sys
import rospy
from ros_colony_morphology.msg import ColonyMetrics
from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyRequest
from cv_bridge import CvBridge
from pathlib import Path
from matplotlib import pyplot as plt

def get_colony_morphology_client(request):
    rospy.wait_for_service('get_colony_morphology')
    try:
        get_colony_morphology = rospy.ServiceProxy('get_colony_morphology', GetColonyMorphology)
        response = get_colony_morphology(request)
        return response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":

    absolute_path = str(Path(__file__).parent.resolve().absolute())
    absolute_path += "/../ext/colony-morphology/dataset/intel_rs_d415_2.png"
    print(absolute_path)

    # read image from dataset
    # absolute_path = os.path.join(os.getcwd(), "../ext/colony-morphology/dataset/intel_rs_d415_2.png")
    cv_img =  cv.imread(str(absolute_path))
    assert cv_img is not None, "File could not be read, check with os.path.exists()"

    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="passthrough")

    # write request
    request = GetColonyMorphologyRequest()

    request.max_cells = 15
    request.image = img_msg

    request.mask_petri_dish = True
    request.dish_diameter = 882
    request.dish_offset = 134

    request.weight_area = 2.0
    request.weight_compactness = 1.0
    request.weight_eccentricity = 0.25
    request.weight_min_distance_nn = 2.0
    request.weight_solidity = 1.0

    request.cell_min_diameter = 0
    request.cell_max_diameter = 0
    request.cell_min_compactness = 0.6
    request.cell_min_solidity = 0.85
    request.cell_max_eccentricity = 0.8

    request.std_weight_area = 1.5

    request.send_image_result = True

    response = get_colony_morphology_client(request)

    ctr = 1
    for cell in response.cell_metrics:
        print(f'### Cell {ctr} ######################')
        print(f'area              = {cell.area}')
        print(f'cell_quality      = {cell.cell_quality}')
        print(f'compactness       = {cell.compactness}')
        print(f'diameter          = {cell.diameter}')
        print(f'centroid_local_x  = {cell.centroid_local[0]}')
        print(f'centroid_local_y  = {cell.centroid_local[1]}')
        print(f'centroid_global_x = {cell.centroid_global[0]}')
        print(f'centroid_global_y = {cell.centroid_global[1]}')
        print('')
        ctr += 1


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

            radius = metric.diameter/2.0 + 5
            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax.add_patch(circle)
            ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')
            index += 1

        plt.tight_layout()
        plt.show()
