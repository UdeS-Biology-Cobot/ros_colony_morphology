#!/usr/bin/env python3
import cv2 as cv
import os
import sys
import rospy
from ros_colony_morphology.msg import ColonyMetrics
from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyRequest
from cv_bridge import CvBridge
from pathlib import Path

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

    request.send_masked_image = False
    request.send_annotated_image = False

    response = get_colony_morphology_client(request)

    ctr = 1
    for cell in response.cell_metrics:
        print(f'### Cell {ctr} ######################')
        print(f'area         = {cell.area}')
        print(f'cell_quality = {cell.cell_quality}')
        print(f'centroid_x   = {cell.centroid_x}')
        print(f'centroid_y   = {cell.centroid_y}')
        print(f'compactness  = {cell.compactness}')
        print(f'diameter     = {cell.diameter}')
        print('')
        ctr += 1


    print(f'number of cells metrics =  {len(response.cell_metrics)}')



