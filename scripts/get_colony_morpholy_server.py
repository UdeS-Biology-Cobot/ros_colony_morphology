#!/usr/bin/env python3
from __future__ import print_function

from ros_colony_morphology.srv import GetColonyMorphology, GetColonyMorphologyResponse
import rospy

from colony_morphology.geometry import *


def callback_compute_morphology(req):
    print(f'request = {req}')
    return GetColonyMorphologyResponse()

def get_colony_morphology_server():
    rospy.init_node('colony_morphogy')
    s = rospy.Service('get_colony_morphology', GetColonyMorphology, callback_compute_morphology)
    print("Ready to compute colony morphology")
    rospy.spin()

if __name__ == "__main__":
    get_colony_morphology_server()
