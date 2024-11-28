cmake_minimum_required(VERSION 3.1.3)
project(ros_colony_morphology)

set(MSG_DEPS sensor_msgs)

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	message_generation
	${MSG_DEPS}
	rospy
)

# ROS messages, services and actions
add_message_files(DIRECTORY msg FILES
	ColonyMetrics.msg
)

add_service_files(DIRECTORY srv FILES
	GetColonyMorphology.srv
)

catkin_python_setup()

generate_messages(DEPENDENCIES ${MSG_DEPS})


catkin_package(
	CATKIN_DEPENDS
	    cv_bridge
		message_runtime
		${MSG_DEPS}
)
