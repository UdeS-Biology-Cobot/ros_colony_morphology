cmake_minimum_required(VERSION 3.1.3)
project(ros_colony_morphology)

set(MSG_DEPS sensor_msgs)

find_package(catkin REQUIRED COMPONENTS
	message_generation
	${MSG_DEPS}
)

# ROS messages, services and actions
add_message_files(DIRECTORY msg FILES
	ColonyMetrics.msg
)

add_service_files(DIRECTORY srv FILES
	GetColonyMorphology.srv
)

generate_messages(DEPENDENCIES ${MSG_DEPS})

catkin_package(
	CATKIN_DEPENDS
		message_runtime
		${MSG_DEPS}
)
