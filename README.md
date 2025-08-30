# ros_colony_morphology
ROS wrapper around the [colony_morphology](https://github.com/captain-yoshi/colony-morphology) python library.

Creates the ROS Service on the `/get_colony_morphology` namespace and :
- Given an image and some [parameters](./srv/GetColonyMorphology.srv#L1-L42)
- Returns a list of [colonies metrics](./msg/ColonyMetrics.msg)

## Build Dependencies
Add colony_morphology into a virtual python environment instead of installing globally.

``` sh
# clone this package + submodule
$ git clone --recurse-submodules https://github.com/captain-yoshi/ros_colony_morphology

# install dependencies
$ rosdep install --from-path src/ros_colony_morphology


$ catkin build
```

## Run
``` sh
# run server
$ rosrun ros_colony_morphology get_colony_morpholy_server.py

# run client, in another terminal
$ rosrun ros_colony_morphology get_colony_morpholy_client.py
```
