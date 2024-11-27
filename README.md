# ros_colony_morphology
ROS wrapper around the colony morpholgy python library.

Creates the ROS Service on the `/get_colony_morphology` namespace and :
- Given an image and some [parameters](./srv/GetColonyMorphology.srv#L1-L42)
- Returns a list of [colonies metrics](./msg/ColonyMetrics.msg)

## Build Dependencies
Add colony_morphology into a virtual python environment instead of installing globally.

``` sh
# clone this package + submodule
$ git clone --recurse-submodules https://github.com/captain-yoshi/ros_colony_morphology


# create python dependencies in a virtual environment
$ cd ros_colony_morphology/ext/colony-morphology

# use version used by ROS
$ python3 -m venv venv

$ source venv/bin/activate

(venv)$ pip install -r requirements.txt

(venv)$ deactivate

$ cd ../../scripts

# add softlink of required python modules to work with the script
$ ln -s ../ext/colony-morphology/venv/lib/python3.8/site-packages/* ./
```

## Run

``` sh
$ rosrun ros_colony_morphology get_colony_morpholy_server.py
```
