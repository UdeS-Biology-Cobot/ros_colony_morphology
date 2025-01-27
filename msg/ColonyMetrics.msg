float64 area
float64 cell_quality
float64 compactness
float64 diameter
float64 nn_centroid_distance   # nearest neighbor distance of centroids
float64 nn_collision_distance  # nearest neighbor collision distance wrt. the equivalent radius area
                               # > 0, not in collision
                               # = 0, no nearest neighbor or in collision
                               # < 0, in collision

# Coordinates of the colony
#     +x points to the right of the image
#     +y points down in the image
float64[] centroid_local   # [x, y] wrt. the modified image, if no changes then same as global
float64[] centroid_global  # [x, y] wrt. the original image
