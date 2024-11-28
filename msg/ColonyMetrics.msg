float64 area
float64 cell_quality
float64 compactness
float64 diameter

# Coordinates of the colony
#     +x points to the right of the image
#     +y points down in the image
float64[] centroid_local   # [x, y] wrt. the modified image, if no changes then same as global
float64[] centroid_global  # [x, y] wrt. the original image
