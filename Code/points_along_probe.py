
# @AdamLoydHarris

# %%
import numpy as np
import json
# %%

def generate_line_points(start_point, rotation_x, rotation_y, increment=20, num_points=10, direction = 1):
    """
    Generate points along a line in 3D space given a starting point, and rotations around x and y axes.
    
    :param start_point: tuple or list of (x, z, y) coordinates of the starting point
    :param rotation_x: rotation angle around the x-axis in degrees
    :param rotation_y: rotation angle around the y-axis in degrees
    :param increment: distance between consecutive points along the line
    :param num_points: number of points to generate along the line
    :return: list of (x, z, y) tuples representing points along the line
    """

    start_point = [start_point[1], start_point[0], start_point[2]]
    # Convert angles from degrees to radians
    rotation_x_rad = np.radians(rotation_x)
    rotation_y_rad = np.radians(rotation_y)
    
    # Compute the direction vector from rotation angles
    # Rotation around x-axis affects z and y
    # Rotation around y-axis affects x and z
    direction_vector = np.array([
        np.cos(rotation_y_rad),
        np.sin(rotation_x_rad),
        -np.sin(rotation_y_rad) * np.cos(rotation_x_rad)
    ])
    
    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)
    
    # Generate points along the line
    points = []
    for i in range(num_points):
        # Calculate the point at the current increment
        point = start_point + direction * i * increment * direction_vector
        point = [int(point[1]), int(point[0]), int(point[2])]
        points.append(tuple(point))
    
    return points

# %%
# Example usage

start_point = np.array([501, 1057, 620])
rotation_y = 0  # degrees
rotation_x = 7  # degrees
increment = 20  # length increment between points
num_points = 6  # number of points to generate
frames_diff_AP = 7
direction = -1 #1=anterior, -1=posterior
origin_DV = -1  #-1 uf your initial point is at the top, 1 if it's at the bottom
mouse = 'ah03'
final_depth_lowering = 1.3
depth_diff = final_depth_lowering * np.cos(np.radians(rotation_x))
print(f"insertion-terminus angle-adjusted DV diff = {depth_diff*100} voxels")
print("_______________________")

line_points = generate_line_points(start_point,rotation_y, 
                                   rotation_x, increment, 
                                   num_points, direction,
                                   )
with open(f'mouse_term_points/{mouse}.json', 'w') as f:
    for i, point in enumerate(line_points):
        print(f"terminus point {i} = {point}")
        print(f"insertion point {i} = ({point[0]}, {int(point[1]+origin_DV*frames_diff_AP)}, {int(point[2]+origin_DV*depth_diff*100)})")
        print("_______________________")
        f.write('\n')
        json.dump(point, f)
        f.write('\n')
        json.dump(f"{point[0]}, {int(point[1]+origin_DV*frames_diff_AP)}, {int(point[2]+origin_DV*depth_diff*100)}", f)
    f.write('\n')
    json.dump(f"direction = {direction}", f)
    f.write('\n')
    json.dump(f"origin_DV = {origin_DV}", f)
# %%

# %%
