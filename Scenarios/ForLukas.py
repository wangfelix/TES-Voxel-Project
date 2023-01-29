import carla
import math
import random
import time
import numpy as np
import cv2
import open3d as o3d
from matplotlib import cm

# Connect the client and set up bp library and spawn points
client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()


#Add the ego_vehicle
vehicle_bp_benz= bp_lib.find('vehicle.mercedes.coupe_2020')
vehicle_bp_benz.set_attribute('color', '255,255,255')
ego_vehicle = world.spawn_actor(vehicle_bp_benz, spawn_points[28])

#Move spectator behind to view
spectator = world.get_spectator()
transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-6, z=2.0)), ego_vehicle.get_transform().rotation)
spectator.set_transform(transform)

#Security Auto vor dem ego_vehicle
vehicle_bp_benz.set_attribute('color', '0,0,0')
enemy_vehicle = world.try_spawn_actor(vehicle_bp_benz, spawn_points[34])

enemy_vehicle.set_autopilot(True)

#Iterate this cell to find desired camera location
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2)) #Change this to move camera
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

#time.sleep(0.2)
#spectator.set_transform(camera.get_transform())
#camera.destroy()

# Callback stores sensor data in a dictionary for use outside callback
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Get gamera dimensions and initialise dictionary
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
camera_data = {'image': np.zeros((image_h, image_w, 4))}

# Start camera recording
camera.listen(lambda image: camera_callback(image, camera_data))

# OpenCV named window for rendering
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])
cv2.waitKey(1)

# Game loop
while True:

    # Imshow renders sensor data to display
    cv2.imshow('RGB Camera', camera_data['image'])

    # Quit if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Close OpenCV window when finished
cv2.destroyAllWindows()
cv2.stop()
