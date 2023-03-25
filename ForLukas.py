import carla
import math
import random
import time

# Connect the client and set up bp library and spawn points
client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()



#Load a World
client.load_world('Town05')


bp_static = world.get_blueprint_library().filter('static.prop.*')
    
transform = carla.Transform(carla.Location(x = 0, y = 0, z = 0.5))
    

anomaly_object = random.choice(bp_static)
a = world.spawn_actor(anomaly_object, transform)

a.set_simulate_physics(True)
a.set_enable_gravity(True)

#time.sleep(0.2)