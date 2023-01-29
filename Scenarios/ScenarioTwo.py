import carla

# Connect to CARLA simulator
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get all the available routes in the map
routes = world.get_map().get_topology()

# Select a random route
import random
route = random.choice(routes)

# Get the starting and ending points of the route
start_waypoint = route[0]
end_waypoint = route[-1]

# Spawn the ego vehicle at a random point on the starting waypoint
spawn_point = carla.Transform(location=start_waypoint.location + carla.Location(x=random.uniform(-5, 5), y=random.uniform(-5, 5), z=0))

blueprint = world.get_blueprint_library().filter("vehicle.audi.a2")[0]
vehicle = world.spawn_actor(blueprint, spawn_point)

# Create a traffic manager
tm = client.get_trafficmanager()

# Set the ego vehicle's route
tm.set_vehicle_route(vehicle, route)

# Run the simulation
while True:
    world.tick()
