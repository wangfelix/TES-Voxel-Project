import glob
import os
import sys

import random
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import cv2
import weakref

import torch

IM_WIDTH = 256
IM_HEIGHT = 256

BEV_DISTANCE = 20

N_ACTIONS = 9

RESET_SLEEP_TIME = 1

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

class Environment:

    def __init__(self, world=None, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=BEV_DISTANCE, cam_rotation=-90, cam_zoom=110, random_spawn=True):
        weak_self = weakref.ref(self)
        self.client = carla.Client(host, port)            #Connect to server
        self.client.set_timeout(30.0)

        # traffic_manager = self.client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # traffic_manager.set_respawn_dormant_vehicles(True)
        # traffic_manager.set_synchronous_mode(True)

        self.random_spawn = random_spawn
        if not world == None: self.world = self.client.load_world(world)
        else: self.world = self.client.load_world("Town01_Opt")

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(weather)

        self.observation = None
        self.observation_seg = None

        self.collision_hist = []

    def init_ego(self):
        self.vehicle_bp = self.bp_lib.find('vehicle.tesla.model3')
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.ss_camera_bp_sg = self.bp_lib.find('sensor.camera.semantic_segmentation')
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')

        # Configure sensors
        self.ss_camera_bp.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp.set_attribute('fov', str(self.cam_zoom))

        # Location for both sensors
        self.ss_cam_location = carla.Location(10,0,self.cam_height)
        self.ss_cam_rotation = carla.Rotation(self.cam_rotation,0,0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        # # Configure segmantic sensors
        self.ss_camera_bp_sg.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp_sg.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp_sg.set_attribute('fov', str(self.cam_zoom))
        
        # collision sensor
        self.col_sensor_location = carla.Location(0,0,0)
        self.col_sensor_rotation = carla.Rotation(0,0,0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)




    def reset(self):
        self.deleteEnv()
        
        self.actor_list = []
        self.collision_hist = []

        # Spawn vehicle
        if self.random_spawn: transform = random.choice(self.spawn_points)
        else: transform = self.spawn_points[1]
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
        self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        # Attach and listen to image sensor (RGB)
        self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data, weak_self))

        # Attach and listen to image sensor (Semantic Seg)
        self.ss_cam_seg = self.world.spawn_actor(self.ss_camera_bp_sg, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.ss_cam_seg)
        self.ss_cam_seg.listen(lambda data: self.__process_sensor_data_Seg(data, weak_self))

        time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        # Attach and listen to collision sensor
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event, weak_self))


        self.episode_start = time.time()
        return self.get_observation()

    def step(self, action):
        # Easy actions: Steer left, center, right (0, 1, 2)
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1))


        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        elif v_kmh < 20:
            reward = -1
        else:
            reward = 1

        return self.get_observation(), reward, done, None

    
    def spawn_anomaly(self, distance=15):
        transform = self.get_Vehicle_transform() #get vehicle location and rotation (0-360 degrees)
        ped_blueprints = self.bp_lib.filter("static.prop.clothcontainer")
        vec = transform.rotation.get_forward_vector()
        transform.location.x = transform.location.x + vec.x * distance
        transform.location.y = transform.location.y + vec.y * distance
        transform.location.z = transform.location.z + vec.z * distance 
        player = self.world.try_spawn_actor(random.choice(ped_blueprints),transform)
        self.actor_list.append(player)
        return player #for destruction

    def change_Weather(self):
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=70.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=70.0,
            fog_distance=3.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(weather)

    def reset_Weather(self):
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(weather)

    def getRelevantWaypoints(self):
        waypoints = self.map.generate_waypoints(2.0)
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                      lane_type=carla.LaneType.Driving)

        return waypoints, wp

    def destroy_actor(self, actor):
        actor.destroy()

    def isActorAlive(self, actor):
        if actor.is_alive:
            return True
        return False
    
    def isEnvAlive(self):
        if len(self.actor_list) == 0:
            return False
        return True
        
    #get vehicle location and rotation (0-360 degrees)
    def get_Vehicle_transform(self):
        return self.vehicle.get_transform()

    #get vehicle location
    def get_Vehicle_positionVec(self):
        position = self.vehicle.get_transform().location
        return np.array([position.x, position.y, position.z])


    def get_observation(self):
        """ Observations in PyTorch format BCHW """
        frame = self.observation
        frame = frame.astype(np.float32) / 255
        frame = self.arrange_colorchannels(frame)

        # seg = self.observation_seg
        # seg = seg.astype(np.float32)
        # seg = self.arrange_colorchannels(seg)
        # return frame, seg
        return frame,None

    def __process_sensor_data(self, image, weak_self):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        self = weak_self()
        if not self: return
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    def __process_sensor_data_Seg(self, image, weak_self):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        self = weak_self()
        if not self: return
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation_seg = i3

    def __process_collision_data(self, event, weak_self):
        self = weak_self()
        if not self: return
        self.collision_hist.append(event)

    # changes order of color channels. Silly but works...
    def arrange_colorchannels(self, image):
        mock = image.transpose(2,1,0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2,1,0)
        return tmp

    def exit_env(self):
        self.deleteEnv()
    
    def deleteEnv(self):
        # for actor in self.actor_list:
        #     # print(actor.type_id )
        #     # print(actor.type_id  == "vehicle.tesla.model3")
        #     if self.isActorAlive(actor) and not actor.type_id == "vehicle.tesla.model3":
        #         actor.stop()
        # for actor in self.actor_list:
        #     # if self.isActorAlive(actor):
        #     # if not actor.type_id == "vehicle.tesla.model3":
        #     actor.destroy()
        
        # for actor in self.actor_list:
        # #     # if self.isActorAlive(actor):
        # #     # if actor.type_id == "vehicle.tesla.model3":
        #     actor.destroy()
        # time.sleep(5)

        # while not len(self.actor_list) == 0:
        #     actor = self.actor_list.pop(0)
        #     actor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        # time.sleep(5)

