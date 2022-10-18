#!/usr/bin/env python
# coding: utf-8


from operator import truediv
import re
from turtle import distance
import cv2
import time
import torch
import numpy as np
import argparse

import os
import sys
from tempfile import gettempdir

from torch.utils.tensorboard import SummaryWriter

from clearml import Task

task = Task.init(project_name="bogdoll/Anomaly_detection_Moritz", task_name="QAgent_Reward", output_uri="s3://tks-zx.fzi.de:9000/clearml")
task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3",
            docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"  # --ipc=host",   
            )
# Remote Execution on FZI XZ
# task.set_base_docker(
#             "tks-zx-01.fzi.de/autonomous-agents/core-carla:21.10",
#             docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
#             docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all",  # --ipc=host",
#         )
# # PyTorch fix for version 1.10, see https://github.com/pytorch/pytorch/pull/69904
# task.add_requirements(
#     package_name="setuptools",
#     package_version="59.5.0",
# )
# task.add_requirements(
#     package_name="moviepy",
#     package_version="1.0.3",
# )
# task.execute_remotely('rtx3090', clone=False, exit_process=True) 


# http://tks-zx-01.fzi.de:8080/workers-and-queues/queues

from AE_model import AutoEncoder
from model_eval import Evaluater

from env_carla import Environment


from training import Training

from training import N_EPISODES
from training import TARGET_UPDATE
from training import EPS_START

# The learned Q value rates (state,action) pairs
# A CNN with a state input can rate possible actions, just as a classifier would

PREVIEW = False
VIDEO_EVERY = 1_000
PATH = "model.pt"
IM_HEIGHT = 256
IM_WIDTH = 256

EGO_X = 246
EGO_Y = 128

def main(withAE=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu": print("!!! device is CPU !!!")
    ae_model = AutoEncoder()
    evaluater = Evaluater(ae_model, device, PATH)
    DISTANCE_MATRIX = init_distance_matrices(EGO_X,EGO_Y)
    print(DISTANCE_MATRIX)

    writer = SummaryWriter()
#     env = Environment(host="tks-fly.fzi.de", port=2000)
    env = Environment(host="localhost", port=2000, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130, random_spawn=False)
    env.init_ego()

    trainer = Training(writer, device, withAE=withAE)

    epsilon = EPS_START
    reward_best = -1000

    for i in range(N_EPISODES):

        reward_per_episode = 0
        start = time.time()
        n_frame = 1

        env.reset()
        env.spawn_anomaly()

        obs_current = env.get_observation()
        obs_current = obs_current[0] #no segemntation
        # if withAE:
        #     # heatmap = evaluater.getHeatMap(obs_current)
        #     detectionMap = evaluater.getColoredDetectionMap(obs_current)
        #     obs_current = np.hstack((obs_current, detectionMap))
        
        obs_current = np.transpose(obs_current, (2,1,0))
        obs_current = np.array([obs_current])
#         print(obs_current.shape)
        obs_current = torch.as_tensor(obs_current)

        chw_list = []

        while True:
            if PREVIEW:
                cv2.imshow("", env.observation)
                cv2.waitKey(1)

#             chw = obs_current.squeeze(0)  # Remove batch information from BCHW
            chw_list.append(obs_current)

            # Perform action on observation and buildup replay memory

            if i % VIDEO_EVERY == 0:
                action = trainer.select_action(obs_current, 0)
            else:
                action = trainer.select_action(obs_current, epsilon)
            obs_next, reward, done, _ = env.step(action)
            obs_next = obs_next[0] #no segemntation

            if withAE:
                detectionMap = evaluater.getDetectionMap(obs_next)
                reward = calcualte_enriched_reward(reward, detectionMap, DISTANCE_MATRIX)
                # print(reward)
                

            reward_per_episode += reward
            reward_torch = torch.tensor([reward], device=device)  # For compatibility with PyTorch

            if (
                time.time() - start
            ) > 30:  # Since the agent can simply stand now, the episode should terminate after 30 seconds
                done = True

            if done:
                obs_next = None
            else:
                # if withAE:
                #     # heatmap = evaluater.getHeatMap(obs_next)
                #     detectionMap = evaluater.getColoredDetectionMap(obs_next)
                #     obs_next = np.hstack((obs_next, detectionMap))

                obs_next = np.transpose(obs_next, (2,1,0))
                obs_next = np.array([obs_next])
                obs_next = torch.as_tensor(obs_next)
            
            # Python tuples () https://www.w3schools.com/python/python_tuples.asp
            trainer.replay_memory.push(obs_current, action, obs_next, reward_torch, done)
            
            obs_current = obs_next

            # Optimization on policy model (I believe this could run in parallel to the data collection task)
            trainer.optimize(i)

            if done:
                end = time.time()
                duration = end - start
                writer.add_scalar("Reward per episode", reward_per_episode, i)
                writer.add_scalar("Duration before crash/seconds", duration, i)
                writer.add_scalar("Frames before crash/frames", n_frame, i)

                if reward_per_episode > reward_best:
                    reward_best = reward_per_episode
                    save_video(chw_list, reward_best, i, writer, evaluater, withAE)
                    # tchw_list = torch.stack(chw_list)  # Adds "list" like entry --> TCHW
                    # tchw_list = torch.squeeze(tchw_list)
                    # name = "DQN Champ: " + str(reward_per_episode)
                    # writer.add_video(
                    #     tag=name, vid_tensor=tchw_list.unsqueeze(0), global_step=i
                    # )  # Unsqueeze adds batch --> BTCHW
                    torch.save(trainer.policy_net.state_dict(), os.path.join(gettempdir(), "dqn_" + str(i) + ".pt"))
                break

            n_frame += 1

        # Save video of episode to ClearML https://github.com/pytorch/pytorch/issues/33226
        if i % VIDEO_EVERY == 0:
            tchw_list = torch.stack(chw_list)  # Adds "list" like entry --> TCHW
            tchw_list = torch.squeeze(tchw_list)
            writer.add_video(
                tag="DQN Agent", vid_tensor=tchw_list.unsqueeze(0), global_step=i
            )  # Unsqueeze adds batch --> BTCHW

        # Update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == 0:
            trainer.target_net.load_state_dict(trainer.policy_net.state_dict())

        # Decay epsilon
        writer.add_scalar("Exploration-Exploitation/epsilon", epsilon, i)
        epsilon = trainer.decay_epsilon(epsilon)

        print(f"Episode: {i}")

    writer.flush()

# generate distance map from the center of the car to all other pixels in the space (works only in BEV)
# remains static for the whole training since the car in the image is never moving
def init_distance_matrices(pos_x, pos_y):
    size = IM_WIDTH
    ring_count = size # we want to double the size. each ring adds a size of 2 
    distance_matrix = np.zeros((1,1))
    for x in range(ring_count):
        distance_matrix = add_ring(distance_matrix, x + 1)


    max_distance = max(distance_matrix.flatten())

    distance_matrix = distance_matrix[size - pos_y: size - pos_y + size, size - pos_x: size - pos_x + size]
    distance_matrix

    max_distance_matrix = np.zeros((size, size)) + max_distance

    distance_matrix = distance_matrix / max_distance_matrix
    distance_matrix

    return distance_matrix

def calcualte_enriched_reward(reward, detectionMap, distanceMap):
    if reward == -1 : return -1 # collision or timeout

    rewardMap = detectionMap * distanceMap # element wise
    total_reward = np.sum(rewardMap)
    penalty = total_reward / (rewardMap.shape[0] * rewardMap.shape[1] - 1) # minus one, because the origion of the car should not be taken into count and is always zero
    reward = 1 - penalty# * 0.1
    reward = np.float32(reward)

    return reward



# given matrix a, adds a ring to it of the given value:
# a   --->    b-b-b
#             b-a-b
#             b-b-b
def add_ring(matrix, value):
    b = np.zeros(tuple(s+2 for s in matrix.shape), matrix.dtype) + value
    b[tuple(slice(1,-1) for s in matrix.shape)] = matrix
    return b


def save_video(chw_list, reward_best, step, writer, evaluater, withVAE):
    if withVAE:
        aug_list = []
        for img in chw_list:
            img = img.numpy()
            img = np.squeeze(img)
            img = np.transpose(img, (2,1,0)) # shape: w,h,3
            detectionMap = evaluater.getColoredDetectionMap(img)
            detectionMap = color_pixel(detectionMap)
            seperator = np.zeros((256,2,3))
            seperator[:,:,0] = 1.
            aug_img = np.hstack((img, seperator, detectionMap))
            aug_img = np.transpose(aug_img, (2,1,0)) # shape: 3,w,h
            aug_img = torch.as_tensor(np.array([aug_img]))
            aug_list.append(aug_img)
        tchw_list = aug_list

    tchw_list = torch.stack(tchw_list)  # Adds "list" like entry --> TCHW
    tchw_list = torch.squeeze(tchw_list)
    tchw_list = tchw_list.unsqueeze(0)
    name = "DQN Champ: " + str(reward_best)
    writer.add_video(
        tag=name, vid_tensor=tchw_list, global_step=step
    )  # Unsqueeze adds batch --> BTCHW

def color_pixel(img):
    img[EGO_X+1, EGO_Y+1, 0] = 0.
    img[EGO_X+1, EGO_Y+1, 1] = 0.
    img[EGO_X+1, EGO_Y+1, 2] = 1.

    img[EGO_X+1, EGO_Y, 0] = 0.
    img[EGO_X+1, EGO_Y, 1] = 0.
    img[EGO_X+1, EGO_Y, 2] = 1.

    img[EGO_X, EGO_Y+1, 0] = 0.
    img[EGO_X, EGO_Y+1, 1] = 0.
    img[EGO_X, EGO_Y+1, 2] = 1.

    img[EGO_X-1, EGO_Y-1, 0] = 0.
    img[EGO_X-1, EGO_Y-1, 1] = 0.
    img[EGO_X-1, EGO_Y-1, 2] = 1.

    img[EGO_X-1, EGO_Y, 0] = 0.
    img[EGO_X-1, EGO_Y, 1] = 0.
    img[EGO_X-1, EGO_Y, 2] = 1.

    img[EGO_X, EGO_Y-1, 0] = 0.
    img[EGO_X, EGO_Y-1, 1] = 0.
    img[EGO_X, EGO_Y-1, 2] = 1.

    img[EGO_X+1, EGO_Y-1, 0] = 0.
    img[EGO_X+1, EGO_Y-1, 1] = 0.
    img[EGO_X+1, EGO_Y-1, 2] = 1.

    img[EGO_X-1, EGO_Y+1, 0] = 0.
    img[EGO_X-1, EGO_Y+1, 1] = 0.
    img[EGO_X-1, EGO_Y+1, 2] = 1.

    img[EGO_X, EGO_Y, 0] = 0.
    img[EGO_X, EGO_Y, 1] = 0.
    img[EGO_X, EGO_Y, 2] = 1.

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--AE", type=bool, default=True) # turn on Autoencoder

    args = parser.parse_args()
    withAE = args.AE
    if withAE == "True" or withAE == "true":
        withAE = True
    elif withAE == "False" or withAE == "false":
        withAE = False

    main(withAE)