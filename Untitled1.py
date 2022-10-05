#!/usr/bin/env python
# coding: utf-8

# In[10]:


from clearml import Task
from clearml import Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Image_Sampler import Sampler

task = Task.init(project_name="bogdoll/Anomaly_detection_Moritz", task_name="example_run", reuse_last_task_id=False)

# Remote Execution on FZI XZ
task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3",
            docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all",  # --ipc=host",
        )
# PyTorch fix for version 1.10, see https://github.com/pytorch/pytorch/pull/69904
# task.add_requirements(
#     package_name="setuptools",
#     package_version="59.5.0",
# )
# task.add_requirements(
#     package_name="moviepy",
#     package_version="1.0.3",
# )
task.execute_remotely('docker', clone=False, exit_process=True)                                                                              # http://tks-zx-01.fzi.de:8080/workers-and-queues/queues

print("Halloooooooooooooooasda ssssssssssssssssssss")
TEST_ID = "8ce5cdd31e8e499db2e07fc70b6136d5"
test_data = Dataset.get(dataset_id=TEST_ID).get_local_copy()
# test_data = np.array(test_data).astype("float32") / 255
img = Sampler.load_Images(test_data).astype("float32") / 255
plt.imshow(img[0])


# In[7]:




