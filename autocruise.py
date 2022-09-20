
"""Uses a heuristic to automatically navigate generated scenes.

fly_camera.fly_dynamic will generate poses using disparity maps that avoid
crashing into nearby terrain.
"""
import pickle
import time

import config
import fly_camera
import imageio
import infinite_nature_lib
import numpy as np
import tensorflow as tf

