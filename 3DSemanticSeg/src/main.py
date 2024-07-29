"""Main execution"""

import logging
import os
import numpy as np
import pyarrow.parquet as pq

import polars as pl
import tensorflow.compat.v1 as tf
import torch

from waymo_open_dataset import dataset_pb2 as wod
from waymo_open_dataset.utils import frame_utils

from models import PlaceholderModel

from utils import utils as utl
from utils import utils_constants as utl_c
from utils import utils_plotting as utl_p
from utils import utils_waymo as utl_w

tf.enable_eager_execution()


if __name__ == "__main__":

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="main.log", encoding="utf-8", level=logging.DEBUG)
    logger.debug("Logger set-up successful")

    # Define constants
    LASER_NAME_MAP = dict(wod.LaserName.Name.items())
    CAMERA_NAME_MAP = dict(wod.CameraName.Name.items())
    LIDAR_RETURN_MAP = dict()
    RANGE_IMAGE_DIM_MAP = utl_c.get_range_image_final_dim_dict()
    SEG_IMAGE_DIM_MAP = utl_c.get_seg_image_final_dim_dict()

    # Import data files
    data_dir = "/workspace/hostfiles/3DSemanticSeg/data"
    data_files = os.listdir(data_dir)
    data_file = data_files[0]  # TODO: generalize
    dataset = utl.load_datafile(data_dir, data_file)

    # Extract frames
    frames = utl.extract_frames_from_datafile(dataset)
    frame = frames[24]  # TODO: generalize

    # Parse range images
    range_images, camera_projections, seg_labels, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )

    # Convert range and segmentation images to tensors
    ## range_images is a dictionary formatted: {laser_index: [return1, return2]}
    ## seg_labels is a dictionary formatted: {laser_index: [return1, return2]}
    range_image_tensor = utl.convert_range_image_to_tensor(
        range_images[LASER_NAME_MAP["TOP"]][0]
    )
    seg_image_tensor = utl.convert_range_image_to_tensor(
        seg_labels[LASER_NAME_MAP["TOP"]][0]
    )

    # Plot example range and segmentation image
    utl_p.plot_range_image_tensor(
        range_image_tensor,
        RANGE_IMAGE_DIM_MAP,
        invert_colormap=True,
    )
    utl_p.plot_range_image_tensor(
        seg_image_tensor, SEG_IMAGE_DIM_MAP, style_params={"cmap": "tab20"}
    )

    # Plot example point cloud
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=0,  # First return
    )
    point_labels = utl_w.convert_range_image_to_point_cloud_labels(
        frame,
        range_images,
        seg_labels,
        ri_index=0,  # First return
    )

    points_all = np.concatenate(points, axis=0)
    point_labels_all = np.concatenate(point_labels, axis=0)
    # TODO: plot point cloud

    # Placeholder model (getting pipeline set up before iterating on model development)
    model = PlaceholderModel()
    range_image_tensor = torch.from_numpy(range_image_tensor.numpy())
    output = model(range_image_tensor)

    # Generate protobuff submission file for validation set
