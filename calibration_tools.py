import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

import box_dimensioner_multicam.realsense_device_manager as rsdm
import box_dimensioner_multicam.helper_functions as hf


def run_calibration(frames, extrinsics):
	
	# # Define some constants 
	# L515_resolution_width = 640 # 1024 # pixels
	# L515_resolution_height = 480 # 768 # pixels
	# L515_frame_rate = 30

	# resolution_width = 640 # 1280 # pixels
	# resolution_height = 480 # 720 # pixels
	# frame_rate = 30 # 15  # fps

	dispose_frames_for_stablisation = 30  # frames
	
	chessboard_width = 7 # squares
	chessboard_height = 10 	# squares
	square_size = 0.035 # meters

	try:
		intrinsics = get_device_intrinsics(frames)
		chessboard_params = [chessboard_height, chessboard_width, square_size]

		# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
		calibrated = False
		while not calibrated:               
			transformation_result_kabsch = perform_pose_estimation(frames, intrinsics, chessboard_params)
			object_point = get_chessboard_corners_in_3d(frames, intrinsics, chessboard_params)

			if not transformation_result_kabsch:
				print("Place the chessboard on the plane where the object needs to be detected..")
			else:
				calibrated = True

		# Save the transformation object for the device to use for measurements
		transformations = transformation_result_kabsch[1].inverse()

		# Transform chessboard points to 3D using the transformation
		points3D = object_point[2][:, object_point[3]]
		points3D = transformations.apply_transformation(points3D)

		# Extract the bounds between which the object's dimensions are needed
		roi_2D = get_boundary_corners_2D(points3D)

		print("Calibration completed... \nPlace the box in the field of view of the device...")

		# Get the calibration info as a dictionary to help with display of the measurements onto the color image
		calibration_info_device = {
			'transformation': transformations,
			'intrinsics': intrinsics,
			'extrinsics': extrinsics
		}

	except KeyboardInterrupt:
		print("The program was interrupted by the user. Closing the program...")

	finally:
		# device_manager.disable_streams()
		cv2.destroyAllWindows()


def get_device_intrinsics(frames):
	"""
	Get the intrinsics of the imager using its depth and RGB frames delivered by the RealSense device

	Parameters:
	-----------
	depth_frame : rs::frame
				  The depth frame grabbed from the imager inside the Intel RealSense
	rgb_frame : rs::frame
				The RGB frame grabbed from the imager inside the Intel RealSense

	Return:
	-----------
	device_intrinsics : dict
	keys  : 'depth', 'rgb'
	values: Intrinsics of the corresponding device for depth and RGB frames
	"""
	device_intrinsics = {
		'depth': frames['depth'].get_profile().as_video_stream_profile().get_intrinsics(),
		'rgb': frames['color'].get_profile().as_video_stream_profile().get_intrinsics()
	}

	return device_intrinsics


def calculate_transformation_kabsch(src_points, dst_points):
    """
    Calculates the optimal rigid transformation from src_points to dst_points
    (regarding the least squares error)

    Parameters:
    -----------
    src_points: array
        (3,N) matrix
    dst_points: array
        (3,N) matrix

    Returns:
    -----------
    rotation_matrix: array
        (3,3) matrix
    translation_vector: array
        (3,1) matrix
    rmsd_value: float
    """
    assert src_points.shape == dst_points.shape
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed to compute the transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rsdm.centroid(src_points)
    dst_points_centered = dst_points - rsdm.centroid(dst_points)

    rotation_matrix = rsdm.kabsch(src_points_centered, dst_points_centered)
    rmsd_value = rsdm.kabsch_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rsdm.centroid(dst_points) - np.matmul(rsdm.centroid(src_points), rotation_matrix)

    return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value


def apply_transformation(transformation, points):
    """
    Applies the transformation to the pointcloud

    Parameters:
    -----------
    transformation: array
        (3,3) rotation matrix and (3,1) translation vector
    points : array
        (3, N) matrix where N is the number of points

    Returns:
    ----------
    points_transformed : array
        (3, N) transformed matrix
    """
    assert points.shape[0] == 3
    n = points.shape[1]
    points_ = np.vstack((points, np.ones((1, n))))
    points_trans_ = np.matmul(transformation, points_)
    points_transformed = np.true_divide(points_trans_[:3, :], points_trans_[[-1], :])
    return points_transformed


def inverse_transformation(transformation):
    """
    Computes the inverse transformation and returns a new Transformation object

    Parameters:
    -----------
    transformation: array
        (3,3) rotation matrix and (3,1) translation vector

    Returns:
    -----------
    inverse: array
        (3,3) rotation matrix and (3,1) translation vector
    """
    rotation_matrix = transformation[:3, :3]
    translation_vector = transformation[:3, 3]

    rot = np.transpose(rotation_matrix)
    trans = -np.matmul(np.transpose(rotation_matrix), translation_vector)
    return np.column_stack((np.row_stack((rot, trans)), [0, 0, 0, 1]))



def get_chessboard_corners_in_3d(frames, intrinsic, chessboard_params):
    depth_frame = rsdm.post_process_depth_frame(frames['depth'])
    rgb_frame = frames['color']
    depth_intrinsics = intrinsic['depth']

    found_corners, points_2d = hf.cv_find_chessboard(depth_frame, rgb_frame, chessboard_params)
    corners_3d = [found_corners, None, None, None]

    if found_corners:
        points_3d = np.zeros((3, len(points_2d[0])))
        valid_points = [False] * len(points_2d[0])

        for index in range(len(points_2d[0])):
            corner = points_2d[:, index].flatten()
            depth = hf.get_depth_at_pixel(depth_frame, corner[0], corner[1])

            if depth != 0 and depth is not None:
                valid_points[index] = True
                [X, Y, Z] = hf.convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], depth_intrinsics)
                points_3d[0, index] = X
                points_3d[1, index] = Y
                points_3d[2, index] = Z

        corners_3d = [found_corners, points_2d, points_3d, valid_points]

    return corners_3d

def perform_pose_estimation(frames, intrinsic, chessboard_params):
    corners_3d = get_chessboard_corners_in_3d(frames, intrinsic, chessboard_params)
    retval = {'success': False, 'transformation': None, 'points_2d': None, 'rmsd': None}

    for [found_corners, points_2d, points_3d, valid_points] in [corners_3d]:
        object_points = hf.get_chessboard_points_3D(chessboard_params)
        retval['success'] = found_corners
        retval['transformation'] = None
        retval['points_2d'] = None
        retval['rmsd'] = None

        if found_corners:
            valid_object_points = object_points[:, valid_points]
            valid_observed_object_points = points_3d[:, valid_points]

            if valid_object_points.shape[1] < 5:
                print("Not enough points have a valid depth for calculating the transformation")
            else:
                [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(
                    valid_object_points, valid_observed_object_points)
                retval['transformation'] = np.column_stack(
                    (np.row_stack((rotation_matrix, translation_vector)), [0, 0, 0, 1]))
                retval['points_2d'] = points_2d
                retval['rmsd'] = rmsd_value
                print("RMS error for calibration is:", rmsd_value, "m")

    return retval

def find_chessboard_boundary_for_depth_image(frames, chessboard_params):
    boundary = {}
    
    depth_frame = rsdm.post_process_depth_frame(frames['depth'])
    rgb_frame = frames['color']

    found_corners, points_2d = hf.cv_find_chessboard(depth_frame, rgb_frame, chessboard_params)
    boundary['device'] = [np.floor(np.amin(points_2d[0, :])).astype(int),
                          np.floor(np.amax(points_2d[0, :])).astype(int),
                          np.floor(np.amin(points_2d[1, :])).astype(int),
                          np.floor(np.amax(points_2d[1, :])).astype(int)]

    return boundary

def get_boundary_corners_2D(points):
	"""
	Get the minimum and maximum point from the array of points
	
	Parameters:
	-----------
	points 	 	 : array
						   The array of points out of which the min and max X and Y points are needed
	
	Return:
	----------
	boundary : array
		The values arranged as [minX, maxX, minY, maxY]
	
	"""
	padding=0.05
	if points.shape[0] == 3:
		assert (len(points.shape)==2)
		minPt_3d_x = np.amin(points[0,:])
		maxPt_3d_x = np.amax(points[0,:])
		minPt_3d_y = np.amin(points[1,:])
		maxPt_3d_y = np.amax(points[1,:])

		boundary = [minPt_3d_x-padding, maxPt_3d_x+padding, minPt_3d_y-padding, maxPt_3d_y+padding]

	else:
		raise Exception("wrong dimension of points!")

	return boundary

def load_settings_json(device, path_to_settings_file):
    """
    Load the settings stored in the JSON file

    """

    with open(path_to_settings_file, 'r') as file:
        json_text = file.read().strip()
    
    
    # Get the active profile and load the json file which contains settings readable by the realsense
    device = device.pipeline_profile.get_device()
    advanced_mode = rs.rs400_advanced_mode(device)
    advanced_mode.load_json(json_text)

    return device

def get_depth_to_color_extrinsics(frames):
    # Get the depth and color sensor profiles from the frames
    depth_sensor = frames['depth'].profile.as_video_stream_profile().get_sensor()
    color_sensor = frames['color'].profile.as_video_stream_profile().get_sensor()

    # Get the extrinsics from the depth sensor to the color sensor
    depth_to_color_extrinsics = depth_sensor.get_extrinsics_to(color_sensor)

    return depth_to_color_extrinsics



# def get_depth_to_color_extrinsics(device):
#     # Get the depth-to-color extrinsics of the RealSense device
#     depth_sensor = device.first_depth_sensor()
#     depth_to_color_extrinsics = depth_sensor.get_extrinsics_to(device.color_sensor)
#     return depth_to_color_extrinsics

# def main():
#     try:
#         # Initialize the RealSense pipeline
#         pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#         # Start the RealSense pipeline
#         profile = pipeline.start(config)
#         device = profile.get_device()

#         # Enable the emitter
#         enable_emitter(device, True)

#         # Load the JSON settings file
#         load_settings_json(device, "path/to/your/settings.json")

#         # Get the depth-to-color extrinsics
#         extrinsics_device = get_depth_to_color_extrinsics(device)

#         # Perform your measurement and display logic here

#     except KeyboardInterrupt:
#         print("The program was interrupted by the user. Closing the program...")

#     finally:
#         # Stop the RealSense pipeline
#         pipeline.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
