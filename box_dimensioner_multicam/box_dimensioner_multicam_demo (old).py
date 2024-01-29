###########################################################################################################################
##					  License: Apache 2.0. See LICENSE file in root directory.										 ##
###########################################################################################################################
##				  Simple Box Dimensioner with multiple cameras: Main demo file										 ##
###########################################################################################################################
## Workflow description:																								 ##
## 1. Place the calibration chessboard object into the field of view of all the realsense cameras.					   ##
##	Update the chessboard parameters in the script in case a different size is chosen.								 ##
## 2. Start the program.																								 ##
## 3. Allow calibration to occur and place the desired object ON the calibration object when the program asks for it.	##
##	Make sure that the object to be measured is not bigger than the calibration object in length and width.			##
## 4. The length, width and height of the bounding box of the object is then displayed in millimeters.				   ##
###########################################################################################################################

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
import box_dimensioner_multicam.measurement_task as mt
from collections import defaultdict
from box_dimensioner_multicam.realsense_device_manager import DeviceManager
from box_dimensioner_multicam.calibration_kabsch import PoseEstimation
from box_dimensioner_multicam.helper_functions import get_boundary_corners_2D
from box_dimensioner_multicam.measurement_task import visualise_measurements as visualise_measurements_multiple

def run_calibration():
	
	# Define some constants 
	L515_resolution_width = 640 # 1024 # pixels
	L515_resolution_height = 480 # 768 # pixels
	L515_frame_rate = 30

	resolution_width = 640 # 1280 # pixels
	resolution_height = 480 # 720 # pixels
	frame_rate = 30 # 15  # fps

	dispose_frames_for_stablisation = 30  # frames
	
	chessboard_width = 7 # squares
	chessboard_height = 10 	# squares
	square_size = 0.035 # meters

	try:
		# Enable the streams from all the intel realsense devices
		L515_rs_config = rs.config()
		L515_rs_config.enable_stream(rs.stream.depth, L515_resolution_width, L515_resolution_height, rs.format.z16, L515_frame_rate)
		L515_rs_config.enable_stream(rs.stream.infrared, 0, L515_resolution_width, L515_resolution_height, rs.format.y8, L515_frame_rate)
		L515_rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

		rs_config = rs.config()
		rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
		rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
		rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

		# Use the device manager class to enable the devices and get the frames
		device_manager = DeviceManager(rs.context(), rs_config, L515_rs_config)
		device_manager.enable_all_devices()
		
		# Allow some frames for the auto-exposure controller to stablise
		for frame in range(dispose_frames_for_stablisation):
			frames = device_manager.poll_frames()

		assert( len(device_manager._available_devices) > 0 )
		"""
		1: Calibration
		Calibrate all the available devices to the world co-ordiiamnates.
		For this purpose, a chessboard printout for use with opencv based calibration process is needed.
		
		"""
		# Get the intrinsics of the realsense device 
		intrinsics_devices = device_manager.get_device_intrinsics(frames)
		
				# Set the chessboard parameters for calibration 
		chessboard_params = [chessboard_height, chessboard_width, square_size] 
		
		# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
		calibrated_device_count = 0
		while calibrated_device_count < len(device_manager._available_devices):
			frames = device_manager.poll_frames()
			pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
			transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
			object_point = pose_estimator.get_chessboard_corners_in3d()
			calibrated_device_count = 0
			for device_info in device_manager._available_devices:
				device = device_info[0]
				if not transformation_result_kabsch[device][0]:
					print("Place the chessboard on the plane where the object needs to be detected..")
				else:
					calibrated_device_count += 1

		# Save the transformation object for all devices in an array to use for measurements
		transformation_devices={}
		chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
		for device_info in device_manager._available_devices:
			device = device_info[0]
			transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
			points3D = object_point[device][2][:,object_point[device][3]]
			points3D = transformation_devices[device].apply_transformation(points3D)
			chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )

		# Extract the bounds between which the object's dimensions are needed
		# It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
		chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
		roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

		print("Calibration completed... \nPlace the box in the field of view of the devices...")


		"""
				2: Measurement and display
				Measure the dimension of the object using depth maps from multiple RealSense devices
				The information from Phase 1 will be used here

				"""

		# Enable the emitter of the devices
		device_manager.enable_emitter(True)

		# Load the JSON settings file in order to enable High Accuracy preset for the realsense
		device_manager.load_settings_json("box_dimensioner_multicam/HighResHighAccuracyPreset.json")

		# Get the extrinsics of the device to be used later
		extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

		# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
		calibration_info_devices = defaultdict(list)
		for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
			for key, value in calibration_info.items():
				calibration_info_devices[key].append(value)

		# calculate_values(device_manager, calibration_info_devices, roi_2D)

		# return [device_manager, calibration_info_devices, roi_2D]

		# Continue acquisition until terminated with Ctrl+C by the user
		while 1:
			 # Get the frames from all the devices
				frames_devices = device_manager.poll_frames()
				# print(frames_devices)
				# exit(0)

				# Calculate the pointcloud using the depth frames from all the devices
				point_cloud = mt.calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)

				# Get the bounding box for the pointcloud in image coordinates of the color imager
				bounding_box_points_color_image, length, width, height = mt.calculate_boundingbox_points(point_cloud, calibration_info_devices )

		# 		# Draw the bounding box points on the color image and visualise the results
				mt.visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)
				
				if cv2.waitKey(1) == ord('q'):
					break

	except KeyboardInterrupt:
		print("The program was interupted by the user. Closing the program...")
	
	finally:
		device_manager.disable_streams()
		cv2.destroyAllWindows()
	
# def calculate_values(calibration_output, frame):
# 	device_manager = calibration_output[0]
# 	calibration_info_devices = calibration_output[1]
# 	roi_2D = calibration_output[2]


# 	 # Get the frames from all the devices
# 	frames_devices = device_manager.poll_frames()
# 	device = ([d[0] for (d, _ ) in frames_devices.items()])[0]

# 	# Calculate the pointcloud using the depth frames from all the devices
# 	point_cloud = calculate_cumulative_pointcloud(device, frame, calibration_info_devices, roi_2D)

# 	# Get the bounding box for the pointcloud in image coordinates of the color imager
# 	bounding_box_points_color_image, length, width, height = mt.calculate_boundingbox_points(point_cloud, calibration_info_devices)

# 	# Draw the bounding box points on the color image and visualise the results
# 	# visualise_measurements_multiple(frames_devices, bounding_box_points_color_image, length, width, height)

# 	return bounding_box_points_color_image, length, width, height, point_cloud

# def visualise_measurements(color_image, bounding_box_points, length, width, height):
# 	print(color_image)
	
# 	if (length != 0 and width !=0 and height != 0):
# 		bounding_box_points_device_upper = bounding_box_points[0:4,:]
# 		bounding_box_points_device_lower = bounding_box_points[4:8,:]
# 		box_info = "Length, Width, Height (mm): " + str(int(length*1000)) + ", " + str(int(width*1000)) + ", " + str(int(height*1000))

# 		# Draw the box as an overlay on the color image		
# 		bounding_box_points_device_upper = tuple(map(tuple,bounding_box_points_device_upper.astype(int)))
# 		for i in range(len(bounding_box_points_device_upper)):	
# 			cv2.line(color_image, bounding_box_points_device_upper[i], bounding_box_points_device_upper[(i+1)%4], (0,255,0), 4)

# 		bounding_box_points_device_lower = tuple(map(tuple,bounding_box_points_device_lower.astype(int)))
# 		for i in range(len(bounding_box_points_device_upper)):	
# 			cv2.line(color_image, bounding_box_points_device_lower[i], bounding_box_points_device_lower[(i+1)%4], (0,255,0), 1)
			
# 		cv2.line(color_image, bounding_box_points_device_upper[0], bounding_box_points_device_lower[0], (0,255,0), 1)
# 		cv2.line(color_image, bounding_box_points_device_upper[1], bounding_box_points_device_lower[1], (0,255,0), 1)
# 		cv2.line(color_image, bounding_box_points_device_upper[2], bounding_box_points_device_lower[2], (0,255,0), 1)
# 		cv2.line(color_image, bounding_box_points_device_upper[3], bounding_box_points_device_lower[3], (0,255,0), 1)
# 		cv2.putText(color_image, box_info, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )
			
# 	# Visualise the results
# 	cv2.imshow('Color image from RealSense', color_image)
# 	cv2.waitKey(1)



# def calculate_cumulative_pointcloud(device, depth_frame, calibration_info_devices, roi_2d, depth_threshold = 0.01):
# 	# Use a threshold of 5 centimeters from the chessboard as the area where useful points are found
# 	point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
# 	# Filter the depth_frame using the Temporal filter and get the corresponding pointcloud for each frame
# 	filtered_depth_frame = mt.post_process_depth_frame(depth_frame, temporal_smooth_alpha=0.1, temporal_smooth_delta=80)	
# 	point_cloud = mt.convert_depth_frame_to_pointcloud( np.asarray( filtered_depth_frame.get_data()), calibration_info_devices[device][1][rs.stream.depth])
# 	point_cloud = np.asanyarray(point_cloud)

# 	# Get the point cloud in the world-coordinates using the transformation
# 	point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)

# 	# Filter the point cloud based on the depth of the object
# 	# The object placed has its height in the negative direction of z-axis due to the right-hand coordinate system
# 	point_cloud = mt.get_clipped_pointcloud(point_cloud, roi_2d)
# 	point_cloud = point_cloud[:,point_cloud[2,:]<-depth_threshold]
# 	point_cloud_cumulative = np.column_stack( ( point_cloud_cumulative, point_cloud ) )
# 	point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
# 	return point_cloud_cumulative



if __name__ == "__main__":
	run_calibration()
