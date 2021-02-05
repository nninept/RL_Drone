import setup_path 
import airsim
import math
import numpy as np
import os
import tempfile
import pprint
import cv2
import time
import sys

class DroneEnvClass:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def step(self, state):
        self.client.enableApiControl(True)
        
        #state = 1/(1+np.exp(-state))

        front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm = state[0].item(),state[1].item(),state[2].item(),state[3].item()
        #self.client.simPause(False)
        self.client.moveByMotorPWMsAsync(front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm, duration = 0.2).join()  #roll, pitch, yaw, throttle, duration / angle -> radians, throttle -> 0~1
        #self.client.simPause(True)
        
        state = self.client.getMultirotorState()
        done = self.client.simGetCollisionInfo().has_collided
        objectPose = self.client.simGetObjectPose('BP_Boost_Ring_C_13')
        cur_state = np.array([
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
            state.kinematics_estimated.angular_acceleration.x_val,
            state.kinematics_estimated.angular_acceleration.y_val,
            state.kinematics_estimated.angular_acceleration.z_val,
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val])
            #objectPose.position.x_val,
            #objectPose.position.y_val,
            #objectPose.position.z_val
        #reward = 1000/(np.abs(state.kinematics_estimated.position.x_val-objectPose.position.x_val) + \
                    #np.abs(state.kinematics_estimated.position.y_val-objectPose.position.y_val))
        #reward = -(np.abs(state.kinematics_estimated.position.x_val-objectPose.position.x_val) + \
                        #np.abs(state.kinematics_estimated.position.y_val-objectPose.position.y_val))
        
        #reward=1/np.abs((100-state.kinematics_estimated.position.z_val))
        reward = 1
        #print(state.kinematics_estimated.position.x_val,
            #objectPose.position.x_val,state.kinematics_estimated.position.y_val,objectPose.position.y_val)
        
        if(done):
            print("추락")
            reward = -100
        elif(airsim.LandedState.Landed == state.landed_state):
            print("추락")
            reward = -100
            done = True
        elif(state.kinematics_estimated.position.z_val <= -100):
            print("고도 제한", state.kinematics_estimated.position.z_val)
            reward = -100
            done = True
        return cur_state, reward, done

    def reset(self):
        self.client.reset()
        #time.sleep(1)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        state = self.client.getMultirotorState()
        objectPose = self.client.simGetObjectPose('BP_Boost_Ring_C_13')
        cur_state = np.array([
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
            state.kinematics_estimated.angular_acceleration.x_val,
            state.kinematics_estimated.angular_acceleration.y_val,
            state.kinematics_estimated.angular_acceleration.z_val,
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val])
            #objectPose.position.x_val,
            #objectPose.position.y_val,
            #objectPose.position.z_val
        
        return cur_state
    
'''
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()

lidar_data = client.getLidarData("Lidar1")
s = pprint.pformat(lidar_data)
print("lidar_data: %s" % s)

lidar_seg_data = client.simGetLidarSegmentation("Lidar1")
s = pprint.pformat(lidar_seg_data)
print("lidar_seg_data: %s" % s)

distance_data = client.getDistanceSensorData("Distance")
s = pprint.pformat(distance_data)
print("distance_data: %s" % s)

airsim.wait_key('Press value roll,pitch,yaw,throttle')
#
for i in range(60):
    client.moveByMotorPWMsAsync(0.8, 0.8, 0.8, 0.3, 0.1).join()
    collision = client.simGetCollisionInfo().has_collided
    print(i, collision)
    if(collision == True):
        break

#client.hoverAsync().join()

state = client.getMultirotorState().position.angular_velocity
print("state: %s" % pprint.pformat(state))
'''
################################################

'''
airsim.wait_key('Press any key to take images')
# get camera images from the car
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
    airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
    airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
    airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
print('Retrieved images: %d' % len(responses))

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
'''
'''
airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
'''