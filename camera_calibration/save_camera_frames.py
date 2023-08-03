#!/usr/bin/env python3

import rospy
import moveit_commander
import glob, os
import numpy as np
from tqdm import tqdm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
import cv2


  
##########################
# Parameters

ROBOT = "rightarm"
CAMERA_TOPIC = 'rgb_publisher/color/image'
SAVE_FOLDER = "saved_frames"
EE_FRAME = "tool0"  

#####################################

    
class SaveFrames():

    def __init__(self):

        self.camera_topic = CAMERA_TOPIC
        self.save_folder = SAVE_FOLDER

        # delete folder if exist
        if os.path.exists(self.save_folder):
            os.system("rm -rf {}".format(self.save_folder))
        os.makedirs(self.save_folder)

        rospy.wait_for_message(self.camera_topic, Image)
        print('Camera Topic : OK')

        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def save_img_and_pose(self, counter):
        
        rospy.sleep(0.5)
        data = rospy.wait_for_message(self.camera_topic, Image)
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        T = self.tfBuffer.lookup_transform("world", f"{ROBOT}_{EE_FRAME}", rospy.Time(), rospy.Duration(3.0))
        pose = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z,
                T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
              
        cv2.imwrite(os.path.join(self.save_folder, "frame_ " + str(counter) + ".png"), img)
        np.savetxt(os.path.join(self.save_folder, "frame_ " + str(counter) + ".txt"), np.array(pose).reshape(1,-1))
  
                


if __name__ == "__main__":

    rospy.init_node("calib_joints")
    
    bridge = CvBridge()

    parent_dir_path = os.path.dirname(os.path.abspath(__file__))
    joints_path = os.path.join(parent_dir_path, "calib_joints")
    group_name = "rightarm"
        
    move_group = moveit_commander.MoveGroupCommander(group_name)
    move_group.set_planner_id("PTP")
    
    s = SaveFrames()

    # load joint values
    files = sorted(glob.glob(os.path.join(joints_path, "*.txt")))
    poses = [np.loadtxt(f) for f in files]
     
    for it, joints_pose in enumerate(tqdm(poses)):

        move_group.set_joint_value_target(joints_pose)
        move_group.go(wait=True)
        move_group.stop()
        if rospy.is_shutdown():
            break
        
        rospy.sleep(0.5)
            
        s.save_img_and_pose(counter=it)