#!/usr/bin/env python3

import os, cv2
import numpy as np
import rospy, tf2_ros
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge


##########################
# Parameters

ROBOT = "ur5_"
SAVE_JOINTS = False
CAMERA_TOPIC = 'oak_camera/rgb'
SAVE_FOLDER = "/home/lar/ros/uc1_ws/src/camera_calibration/right_arm26-07"
EE_FRAME = ROBOT + "flange"
JOINTS_TOPIC = f"{ROBOT}/joint_states"

##########################

joint_names_sequence = [f"{ROBOT}_shoulder_pan_joint", 
                        f"{ROBOT}_shoulder_lift_joint", 
                        f"{ROBOT}_elbow_joint", 
                        f"{ROBOT}_wrist_1_joint",
                        f"{ROBOT}_wrist_2_joint", 
                        f"{ROBOT}_wrist_3_joint"]
                        
class SaveFrames():

    def __init__(self):

        self.camera_topic = CAMERA_TOPIC
        self.joints_topic = JOINTS_TOPIC
        self.save_folder = SAVE_FOLDER
        self.ee_frame = EE_FRAME

        # delete folder if exist
        if os.path.exists(self.save_folder):
            os.system("rm -rf {}".format(self.save_folder))
        os.makedirs(self.save_folder)

        rospy.wait_for_message(self.camera_topic, Image)
        print('Camera Topic : OK')

        # rospy.wait_for_message(self.joints_topic, JointState)
        # print('Joints Topic : OK')

        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def save_img_and_pose(self, counter):
        
        rospy.sleep(0.5)
        data = rospy.wait_for_message(self.camera_topic, Image)
        img = cv2.cvtColor(self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)

        try:
            T = self.tfBuffer.lookup_transform("world", self.ee_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        pose = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z,
                T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
        
        # data_joint = rospy.wait_for_message(self.joints_topic, JointState)

        # joint_values = [None for _ in joint_names_sequence]
        # for i, name in enumerate(data_joint.name):
        #     if name in joint_names_sequence: #and ROBOT in name:
        #         joint_values[joint_names_sequence.index(name)] = data_joint.position[i]


        cv2.imwrite(os.path.join(self.save_folder, "frame_ " + str(counter) + ".png"), img)
        np.savetxt(os.path.join(self.save_folder, "frame_ " + str(counter) + ".txt"), np.array(pose).reshape(1,-1))
        # np.savetxt(os.path.join(self.save_folder, "joints_ " + str(counter) + ".txt"), np.array(joint_values).reshape(1,-1))


if __name__ == '__main__' :

    rospy.init_node('save_camera_frames')

    #######################################
    # Init
    ####################################### 
    f = SaveFrames()    

    counter = 0
    while not rospy.is_shutdown():
        input("Press ENTER to save frame {}".format(counter))
        f.save_img_and_pose(counter)
        counter += 1
