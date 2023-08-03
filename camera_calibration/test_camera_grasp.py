#!/usr/bin/env python3

import rospy
import cv2, copy
import numpy as np
from sklearn.decomposition import PCA

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation
import tf2_ros
import moveit_commander

from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperAction, CommandRobotiqGripperGoal
from control_msgs.msg import FollowJointTrajectoryActionFeedback

import actionlib


class Tester():


    def __init__(self, cam_frame = "rightarm_oak_rgb_camera_frame", ee_frame = "rightarm_hande_tip"):
        
        self.cam_frame = cam_frame
        self.ee_frame = ee_frame

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/rgb_publisher/color/image', Image, self.image_callback)
        self.image = None

        self.hsv_lower = np.array([26, 50, 70])
        self.hsv_higher = np.array([86, 255, 255])

        self.move_group = moveit_commander.MoveGroupCommander("rightarm") 
        self.move_group.set_planner_id("LIN")
        
        self.arm_feedback = rospy.Subscriber('/rightarm/scaled_pos_joint_traj_controller/follow_joint_trajectory/feedback', 
                                            FollowJointTrajectoryActionFeedback, self.feedback_callback)


        print("move group ok!")


        self.joints_homing = np.array([-4.7312509457217615, -1.5177128950702112, 1.1539268493652344, 
                                    -1.201845947896139, -1.5522459189044397, -1.549436394368307])
        self.homing()

        # get camera info
        cam_info = rospy.wait_for_message('/rgb_publisher/color/camera_info', CameraInfo)
        self.K = np.array(cam_info.K).reshape(3,3)
        self.D = np.array(cam_info.D)[:5]
        self.H = cam_info.height
        self.W = cam_info.width
        
        # camera tf
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.gripper_client = actionlib.SimpleActionClient('/command_robotiq_action', CommandRobotiqGripperAction)       
        


    def feedback_callback(self, data):
        self.goal_status = data.status.status
        print(self.goal_status)
        
    def homing(self) :
        print('homing:')
        self.move_group.go(self.joints_homing, wait=True)       
        self.move_group.stop()
        print('homing: OK')

    def go_to_pose_goal(self, target_pose):
        self.move_group.set_pose_target(target_pose, self.ee_frame)
        self.move_group.go(wait = True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def image_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def background_segmentation(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_higher)
        return cv2.bitwise_not(mask)

    def get_xyz(self, px, py, depth, camera_matrix):
        x = depth * (px - camera_matrix[0,2]) / camera_matrix[0,0]
        y = depth * (py - camera_matrix[1,2]) / camera_matrix[1,1] 
        return x, y, depth
    
    def from_pose_to_matrix(self, pose):
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat([pose[-4], pose[-3], pose[-2], pose[-1]]).as_matrix()
        T[:3,3] = np.array([pose[0], pose[1], pose[2]])
        return T 



    def run(self):

        self.homing()

        if self.image is not None:
            mask = self.background_segmentation()

            white_pixels = np.where(mask == 255)          
            X = np.array([white_pixels[0], white_pixels[1]]).T
            pca = PCA(n_components=2)
            pca.fit(X)

            mean_pos = pca.mean_
            main_axis = pca.components_[0]
            angle = np.arctan2(main_axis[1], main_axis[0])

            if False:
                import matplotlib.pyplot as plt
                def draw_vector(v0, v1, ax=None):
                    ax = ax or plt.gca()
                    arrowprops=dict(arrowstyle='->',
                                    linewidth=2,
                                    shrinkA=0, shrinkB=0)
                    ax.annotate('', v1, v0, arrowprops=arrowprops)

                # plot data
                plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
                for length, vector in zip(pca.explained_variance_, pca.components_):
                    v = vector * 3 * np.sqrt(length)
                    draw_vector(pca.mean_, pca.mean_ + v)
                plt.axis('equal')
                plt.show()


            print("angle: ", angle, np.degrees(angle))

            # world coordinates (camera frame)
            T = self.tfBuffer.lookup_transform("world", self.cam_frame, rospy.Time(), rospy.Duration(3.0))
            pose = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z,
            T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
            T = self.from_pose_to_matrix(pose)

            x, y, z = self.get_xyz(mean_pos[1], mean_pos[0], depth=T[2,3], camera_matrix=self.K)


            T_object = np.eye(4)
            T_object[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
            T_object[:3, 3] = np.array([x, y, z])

            
            
            # object pose (world frame)
            T_world = np.matmul(T, T_object)
            quat = Rotation.from_matrix(T_world[:3, :3]).as_quat()
            xyz = T_world[:3, 3]  
            xyz[2] = max(0.03, xyz[2])   

            pose = Pose(Point(xyz[0], xyz[1], xyz[2]), Quaternion(quat[0], quat[1], quat[2], quat[3]))
            

            print('moving to top')
            print("pose: ", pose.position)            

            pose2 = copy.deepcopy(pose)
            pose2.position.z = 0.1
            self.go_to_pose_goal(pose2)


            # open gripper
            print('opening gripper')
            goal = CommandRobotiqGripperGoal()
            goal.position = 0.30
            goal.speed = 0.1
            goal.force = 0.1
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result()


            # move to object
            print('moving to object')
            print(pose.position)
            self.go_to_pose_goal(pose)

            # close gripper
            print('closing gripper')
            goal = CommandRobotiqGripperGoal()
            goal.position = 0.0
            goal.speed = 0.1
            goal.force = 0.1
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result()

            self.go_to_pose_goal(pose2)


            # open gripper
            print('opening gripper')
            goal = CommandRobotiqGripperGoal()
            goal.position = 0.30
            goal.speed = 0.1
            goal.force = 0.1
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result()

            

if __name__ == "__main__":

    rospy.init_node("test_camera_grasp_node")

    tester = Tester()

    while not rospy.is_shutdown():
        tester.run()

        quit()