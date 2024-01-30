#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from tf_transformations import euler_from_quaternion
import numpy as np


class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        self.init_params()

        self.add_on_set_parameters_callback(callback=self.on_params_changed)

        self.create_pub_sub_timer()
        
        self.setpoint = Point()
        self.setpoint_timed_out = True
        
        self.pose_counter = 0

        self.errors_prev = 0
        self.t_prev = None

        self.P = self.I = self.D = np.zeros(3)

    def init_params(self):
        self.declare_parameters(namespace='',
                                parameters=[('K_P', rclpy.Parameter.Type.DOUBLE_ARRAY),
                                            ('K_I', rclpy.Parameter.Type.DOUBLE_ARRAY),
                                            ('K_D', rclpy.Parameter.Type.DOUBLE_ARRAY)
                                ])
        
        _K_P = self.get_parameter('K_P').get_parameter_value().double_array_value
        _K_I = self.get_parameter('K_I').get_parameter_value().double_array_value
        _K_D = self.get_parameter('K_D').get_parameter_value().double_array_value
        
        self.K_P = np.array([[_K_P[0], 0.0, 0.0],
                             [0.0, _K_P[1], 0.0],
                             [0.0, 0.0, _K_P[2]]])
        
        self.K_I = np.array([[_K_I[0], 0.0, 0.0],
                             [0.0, _K_I[1], 0.0],
                             [0.0, 0.0, _K_I[2]]])
        
        self.K_D = np.array([[_K_D[0], 0.0, 0.0],
                             [0.0, _K_D[1], 0.0],
                             [0.0, 0.0, _K_D[2]]])
        
    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            if param.name == 'K_P':
                _K_P = self.get_parameter('K_P').get_parameter_value().double_array_value
                self.K_P = np.array([[_K_P[0], 0.0, 0.0],
                                     [0.0, _K_P[1], 0.0],
                                     [0.0, 0.0, _K_P[2]]])
            elif param.name == 'K_I':
                _K_I = self.get_parameter('K_I').get_parameter_value().double_array_value
                self.K_I = np.array([[_K_I[0], 0.0, 0.0],
                                     [0.0, _K_I[1], 0.0],
                                     [0.0, 0.0, _K_I[2]]])
            elif param.name == 'K_D':
                _K_D = self.get_parameter('K_D').get_parameter_value().double_array_value
                self.K_D = np.array([[_K_D[0], 0.0, 0.0],
                                     [0.0, _K_D[1], 0.0],
                                     [0.0, 0.0, _K_D[2]]])
            else:
                return SetParametersResult(successful=False, reason='Parameter name not found')
            
        self.I = 0
        
        return SetParametersResult(successful=True, reason='Parameter set')

    def create_pub_sub_timer(self):
        self.thrust_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='thrust_setpoint', 
                                                qos_profile=1)
        
        self.position_setpoint_sub = self.create_subscription(msg_type=PointStamped, 
                                                              topic='~/setpoint', 
                                                              callback=self.on_position_setpoint, 
                                                              qos_profile=1)
        
        self.pose_sub = self.create_subscription(msg_type=PoseWithCovarianceStamped,
                                                 topic='vision_pose_cov',
                                                 callback=self.on_pose, 
                                                 qos_profile=1)
        
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        
    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('setpoint timed out. waiting for new setpoints.')
        self.setpoint_timed_out = True

    def on_position_setpoint(self, msg: PointStamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = msg.point

    def on_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.apply_control(position, yaw)

    def apply_control(self, position: Point, yaw: float):
        t = self.get_clock().now().nanoseconds * 1e-9
        errors = np.array([self.setpoint.x - position.x, self.setpoint.y - position.y, self.setpoint.z - position.z])

        self.P = errors

        if self.t_prev is not None:
            self.I += errors * (t - self.t_prev)
            self.D = (errors - self.errors_prev) / (t - self.t_prev)

        u = self.K_P @ self.P + self.K_I @ self.I + self.K_D @ self.D

        self.get_logger().info(f'u: {u}')

        msg = ActuatorSetpoint()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.x = math.cos(-yaw) * u[0] - math.sin(-yaw) * u[1]
        msg.x = min(0.5, max(-0.5, msg.x))
        msg.y = math.sin(-yaw) * u[0] + math.cos(-yaw) * u[1]
        msg.y = min(0.5, max(-0.5, msg.y))
        msg.z = u[2]

        self.thrust_pub.publish(msg)

        self.errors_prev = errors
        self.t_prev = t



def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
