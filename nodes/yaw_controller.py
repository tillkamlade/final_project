#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion


class YawController(Node):

    def __init__(self):
        super().__init__(node_name='yaw_controller')

        self.init_params()

        self.add_on_set_parameters_callback(callback=self.on_params_changed)

        self.create_pub_sub_timer()

        # default value for the yaw setpoint
        self.setpoint = math.pi / 2.0
        self.setpoint_timed_out = True

        self.P = self.I = self.D = 0

        self.t_prev = None
        self.error_prev = 0

    def init_params(self):
        self.declare_parameters(namespace='',
                                parameters=[('K_P', rclpy.Parameter.Type.DOUBLE),
                                            ('K_I', rclpy.Parameter.Type.DOUBLE),
                                            ('K_D', rclpy.Parameter.Type.DOUBLE)])

        self.K_P = self.get_parameter('K_P').get_parameter_value().double_value
        self.K_I = self.get_parameter('K_I').get_parameter_value().double_value
        self.K_D = self.get_parameter('K_D').get_parameter_value().double_value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            if param.name == 'K_P':
                self.K_P = param.value
            elif param.name == 'K_I':
                self.K_I = param.value
            elif param.name == 'K_D':
                self.K_D = param.value
            else:
                return SetParametersResult(successful=False, reason='Parameter name not found')
        return SetParametersResult(successful=True, reason='Parameter set')

    def create_pub_sub_timer(self):
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)
        
        self.torque_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='torque_setpoint',
                                                qos_profile=1)
        
        self.vision_pose_sub = self.create_subscription(msg_type=PoseWithCovarianceStamped,
                                                        topic='vision_pose_cov',
                                                        callback=self.on_vision_pose,
                                                        qos_profile=qos)
        
        self.setpoint_sub = self.create_subscription(msg_type=Float64Stamped,
                                                     topic='~/setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=qos)
        
        self.timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)

    def on_setpoint_timeout(self):
        self.timeout_timer.cancel()
        self.get_logger().warn('Setpoint timed out. Waiting for new setpoints')
        self.setpoint_timed_out = True

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def on_setpoint(self, msg: Float64Stamped):
        self.timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        self.setpoint_timed_out = False
        self.setpoint = self.wrap_pi(msg.data)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        if self.setpoint_timed_out:
            return
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert the quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        #yaw = self.wrap_pi(yaw)

        control_output = self.compute_control_output(yaw)
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.publish_control_output(control_output, timestamp)

    def compute_control_output(self, yaw):
        # very important: normalize the angle error!
        t = self.get_clock().now().nanoseconds * 1e-9
        error = self.wrap_pi(self.setpoint - yaw)

        self.P = error
        
        if self.t_prev is not None:
            self.I += error * (t - self.t_prev)
            self.D = (error - self.error_prev) / (t - self.t_prev)

        u = self.K_P * self.P + self.K_I * self.I + self.K_D * self.D

        self.t_prev = t
        self.error_prev = error

        return u

    def publish_control_output(self, control_output: float,
                               timestamp: rclpy.time.Time):
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.ignore_x = True
        msg.ignore_y = True
        msg.ignore_z = False  # yaw is the rotation around the vehicle's z axis

        msg.z = control_output
        self.torque_pub.publish(msg)


def main():
    rclpy.init()
    node = YawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
