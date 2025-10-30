#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3, SO2
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        # todo HW02: implement fk for the flange

        #theory is for 
        # R-first rotate for q[i]parameter - variable angle
        #         then translate for link_parameters[i] - constant length
        # P-first rotate for link_parameters[i] - constant angle
        #         then translate for q[i]parameter - variable length
        ret = SE2()
        ret.rotation = self.base_pose.rotation
        ret.translation = self.base_pose.translation.copy()
        for i in range(len(self.q)):
            if self.structure[i] == 'R':
                ret *= SE2(rotation=self.q[i]) * SE2(translation=[self.link_parameters[i], 0])
            else:  # self.structure[i] == "P":
                ret *= SE2(rotation=self.link_parameters[i]) * SE2(translation=[self.q[i], 0])
        return ret

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = []
        rot = self.base_pose.rotation
        trans = self.base_pose.translation.copy()
        frames.append(SE2(rotation=rot, translation=trans.copy()))
        for i in range(len(self.q)):
            if self.structure[i] == 'R':
                rot = rot * SO2(self.q[i])
                trans = trans + rot.act([self.link_parameters[i], 0])

            else: #if self.structure[i] == 'P':
                rot = rot * SO2(self.link_parameters[i])
                trans = trans + rot.act([self.q[i], 0])
            
            frames.append(SE2(rotation=rot, translation=trans.copy()))
        
        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation
        
        print("")
        print("Computing analytical jacobian")

        n = len(self.q)

        #precompute cumulative angles
        theta_base = self.base_pose.rotation.angle
        theta_star = np.zeros(n)  #cumulative angle
        for i in range(n):
            theta_star[i] = theta_base
            for j in range(i+1):
                if self.structure[j] == 'R':
                    theta_star[i] += self.q[j]
                elif self.structure[j] == 'P':
                    theta_star[i] += self.link_parameters[j]

        
        for i in range(n):
            #declare partial derivatives
            dx_dqi = 0.0
            dy_dqi = 0.0
            dtheta_dqi = 0.0
            #compute partial derivatives based on joint type
            if self.structure[i] == 'R':
                dtheta_dqi = 1.0
                for j in range(i, n):
                    dx_dqi += -self.link_parameters[j] * np.sin(theta_star[j])
                    dy_dqi += self.link_parameters[j] * np.cos(theta_star[j])
            elif self.structure[i] == 'P':
                dtheta_dqi = 0.0
                dx_dqi = np.cos(theta_star[i])
                dy_dqi = np.sin(theta_star[i])

            
            else:
                raise ValueError("Unknown joint type in structure")
            
            jac[0, i] = dx_dqi
            jac[1, i] = dy_dqi
            jac[2, i] = dtheta_dqi
        
        #print(jac)
        for i in range(n):
            print(f"  MY_Joint {i}: Pinocchio Jacobian column: {jac[:, i]}  , Joint type: {self.structure[i]}")

        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation
        print("Computing finite difference jacobian")
        print("")
        n = len(self.q)
        q_copy = self.q.copy() #backup
        dq = self.q.dtype.type(delta)

        for i in range(n):
            #restore q+ a q-
            q_plus, q_minus = q_copy.copy(), q_copy.copy()

            #move only ith joint by +dq and -dq
            q_plus[i] += dq
            q_minus[i] -= dq

            #compute poses for q+ and q-
            #positive perturbation
            self.set_configuration(q_plus)
            pose_plus = self.flange_pose()
            #negative perturbation
            self.set_configuration(q_minus)
            pose_minus = self.flange_pose()

            #central difference
            jac[0, i] = (pose_plus.translation[0] - pose_minus.translation[0]) / (2 * dq) #dx/dqi
            jac[1, i] = (pose_plus.translation[1] - pose_minus.translation[1]) / (2 * dq) #dy/dqi
            jac[2, i] = (pose_plus.rotation.angle - pose_minus.rotation.angle) / (2 * dq) #dtheta/dqi

        #restore q
        self.set_configuration(q_copy)

        #print(jac)
        for i in range(n):
            print(f"Joint {i}: dx/dq={jac[0,i]}, dy/dq={jac[1,i]}, dtheta/dq={jac[2,i]}")

        return jac







    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW05 implement numerical IK

        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW05 implement analytical IK for RRR manipulator
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            pass
        return []

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
