#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        # todo HW01: implement Rodrigues' formula, t.rot = ...
        angle = np.linalg.norm(v)
        if np.isclose(angle, 0.0):
            t.rot = np.eye(3)
        else:
            v_norm = v/angle #must normalize the vector for rodrigues
            skew_sym_matrix: np.ndarray = np.array([
                [0, -v_norm[2], v_norm[1]],
                [v_norm[2], 0, -v_norm[0]],
                [-v_norm[1], v_norm[0], 0]
            ])
            t.rot = np.eye(3) + np.sin(angle)*skew_sym_matrix + (1 - np.cos(angle))*(skew_sym_matrix@skew_sym_matrix)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        # todo HW01: implement computation of rotation vector from this SO3
        if np.allclose(self.rot, np.eye(3)):
            v = np.zeros(3)
            angle = 0
        elif np.isclose(np.trace(self.rot), -1.0):
            angle = np.pi
            if (self.rot[2][2] != -1):
                v = np.array([self.rot[0][2], self.rot[1][2], 1+self.rot[2][2]]) * (1/(np.sqrt(2*(1+self.rot[2][2]))))
            elif (self.rot[1][1] != -1):
                v = np.array([self.rot[0][1], 1+self.rot[1][1], self.rot[2][1]]) * (1/(np.sqrt(2*(1+self.rot[1][1]))))
            elif (self.rot[0][0] != -1):
                v = np.array([1+self.rot[0][0], self.rot[1][0], self.rot[2][0]]) * (1/(np.sqrt(2*(1+self.rot[0][0]))))
            else:
                print("Error: no if executed")
                v = np.zeros(3)
        else:
            angle = np.arccos((1/2)*(np.trace(self.rot)-1))
            skew_sym_matrix: np.ndarray = np.array(1/(2*np.sin(angle))*(self.rot - self.rot.T))
            v = np.array([skew_sym_matrix[2][1], skew_sym_matrix[0][2], skew_sym_matrix[1][0]])
        return angle*v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        res = SO3()
        res.rot = self.rot @ other.rot
        return res

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        # todo: HW01: implement inverse, do not use np.linalg.inverse()
        res = SO3()
        res.rot = self.rot.T
        return res

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        # todo: HW1opt: implement rx
        res_x = SO3()
        res_x.rot = np.array([
            [1,0,0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        return res_x
        #raise NotImplementedError("RX needs to be implemented.")

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # todo: HW1opt: implement ry
        res_y = SO3()
        res_y.rot = np.array([
            [np.cos(angle),0,np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        return res_y
        #raise NotImplementedError("RY needs to be implemented.")

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # todo: HW1opt: implement rz
        res_z = SO3()
        res_z.rot = np.array([
            [np.cos(angle),-np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return res_z
        #raise NotImplementedError("RZ needs to be implemented.")

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternion
        #TODO this is not ready yet
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]
        res_fq = SO3()
        angle = 2*np.arccos(qw)
        v = np.array([qx, qy, qz])
        v_norm = v/np.linalg.norm(v)
        skew_sym_matrix: np.ndarray = np.array([
            [0, -v_norm[2], v_norm[1]],
            [v_norm[2], 0, -v_norm[0]],
            [-v_norm[1], v_norm[0], 0]
        ])
        res_fq.rot = np.array([skew_sym_matrix[2][1], skew_sym_matrix[0][2], skew_sym_matrix[1][0]])
        return res_fq
        #raise NotImplementedError("From quaternion needs to be implemented")

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # todo: HW1opt: implement to quaternion
        qw = (1/2)*np.sqrt(1+np.trace(self.rot))
        qx = (1/(4*qw))*(self.rot[2][1] - self.rot[1][2])
        qy = (1/(4*qw))*(self.rot[0][2] - self.rot[2][0])
        qz = (1/(4*qw))*(self.rot[1][0] - self.rot[0][1])
        res_q: np.ndarray = np.array([qx, qy, qz, qw])
        return res_q
        #raise NotImplementedError("To quaternion needs to be implemented")

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        res = SO3()
        axis = np.asarray(axis)
        angle = angle
        #here do skewsym -> rodrigues

        return res
        #raise NotImplementedError("Needs to be implemented")

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # todo: HW1opt: implement to angle axis
        v = self.log(self)
        angle = np.linalg.norm(v)
        v_norm: np.ndarray = v/angle
        return [angle, v_norm]


        #raise NotImplementedError("Needs to be implemented")

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        raise NotImplementedError("Needs to be implemented")

    def __hash__(self):
        return id(self)
