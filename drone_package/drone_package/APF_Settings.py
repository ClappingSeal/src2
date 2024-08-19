import numpy as np


class APFEnvironment:
    def __init__(self, pos, min_altitude=-100, max_altitude=100):
        self.pos = np.array(pos, dtype=float)
        self.r = 20
        self.limit = 6
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

    def goal_vector(self, goal):
        goal = np.array(goal)
        return goal - self.pos

    def within_obs(self, obs_pos):
        obs_pos = [np.array(pos) for pos in obs_pos]
        within_radius = [pos for pos in obs_pos if np.linalg.norm(pos - self.pos) <= self.r]

        if not within_radius:
            within_radius = [np.array([self.pos[0], self.pos[1], self.pos[2]])]  # 로봇의 현재 위치를 포함하는 더미 항목 추가

        return within_radius

    def rev_obs_center(self, obs_pos):
        avg_vector = np.mean(self.within_obs(obs_pos), axis=0)
        return avg_vector - self.pos

    def obs_disp(self, obs_pos):
        obs_pos = [np.array(pos) for pos in obs_pos]
        within_radius = [pos for pos in obs_pos if np.linalg.norm(pos - self.pos) <= self.r]

        if not within_radius:
            return np.zeros(3)

        obs_array = np.array(within_radius)
        return np.var(obs_array, axis=0)

    def closest_obs(self, obs_pos):
        obs_pos = [np.array(pos) for pos in obs_pos]
        if not obs_pos:
            return None
        distances = [np.linalg.norm(self.pos - pos) for pos in obs_pos]
        closest_index = np.argmin(distances)
        closest_pos = obs_pos[closest_index]
        return closest_pos

    def att_force(self, goal, att_gain=1):
        goal_vector = self.goal_vector(goal)
        norm = np.linalg.norm(goal_vector)
        if norm == 0:
            return np.zeros_like(goal_vector)
        force = goal_vector / norm * att_gain
        return force

    def rep_force(self, obs_pos, rep_gain=0.1):
        obs_within_radius = obs_pos
        force = np.zeros(3)

        for obs in obs_within_radius:
            distance_vector = self.pos - obs
            distance = np.linalg.norm(distance_vector)
            if distance == 0:
                continue
            if (self.limit < distance) and (distance < self.r):
                repulsive_force_magnitude = rep_gain * ((1 / pow((distance - self.limit), 0.5)) + 2)
                repulsive_force_direction = distance_vector / distance
                force += repulsive_force_magnitude * repulsive_force_direction
            elif distance <= self.limit:
                repulsive_force_magnitude = rep_gain * 1000
                repulsive_force_direction = distance_vector / distance
                force += repulsive_force_magnitude * repulsive_force_direction

        # z축 고도 제한 적용
        if self.pos[2] >= self.max_altitude and force[2] > 0:
            force[2] = 0
        if self.pos[2] <= self.min_altitude and force[2] < 0:
            force[2] = 0

        return force

    def apf(self, goal, obs_pos):
        total_force = self.att_force(goal) + self.rep_force(obs_pos)
        norm = np.linalg.norm(total_force)

        if norm == 0:
            return np.zeros_like(total_force)

        if self.pos[2] > self.max_altitude and total_force[2] > 0:
            total_force[2] = 0

        elif self.pos[2] < self.min_altitude and total_force[2] < 0:
            total_force[2] = 0

        if norm > 1:
            total_force = total_force / norm

        return total_force * 0.1

    def apf_rev_rotate(self, goal, obs_pos):
        goal = np.array(goal)
        apf_vector = self.att_force(goal) + self.rep_force(obs_pos)

        # XY plane rotation
        angle_xy = np.pi / 2 - np.arctan2(apf_vector[1], apf_vector[0])
        rotation_matrix_xy = np.array([
            [np.cos(angle_xy), -np.sin(angle_xy), 0],
            [np.sin(angle_xy), np.cos(angle_xy), 0],
            [0, 0, 1]
        ])

        # YZ plane rotation
        angle_yz = np.pi / 2 - np.arctan2(apf_vector[2], apf_vector[1])
        rotation_matrix_yz = np.array([
            [1, 0, 0],
            [0, np.cos(angle_yz), -np.sin(angle_yz)],
            [0, np.sin(angle_yz), np.cos(angle_yz)]
        ])

        # ZX plane rotation
        angle_zx = np.pi / 2 - np.arctan2(apf_vector[0], apf_vector[2])
        rotation_matrix_zx = np.array([
            [np.cos(angle_zx), 0, np.sin(angle_zx)],
            [0, 1, 0],
            [-np.sin(angle_zx), 0, np.cos(angle_zx)]
        ])

        rotation_matrix = rotation_matrix_xy @ rotation_matrix_yz

        rev_att_vector = self.att_force(goal)
        rev_rep_vector = self.rep_force(obs_pos)
        closest_obs_pos = self.closest_obs(obs_pos)

        rot_rev_att_vector = np.dot(rotation_matrix, rev_att_vector.T)
        rot_rev_rep_vector = np.dot(rotation_matrix, rev_rep_vector.T)

        if closest_obs_pos is not None:
            rev_closest_obs_pos = closest_obs_pos - self.pos
            rot_rev_closest_obs_pos = np.dot(rotation_matrix, rev_closest_obs_pos.T)
            rot_rev_closest_obs_pos = 1 / rot_rev_closest_obs_pos
        else:
            rot_rev_closest_obs_pos = np.array([0, 0, 0])

        return rot_rev_att_vector, rot_rev_rep_vector, rot_rev_closest_obs_pos

    def apf_inverse_rotate(self, goal, obs_pos, b_vector):
        goal = np.array(goal)
        apf_vector = self.att_force(goal) + self.rep_force(obs_pos)

        # XY plane rotation
        angle_xy = np.pi / 2 - np.arctan2(apf_vector[1], apf_vector[0])
        rotation_matrix_xy = np.array([
            [np.cos(angle_xy), -np.sin(angle_xy), 0],
            [np.sin(angle_xy), np.cos(angle_xy), 0],
            [0, 0, 1]
        ])

        # YZ plane rotation
        angle_yz = np.pi / 2 - np.arctan2(apf_vector[2], apf_vector[1])
        rotation_matrix_yz = np.array([
            [1, 0, 0],
            [0, np.cos(angle_yz), -np.sin(angle_yz)],
            [0, np.sin(angle_yz), np.cos(angle_yz)]
        ])

        # ZX plane rotation
        angle_zx = np.pi / 2 - np.arctan2(apf_vector[0], apf_vector[2])
        rotation_matrix_zx = np.array([
            [np.cos(angle_zx), 0, np.sin(angle_zx)],
            [0, 1, 0],
            [-np.sin(angle_zx), 0, np.cos(angle_zx)]
        ])

        rotation_matrix = rotation_matrix_xy @ rotation_matrix_yz

        inverse_rotation_matrix = rotation_matrix.T
        inverse_rotated_b_vector = np.dot(inverse_rotation_matrix, b_vector)

        return inverse_rotated_b_vector

    def apf_drl(self, goal, obs_pos, a, b):
        total_force = 2 * (a * self.att_force(goal) + (1 - a) * self.rep_force(obs_pos)) + b
        norm2 = np.linalg.norm(total_force)

        if norm2 > 1:
            total_force = total_force / norm2

        if norm2 == 0:
            return np.zeros_like(total_force)

        if self.pos[2] > self.max_altitude and total_force[2] > 0:
            total_force[2] = 0

        elif self.pos[2] < self.min_altitude and total_force[2] < 0:
            total_force[2] = 0

        return total_force * 0.1

# env = APFEnvironment([1, 1, 10])
# print(env.apf([10, 0, 8], [[40, 1, 12], [10, 20, 14]]))
# print(env.apf_rev_rotate([10, 0, 8], [[40, 1, 12], [10, 20, 14]]))

env = APFEnvironment(current_position)
next_position = current_postion + np.array(env.apf(self.goal_position, self.other_drones_positions))
self.goto(next_position[0], next_position[1], next_position[2])