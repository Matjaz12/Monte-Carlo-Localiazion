import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_landmarks(filename):
    """
    Read landmark positions from a file
    :param filename: name of the file
    :return: array of landmark positions [[x, y]]
    """
    return np.loadtxt(filename)[:, 1:]


def read_sensor_data(filename):
    """
    Read sensor data from a file
    :param filename: name of the file
    :return: a dictionary of odometry and sensor data, 
    odometry: [[d_r1, d_t, d_r2]], sensor: [[[id, range, bearing]]]
    """
    df = pd.read_csv(filename, sep=" ", header=None)
    odom_idxs = np.where(df[0] == "ODOMETRY")[0]

    data = {"odometry": [], "sensor": []}
    for i in range(len(odom_idxs)):
        odometry = df.iloc[odom_idxs[i], 1:].to_numpy(dtype=np.float64)

        if i == len(odom_idxs) - 1:
            sensor = df.iloc[odom_idxs[i]+1:, 1:].to_numpy(dtype=np.float64)
        else:
            sensor = df.iloc[odom_idxs[i]+1:odom_idxs[i+1], 1:].to_numpy(dtype=np.float64)

        data["odometry"].append(odometry)
        data["sensor"].append(sensor)

    data["odometry"] = np.array(data["odometry"])
    data["sensor"] = np.array(data["sensor"], dtype=object)
    return data


class MCL:
    def __init__(
            self, landmarks, map_limits, alpha=[0.1, 0.1, 0.05, 0.05], 
            sigma_range=0.2, sigma_bearing=0.1, n_particles=1000
        ):

        self.landmarks = landmarks
        self.map_limits = map_limits
        self.alpha = alpha
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.n_particles = n_particles

        self._init_particles()

    def __call__(self, odometry, sensor):
        """Predict the pose of the robot given odometry and sensor data."""
        return self.predict(odometry, sensor)
    
    def predict(self, odometry, sensor):
        """Predict the pose of the robot given odometry and sensor data."""
        self._motion_model(odometry)
        self._sensor_model(sensor)
        self._resample()
        return self._mean_pose()    

    def _init_particles(self):
        """Initialize particles, each particle is given by [x, y, theta, weight]"""
        self.particles = np.zeros((self.n_particles, 4))
        self.particles[:, 0] = np.random.uniform(self.map_limits[0], self.map_limits[1], self.n_particles)
        self.particles[:, 1] = np.random.uniform(self.map_limits[2], self.map_limits[3], self.n_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.n_particles)
        self.particles[:, 3] = 1 / self.n_particles

    def _motion_model(self, odometry):
        """
        Apply the motion model to each particle
        :param odometry: odometry data [d_r1, d_t, d_r2]
        """

        d_r1, d_t, d_r2 = odometry
        a1, a2, a3, a4 = self.alpha
        
        # add noise to odometry data
        d_r1_hat = d_r1 + np.random.randn(self.n_particles) * (a1*abs(d_r1) + a2*abs(d_t))
        d_r2_hat = d_r2 + np.random.randn(self.n_particles) * (a1*abs(d_r2) + a2*abs(d_t))
        d_t_hat = d_t + np.random.randn(self.n_particles)  * (a3*abs(d_t) + a4*(abs(d_r1) + abs(d_r2)))

        # update particles
        self.particles[:, 0] += d_t_hat * np.cos(self.particles[:, 2] + d_r1_hat)
        self.particles[:, 1] += d_t_hat * np.sin(self.particles[:, 2] + d_r1_hat)
        self.particles[:, 2] += d_r1_hat + d_r2_hat

    def _sensor_model(self, sensor_data):
        """
        Compute observation likelihood of all particles, given
        the particle pose and landmark positions and sensor measurements.
        :param sensor_data: sensor data [[id, range, bearing]]
        """

        def _prob(x, mean=0, var=1): 
            return (1 / np.sqrt(2*np.pi*var)) * np.exp(-0.5*(x - mean)**2/var)

        # select landmarks which were observed by the robot
        landmarks = self.landmarks[[int(data[0]) - 1 for data in sensor_data]]

        for idx, particle in enumerate(self.particles):
            # compute the expected range and bearing between the curr. particle and landmarks
            d_hat = np.sqrt((landmarks[:, 0] - particle[0]) ** 2 + (landmarks[:, 1] - particle[1]) ** 2)
            alpha_hat = np.arctan2(landmarks[:, 1] - particle[1], landmarks[:, 0] - particle[0]) - particle[2]

            # compute log likelihood and update the weight of the curr. particle
            p_range = _prob(x=d_hat - sensor_data[:, 1], var=self.sigma_range**2)
            p_bearing = _prob(x=alpha_hat - sensor_data[:, 2], var=self.sigma_bearing**2)
            self.particles[idx, 3] = np.log(np.concatenate([p_range, p_bearing]) + 1.0).sum()
        
        # print(f"sum(w)={self.particles[idx, 3]}")            
        self.particles[:, 3] /= self.particles[:, 3].sum()

    def _resample(self):
        """Resample using stochastic universal sampling"""

        # compute the cdf
        n = self.particles.shape[0]
        sampled_idxs = []
        c_i = self.particles[:, 3][0]

        # initialize threshold
        u, i = 1 / n * np.random.rand(), 0
        for j in range(0, n):
            # skip until the threshold is reached
            while u > c_i: 
                i += 1
                c_i += self.particles[:, 3][i]

            # increment the threshold and store particle
            sampled_idxs.append(i)
            u = u + 1 / n

        assert len(sampled_idxs) == n, f"{len(sampled_idxs)} != {n} resampling failed"
        self.particles = self.particles[sampled_idxs]

    def _mean_pose(self):
        """Compute the mean pose of a particle set."""
        mean_x = self.particles[:, 0].mean()
        mean_y = self.particles[:, 1].mean()
        mean_theta = np.arctan2(np.sin(self.particles[:, 2]).mean(), np.cos(self.particles[:, 2]).mean())
        return np.array([mean_x, mean_y, mean_theta])


fig, ax = plt.subplots()
def init():
    ax.set_title("Monte Carlo Localization (MCL)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()
    return ax,


def simulate(idx):
    odom, sens = sensor_data["odometry"][idx], sensor_data["sensor"][idx]
    estimated_pose = mcl(odom, sens)
    plt.clf()
    plt.gca().set(frame_on=False)
    plt.title("Monte Carlo Localization (MCL)")
    plt.quiver(mcl.particles[:, 0], mcl.particles[:, 1], np.cos(mcl.particles[:, 2]),
               np.sin(mcl.particles[:, 2]), angles='xy', scale_units='xy', color="#3776ab")
    plt.scatter(x=landmarks[:, 0], y=landmarks[:, 1], color='black', label="landmarks", marker="o", s=60)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]),
               np.sin(estimated_pose[2]), angles='xy', scale_units='xy', color="gray", label="estimated pose")
    plt.axis(map_limits)
    plt.xticks([]); plt.yticks([])
    plt.legend()
    return ax, 


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--n_particles", type=int, default=1000, help="number of particles")
    argparse.add_argument("--sigma_range", type=float, default=0.2, help="standard deviation of the range measurement")
    argparse.add_argument("--sigma_bearing", type=float, default=0.1, help="standard deviation of the bearing measurement")
    argparse.add_argument("--world", type=str, default="./world.dat", help="world file")
    argparse.add_argument("--sensor_data", type=str, default="./sensor_data.dat", help="sensor data file")
    args = argparse.parse_args()

    landmarks = read_landmarks(args.world)
    sensor_data = read_sensor_data(args.sensor_data)
    map_limits = [-1, 12, 0, 10]

    mcl = MCL(landmarks, map_limits, n_particles=args.n_particles, sigma_range=args.sigma_range, sigma_bearing=args.sigma_bearing)
    print("Simulating the MCL...")
    sim = FuncAnimation(fig, simulate, init_func=init, frames=len(sensor_data["odometry"]), interval=20, blit=True)
    sim.save('mcl_sim.gif', writer='imagemagick')
