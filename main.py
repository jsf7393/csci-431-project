import open3d as o3d
import numpy as np
import os


def main():
    directory = "./dataset/PointClouds/"

    first = True

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    files = os.listdir(directory)

    for i in range(1, len(files)):
        prev = o3d.io.read_point_cloud(directory + files[i-1])
        tmp = o3d.io.read_point_cloud(directory + files[i])
        dist = tmp.compute_point_cloud_distance(prev)
        dist = np.asarray(dist)
        ind = np.where(dist < 0.01)[0]
        if first:
            current = tmp.select_by_index(ind, invert=True)
            vis.add_geometry(current)
            first = False
        else:
            current.points = tmp.select_by_index(ind, invert=True).points
            vis.update_geometry(current)
            vis.poll_events()
            vis.update_renderer()

    vis.destroy_window()


main()
