import open3d as o3d
import numpy as np
import os


def main():
    directory = "./dataset/PointClouds"
    #
    # source = o3d.io.read_point_cloud(directory + "/0.pcd")

    first = True
    pcd = o3d.geometry.PointCloud()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for filename in os.scandir(directory):
        if first:
            pcd = o3d.io.read_point_cloud(filename.path)
            vis.add_geometry(pcd)
            first = False
        else:
            pcd.points = o3d.io.read_point_cloud(filename.path).points
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
    vis.destroy_window()


main()
