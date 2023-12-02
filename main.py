import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import time

bb = dict()
car_details = []
prev_centers = dict()
curr_centers = dict()


def draw_bounding_boxes_and_compute(vis, clusters, pcd):
    global prev_centers
    global curr_centers
    global car_details
    car_details = []
    if len(curr_centers) > 0:
        prev_centers = curr_centers.copy()
    for i in range(0, 6):
        details = []
        details.append(i)
        clust = np.asarray(clusters)
        ind = np.where(clust == i)[0]
        aabb = o3d.geometry.AxisAlignedBoundingBox()
        aabb = aabb.create_from_points(points=pcd.select_by_index(ind).points)
        aabb.color = (1, 0, 0)
        if i not in bb.keys():
            line_set = o3d.geometry.LineSet()
            line_set = line_set.create_from_axis_aligned_bounding_box(aabb)
            bb[i] = line_set
            vis.add_geometry(line_set)
        else:
            bb[i].points = (bb[i].create_from_axis_aligned_bounding_box(aabb)).points
            vis.update_geometry(bb[i])

        center = bb[i].get_center()
        curr_centers[i] = center
        details.append(center[0])
        details.append(center[1])
        details.append(center[2])
        if (len(prev_centers)) > 0 and len(ind) > 0:
            vectors = calculate_vectors(i)
            details.append(vectors[0])
            details.append(vectors[1])
            details.append(vectors[2])
        else:
            details.append(0)
            details.append(0)
            details.append(0)
        min_bounds = bb[i].get_min_bound()
        max_bounds = bb[i].get_max_bound()
        details.append(min_bounds[0])
        details.append(max_bounds[0])
        details.append(min_bounds[1])
        details.append(max_bounds[1])
        details.append(min_bounds[2])
        details.append(max_bounds[2])
        car_details.append(details)


def create_csv(file_num):
    """
    create a csv file for frame (called by other function (once per frame))
    :param file_num: frame number
    :return:
    """
    directory = "./perception_results"
    file_name = "frame_" + str(file_num) + ".csv"
    path = os.path.join(directory, file_name)
    header = ['vehicle_id', 'position_x', 'position_y', 'position_z', 'mvec_x', 'mvec_y', 'mvec_z', 'bbox_x_min',
              'bbox_x_max', 'bbox_y_min', 'bbox_y_max', 'bbox_z_min', 'bbox_z_max']
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(car_details)


def calculate_vectors(i):
    curr_center = curr_centers.get(i)
    prev_center = prev_centers.get(i)
    vectors = []
    if i in prev_centers.keys():
        vectors.append(curr_center[0] - prev_center[0])
        vectors.append(curr_center[1] - prev_center[1])
        vectors.append(curr_center[2] - prev_center[2])
        return vectors
    return [0, 0, 0]


def main():
    result_dir = "./perception_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    directory = "./dataset/PointClouds/"

    first = True

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Car Tracking")

    files = os.listdir(directory)
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    pcds = [o3d.io.read_point_cloud(directory + files[0])]

    times = []

    for i in range(1, len(files), 1):
        start = time.time()

        file_num = files[i-1].split(".")[0]

        prev = pcds[i-1]
        tmp = o3d.io.read_point_cloud(directory + files[i])
        pcds.append(tmp)

        dist = tmp.compute_point_cloud_distance(prev)
        dist = np.asarray(dist)
        ind = np.where(dist > 0.001)[0]

        if first:
            current = tmp.select_by_index(ind)

            current = current.remove_radius_outlier(nb_points=2, radius=2.0)[0]
            clusters = np.array(current.cluster_dbscan(eps=2.0, min_points=2))
            max_clust = 0
            if clusters.size > 0:
                max_clust = clusters.max()
            print(f"{max_clust + 1} clusters")
            colors = plt.get_cmap("tab20")(clusters / (max_clust if max_clust > 0 else 1))
            colors[clusters < 0] = 0
            current.colors = o3d.utility.Vector3dVector(colors[:, :3])

            draw_bounding_boxes_and_compute(vis, clusters, current)
            times.append(time.time() - start)
            print(times[i - 1])
            create_csv(file_num)

            vis.add_geometry(current)

            first = False
        else:
            current.points = tmp.select_by_index(ind).points

            current.points = (current.remove_radius_outlier(nb_points=2, radius=2.0)[0]).points
            clusters = np.array(current.cluster_dbscan(eps=2.0, min_points=2))
            max_clust = 0
            if clusters.size > 0:
                max_clust = clusters.max()
            print(f"{max_clust + 1} clusters")
            colors = plt.get_cmap("tab20")(clusters / (max_clust if max_clust > 0 else 1))
            colors[clusters < 0] = 0
            current.colors = o3d.utility.Vector3dVector(colors[:, :3])

            draw_bounding_boxes_and_compute(vis, clusters, current)
            times.append(time.time() - start)
            print(times[i - 1])
            create_csv(file_num)

            vis.update_geometry(current)
            vis.poll_events()
            vis.update_renderer()

    print(f"Average frame time: {sum(times) / len(times)}")

    vis.destroy_window()


main()
