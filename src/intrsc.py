"""Imago intersections module."""

from math import cos, tan, pi
from operator import itemgetter
import colorsys

from PIL import ImageDraw

import filters
import k_means
import output
import linef
import cv2
import numpy as np

def dst(line):
    """Return normalized line."""
    if line[0] < pi / 2:
        line = line[0] + pi, - line[1]
    return line

def dst_sort(lines):
    """Return lines sorted by distance."""
    l_max = max(l[0] for l in lines)
    l_min = min(l[0] for l in lines)
    if l_max - l_min > (3. / 4) * pi:
        lines = [dst(l) for l in lines]
    lines.sort(key=itemgetter(1))
    return lines

def b_intersects(image, lines, show_all, do_something, logger):
    """Compute intersections."""
    # TODO refactor show_all, do_something
    # TODO refactor this into smaller functions
    logger("finding the stones")
    lines = [dst_sort(l) for l in lines]
    an0 = (sum([l[0] for l in lines[0]]) / len(lines[0]) - pi / 2)
    an1 = (sum([l[0] for l in lines[1]]) / len(lines[1]) - pi / 2)
    if an0 > an1:
        lines = [lines[1], lines[0]]

    intersections = intersections_from_angl_dist(lines, image.size)

    if show_all:
        image_g = image.copy()
        draw = ImageDraw.Draw(image_g)
        for line in intersections:
            for (x, y) in line:
                draw.point((x , y), fill=(120, 255, 120))
        do_something(image_g, "intersections")

    return intersections

def board(image, intersections, show_all, do_something, logger):
    """Find stone colors and return board situation."""

#    image_c = filters.color_enhance(image)
#    if show_all:
#        do_something(image_c, "white balance")

    image_c = image
    
    board_raw = []
    
    for line in intersections:
        board_raw.append([stone_color_raw(image_c, intersection) for intersection in
                      line])
    board_raw = sum(board_raw, [])

    ### Show color distribution

    if show_all:
        import matplotlib.pyplot as pyplot
        from PIL import Image
        fig = pyplot.figure(figsize=(8, 6))
        luma = [s[0] for s in board_raw]
        saturation = [s[1] for s in board_raw]
        pyplot.scatter(luma, saturation, 
                       color=[s[2] for s in board_raw])
        pyplot.xlim(0,1)
        pyplot.ylim(0,1)
        fig.canvas.draw()
        size = fig.canvas.get_width_height()
        buff = fig.canvas.tostring_argb()
        image_p = Image.frombytes('RGBA', size, buff, 'raw')
        do_something(image_p, "color distribution")

    color_data = [(s[0], s[1]) for s in board_raw]

    init_x = sum(c[0] for c in color_data) / float(len(color_data))

    clusters, score = k_means.cluster(3, 2,list(zip(color_data, range(len(color_data)))),
                               [[0., 0.5], [init_x, 0.5], [1., 0.5]])

    if show_all:
        fig = pyplot.figure(figsize=(8, 6))
        pyplot.scatter([d[0][0] for d in clusters[0]], [d[0][1] for d in clusters[0]],
                                                 color=(1,0,0,1))
        pyplot.scatter([d[0][0] for d in clusters[1]], [d[0][1] for d in clusters[1]],
                                                 color=(0,1,0,1))
        pyplot.scatter([d[0][0] for d in clusters[2]], [d[0][1] for d in clusters[2]],
                                                 color=(0,0,1,1))
        pyplot.xlim(0,1)
        pyplot.ylim(0,1)
        fig.canvas.draw()
        size = fig.canvas.get_width_height()
        buff = fig.canvas.tostring_argb()
        image_p = Image.frombytes('RGBA', size, buff, 'raw')
        do_something(image_p, "color clustering")

    clusters[0] = [(p[1], 'B') for p in clusters[0]]
    clusters[1] = [(p[1], '.') for p in clusters[1]]
    clusters[2] = [(p[1], 'W') for p in clusters[2]]

    board_rl = sum(clusters, [])
    board_rl.sort()
    board_rg = (p[1] for p in board_rl)
    
    board_r = []

    #TODO 19 should be a size parameter
    try:
        for i in range(19):
            for _ in range(19):
                board_r.append(next(board_rg))
    except StopIteration:
        pass
    
    return output.Board(19, board_r)

def mean_luma(cluster):
    """Return mean luminanace of the *cluster* of points."""
    return sum(c[0][0] for c in cluster) / float(len(cluster))

def to_general(line, size):
    # TODO comment
    (x1, y1), (x2, y2) = linef.line_from_angl_dist(line, size)
    return (y2 - y1, x1 - x2, x2 * y1 - x1 * y2)

def intersection(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    delim = float(a1 * b2 - b1 * a2)
    x = (b1 * c2 - c1 * b2) / delim
    y = (c1 * a2 - a1 * c2) / delim
    return x, y

# TODO remove the parameter get_all
def intersections_from_angl_dist(lines, size, get_all=True):
    """Take grid-lines and size of the image. Return intersections."""
    lines0 = list(map(lambda l: to_general(l, size), lines[0]))
    lines1 = list(map(lambda l: to_general(l, size), lines[1]))
    intersections = []
    for l1 in lines1:
        line = []
        for l2 in lines0:
            line.append(intersection(l1, l2))
        intersections.append(line)
    return intersections
   
def rgb2lumsat(color):
    """Convert RGB to luminance and HSI model saturation."""
    r, g, b = color
    luma = (0.30 * r + 0.59 * g + 0.11 * b) / 255.0
    max_diff = max(color) - min(color)
    if max_diff == 0:
        saturation = 0
    else:
        saturation = 1. - ((3. * min(color)) / sum(color)) 
    return luma, saturation

def stone_color_raw(image, pt):
    """Given image and coordinates, return stone color."""
    x, y = pt
    size = 3 
    points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            try:
                points.append(image.getpixel((x + i, y + j)))
            except IndexError:
                pass
    norm = float(len(points))
    if norm == 0:
        return 0, 0, (0, 0, 0) #TODO trow exception here
    norm = float(norm*255)
    color = (sum(p[0] for p in points) / norm,
             sum(p[1] for p in points) / norm,
             sum(p[2] for p in points) / norm)
    hue, luma, saturation = colorsys.rgb_to_hls(*color)
    color = colorsys.hls_to_rgb(hue, 0.5, 1.)
    return luma, saturation, color, hue

def adjust_for_stone_thickness(intersections, image, show_all, do_something):
    """Given grid intersections on the board, use homography to find stone centers above the grid"""
    # this doesn't have to be perfect, so it's ok making some reasonable assumptions
    goban_width_mm = 439.0 # standard 19x19 goban grid is 424x454mm, average to square
    go_stone_thickness_mm = 10.0 # Typical: size 28-36 (7.5 - 10.1mm), can be size 50 (14.3mm)

    # typical phone camera with 80deg FOV, square pixels, and 640x480 resolution
    fx = fy = 381.36 # f_pixels = W_pixels / (2 * tan(theta/2)) = 640 / (2 * tan(80deg/2))
    cx, cy = 320.0, 240.0 # center of 640x480 image
    K = np.array([[fx,  0, cx], [ 0, fy, cy], [ 0,  0,  1]], dtype=np.float32) # cam intrinsics

    # Define source quadrilateral (world/surface in mm), assume flat, square grid
    pts_src = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32) * 0.5 * goban_width_mm

    # Destination quadrilateral (perspective, in pixel coords)
    hull = cv2.convexHull(np.array(sum(intersections, []), dtype=np.float32))
    pts_dst = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)

    H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    H_inv = np.linalg.inv(H)  # Invert to go from Pixel Space -> World Space
  
    # Extract true physical rotation and translation directly from H and K
    # Stripping the camera intrinsics from the homography yields [r1, r2, t] up to a scale factor
    r1_r2_t = np.linalg.inv(K) @ H

    # Determine the physical scale factor based on the first rotation column vector
    scale = np.linalg.norm(r1_r2_t[:, 0])
    r1_r2_t_scaled = r1_r2_t / scale

    r1 = r1_r2_t_scaled[:, 0:1]
    r2 = r1_r2_t_scaled[:, 1:2]
    r3 = np.cross(r1.flatten(), r2.flatten()).reshape(3, 1) 
    R = np.hstack((r1, r2, r3)) # reconstructed 3D rotation matrix
    t = r1_r2_t_scaled[:, 2:3] # translation in real-world mm

    #from scipy.spatial.transform import Rotation
    #euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True).astype(int)
    #print(f"R: {R}, eulers:{euler_angles}, t: {t.T}")

    v_world = np.array([[0.0], [0.0], [-go_stone_thickness_mm*0.5]]) # half-thickness up in neg Z-axis

    adjusted_intersections = []
    for line in intersections:
        new_line = []
        for point in line:
            p_pixel_2d = np.array([point[0], point[1], 1.0], dtype=np.float32).reshape(3, 1)

            # Transform the grid pixel coordinates into 2D world coordinates
            P_world_homogenous = H_inv @ p_pixel_2d
            P_world_homogenous /= P_world_homogenous[2] # Normalize by the scale factor

            # Construct the 3D world grid point (setting Z to 0 because it's flat on the board)
            P_world = np.array([[P_world_homogenous[0][0]], [P_world_homogenous[1][0]], [0.0]], dtype=np.float32)

            P_offset_cam = R @ (P_world + v_world) + t # Transform offset point to Camera 3D space

            # Project offset grid point (stone center) from 3D camera space to 2D pixel coords
            p_offset_pixel = K @ P_offset_cam
            p_offset_pixel /= p_offset_pixel[2] # Divide by Z

            new_point = (int(round(p_offset_pixel[0][0])), int(round(p_offset_pixel[1][0])))
            new_line.append(new_point)

        adjusted_intersections.append(new_line)
    if show_all:
        image_g = image.copy()
        draw = ImageDraw.Draw(image_g)
        r = 2
        for line in intersections:
            for (x, y) in line:
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(120, 255, 120)) # green for original intersections
        for line in adjusted_intersections:
            for (x, y) in line:
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(120, 120, 255)) # blue for adjusted intersections
        do_something(image_g, "intersections (green = grid, blue = stone thickness)", name="intersections")

    return adjusted_intersections

