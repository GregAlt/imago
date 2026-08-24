"""Imago intersections module."""

from math import cos, tan, pi
from operator import itemgetter
import colorsys

from PIL import ImageDraw
from PIL import Image

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

def board(unwarped, H, intersections, crosses, circles, show_all, do_something, logger):
    """Find stone colors and return board situation."""
    board_raw = []

    im = np.asarray(unwarped)
    for int_line, cross_line, circle_line in zip(intersections, crosses, circles, strict=True):
        for intersection, cross, circle in zip(int_line, cross_line, circle_line, strict=True):
            if circle and not cross and circle[2] >= 0.7:
                c, r, a, d = circle
                pixels = get_circle_pixels(im, c, r)
            else:
                # TODO also take into account circles and crosses
                (x,y,z) = H @ np.array([intersection[0],intersection[1],1])
                intersection_unwarped = int(round(x/z)), int(round(y/z))
                pixels = get_square_pixels(im, intersection_unwarped, 3)
            rgb = np.median(pixels, axis=0)
            board_raw.append([process_color(rgb/255.0)])

    board_raw = sum(board_raw, [])

    ### Show color distribution

    if show_all:
        import matplotlib.pyplot as pyplot
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

def get_circle_pixels(im, c, r):
    cx, cy = c
    h, w = im.shape[:2]
    y, x = np.ogrid[:h, :w]
    return im[(x - cx)**2 + (y - cy)**2 <= r**2]

def process_color(rgb):
    hue, luma, saturation = colorsys.rgb_to_hls(*rgb) 
    saturation *=  1.0 - abs(2.0 * luma - 1.0) # correct to be more perceptual, without high spread near white
    color = colorsys.hls_to_rgb(hue, 0.5, 1.)    
    return luma, saturation, color, hue

def get_square_pixels(image, pt, size_2):
    """Given image, coordinates, and half size, return square block of pixels."""
    x, y = pt
    h, w, c = image.shape
    y_min, y_max = max(0, y - size_2), min(h, y + size_2 + 1)
    x_min, x_max = max(0, x - size_2), min(w, x + size_2 + 1)
    block = image[y_min:y_max, x_min:x_max].reshape(-1, c)
    return block

def calc_camera_intrinsics_estimate():
    # typical phone camera with 80deg FOV, square pixels, and 640x480 resolution
    # 2x and 3x work, but 1x and 4x causes some tests to fail
    fx = fy = 2*381.36 # f_pixels = W_pixels / (2 * tan(theta/2)) = 640 / (2 * tan(80deg/2)) = 381.36 [2x means about 45.52deg]
    cx, cy = 320.0, 240.0 # center of 640x480 image
    K = np.array([[fx,  0, cx], [ 0, fy, cy], [ 0,  0,  1]], dtype=np.float32) # cam intrinsics
    return K

def calc_homography_from_intersections(intersections, K):
    # this doesn't have to be perfect, so it's ok making some reasonable assumptions
    goban_width_mm = 439.0 # standard 19x19 goban grid is 424x454mm, average to square

    # Destination quadrilateral (perspective, in pixel coords)
    hull = cv2.convexHull(np.array(sum(intersections, []), dtype=np.float32))
    pts_dst = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)

    dst_tmp = pts_dst - pts_dst.mean(axis=0)
    angles_deg = np.rad2deg(np.arctan2(dst_tmp[:, 1], dst_tmp[:, 0]))
    #print(angles_deg.astype(int))

    # Define source quadrilateral (world/surface in mm), assume flat, square grid
    pts_src = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32) * 0.5 * goban_width_mm

    H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    return H

def extract_3d_transform_from_homography(K, H):
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
    return R, t

def get_vertically_adjusted_intersections(intersections, height, H, K, show_all):
    R, t = extract_3d_transform_from_homography(K, H)
    if show_all:
        from scipy.spatial.transform import Rotation
        euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True).astype(int)
        print(f"Camera angles in degrees: {euler_angles} Pitch is first, 0 is straight down, -90 is edge on")

    H_inv = np.linalg.inv(H)  # Invert to go from Pixel Space -> World Space
    v_world = np.array([[0.0], [0.0], [-height]]) # half-thickness up in neg Z-axis
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
    return adjusted_intersections

def draw_intersections_on_image(intersections, H, draw, r, color):
    for line in intersections:
        for p in line:
            (x,y,z) = H @ np.array([p[0],p[1],1])
            x /= z
            y /= z
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color) 

def find_crosses(intersections, H, image):
    pad = 32
    window = 12
    crosses=[]
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for line in intersections:
        cross_line=[]
        for p in line:
            cross = None
            (x,y,z) = H @ np.array([p[0],p[1],1])
            x = int(round(x/z))
            y = int(round(y/z))
            slice = padded_image[y+pad-window : y+pad+window, x+pad-window : x+pad+window]    

            row_sums = np.sum(slice, axis=1)
            col_sums = np.sum(slice, axis=0)
            peak_row_idx = np.argmax(row_sums)
            peak_row_val = row_sums[peak_row_idx]
            peak_col_idx = np.argmax(col_sums)
            peak_col_val = row_sums[peak_col_idx]
            row_background = np.delete(row_sums, peak_row_idx)
            bg_row_mean = np.mean(row_background)
            bg_row_std = np.std(row_background) if np.std(row_background) > 0 else 1.0            
            col_background = np.delete(col_sums, peak_col_idx)
            bg_col_mean = np.mean(col_background)
            bg_col_std = np.std(col_background) if np.std(col_background) > 0 else 1.0  
            row_relative_score = (peak_row_val - bg_row_mean) / bg_row_std          
            col_relative_score = (peak_col_val - bg_col_mean) / bg_col_std  
            delta = (peak_row_idx-window, peak_col_idx-window)
            score = (row_relative_score, col_relative_score)
            if abs(delta[0])<=3 and abs(delta[1])<=3 and score[0] > -1 and score[1] > -1:
                rc = (x-window+peak_col_idx, y-window+peak_row_idx)
                start = (x-window, y-window)
                cross = (rc, start, window, delta, score)
            cross_line.append(cross)
        crosses.append(cross_line)

    return crosses



def do_hough_circles(intersections, H, image):
    from skimage.transform import hough_circle, hough_circle_peaks

    # find circles for stones (will have some false negative/positives!)
    pad = 32
    window = 18
    circles = []
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for line in intersections:
        circle_line = []
        for p in line:
            circle = None
            (x,y,z) = H @ np.array([p[0],p[1],1])
            x,y = int(round(x/z)), int(round(y/z))
            slice = padded_image[y+pad-window : y+pad+window, x+pad-window : x+pad+window]    

            radii = np.arange(8, 16) # (8 to 16 inclusive) 
            hough = hough_circle(slice, radii)
            if hough is not None and hough.size > 0:
                accum, cx, cy, radii = hough_circle_peaks(hough, radii, total_num_peaks=1)
                if len(radii) > 0 and accum > 0.5:
                    c, radius = np.array([cx[0],cy[0]]), radii[0]
                    if radius > 9:
                        dist = np.linalg.norm(c - window)
                        if dist-radius < -2:
                            circle = ((x-window+c[0], y-window+c[1]), radius, accum, dist)
            circle_line.append(circle)
        circles.append(circle_line)
    return circles

def adjust_for_stone_thickness(intersections, image, im_h, show_all, do_something, logger):
    """Given grid intersections on the board, use homography to find stone centers above the grid"""
    K = calc_camera_intrinsics_estimate()
    H = calc_homography_from_intersections(intersections, K)  
    H_inv = np.linalg.inv(H)  # Invert to go from Pixel Space -> World Space

    go_stone_thickness_mm = 10.0 # Typical: size 28-36 (7.5 - 10.1mm), can be size 50 (14.3mm)
    adjusted_intersections = get_vertically_adjusted_intersections(intersections, go_stone_thickness_mm*0.5, H, K, show_all)
    H2 = calc_homography_from_intersections(adjusted_intersections, K)  

    if show_all:
        # draw grid intersections and raised stone centers on original image
        image_g = image.copy()
        draw = ImageDraw.Draw(image_g)
        draw_intersections_on_image(intersections, np.identity(3), draw, 2, (120, 255, 120)) # green for original intersections
        draw_intersections_on_image(adjusted_intersections, np.identity(3), draw, 2, (120, 120, 255)) # blue for adjusted intersections
        do_something(image_g, "intersections (green = grid, blue = stone thickness)", name="intersections")

    return adjusted_intersections, H, H2

