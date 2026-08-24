"""Computing the grid"""

from math import sqrt, acos, copysign
from geometry import l2ad, line, intersection
import numpy as np
import cv2

def gen_line(n, start, end):
    yield start
    for i in range(1, n - 1):
        yield (start[0] + i * (end[0] - start[0]) / float(n - 1),
               start[1] + i * (end[1] - start[1]) / float(n - 1))
    yield end

# TODO: dedupe function
def sort_points_CW_from_TL(pts):
    # Sort points based on arctan2 of angles (-180 to 180)  [TL, TR, BR, BL]
    # Note: TL of image is 0,0. CW -179 center left, -90 top, 0 right, +90 bottom, +179 center left
    tmp = pts - pts.mean(axis=0)
    angles_deg = np.rad2deg(np.arctan2(tmp[:, 1], tmp[:, 0]))
    sort_indices = np.argsort(angles_deg)
    pts = pts[sort_indices]
    return pts

def get_square_points_CW_from_TL():
    '''Square corner points, centered at 0,0, side length 2, ordered clockwise: [TL, TR, BR, BL]'''
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

def lines(corners):
    # TODO - this homography calculation needs to agree with intrsc.b_intersects and calc_homography_from_intersections
    # to make the rendered unwarped board and the final logical stone layout are aligned and not rotated by 90
    # this all currently works but is brittle and could use some principled cleanup.
    # lots of wrong assumptions about CW vs CCW, +Y being up or down, method of sorting angles, canonical square corners, etc
    # manual_lines.py line 18, lines(corners)
    # grid_newf.py line 148, gen_corners(d1
    # grid_newf.py line 194, find( (sorting grid_lines at bottom))
    # intrsc.py line 229, calc_homography_from_intersections
    # intrsc.py line 32, b_intersects
    # tests that give different results: t0\skip\gregtest004.jpg, t2\test_010.jpg, t2\test_051.jpg

    corners = sort_points_CW_from_TL(np.array(corners))
    gcorners = get_square_points_CW_from_TL() * 50.0 + 50.0
    
    l1 = list(gen_line(19, gcorners[0], gcorners[3])) # TL to BL
    l2 = list(gen_line(19, gcorners[3], gcorners[2])) # BL to BR
    l3 = list(gen_line(19, gcorners[1], gcorners[2])) # TR to BR
    l4 = list(gen_line(19, gcorners[0], gcorners[1])) # TL to TR
    
    mC, _ = cv2.findHomography(gcorners, corners)
    
    def perspective(point):
        x, y, z = np.matmul(mC, np.array([point[0], point[1], 1]))
        return (x/z, y/z)

    l1_, l2_, l3_, l4_ = map(lambda x: list(map(perspective, x)), [l1, l2, l3, l4])
    return (list(zip(l1_, l3_)), list(zip(l2_, l4_)))
