#!/usr/bin/env python

"""Go image recognition."""

import sys
import os
import argparse
import pickle
import random

try:
    from PIL import Image, ImageDraw
    import numpy as np
    import cv2
    from skimage.morphology import closing, footprint_rectangle
except ImportError as msg:
    print(msg, file=sys.stderr)
    sys.exit(1)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import linef
import intrsc
import gridf_new as gridf
import output
import manual

def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', metavar='file', nargs='+',
                        help="image to analyse")
    parser.add_argument('-w', type=int, default=640,
                    help="scale image to the specified width before analysis")
    parser.add_argument('-m', '--manual', dest='manual_mode',
                        action='store_true',
                        help="manual grid selection")
    parser.add_argument('-d', '--debug', dest='show_all',
                        action='store_true',
                        help="show every step of the computation")
    parser.add_argument('-s', '--save', dest='saving', action='store_true',
                        help="save images instead of displaying them")
    parser.add_argument('-c', '--cache', dest='l_cache', action='store_true',
                        help="use cached lines")
    parser.add_argument('-S', '--sgf', dest='sgf_output', action='store_true',
                        help="output in SGF")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help="report progress")
    parser.add_argument('--rng-seed', dest='rng_seed', help="Specify random number generator seed, for consistent test results.")
    return parser

def unwarp_image(image, H):
    '''take a photo with perspective and unwarp to square world space using homography, H'''
    H_inv = np.linalg.inv(H)  # Invert to go from Pixel Space -> World Space
    T = np.array([
        [1, 0, 240],
        [0, 1, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    adjusted_H = T.dot(H_inv)
    unwarped = cv2.warpPerspective(np.array(image), adjusted_H, (480, 480), flags=cv2.INTER_CUBIC )
    return unwarped, adjusted_H

def apply_homography(H, p):
    """Applies a 3x3 homography matrix to a 2D point using projective division."""
    x, y, z = H @ np.array([p[0], p[1], 1]) # Convert to homogeneous coordinates and multiply
    return (x / z, y / z) # Projective division (normalize by scale factor z)

def remove_grid_lines(image):
    '''use morphological closing operator to remove thin grid lines from image'''
    footprint = footprint_rectangle((4, 4)) # 3 works better for most, but 4 is needed for some images
    processed_channels = []
    for channel in range(3):
        processed_channels.append(closing(np.array(image)[:, :, channel], footprint))
    image_clean = Image.fromarray(np.stack(processed_channels, axis=-1))
    return image_clean

def fix_filename(filename):
    return filename.replace("\\", "/").replace("./", "") if os.name == 'nt' else filename

def write_detected_corners(pts, filename):
    np.savetxt(fix_filename(filename), pts, delimiter=',', fmt='%.1f', header='pixel_x,pixel_y')

def read_cached_corners(filename):
    try:
        corners = np.loadtxt(fix_filename(filename), delimiter=',')
    except FileNotFoundError:
        corners = None
    return corners

def draw_crosses_and_circles(unwarped_orig, H, H2, circles, crosses, do_something):
    unwarped_im = Image.fromarray(unwarped_orig).convert('RGB')
    unwarped_g = unwarped_im.copy()
    draw = ImageDraw.Draw(unwarped_g)

    for line in circles:
        for c in line:
            if c is not None:
                center, r, accum, dist = c
                cx, cy = center
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=None, outline="blue", width=2)

    # cross endpoints are in the grid unwarped image. Need to convert to stone unwarped
    # So: grid unwarped -> warped original -> stone unwarped
    g2s = np.linalg.inv(H2) @ H 
    for line in crosses:
        for cross in line:
            if cross is not None:
                c, start, win, delta, score = cross
                col,row = c
                v1 = apply_homography(g2s, (col, start[1]))
                v2 = apply_homography(g2s, (col, start[1]+2*win))
                draw.line((v1,v2), fill="green", width=2)
                h1 = apply_homography(g2s, (start[0], row))
                h2 = apply_homography(g2s, (start[0]+2*win, row))
                draw.line((h1,h2), fill="green", width=2)
    do_something(unwarped_g, "circles and crosses")

def find_lines_and_corners(args, image, im_h, show_all, do_something, logger):
    corners = None
    ##### TODO make cached corners optional, also compare with known correct corners
    # corners = read_cached_corners(args.files[0][:-4] + "_corners.txt")
    # print("using cached corners", file=sys.stderr)
    #####

    if corners is not None:
        lines = manual.lines_from_corners(corners, image.size)
    elif args.manual_mode:
        try:
            lines, corners = manual.find_lines(image)
        except manual.UserQuitError:
            #TODO ask user to try again
            return 1
    else:
        if args.l_cache:
            filename = fix_filename("saved/cache/" + args.files[0][:-4] + "_" +
                       str(image.size[0]))
            cache_dir = "/".join(filename.split('/')[:-1])
            if os.path.exists(filename):
                lines, l1, l2, bounds, hough = pickle.load(open(filename, 'rb'))
                print("using cached results", file=sys.stderr)
            else:
                lines, l1, l2, bounds, hough = linef.find_lines(image, im_h, do_something, logger)
                if not os.path.isdir(cache_dir):
                    os.makedirs(cache_dir)
                d_file = open(filename, 'wb')
                pickle.dump((lines, l1, l2, bounds, hough), d_file)
                d_file.close()
        else:
            lines, l1, l2, bounds, hough = linef.find_lines(image, im_h, do_something, logger)

        grid, lines = gridf.find(lines, image.size, l1, l2, bounds, hough,
                                 show_all, do_something, logger)
        if show_all:
            im_g = image.copy()
            draw = ImageDraw.Draw(im_g)
            for l in grid[0] + grid[1]:
                draw.line(l, fill=(64, 255, 64), width=1)
            do_something(im_g, "grid", name="grid")

    return lines, corners


# TODO factor this into smaller functions
def main():
    """Main function of the program."""
    
    parser = argument_parser()
    args = parser.parse_args()

    show_all = args.show_all
    verbose = args.verbose

    random.seed(args.rng_seed)

    try:
        image = Image.open(args.files[0])
    except IOError as msg:
        print(msg, file=sys.stderr)
        return 1
    if image.mode == 'P':
        image = image.convert('RGB')
    
    if image.size[0] > args.w:
        image = image.resize((args.w, int((float(args.w)/image.size[0]) *
                              image.size[1])), Image.LANCZOS)

    if not show_all:
        def nothing(a, b):
            pass
        do_something = nothing
    elif args.saving:
        do_something = Imsave("saved/" + args.files[0][:-4] + "_" +
                               str(image.size[0]) + "/").save
    else:
        import im_debug
        do_something = im_debug.show

    if verbose:
        import time
        class Logger:
            def __init__(self):
                self.t = 0

            def __call__(self, m):
                t_n = time.time()
                if self.t > 0:
                    print("\t" + str(t_n - self.t), file=sys.stderr)
                print(m, file=sys.stderr)
                self.t = t_n
        logger = Logger()

    else:
        def logger(m):
            pass

    im_h = linef.prepare(image, do_something, logger)
    lines, corners = find_lines_and_corners(args, image, im_h, show_all, do_something, logger)
    grid_intersections = intrsc.b_intersects(image, lines, show_all, do_something, logger)
    pts = manual.corners_from_intersections(grid_intersections)
    write_detected_corners(pts, args.files[0][:-4] + "_corners.out")
    adjusted_intersections, H, H2 = intrsc.adjust_for_stone_thickness(grid_intersections, image, im_h, show_all, do_something, logger)
    gridless = remove_grid_lines(image)
    stone_edge = linef.prepare(gridless, do_something, logger) 
    unwarped_grid_edge, adjusted_H = unwarp_image(im_h, H) # original edge image (using grid homography)
    unwarped_stone_edge, adjusted_H2 = unwarp_image(stone_edge, H2) # gridless edge image (using stone homography)
    unwarped_orig, adjusted_H2 = unwarp_image(image, H2) # original image (using stone homography)
    crosses = intrsc.find_crosses(grid_intersections, adjusted_H, unwarped_grid_edge)
    circles = intrsc.do_hough_circles(adjusted_intersections, adjusted_H2, unwarped_stone_edge)

    if show_all:
        draw_crosses_and_circles(unwarped_orig, H, H2, circles, crosses, do_something)

    board = intrsc.board(unwarped_orig, adjusted_H2, adjusted_intersections, crosses, circles, show_all, do_something, logger)

    logger("finished")

    # TODO! refactor this mess:
    if len(args.files) == 1:

        if args.sgf_output:
            print(board.asSGFsetPos())
        else:
            print(board)
    
    else:
        game = output.Game(19, board) #TODO size parameter
        for f in args.files[1:]:
            try:
                image = Image.open(f)
            except IOError as msg:
                print >> sys.stderr, msg
                continue
            if verbose:
                print >> sys.stderr, "Opening", f
            if image.mode == 'P':
                image = image.convert('RGB')
            if image.size[0] > args.w:
                image = image.resize((args.w, int((float(args.w)/image.size[0]) *
                              image.size[1])), Image.ANTIALIAS)
            board = intrsc.board(unwarped_orig, adjusted_H2, adjusted_intersections, crosses, circles, show_all, do_something, logger)
            if args.sgf_output:
                game.addMove(board)
            else:
                print(board)

        if args.sgf_output:
            print(game.asSGF())

    return 0

class Imsave():
    def __init__(self, saving_dir):
        self.saving_dir = saving_dir
        self.saving_num = 0

    def save(self, image, title='', name=None):
        im_format = ('.png', 'PNG')
        if name:
            filename = self.saving_dir + name + im_format[0]
        else:
            filename = self.saving_dir + "{0:0>3}".format(self.saving_num) + im_format[0]
            self.saving_num += 1
        if not os.path.isdir(self.saving_dir):
            os.makedirs(self.saving_dir)
        image.save(filename, im_format[1])

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt: #TODO does this work?
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
