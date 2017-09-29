import sys
import cv2
import numpy as np
import math

THRESHOLD = 113.3
GRID_SPACING = 1
BLUR_FACTOR = 3
RADIUS = [5, 3]
STROKE_ANGLE = 180 + 45
STROKE_WIDTH = 5
MIN_STROKE_WIDTH = 2
MAX_STROKE_WIDTH = 7
CURVATURE = 1

THRESHOLD = 140
GRID_SPACING = 1
BLUR_FACTOR = 1
RADIUS = [9, 7, 3, 1]
MIN_STROKE_WIDTH = 7
MAX_STROKE_WIDTH = 17
CURVATURE = 0.1


def _diff(a, b):
    ar, ag, ab = np.split(a, 3, axis=2)
    br, bg, bb = np.split(b, 3, axis=2)
    diff = np.clip(np.abs(np.power(ar - br, 2) +
                          np.power(ag - bg, 2) + np.power(ab - bb, 2)), 0., 255.)
    return np.squeeze(diff)


def vangoginize(img):
    canvas = np.zeros(img.shape)
    canvas.fill(-255)

    for r in RADIUS:
        reference_img = cv2.GaussianBlur(
            img, (int(BLUR_FACTOR * r), int(BLUR_FACTOR * r)), 0)
        render_layer(canvas, reference_img, r)

    cv2.imwrite('generated.png', canvas)


def render_layer(canvas, reference_img, r):
    strokes = []

    diff = _diff(canvas, reference_img)

    step = GRID_SPACING * r

    for x in xrange(0, canvas.shape[1], step):
        for y in xrange(0, canvas.shape[0], step):
            # getting near-point region of step width
            roi = np.ix_([max(int(x - float(step) / 2), 0), max(int(y - float(step) / 2), 0)],
                         [min(int(x + float(step) / 2), canvas.shape[1] - 1), min(int(y + float(step) / 2), canvas.shape[0] - 1)])
            region = diff[roi]
            error = np.sum(region) / step**2
            if error > int(THRESHOLD / r):
                # find the largest error point
                (max_y, max_x) = np.unravel_index(
                    region.argmax(), region.shape)
                max_y = int(y - float(step) / 2) + max_y
                max_x = int(x - float(step) / 2) + max_x
                strokes.append((r, max_x, max_y, reference_img))

    # Getting gradient unit-vectors
    #Gx = cv2.Sobel(rimg, cv2.CV_64F, 1, 0).mean(axis=2)
    #Gy = cv2.Sobel(rimg, cv2.CV_64F, 0, 1).mean(axis=2)
    Gx, Gy = np.array(np.gradient(reference_img.mean(axis=2)))
    Gx /= Gx.max()
    Gy /= Gy.max()

    # shuffling stroke to make it more natural
    np.random.shuffle(strokes)
    for r, x, y, rimg in strokes:
        make_spline_stroke(canvas, r, Gx, Gy, x, y, rimg)


def make_stroke(canvas, r, x, y, reference_img):
    color = np.squeeze(reference_img[y, x, :]).astype(int).tolist()
    ny = y + int(-math.sin((STROKE_ANGLE * math.pi) / 180) * STROKE_WIDTH)
    nx = x + int(math.cos((STROKE_ANGLE * math.pi) / 180) * STROKE_WIDTH)
    cv2.line(canvas, (x, y), (nx, ny), color, r)


def make_spline_stroke(canvas, r, Gx, Gy, x0, y0, rimg):
    def c(im, x1, y1):
        return np.squeeze(im[y1, x1, :]).astype(int)

    def brush(x1, y1):
        cv2.circle(canvas, (int(x1), int(y1)), r, color, -1)

    def clip(a, _min=0, _max=255):
        return max(min(a, _max - 1), _min)

    color = np.squeeze(rimg[y0, x0, :]).astype(int).tolist()
    x, y = x0, y0
    ldx, ldy = 0, 0
    brush(x, y)

    for i in xrange(MAX_STROKE_WIDTH):
        if i > MIN_STROKE_WIDTH and \
           np.abs(np.sum(c(rimg, x, y) - c(canvas, x, y))) < \
           np.abs(np.sum(c(rimg, x, y) - color)):
            return

        mag = math.sqrt(Gy[y][x]**2 + Gx[y][x]**2)
        if mag == 0:
            return

        gx, gy = Gx[y][x], Gy[y][x]
        dx, dy = -gy, gx

        if (ldx * dx + ldy * dy) < 0:
            dx, dy = -dx, -dy

        dx = CURVATURE * dx + (1 - CURVATURE) * ldx
        dy = CURVATURE * dy + (1 - CURVATURE) * ldy

        dx = dx / math.sqrt(dx**2 + dy**2)
        dy = dy / math.sqrt(dx**2 + dy**2)

        x, y = int(x + r * dx), int(y + r * dy)
        ldx, ldy = dx, dy

        x, y = clip(x, 0, canvas.shape[1]), clip(y, 0, canvas.shape[0])

        brush(x, y)


if __name__ == "__main__":
    vangoginize(cv2.imread(sys.argv[1]))
