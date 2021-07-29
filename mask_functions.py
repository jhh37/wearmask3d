# WearMask3D
# Copyright 2021 Hanjo Kim and Minsoo Kim. All rights reserved.
# http://github.com/jhh37/wearmask3d
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: rlakswh@gmail.com      (Hanjo Kim)
#         devkim1102@gmail.com   (Minsoo Kim)

import cv2
import geomdl
import torch
import torchvision.transforms as transforms
from PIL import Image
from geomdl import exchange, NURBS
from pygame.constants import *

from misc_utils import get_models, mask_transformation
from obj_loader import *
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, predict_dense


def get_surface_v1(mType, points, delta = 0.025):

    chin_left = points[2]
    chin_right = points[14]
    nose_tip = points[30]
    nose_top = points[27]
    lip_bottom = points[57]

    chin_middle = (chin_left + chin_right) / 2
    normalVector = nose_tip - chin_middle
    verticalVector = lip_bottom - nose_top
    horVector = chin_right - chin_left

    c = []
    for i in range(0, 28):
        c.append(np.array([0, 0, 0, 1.0]))

    if (mType == 4):  # smooth (center-focused)
        c1 = 0.50
        c2 = 0.10
        c3 = 0.25
        c4 = 0.30
        c5 = 0.07
        c6 = 0.10
        c7 = 0.015
        c8 = 0.10
        degree_u = 2
        degree_v = 2
        knotvector_u = [0, 0, 0, 0.5, 1, 1, 1]
        knotvector_v = [0, 0, 0, 0.30, 0.50, 0.50, 0.70, 1, 1, 1]
    elif (mType == 3):  # smooth
        c1 = 0.50
        c2 = 0.10
        c3 = 0.25
        c4 = 0.30
        c5 = 0.07
        c6 = 0.00
        c7 = 0.04
        c8 = 0.15
        degree_u = 2
        degree_v = 2
        knotvector_u = [0, 0, 0, 0.5, 1, 1, 1]
        knotvector_v = [0, 0, 0, 0.15, 0.45, 0.55, 0.85, 1, 1, 1]
    elif (mType == 2):  # angulated
        c1 = 0.50
        c2 = 0.10
        c3 = 0.17
        c4 = 0.20
        c5 = 0.08
        c6 = 0.00
        c7 = 0.04
        c8 = 0.15
        degree_u = 1
        degree_v = 2
        knotvector_u = [0, 0, 0.2, 0.75, 1, 1]
        knotvector_v = [0, 0, 0, 0.15, 0.5, 0.5, 0.85, 1, 1, 1]

    c[0] = chin_left - horVector * c7 * 1.5 + normalVector * 0.1
    c[6] = chin_right + horVector * c7 * 1.5 + normalVector * 0.1
    c[3] = nose_tip
    c[1] = c[0] + normalVector * c1 + horVector * c7
    c[5] = c[6] + normalVector * c1 - horVector * c7
    c[2] = (c[1] + c[3]) / 2
    c[4] = (c[5] + c[3]) / 2

    for i in range(0, 7):
        c[i] -= verticalVector * 0.4
        c[i + 21] = c[i] + verticalVector * 1.4

    c[1] -= normalVector * c2
    c[2] -= normalVector * c2
    c[3] -= normalVector * c2
    c[4] -= normalVector * c2
    c[5] -= normalVector * c2

    c[0] += normalVector * c6 * 2
    c[1] += normalVector * c6
    c[5] += normalVector * c6
    c[6] += normalVector * c6 * 2
    c[7] += normalVector * c6 * 2
    c[8] += normalVector * c6
    c[12] += normalVector * c6
    c[13] += normalVector * c6 * 2

    c[21] += horVector * c8
    c[22] -= -horVector * c8 + normalVector * 0.2
    c[23] -= normalVector * 0.2
    c[24] -= normalVector * 0.2
    c[25] -= normalVector * 0.2
    c[26] -= horVector * c8 + normalVector * 0.2
    c[27] -= horVector * c8

    for i in range(7, 14):
        c[i] = (7 * c[i - 7] + 3 * c[i + 14]) / 10
        c[i + 7] = (1 * c[i - 7] + 4 * c[i + 14]) / 5

    c[8] += normalVector * 0.1
    c[9] += normalVector * c3
    c[10] += normalVector * c4
    c[11] += normalVector * c3
    c[12] += normalVector * 0.1

    c[15] += normalVector * 0.08
    c[16] += normalVector * 0.12
    c[17] += normalVector * 0.12
    c[18] += normalVector * 0.12
    c[19] += normalVector * 0.08

    c[22] -= normalVector * 0.1
    c[23] -= normalVector * c5
    c[24] -= normalVector * c5
    c[25] -= normalVector * c5
    c[26] -= normalVector * 0.1

    ctrlPts = []
    for i in range(0, 28):
        ctrlPts.append(c[i].tolist())

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    surf = NURBS.Surface()

    surf.degree_u = degree_u
    surf.degree_v = degree_v

    surf.set_ctrlpts(ctrlPts, 4, 7)  # u=4 / v= 7

    surf.knotvector_u = knotvector_u
    surf.knotvector_v = knotvector_v

    surf.delta = delta # default = 0.025

    return exchange.export_obj_str(surf)


def get_surface_v0(mType, points, delta = 0.03):
    chin_left = points[2]
    chin_right = points[14]
    nose_tip = points[30]
    nose_top = points[27]
    lip_bottom = points[57]

    chin_middle = (chin_left + chin_right) / 2
    normalVector = nose_tip - chin_middle
    verticalVector = lip_bottom - nose_top
    horVector = chin_right - chin_left

    c = []
    for i in range(0, 10):
        c.append([0, 0, 0, 0])

    c[0] = [chin_left[0] - horVector[0] * 0.036, chin_left[1] - horVector[1] * 0.036,
            chin_left[2] - horVector[2] * 0.036, 1.0]
    c[1] = [c[0][0] + normalVector[0], c[0][1] + normalVector[1], c[0][2] + normalVector[2], 1.0]
    c[4] = [chin_right[0] + horVector[0] * 0.036, chin_right[1] + horVector[1] * 0.036,
            chin_right[2] + horVector[2] * 0.036, 1.0]
    c[2] = [nose_tip[0], nose_tip[1], nose_tip[2], 1.0]
    c[3] = [c[4][0] + normalVector[0], c[4][1] + normalVector[1], c[4][2] + normalVector[2], 1.0]

    for i in range(0, 5):
        for j in range(0, 3):
            c[i][j] -= verticalVector[j] * 0.4

        for j in range(0, 3):
            c[i + 5][j] = c[i][j] + verticalVector[j] * 1.4
        c[i + 5][3] = 1.0

    ctrlPts = []
    for i in range(0, 10):
        ctrlPts.append(c[i])

    for i in range(0, 5):
        c.append([0, 0, 0, 0])

    for i in range(10, 15):
        c[i][0] = c[i - 5][0]
        c[i][1] = c[i - 5][1]
        c[i][2] = c[i - 5][2]
        c[i][3] = 1.0
        ctrlPts.append(c[i])

    for j in range(0, 3):
        c[1][j] -= normalVector[j] * 0.1
        c[2][j] -= normalVector[j] * 0.1
        c[3][j] -= normalVector[j] * 0.1
        c[10][j] += horVector[j] * 0.2
        c[11][j] -= -horVector[j] * 0.2 + normalVector[j] * 0.2
        c[12][j] -= normalVector[j] * 0.2
        c[13][j] -= horVector[j] * 0.2 + normalVector[j] * 0.2
        c[14][j] -= horVector[j] * 0.2

    for i in range(6, 9):
        c[i][0] = (c[i - 5][0] + c[i + 5][0]) / 2
        c[i][1] = (c[i - 5][1] + c[i + 5][1]) / 2
        c[i][2] = (c[i - 5][2] + c[i + 5][2]) / 2
        c[i][3] = 1.0

    for j in range(0, 3):
        c[6][j] += normalVector[j] * 0.2
        c[7][j] += normalVector[j] * 0.4
        c[8][j] += normalVector[j] * 0.2

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    surf = NURBS.Surface()

    surf.degree_u = 2
    surf.degree_v = 2

    surf.set_ctrlpts(ctrlPts, 3, 5)

    surf.knotvector_u = [0, 0, 0, 1, 1, 1]
    surf.knotvector_v = [0, 0, 0, 0.5, 0.5, 1, 1, 1]

    surf.delta = delta

    return exchange.export_obj_str(surf)


def surf_2_obj_str(objStr, mask_shape, numLineVertices):
    firstF = True
    sur_obj_string = ""
    for idx, line in enumerate(objStr.splitlines()):
        if (idx == 0):
            sur_obj_string += 'mtllib nurbs_surf.mtl\n'
        if (line[0] == 'f'):
            if (firstF):
                for y in range(0, numLineVertices):
                    for x in range(0, numLineVertices):
                        xCoord = 1 - x / (numLineVertices - 1)
                        y_ = 1 - y / (numLineVertices - 1)
                        if (y_ >= 0.5):
                            if (mask_shape == 1):
                                yCoord = 2 * (y_ - 0.5) ** 2 + 0.5
                            elif (mask_shape == 3):
                                yCoord = -y_ * y_ + 2.5 * y_ - 0.5
                            elif (mask_shape == 4):
                                if (xCoord >= 0.5):
                                    yCoord = y_ + (2 * xCoord - 1) * 0.1
                                else:
                                    yCoord = y_ + (1 - 2 * xCoord) * 0.1
                            else:
                                yCoord = y_
                        else:
                            if (mask_shape == 2):
                                yCoord = -2 * (y_ - 0.5) * (y_ - 0.5) + 0.5
                            else:
                                yCoord = y_
                        sur_obj_string += 'vt {} {}\n'.format(xCoord, yCoord)
                sur_obj_string += 'usemtl material00\n'
                firstF = False

            fList = line.split()
            sur_obj_string += 'f {}/{} {}/{} {}/{}\n'.format(fList[1], fList[1], fList[2], fList[2], fList[3],
                                                            fList[3])
        else:
            sur_obj_string += line + '\n'

    return sur_obj_string


def get_face_region(face_regressor, img_cv, rect):
    pts = face_regressor(img_cv, rect).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box = parse_roi_box_from_landmark(pts)

    img = crop_img(img_cv, roi_box)
    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)

    return img, roi_box


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_geomdl_newer_than_52():
    ver = geomdl.__version__.split('.')
    if (int(ver[0]) >= 5 or (int(ver[0]) == 5 and int(ver[1]) >= 3)):
        return True
    else:
        return False


def batch_fit_masks(configs, file_list, cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    pid = os.getpid()

    pygame.init()

    # get face detection models
    model, face_detector, face_regressor = get_models()
    model.cuda().eval()

    # torch tensor transformation
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    bBr = 1

    # get configurations
    dataset_path = configs["srcPath"]
    save_dataset_path = configs["dstPath"]
    version = configs["version"]

    # current version check
    if (version == '1.1'):
        from mask_functions import get_surface_v1 as get_surface

        if is_geomdl_newer_than_52():
            numLineVertices = 40  # 5.3.1
        else:
            numLineVertices = 41  # 5.2.10

    elif (version == '1.0'):
        from mask_functions import get_surface_v0 as get_surface

        if is_geomdl_newer_than_52():
            numLineVertices = 33  # 5.3.1
        else:
            numLineVertices = 34  # 5.2.10


    for file_idx, file_path in enumerate(file_list):
        if file_idx % 1000 == 0:
            print(f'PID {pid}: {file_idx} / {len(file_list)} images processed')
        nameLower = file_path.lower()
        if not nameLower.endswith('.jpg') and not nameLower.endswith('.png'):
            continue

        img_cv = cv2.imread(file_path)
        img_ori = Image.open((file_path))

        height, width, _ = img_cv.shape

        lp = variance_of_laplacian(img_cv)
        br = calculate_brightness(img_ori)

        if (bBr != 1):
            br = 1.0

        # transform mask image fit to face image
        mask_surf, mask_shape, mask_surf_type = mask_transformation(br, lp)

        rects = face_detector(img_cv, 1)

        # no face detected
        if len(rects) == 0:
            continue

        pts_res = []
        vertices_lst = []

        for rect in rects:
            # get face region & resize face img
            img, roi_box = get_face_region(face_regressor, img_cv, rect)

            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)
            pts_res.append(pts68)

            vertices = predict_dense(param, roi_box)
            vertices_lst.append(vertices)

            points = np.empty((1, 4), dtype=float)
            for i in range(0, pts68[0].size):
                points = np.append(points, np.array([[pts68[1][i], pts68[0][i], pts68[2][i], 1.0]]),
                                   axis=0)

            points = np.delete(points, [0, 0], axis=0)

            mType = mask_surf_type
            objStr = get_surface(mType, points)

            sur_obj_string = surf_2_obj_str(objStr, mask_shape, numLineVertices)
            srf = pygame.display.set_mode((width, height), OPENGL | RESIZABLE | DOUBLEBUF | HIDDEN)
            size = srf.get_size()

            imgSurf = pygame.image.load(file_path).convert()
            image = pygame.image.tostring(imgSurf, 'RGBA', 1)

            obj = MaskSurfObj(fileString=sur_obj_string, imgHeight=height, swapxy=True, mask_surf = mask_surf)

            # real rendering using openGL lib
            buffer = gl_rendering(image, obj, size, img_shape=(width,height))

            # image save
            screen_surf = pygame.image.fromstring(buffer, size, "RGBA", True)
            pygame.image.save(screen_surf, file_path.replace(dataset_path, save_dataset_path))

    if len(file_list) > 0:
        print(f'PID {pid}: {file_idx+1} / {len(file_list)} images processed')
    print(f'PID {pid}: mask augmentation completed')


def gl_rendering(image, obj, size, img_shape=(224,224)):
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    ambient = [0.0, 0.0, 0.0, 0.0]
    diffuse = [0.8, 0.8, 0.8, 0.0]
    specular = [0.0, 0.0, 0.0, 0.0]
    position = [-0.4, -0.5, 1.0, 0.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular)
    glLightfv(GL_LIGHT0, GL_POSITION, position)

    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_shape[0], img_shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

    idx = glGenLists(1)
    glDisable(GL_LIGHTING)
    glNewList(idx, GL_COMPILE)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texid)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(0, 0, -200)
    glTexCoord2f(1, 0)
    glVertex3f(img_shape[0], 0, -200)
    glTexCoord2f(1, 1)
    glVertex3f(img_shape[0], img_shape[1], -200)
    glTexCoord2f(0, 1)
    glVertex3f(0, img_shape[1], -200)
    glEnd()
    glEndList()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glOrtho(0, img_shape[0], 0, img_shape[1], -10000, 10000)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glCallLists([idx, obj.gl_list])

    buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)

    return buffer