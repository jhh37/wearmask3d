import os
import torch
import torchvision.transforms as transforms
import models.mobilenet_v1 as mobilenet_v1
import cv2
import dlib
import json
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, predict_dense
from utils.estimate_pose import parse_pose

from PIL import Image, ImageEnhance
import torch.backends.cudnn as cudnn
import random

from pygame.constants import *
from obj_loader import *

import geomdl
from geomdl import NURBS
from geomdl import exchange

from tqdm import tqdm


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


def isGeomdlNewVersion():
    ver = geomdl.__version__.split('.')
    if (int(ver[0]) >= 5 or (int(ver[0]) == 5 and int(ver[1]) >= 3)):
        return True
    else:
        return False


def main():
    pygame.init()

    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)

    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)

    cudnn.benchmark = True
    model = model.cuda()
    model.eval()

    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    bLp = 1
    bBr = 1

    with open('config.json') as json_file:
        configs = json.load(json_file)

    dataset_path = configs["srcPath"]
    save_dataset_path = configs["dstPath"]
    version = configs["version"]
    brTh = configs["brightnessThreshold"]
    lpTh = configs["laplacianVarianceThreshold"]
    masks = configs["masks"]

    nMask = len(masks)
    maskWeightSum = 0

    for i in range(nMask):
        maskWeightSum += masks[i]['weight']

    imgNumber = 0
    for root, dirs, files in os.walk(dataset_path):
        imgNumber += len(files)

    tBar = tqdm(total=imgNumber, initial=0)
    for root, dirs, files in os.walk(dataset_path, topdown=True):
        new_root = root.replace(dataset_path, save_dataset_path)

        if not os.path.exists(new_root):
            os.makedirs(new_root)

        for name in files:
            nameLower = name.lower()
            if nameLower.endswith('.jpg') or nameLower.endswith('.png'):

                img_fp = os.path.join(root, name)
                img_cv = cv2.imread(img_fp)
                img_ori = Image.open((img_fp))


                height, width, _ = img_cv.shape

                lp = variance_of_laplacian(img_cv)
                br = calculate_brightness(img_ori)

                if (bBr != 1):
                    br = 1.0

                rndVal = random.randrange(0, maskWeightSum)
                wAcc = 0
                maskIdx = 0
                for i in range(nMask):
                    wAcc += masks[i]['weight']
                    if (rndVal < wAcc):
                        maskIdx = i
                        break

                maskFileName = masks[maskIdx]['name']
                maskShape = masks[maskIdx]['shape']
                maskSurfaceType = masks[maskIdx]['surface']
                maskMinSize = masks[maskIdx]['minSize']

                fileName = name

                enhancer = ImageEnhance.Brightness(Image.open(maskFileName))
                enhanced_im = enhancer.enhance(0.7 + 0.3*min(brTh,br)/brTh)

                if (bLp == 1 and lp < lpTh):
                    w,h = enhanced_im.size
                    enhanced_im = enhanced_im.resize((int(w*max(maskMinSize,lp/lpTh)), int(h*max(maskMinSize,lp/lpTh))))


                enhanced_im.save("mask08.png")




                rects = face_detector(img_cv, 1)

                if len(rects) == 0:
                    continue

                pts_res = []
                Ps = []
                poses = []
                vertices_lst = []


                for rect in rects:
                    pts = face_regressor(img_cv, rect).parts()
                    pts = np.array([[pt.x, pt.y] for pt in pts]).T
                    roi_box = parse_roi_box_from_landmark(pts)


                    img = crop_img(img_cv, roi_box)

                    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                    input = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        input = input.cuda()
                        param = model(input)
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    pts68 = predict_68pts(param, roi_box)

                    pts_res.append(pts68)
                    P, pose = parse_pose(param)
                    Ps.append(P)
                    poses.append(pose)

                    vertices = predict_dense(param, roi_box)
                    vertices_lst.append(vertices)

                    points = np.empty((1, 4), dtype=float)
                    for i in range(0, pts68[0].size):
                        points = np.append(points, np.array([[pts68[1][i], pts68[0][i], pts68[2][i], 1.0]]),
                                           axis=0)

                    points = np.delete(points, [0, 0], axis=0)

                    chin_left = points[2]
                    chin_right = points[14]
                    chin_bottom = points[8]
                    nose_tip = points[30]
                    nose_top = points[27]
                    lip_bottom = points[57]

                    chin_middle = (chin_left + chin_right) / 2
                    normalVector = nose_tip - chin_middle
                    verticalVector = lip_bottom - nose_top
                    horVector = chin_right - chin_left


                    if (version == '1.1'):
                        mType = maskSurfaceType

                        c = []
                        for i in range(0, 28):
                            c.append(np.array([0, 0, 0, 1.0]))

                        if (mType == 4): # smooth (center-focused)
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
                        elif (mType == 2): # angulated
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

                        c[0] = chin_left - horVector*c7*1.5 + normalVector*0.1
                        c[6] = chin_right + horVector*c7*1.5 + normalVector*0.1
                        c[3] = nose_tip
                        c[1] = c[0] + normalVector*c1 + horVector*c7
                        c[5] = c[6] + normalVector*c1 - horVector*c7
                        c[2] = (c[1]+c[3])/2
                        c[4] = (c[5]+c[3])/2

                        for i in range(0, 7):
                            c[i] -= verticalVector * 0.4
                            c[i + 21] = c[i] + verticalVector * 1.4


                        c[1] -= normalVector * c2
                        c[2] -= normalVector * c2
                        c[3] -= normalVector * c2
                        c[4] -= normalVector * c2
                        c[5] -= normalVector * c2

                        c[0] += normalVector * c6*2
                        c[1] += normalVector * c6
                        c[5] += normalVector * c6
                        c[6] += normalVector * c6*2
                        c[7] += normalVector * c6*2
                        c[8] += normalVector * c6
                        c[12] += normalVector * c6
                        c[13] += normalVector * c6*2

                        c[21] += horVector * c8
                        c[22] -= -horVector * c8 + normalVector * 0.2
                        c[23] -= normalVector * 0.2
                        c[24] -= normalVector * 0.2
                        c[25] -= normalVector * 0.2
                        c[26] -= horVector * c8 + normalVector * 0.2
                        c[27] -= horVector * c8


                        for i in range(7, 14):
                            c[i] = (7*c[i - 7] + 3*c[i + 14]) / 10
                            c[i+7] = (1*c[i - 7] + 4*c[i + 14]) / 5


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

                        surf.delta = 0.025

                        if isGeomdlNewVersion():
                            numLineVertices = 40 # 5.3.1
                        else:
                            numLineVertices = 41 # 5.2.10


                    elif (version == '1.0'):
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

                        surf.delta = 0.03

                        if isGeomdlNewVersion():
                            numLineVertices = 33 # 5.3.1
                        else:
                            numLineVertices = 34 # 5.2.10


                    objStr = exchange.export_obj_str(surf)

                    firstF = True
                    surfObjString = ""
                    for idx, line in enumerate(objStr.splitlines()):
                        if (idx == 0):
                            surfObjString += 'mtllib nurbs_surf.mtl\n'
                        if (line[0] == 'f'):
                            if (firstF):
                                for y in range(0, numLineVertices):
                                    for x in range(0, numLineVertices):
                                        xCoord = 1 - x/(numLineVertices-1)
                                        y_ = 1 - y/(numLineVertices-1)
                                        if (y_ >= 0.5):
                                            if (maskShape == 1):
                                                yCoord = 2*(y_-0.5)**2 + 0.5
                                            elif (maskShape == 3):
                                                yCoord = -y_ * y_ + 2.5 * y_ - 0.5
                                            elif (maskShape == 4):
                                                if (xCoord >= 0.5):
                                                    yCoord = y_ + (2*xCoord - 1) * 0.1
                                                else:
                                                    yCoord = y_ + (1 - 2*xCoord) * 0.1
                                            else:
                                                yCoord = y_
                                        else:
                                            if (maskShape == 2):
                                                yCoord = -2*(y_-0.5)*(y_-0.5) + 0.5
                                            else:
                                                yCoord = y_
                                        surfObjString += 'vt {} {}\n'.format(xCoord, yCoord)
                                surfObjString += 'usemtl material00\n'
                                firstF = False

                            fList = line.split()
                            surfObjString += 'f {}/{} {}/{} {}/{}\n'.format(fList[1], fList[1], fList[2], fList[2], fList[3],
                                                                 fList[3])
                        else:
                            surfObjString += line + '\n'




                    srf = pygame.display.set_mode((width, height), OPENGL | RESIZABLE | DOUBLEBUF | HIDDEN)

                    imgSurf = pygame.image.load(img_fp).convert()
                    image = pygame.image.tostring(imgSurf, 'RGBA', 1)

                    obj = OBJ(fileString=surfObjString, imgHeight=height, swapxy=True)

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
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                    GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                    GL_NEAREST)
                    glBindTexture(GL_TEXTURE_2D, texid)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                                 GL_UNSIGNED_BYTE, image)


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
                    glVertex3f(width, 0, -200)
                    glTexCoord2f(1, 1)
                    glVertex3f(width, height, -200)
                    glTexCoord2f(0, 1)
                    glVertex3f(0, height, -200)
                    glEnd()
                    glEndList()

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()

                    glOrtho(0, width, 0, height, -10000, 10000)


                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()

                    glCallLists([idx, obj.gl_list])

                    size = srf.get_size()
                    buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
                    screen_surf = pygame.image.fromstring(buffer, size, "RGBA", True)
                    pygame.image.save(screen_surf, os.path.join(new_root, fileName))

            tBar.update(1)
            os.remove("mask08.png")



if __name__ == '__main__':
    main()
