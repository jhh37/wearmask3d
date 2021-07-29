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

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time

def MTL(filename, surf):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise (ValueError, "mtl file doesn't start with newmtl stmt")
        else:
            mtl[values[0]] = map(float, values[1:])

    mtl[values[0]] = values[1]
    # surf = pygame.image.load(os.path.dirname(filename) + '/' + mtl['map_Kd'])
    image = pygame.image.tostring(surf, 'RGBA', 1)
    ix, iy = surf.get_rect().size
    texid = mtl['texture_Kd'] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, image)

    return contents


class MaskSurfObj:
    def surface_normal(self, poly):
        n = [0.0, 0.0, 0.0]

        for i, v_curr in enumerate(poly):
            v_next = poly[(i + 1) % len(poly)]
            n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
            n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
            n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

        norm = (n[0]**2 + n[1]**2 + n[2]**2) ** 0.5
        normalised = [val / norm for val in n]
        s = (normalised[0]**2 + normalised[1]**2 + normalised[2]**2) ** 0.5
        return normalised


    def __init__(self, filename=None, fileString=None,
                 imgHeight=0, swapxy=False, swapyz=False, cw=False, mask_surf=None):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None

        if (filename):
            lines = open(filename, "r")
            path = os.path.dirname(filename)
        else:
            lines = fileString.splitlines()
            path = os.path.join(os.path.abspath(os.getcwd()))

        for line in lines:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapxy:
                    v = v[1], imgHeight-v[0], v[2]
                if swapyz:
                    v = v[0]/10, v[2]/10, v[1]/10
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapxy:
                    v = v[1], v[0], v[2]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(path + '/' + values[1], mask_surf)
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(int(w[0]))
                self.faces.append((face, norms, texcoords, material))

        if len(self.normals) == 0:
            self.normals = [[0,0,0] for _ in range(len(self.vertices))]
            tNum = [0]*len(self.vertices)
            for face in self.faces:
                vertices, normals, texture_coords, material = face
                triVertices = []
                for i in range(3):
                    triVertices.append(self.vertices[vertices[i] - 1])
                normal = self.surface_normal(triVertices)
                for i in range(3):
                    k = vertices[i]-1
                    for j in range(3):
                        self.normals[k][j] = (self.normals[k][j]*tNum[k] + normal[j])/(tNum[k]+1)
                    tNum[k] += 1

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        #glEnable(GL_LIGHTING)
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glEnable(GL_TEXTURE_2D)

        # for rendering 3DMM
        glDepthMask(GL_TRUE);

        mat_ambient = [1.0, 1.0, 1.0, 0.0]
        mat_diffuse = [1.0, 1.0, 1.0, 0.0]

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse)

        if (cw):
            glFrontFace(GL_CW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            if hasattr(self, 'mtl'):
                mtl = self.mtl[material]
                if 'texture_Kd' in mtl:
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                else:
                    glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()

        glFrontFace(GL_CCW)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glEndList()