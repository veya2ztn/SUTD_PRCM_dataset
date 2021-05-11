from .plgpattern import *
import cv2
from PIL import Image
def generator_PLR_verts(center,out_radius,inn_radius):
    center_x = center
    center_y = center
    irregularity=np.random.random()*0.5
    spikeyness  =np.random.random()*0.5
    numVerts    =np.random.randint(3,20)
    out_verts = generatePolygon( ctrX=center_x, ctrY=center_y, aveRadius=out_radius, irregularity=irregularity, spikeyness=spikeyness, numVerts=numVerts )

    irregularity=np.random.random()*0.5
    spikeyness  =np.random.random()*0.5
    numVerts    =np.random.randint(3,8)
    inn_verts   = generatePolygon( ctrX=center_x, ctrY=center_y, aveRadius=inn_radius, irregularity=irregularity, spikeyness=spikeyness, numVerts=numVerts )
    #inn_verts   = [(x+center_x-inner_x,y+center_y-inner_y) for x,y in inn_verts]

    for pt in inn_verts:
        if cv2.pointPolygonTest(np.array(out_verts),pt,False) != 1:
            return False
    for pt in out_verts:
        if cv2.pointPolygonTest(np.array(inn_verts),pt,False) != -1:
            return False
    return out_verts,inn_verts

def generator_PLR_verts_no_overlop(center,out_radius,inn_radius):
    while True:
        out = generator_PLR_verts(center,out_radius,inn_radius)
        if out:break
    return out

def random_pad_resize(im):
    padding=np.random.randint(3)
    if padding==0:
        array=1-asarray(clip2zero(im).resize((16,16)))[:,:,0]//255
    elif padding==1:
        array=1-asarray(clip2zero(im).resize((14,14)))[:,:,0]//255
        array=np.pad(array,((1,1),(1,1)))
    elif padding==2:
        array=1-asarray(clip2zero(im).resize((12,12)))[:,:,0]//255
        array=np.pad(array,((2,2),(2,2)))
    return array
