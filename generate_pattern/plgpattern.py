import math, random
from PIL import Image,ImageDraw
from numpy import asarray
from fastprogress import master_bar,progress_bar
import numpy as np
def clip(x, min, max) :
    if( min > max ) :  return x
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ):
    '''Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]
    return points


def clip2zero(im):
    array=1-asarray(im)[:,:,0]//255
    assert len(array.shape)==2
    x,y=np.where(array==1)
    im=im.crop([y.min(),x.min(),y.max(),x.max()])
    return im

black = (0,0,0)
white=(255,255,255)
red = (255,0,0)

def generate_polygon():
    irregularity=np.random.random()*0.5
    spikeyness  =np.random.random()*0.5
    numVerts    =np.random.randint(3,20)
    #verts = generatePolygon( ctrX=250, ctrY=250, aveRadius=100, irregularity=0.35, spikeyness=0.2, numVerts=16 )
    verts = generatePolygon( ctrX=250, ctrY=250, aveRadius=130, irregularity=irregularity, spikeyness=spikeyness, numVerts=numVerts )
    im = Image.new('RGB', (500, 500), white)
    imPxAccess = im.load()
    draw = ImageDraw.Draw(im)
    tupVerts = map(tuple,verts)
    # either use .polygon(), if you want to fill the area with a solid colour
    draw.polygon( verts, outline=black,fill=black )
    return im
