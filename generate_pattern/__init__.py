from fastprogress import master_bar,progress_bar
import numpy as np
from .ptnpattern import combination
from .plgpattern import *
from .plrpattern import generator_PLR_verts_no_overlop,random_pad_resize,Image
def get_PTN_pattern(total_num,pattern_num="random"):
    total_image=np.zeros((total_num,16,16))
    mb = master_bar(range(1))
    ptn= combination()
    for i in mb:
        pb = progress_bar(range(total_num),parent=mb)
        for i in pb:
            num=np.random.randint(8,24) if pattern_num == "random" else pattern_num
            total_image[i]=ptn.get_no_cover_pattern(num)
    return total_image.astype('int')

def get_PLG_pattern(total_num,out_size_list=[16],pad_num=1):
    mb = master_bar(range(1))
    total_image=[]
    for _ in mb:
        pb = progress_bar(range(total_num),parent=mb)
        for i in pb:
            im = generate_polygon()
            total_image.append(im)

    mb = master_bar(range(len(out_size_list)))
    images_list=[]
    for idx in mb:
        pb = progress_bar(range(len(total_image)),parent=mb)
        outsize = out_size_list[idx]
        images  =np.zeros((total_num,outsize,outsize)).astype('int')
        for i in pb:
            im = total_image[i]
            padding = np.random.randint(3) if pad_num == 'rand3' else pad_num
            real_size = outsize-2*padding
            array=1-asarray(clip2zero(im).resize((real_size,real_size)))[:,:,0]//255
            array=np.pad(array,((padding,padding),(padding,padding)))
            images[i]=array.astype('int')
        images_list.append(images)
    return images_list

def get_PLR_pattern(total_num,out_size_list=[16],pad_num='rand3'):
    mb = master_bar(range(1))
    total_image=[]
    for _ in mb:
        pb = progress_bar(range(total_num),parent=mb)
        for i in pb:
            center     =250
            out_radius =200
            inn_radius =np.random.randint(80,160)
            out_verts,inn_verts = generator_PLR_verts_no_overlop(center,out_radius,inn_radius)
            im   = Image.new('RGB', (2*center, 2*center), white)
            draw = ImageDraw.Draw(im)
            draw.polygon(out_verts, outline=black,fill=black )
            draw.polygon(inn_verts, outline=white,fill=white )
            total_image.append(im)

    mb = master_bar(range(len(out_size_list)))
    images_list=[]
    for idx in mb:
        pb = progress_bar(range(len(total_image)),parent=mb)
        outsize = out_size_list[idx]
        images  =np.zeros((total_num,outsize,outsize)).astype('int')
        for i in pb:
            im = total_image[i]
            padding = np.random.randint(3) if pad_num == 'rand3' else pad_num
            real_size = outsize-2*padding
            array=1-asarray(clip2zero(im).resize((real_size,real_size)))[:,:,0]//255
            array=np.pad(array,((padding,padding),(padding,padding)))
            images[i]=array.astype('int')
        images_list.append(images)
    return images_list

def get_RDN_pattern(total_num,size = 16):
    return np.random.randint(0,2,size=(total_num,size,size))
