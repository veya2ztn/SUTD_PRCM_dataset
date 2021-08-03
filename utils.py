import numpy as np
import os,json,time,math,shutil,random
import torch

class cPrint:
    def __init__(self,verbose=True):
        self.verbose = verbose
    def __call__(self,string):
        if  self.verbose:print(string)

def download_dropbox_url(url,filepath,redownload=False):
    import requests
    DATAROOT,basename = os.path.split(filepath)
    redownload = (redownload=='redownload')
    if os.path.exists(filepath) and (not redownload):
        print(f"{filepath} has already downloaded, set download='redownload' to force download")
    else:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        print(f"now we download {basename} from url {url}")
        r = requests.get(url, stream=True, headers=headers)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"now we unzip {basename} at {DATAROOT}")
        os.system(f"unzip -d {DATAROOT} {filepath}")

def np2torch(tensor):
    if isinstance(tensor,torch.Tensor):return tensor
    if 'complex' in tensor.dtype.name:
        return torch.Tensor(np.stack([tensor.real,tensor.imag],-1)).float()
    else:
        return torch.Tensor(tensor).float()
def read_curve_data(path):
    store=[[],[],[],[],[]]
    i=0
    with open(path,'r') as f:
        for line in f:
            if i==0:
                i=1
                continue
            for i,val in enumerate(line.strip().split(',')):
                store[i].append(float(val))
    feq,s1r,s1i,s2r,s2i=store
    s1=np.array([np.complex(r,i) for r,i in zip(s1r,s1i)])
    s2=np.array([np.complex(r,i) for r,i in zip(s2r,s2i)])
    return feq,s1,s2

def curver_filte_smoothpeak(tensor0,low_resp=0.1,smooth=0.01):
    if isinstance(tensor0,torch.Tensor):tensor0=tensor0.numpy()
    maxten    = np.max(tensor0,1)
    maxfilter = np.where(maxten>0.1)[0]
    #tensor0   = tensor0[maxfilter]
    tensor=np.pad(tensor0,((0,0),(1,1)),"edge")
    grad_r = tensor[...,2:]-tensor[...,1:-1]
    grad_l = tensor[...,1:-1]-tensor[...,:-2]
    out = np.abs((grad_l - grad_r))
    maxout=np.max(out,1)
    smoothfilter=np.where(maxout<0.01)[0]
    filted_index=np.intersect1d(maxfilter,smoothfilter)
    return filted_index

def random_v_flip(data):
    batch,c,w,h = data.shape
    index=torch.randint(2,(batch,))==1
    data[index]=data[index].flip(2)
def random_h_flip(data):
    batch,c,w,h = data.shape
    index=torch.randint(2,(batch,))==1
    data[index]=data[index].flip(3)

def find_peak(tensor,include_boundary=True,complete=False,level=10):
    #this function will return all the peak, include
    #   /\  and  the center of     /——\
    # /  \                       /      \
    # we will operte the last dim for input tensor. (X,X,N)
    # we will operte on the numpy format
    #tensor = copy.deepcopy(train_loader.dataset.curvedata)
    # if the max postion at the boundary, it will be consider as a peak

    totensor=False
    if isinstance(tensor,torch.Tensor):
        totensor=True
        tensor = tensor.numpy() # limit for the cpu tensor
    tensor = tensor.round(4)    # limit the precision

    out = 0

    if include_boundary:
        new_tensor = np.zeros_like(tensor)
        new_tensor[(*np.where(tensor.argmax(-1)==0),0)]=1
        out = out+new_tensor

        new_tensor = np.zeros_like(tensor)
        new_tensor[(*np.where(tensor.argmax(-1)==tensor.shape[-1]-1),-1)]=1
        out = out+new_tensor

    p_zero  = np.zeros_like(tensor[...,0:1])
    p_one   = np.ones_like(tensor[...,0:1])

    btensor = np.concatenate([p_one,tensor,p_zero],-1)


    grad_r  = (btensor[...,1:-1]-btensor[...,:-2])
    grad_r  = np.sign(grad_r)


    # find the good peak # fast way
    #grad_l = tensor[...,2:]-tensor[...,1:-1]
    #grad_l = np.sign(grad_l)
    #out = ((grad_r - grad_l) == 2)+ 0

    # find the plat
    search_seq = []
    for i in range(level):
        search_seq+=[[1]+[0]*i+[-1]]
    #search_seq += [[1,-1],[1,0,-1],[1,0,0,-1],[1,0,0,0,-1],[1,0,0,0,0,-1],[1,0,0,0,0,0,-1],[1,0,0,0,0,0,0,-1]]
    # for our data, there is only few or no large plat if we desample data from 1001 to 128
    for seq in search_seq: out=out+find_seq(grad_r,seq)
    #     plat0=find_seq(grad_r,[1,-1])
    #     plat1=find_seq(grad_r,[1,0,-1])
    #     plat2=find_seq(grad_r,[1,0,0,-1])
    #     plat3=find_seq(grad_r,[1,0,0,0,-1])
    #     plat4=find_seq(grad_r,[1,0,0,0,0,-1])
    out = np.sign(out)
    #out = out*active[...,:-1]
    if totensor:out = torch.Tensor(out)
    if complete:
        no_peak_id =  np.where(out.sum(-1)==0)
        out[(*no_peak_id,-1)]=1
    return out

def find_seq(grad,seq,return_mode='c',return_type='onehot'):
    seq      = np.array(seq)
    Na, Nseq = grad.shape[-1], seq.size
    r_seq    = np.arange(Nseq)
    M = (grad[...,np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(-1)+0
    out = np.stack(np.where(M==1))
    pos = out[-1]
    if   return_mode == 'c':pos=pos+ Nseq//2-1
    elif return_mode == 'l':pos=pos-1
    elif return_mode == 'r':pos=pos+Nseq-1
    if return_type == 'index':return
    elif return_type == 'onehot':
        new_tensor = np.zeros_like(grad)
        new_tensor[(*out[:-1],pos)]=1
        return new_tensor
def has_peak(tensor):
    return (find_peak(tensor).sum(1)>0)+0


def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r
normer = np.linalg.norm
def _c(ca, i, j, p, q):
    if ca[i, j] > -1:return ca[i, j]
    elif i == 0 and j == 0:ca[i, j] = normer(p[i]-q[j])
    elif i > 0 and j == 0:ca[i, j]  = max(_c(ca, i-1, 0, p, q), normer(p[i]-q[j]))
    elif i == 0 and j > 0:ca[i, j]  = max(_c(ca, 0, j-1, p, q), normer(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            normer(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]
def frdist(p, q):
    """
        Computes the discrete Fréchet distance between
        two curves. The Fréchet distance between two curves in a
        metric space is a measure of the similarity between the curves.
        The discrete Fréchet distance may be used for approximately computing
        the Fréchet distance between two arbitrary curves,
        as an alternative to using the exact Fréchet distance between a polygonal
        approximation of the curves or an approximation of this value.
        This is a Python 3.* implementation of the algorithm produced
        in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
        Tech. Report CD-TR 94/64, Information Systems Department, Technical
        University of Vienna.
        http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
        Function dF(P, Q): real;
            input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
            return: δdF (P, Q)
            ca : array [1..p, 1..q] of real;
            function c(i, j): real;
                begin
                    if ca(i, j) > −1 then return ca(i, j)
                    elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                    elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                    elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                    elsif i > 1 and j > 1 then ca(i, j) :=
                    max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                    else ca(i, j) = ∞
                    return ca(i, j);
                end; /* function c */
            begin
                for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
                return c(p, q);
            end.
        Parameters
        ----------
        P : Input curve - two dimensional array of points
        Q : Input curve - two dimensional array of points
        Returns
        -------
        dist: float64
            The discrete Fréchet distance between curves `P` and `Q`.
        Examples
        --------
        >>> from frechetdist import frdist
        >>> P=[[1,1], [2,1], [2,2]]
        >>> Q=[[2,2], [0,1], [2,4]]
        >>> frdist(P,Q)
        >>> 2.0
        >>> P=[[1,1], [2,1], [2,2]]
        >>> Q=[[1,1], [2,1], [2,2]]
        >>> frdist(P,Q)
        >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:raise ValueError('Input curves are empty.')
    if len_p != len_q:raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist


def check_has_file(_dir,pattern):
    _list = []
    for n in os.listdir(_dir):
        match=pattern.match(n)
        if match:_list.append(match.group())
    return _list

import re
pattern_curve = re.compile(r'Integrate_curve_[\d]*.npy')
pattern_image = re.compile(r'Integrate_image_[\d]*.npy')

def convertlist(_dir):
    if isinstance(_dir,list):return _dir
    if not isinstance(_dir,str):
        print("the dataset dir either is a list or a dir for these list")
        raise
    if os.path.isdir(_dir):
        CURVE_PATH=[]
        IMAGE_PATH=[]
        for curve_image_path in os.listdir(_dir):
            abs_path = os.path.join(_dir,curve_image_path)
            if not os.path.isdir(abs_path):continue
            for n in os.listdir(abs_path):
                curve_match=pattern_curve.match(n)
                image_match=pattern_image.match(n)
                if curve_match:curve_data=curve_match.group()
                if image_match:image_data=image_match.group()
            CURVE_PATH.append(os.path.join(abs_path,curve_data))
            IMAGE_PATH.append(os.path.join(abs_path,image_data))
        return CURVE_PATH,IMAGE_PATH
    else:
        print(_dir)
        print("the dataset dir either is a list or a dir for these list")
        raise

def tuple2str(tup):
    a=",".join([str(i) for i in tup])
    return "("+a+")"

def get_contour_position(images):
    img_big=np.pad(images,((0,0),(1,1),(1,1)))
    right=(img_big[...,2:,1:-1]-images)**2
    left =(img_big[...,:-2,1:-1]-images)**2
    top  =(img_big[...,1:-1,2:]-images)**2
    bot  =(img_big[...,1:-1,:-2]-images)**2
    mask = (right+left+top+bot)*images
    return mask

def get_contour_data(images,dim=50):
    contours_mask=get_contour_position(images)
    real_theta_vector = []
    real_norm_vector  = []
    expand_norm_vector= []

    for i in range(len(contours_mask)):
        mask  = contours_mask[i]
        y,x = np.where(mask>0)
        contour= np.stack([x,y],-1)
        center = [7.5,7.5]
        coodr  = (contour-center)*[1,-1]
        norm = np.linalg.norm(coodr,axis=-1)
        theta= np.arctan2(coodr[...,1],coodr[...,0])
        order=np.argsort(theta)
        norm_s=norm[order]
        theta_s=theta[order]
        norm_a =np.concatenate([norm_s,norm_s,norm_s])
        theta_a=np.concatenate([theta_s-2*np.pi,theta_s,theta_s+2*np.pi])
        itpd=interp1d(theta_a,norm_a,kind='linear')
        angles=np.linspace(0,2,dim)*np.pi
        y_out  = itpd(angles)
        real_theta_vector.append(theta_s)
        real_norm_vector.append(norm_s)
        expand_norm_vector.append(y_out)

    real_norm_vector=np.stack([np.pad(kk,(0,100-len(kk)),constant_values=-1) for kk in real_norm_vector])
    real_theta_vector=np.stack([np.pad(kk,(0,100-len(kk)),constant_values=-1) for kk in real_theta_vector])
    real_vector = np.stack([real_norm_vector,real_theta_vector],-1)
    expand_norm_vector=np.array(expand_norm_vector)
    return real_vector,expand_norm_vector

def get_unicode_of_image(image):
    key = "".join([str(d) for d in image])
    return key
def check_image_repeat(curve_path_list,image_path_list):
    from fastprogress import master_bar,progress_bar

    if not isinstance(curve_path_list,list) and image_path_list is None:
        # this code reveal the curve file and image file from a high level path ../Data##
        curve_path_list,image_path_list = convertlist(curve_path_list)
    if isinstance(curve_path_list,str) and isinstance(image_path_list,str):
        # this code for single image and curve file
        if os.path.isfile(curve_path_list) and os.path.isfile(image_path_list):
            curve_path_list=[curve_path_list]
            image_path_list=[image_path_list]

    image_pool = {}
    repeat_list= []
    replace_list={}
    mb = master_bar(range(len(curve_path_list)))
    for idx in mb:
        curve_path = curve_path_list[idx]
        image_path = image_path_list[idx]
        _,basename = os.path.split(image_path)
        images     = np.load(image_path)
        pb = progress_bar(range(len(images)),parent=mb)
        for i in pb:
            image = images[i]
            key = get_unicode_of_image(image)
            if key in image_pool:
                repeat_list.append([f"{basename}_{i}",image_pool[key]])
                if image_path not in replace_list:replace_list[image_path]=[]
                replace_list[image_path].append(i)
                print(f"{basename}_{i}->{image_pool[key]}")
            else:
                image_pool[key]=f"{basename}_{i}"
    return image_pool,repeat_list,replace_list

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
cmap = ['blue', 'green','orange','grey', 'purple',  'red','pink']
class PeakWiseCurve:
    def __init__(self,xdata,curve,peak_locs_index=None,fit_para=None):
        self.xdata = xdata
        self.curve = curve
        if peak_locs_index is None:
            assert len(curve.shape)==1
            self.peak_locs_index = np.where(find_peak(curve[None]))[1]
        else:
            assert isinstance(peak_locs_index,list)
            self.peak_locs_index = peak_locs_index
        self.peak_locs = self.xdata[self.peak_locs_index]
        if fit_para is None:
            self.peak_locs,self.fit_para = self.find_peak_para(self.xdata,self.curve,self.peak_locs)
            if self.peak_locs is None:self.peak_locs_index=None
        else:
            assert isinstance(fit_para,list)
            assert len(fit_para) == 3*len(self.peak_locs)+2
            self.fit_para = fit_para
    @property
    def width(self):
        if self.fit_para is None:return None
        array = self.fit_para[2:]
        peaks = len(self.peak_locs)
        peak_infos=[]
        for i in range(peaks):
            l = self.peak_locs[i]
            #h = array[3*i+0]
            b = np.sqrt(array[3*i+1])
            m_square = array[3*i+2]
            width = 2*b*np.sqrt(np.power(2,1/(1+m_square)-1))
            peak_infos.append([l,width])
        return peak_infos
    @property
    def peak_para(self):
        if self.fit_para is None:return None
        array = self.fit_para[2:]
        peaks = len(self.peak_locs)
        peak_infos=[]
        for i in range(peaks):
            l = self.peak_locs[i]
            h = array[3*i+0]
            w = np.sqrt(array[3*i+1])# because we dont use square in fitting
            p = np.sqrt(array[3*i+2])# because we dont use square in fitting
            peak_infos.append([l,(h,w,p)])
        return peak_infos
    @staticmethod
    def basic_func(x,peak_locs,o,k, args):
        if peak_locs is None:return 0*x
        p=0
        for i,peak_loc in enumerate(peak_locs):
            #p+=args[3*i+0]*np.power(1 + (x - xdata[peak_loc])**2/(args[3*i+1]**2),-(1 + args[3*i+2]**2))
            p+=args[3*i+0]*np.power(1 + (x - peak_loc)**2/(args[3*i+1]),-(1 + args[3*i+2]))
        p+=o+k*x
        return p

    @staticmethod
    def choice_fun(peak_locs):
        num = len(peak_locs)
        if num == 1:
            def func1(x,o,k,a1,b1,c1):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1])
            return func1
        if num == 2:
            def func2(x,o,k,a1,b1,c1,a2,b2,c2):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1,a2,b2,c2])
            return func2
        if num == 3:
            def func3(x,o,k,a1,b1,c1,a2,b2,c2,a3,b3,c3):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1,a2,b2,c2,a3,b3,c3])
            return func3
        if num == 4:
            def func4(x,o,k,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4])
            return func4
        if num == 5:
            def func5(x,o,k,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5])
            return func5
        if num == 6:
            def func6(x,o,k,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6):
                return PeakWiseCurve.basic_func(x,peak_locs,o,k, [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
            return func6
        raise NotImplementedError


    @staticmethod
    def find_peak_para(xdata,curve,peak_locs):
        full_data = np.stack([xdata,curve]).transpose()
        nozerosdata=full_data[curve>0.001]

        x = nozerosdata[:,0]
        y = nozerosdata[:,1]
        if (len(nozerosdata)<20) or (len(peak_locs)==0):
            peak_locs= None
            popt     = None
        else:
            func    = PeakWiseCurve.choice_fun(peak_locs)
            try:
                popt, _ = curve_fit(func, x, y,bounds=[0,5])
            except:
                popt=None

        return [peak_locs,popt]

    def show_fit(self):
        x     = self.xdata
        curve = self.curve
        plt.plot(x, curve)
        plt.plot(x, self.choice_fun(self.peak_locs)(x, *self.fit_para), 'r*')

    def show_fit_detail(self,final=False):
        x     = self.xdata
        curve = self.curve
        o,k   = self.fit_para[:2]
        plt.plot(x, curve,'r*')
        plt.plot(x, o+k*x,'b')
        for i,(loc,(h,b,m)) in enumerate(self.peak_para):
            plt.plot(x, PeakWiseCurve.choice_fun([loc])(x, 0,0,h,b**2,m**2),color=cmap[i])
        if final:plt.plot(x, self.choice_fun(self.peak_locs)(x, *self.fit_para), 'r*')

    def state_array(self,max_peaks=6):
        if self.peak_locs_index is not None:
            peaks_num = len(self.peak_locs_index)
            locs_code = np.pad(self.peak_locs_index,(0,max_peaks-peaks_num),constant_values=-1)
        else:
            locs_code = -np.ones(max_peaks)
        if self.fit_para is not None:
            paras_num = len(self.fit_para)
            para_code = np.pad(self.fit_para,(0,3*max_peaks+2-paras_num),constant_values=-1)
        else:
            para_code = -np.ones(3*max_peaks+2)
        return np.concatenate([locs_code,para_code])
