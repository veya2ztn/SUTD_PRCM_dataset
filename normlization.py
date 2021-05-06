import numpy as np
import os,json,time,math,shutil,random
import torch
from scipy.interpolate import interp1d
class heq_base:
    def __init__(self,**kargs):
        self.original_data_hist  = None
        self.original_data_cdf   = None
        self.original_data_x_arr = None
    def show_data_profile(self):
        hist  =self.original_data_hist
        cdf   =self.original_data_cdf
        x_part=self.original_data_x_arr
        y = cdf.min()+cdf.max()*(x_part-x_part.min())/(x_part.max()-x_part.min())
        plt.figure(1)
        plt.plot(x_part,hist)
        plt.figure(2)
        plt.plot(x_part,cdf)
        plt.plot(x_part,y,color='r')
    def absoulatly_increse(self,cdf):
        same_index_now=[]
        for index,num in enumerate(cdf):
            if index == 0:last_num = num
            if last_num < num:
                if len(same_index_now)>1:
                    cdf[same_index_now]=np.linspace(last_num,num,len(same_index_now)+1)[:-1]
                last_num=num
                same_index_now = []
            same_index_now.append(index)
        return cdf
    def forf(self,data):
        x_min = self.forward_range[0]
        x_max = self.forward_range[1]
        return self.convert(data,self.forward,x_min,x_max)
    def invf(self,data):
        x_min = self.inverse_range[0]
        x_max = self.inverse_range[1]
        return self.convert(data,self.inverse,x_min,x_max)
class heq(heq_base):
    def __init__(self,data,bins=3000,min_response_mean=0.0007,min_response_var=0.001,**kargs):
        # data must be 1-D array
        self.regist_data_shape=data.shape
        if data.min()==data.max() or (data.mean()<min_response_mean) or (data.var()<min_response_var):
            #print("mean:{:.4f} var:{:.4f}".format(data.mean().item(),data.var().item()))
            self.forward = lambda x:x
            self.inverse = lambda x:x
            self.inverse_range = self.forward_range = (data.min(),data.max())
            return
        x_array   = np.linspace(data.min(),data.max(),bins)
        hist, _   = np.histogram(data,bins=bins)
        cdf       = np.cumsum(hist)
        cdf       = cdf/cdf.max()
        cdf       = self.absoulatly_increse(cdf)
        self.original_data_hist = hist
        self.original_data_cdf  = cdf
        self.original_data_x_arr= x_array
        should    = (cdf-cdf.min())*(x_array.max()-x_array.min())/cdf.max()+ x_array.min()
        self.forward    = interp1d(x_array,should,kind='cubic')
        self.inverse    = interp1d(should,x_array,kind='cubic')
        self.inverse_range=(should.min(),should.max())
        self.forward_range=(x_array.min(),x_array.max())
    def convert(self,data,machine,x_min,x_max):
        assert data.shape[1:] == self.regist_data_shape[1:]
        input_shape= data.shape
        if isinstance(data,torch.Tensor):
            data     = torch.clamp(data,x_min,x_max)
            data     = machine(data)
            data     = torch.Tensor(data)
        else:
            data     = np.clip(data,x_min,x_max)
            data     = machine(data)
        return data
class heq_height(heq_base):
    def __init__(self,data,bins=1000,**kargs):
        # real data shape usually is (n,1,128)
        self.regist_data_shape=data.shape
        data      = data.reshape(data.shape[0],-1)
        if isinstance(data,torch.Tensor):height     = data.max(-1)[0]-data.min(-1)[0]
        else:height     = data.max(-1)-data.min(-1)

        x_array   = np.linspace(height.min(),height.max(),bins)
        hist, _   = np.histogram(height,bins=bins)
        cdf       = np.cumsum(hist)
        cdf       = cdf/cdf.max()
        cdf       = self.absoulatly_increse(cdf)
        self.original_data_hist = hist
        self.original_data_cdf  = cdf
        self.original_data_x_arr= x_array
        should=(cdf-cdf.min())*(x_array.max()-x_array.min())/cdf.max()+ x_array.min()
        self.inverse_range=(should.min(),should.max())
        self.forward_range=(x_array.min(),x_array.max())
        self.forward    = interp1d(x_array,should,kind='cubic')
        self.inverse   = interp1d(should,x_array,kind='cubic')

    def convert(self,data,machine,x_min,x_max):
        eps=1e-4
        assert data.shape[1:] == self.regist_data_shape[1:]
        input_shape= data.shape
        if isinstance(data,torch.Tensor):
            height     = data.max(-1)[0]-data.min(-1)[0]
            height     = torch.clamp(height,x_min+eps,x_max-eps)
            height_next= machine(height)
            height_next= torch.Tensor(height_next)
        else:
            height     = data.max(-1)-data.min(-1)
            height     = np.clip(height,x_min+eps,x_max-eps)
            height_next= machine(height)
        ratio      = height_next/height
        ratio      = ratio.reshape(ratio.shape[0],1,1)
        data       = data*ratio
        return data
class heq_element:
    def __init__(self,data,bins=1000,**kargs):
        assert len(data.shape)==3
        self.heqers = [heq(x,bins) for x in data.permute(2,0,1)]
    def forf(self,data):
        assert len(data.shape)==3
        newcurvedata=[theq.forf(d) for theq,d in zip(self.heqers,data.permute(2,0,1))]
        newcurvedata=np.stack(newcurvedata,-1)
        return data.__class__(newcurvedata)
    def invf(self,data):
        assert len(data.shape)==3
        newcurvedata=[theq.invf(d) for theq,d in zip(self.heqers,data.permute(2,0,1))]
        newcurvedata=np.stack(newcurvedata,-1)
        return data.__class__(newcurvedata)

class gauss_normer:
    def __init__(self,data,eps=0.01,**kargs):
        mean = data.mean(0)
        var  = torch.sqrt(data.var(0))
        self.mean = mean
        self.var  = var+eps
        self.eps  = eps
    def forf(self,x):
        return (x-self.mean)/(self.var)
    def invf(self,x):
        return x*self.var+self.mean

class response_gauss_normer(gauss_normer):
    def __init__(self,data,meaneps=0.001,vareps=0.01,**kargs):
        mean = data.mean(0)
        var  = torch.sqrt(data.var(0))
        var[mean<meaneps]= 1
        var[var<vareps]  = 1
        self.mean = mean
        self.var  = var

class max_normer:
    def __init__(self,data,eps=0.01,**kargs):
        self.eps      = eps
        self.maxvalue = data.abs().max(dim=0)[0]+eps
    def forf(self,x):
        return x/self.maxvalue
    def invf(self,x):
        return x*self.maxvalue

class mean2zero_normer:
    def __init__(self,data,**kargs):
        self.mean = data.mean(0)
    def forf(self,x):
        return x-self.mean
    def invf(self,x):
        return x+self.mean

class mean_normer:
    def __init__(self,data,eps=0.01,**kargs):
        self.eps      = eps
        self.mean = data.mean(0)+eps
    def forf(self,x):
        return x/self.mean
    def invf(self,x):
        return x*self.mean

class minmax_normer:
    def __init__(self,data,eps=0.01,**kargs):
        self.eps      = eps
        self._min = data.min(0)[0]
        self._max = data.max(0)[0]
        self.delta= self._max-self._min+eps
    def forf(self,x):
        return (x-self._min)/self.delta
    def invf(self,x):
        return x*self.delta+self._min

class minmaxgauss_normer:
    def __init__(self,data,eps=0.01,**kargs):
        self.eps      = eps
        self._min =_min= data.min(0)[0]
        self._max =_max= data.max(0)[0]
        self.delta=delta==_max-_min+eps
        temp = (data-_min)/delta
        self.mean = temp.mean(0)
        self.var  = torch.sqrt(temp.var(0))+eps
    def forf(self,x):
        return ((x-self._min)/self.delta-self.mean)/self.var
    def invf(self,x):
        return (x*self.var+self.mean)*self.delta+self._min

class Identity_normer:
    def forf(self,x):return x
    def invf(self,x):return x

class MinersOne_normer:
    def forf(self,x):return 1-x
    def invf(self,x):return 1-x

# class distribution_normer:
#     def __init__(self,data,eps=0.00001,**kargs):
#         assert data.min()>=0
#         self.eps = eps
#         shape    = tuple(range(1,len(data.shape)))
#         self.sum = data.sum(shape,keepdims=True)+eps
#     def forf(self,x):
#         return x/self.sum
#     def invf(self,x):
#         return x*self.sum

def normlizationer(data,method,eps=0.01,**kargs):
    if   method is None:return Identity_normer()
    elif method == "none":return Identity_normer()
    elif method == "norm1in01":return MinersOne_normer()
    elif method in norm_dict: return norm_dict[method](data,eps=0.01,**kargs)
    else:
        print(f"the normf should be {list(norm_dict.keys())} or none")
        raise NotImplementedError
norm_dict={"max":max_normer,
           "mean2zero":mean2zero_normer,
           "mean":mean_normer,
           "minmax":minmax_normer,
           "gauss":gauss_normer,
           "minmaxgauss":minmaxgauss_normer,
           "resp_gauss":response_gauss_normer,
           "heq":heq,
           "heq-h":heq_height,
           "heq-e":heq_element}
