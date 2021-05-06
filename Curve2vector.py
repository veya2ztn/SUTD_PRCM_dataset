import numpy as np
import torch
import scipy.fftpack as fftpack
from scipy.interpolate import interp1d
import torch.nn.functional as F

def fftshift1D(tensor:torch.Tensor):
    '''
    shift the origin to corner
    '''
    assert tensor.shape[-1]==2
    dim_l = tensor.shape[-2]
    tensor=torch.roll(tensor,dim_l//2,-2)
    return tensor

def ifftshift1D(tensor:torch.Tensor):
    assert tensor.shape[-1]==2
    dim_l = tensor.shape[-2]
    tensor=torch.roll(tensor,-(dim_l//2),-2)
    return tensor

class CurveProcesser:
    def __init__(self,curve_len,**kargs):
        self.curve_len   = curve_len
        self.class_name  = None
        self.curve_dtype = ['no define']
        self.vector_dtype= ['no define']
        self.curve_field = 'no define'
        self.vector_field= 'no define'
        self.filter_type = 'no define'
        self.Normfactor  = 1
        self.feature     = "all"

    @property
    def name(self):
        self.class_name = "" if self.class_name is None else self.class_name
        return "({}.{})".format(self.class_name,self.method_name)

    def save_name(self):
        return self.name

    def torch2np(self,tensor):
        if isinstance(tensor,np.ndarray):return tensor
        tensor = tensor.detach().cpu().numpy()
        if tensor.shape[-1]==2:
            tensor = tensor[...,0]+1j*tensor[...,1]
        return tensor

    def np2torch(self,tensor):
        if isinstance(tensor,torch.Tensor):return tensor
        if 'complex' in tensor.dtype.name:
            return torch.Tensor(np.stack([tensor.real,tensor.imag],-1)).float()
        else:
            return torch.Tensor(tensor).float()

    def change_tensor_type(self,tensor,dtype):
        if isinstance(tensor,dtype):return tensor
        if isinstance(tensor,np.ndarray):
            tensor = self.np2torch(tensor)
        else:
            tensor = self.torch2np(tensor)
        return tensor

    def check_data(self,tensor,ComplexQ):
        '''
        check the data format.
        complex <- np.complex_array or torch.Tensor tensor.shape[-1]=2
        real  <- np.array or Tensor
                (B,D) or (B,D,1) or (B,C,D,1)
        will transform (B,D,1) -> (B,D)
        all the operation will operated on the last dimension
        '''
        TensorQ = isinstance(tensor,torch.Tensor)
        nArrayQ = isinstance(tensor,np.ndarray)
        assert TensorQ or nArrayQ
        if tensor.shape[-1]==1:tensor=tensor[...,0]
        if nArrayQ and ComplexQ:
            assert 'complex' in tensor.dtype.name
        if TensorQ and ComplexQ:
            assert tensor.shape[-1]==2
        if not ComplexQ:assert tensor.shape[-1]!=2
        return tensor

    def check_curve_data(self,tensor):
        TensorQ = isinstance(tensor,torch.Tensor)
        nArrayQ = isinstance(tensor,np.ndarray)
        CurveComplexQ = (self.curve_field == 'complex')
        tensor = self.check_data(tensor,CurveComplexQ)
        if TensorQ and CurveComplexQ:assert tensor.shape[-2] == self.curve_len
        if nArrayQ and CurveComplexQ:assert tensor.shape[-1] == self.curve_len
        if type(tensor) not in self.curve_dtype:
            tensor = self.change_tensor_type(tensor,self.curve_dtype[0])
        return tensor

    def check_vector_data(self,tensor):
        VectorComplexQ = (self.vector_field == 'complex')
        tensor = self.check_data(tensor,VectorComplexQ)
        if type(tensor) not in self.vector_dtype:
            tensor = self.change_tensor_type(tensor,self.vector_dtype[0])
        return tensor

    def curve2vector(self,tensor,num):
        raise NotImplementedError

    def vector2curve(self,freq,length=None,keep=True):
        raise NotImplementedError

    def output_dim(self,num):
        raise NotImplementedError

    def reduce(self,vector,mc):
        raise NotImplementedError

    def curve_loss(self,curve1,curve2,axis=-1,Quiet=False,mode='abs'):
        '''
        curve (B,branch,L) or (B,branch,L,2)
        '''
        shape1 = curve1.shape
        shape2 = curve2.shape
        assert shape1 == shape2
        if   shape1[-1]==1:
            curve1=curve1[...,0]
            curve2=curve2[...,0]
        curve1=self.change_tensor_type(curve1,np.ndarray)
        curve2=self.change_tensor_type(curve2,np.ndarray)
        if mode == 'abs':
            return np.abs(curve1-curve2).mean(axis)
        elif mode == 'square':
            return np.square(curve1-curve2).mean(axis)
        elif mode == 'selfattition':
            # curve2 should be target
            return (np.abs(curve1-curve2)*curve2).mean(axis)
        elif mode == 'selfenhattition':
            # curve2 should be target
            return (np.abs(curve1-curve2)*np.power(3,curve2)).mean(axis)

class IdentyProcess(CurveProcesser):
    def __init__(self,curve_len=1001,method='identy',**kargs):
        super().__init__(curve_len,**kargs)
        self.method      = method
        self.class_name  = "Identy"
        method_allow     = ['identy']
        if method not in method_allow:raise NotImplementedError
        self.save_name   = ""
        self.method_name = ""

    def output_dim(self,num):
        raise

    def reduce(self,vector,mc):
        raise

    def curve2vector(self,curve,num=None):
        return curve

    def vector2curve(self,vector,num=None,keep=True):
        return vector

class CurveFourier(CurveProcesser):
    '''
    Expand the 1D curve data by 'method' way
       - 'fft' for the Fouier Coef
       - 'taylor' for Taylor Expansion
       -
    the output form will be same as input
    '''
    def __init__(self,curve_len=1001,method='fft',**kargs):
        super().__init__(curve_len,**kargs)
        #if   method is 'fft':   expander=fft.fft
        #elif method=='taylor':expander=fft
        #now only support fft
        method_allow   = ['fft','dct','rfft']
        if method not in method_allow:raise NotImplementedError
        self.method      = method
        self.method_name = method
        self.class_name  = "Fourier"
        self.default_num = 100
        if method == 'fft':
            self.curve_dtype = [np.ndarray,torch.Tensor]
            self.vector_dtype= [np.ndarray,torch.Tensor]
            self.curve_field = 'complex'
            self.vector_field= 'complex'
            self.torch_transform = lambda x:fftshift1D(torch.fft(x,1))# 1 for 1D fft
            self.torch_inverse_t = lambda x:torch.ifft(ifftshift1D(x),1)
            self.numpy_transform = lambda x: fftpack.fftshift(fftpack.fft(x,axis=-1),axes=-1)
            self.numpy_inverse_t = lambda x:fftpack.ifft(fftpack.ifftshift(x,axes=-1),axis=-1)

        if method == 'rfft':
            self.curve_dtype = [np.ndarray,torch.Tensor]
            self.vector_dtype= [np.ndarray,torch.Tensor]
            self.curve_field = 'real'
            self.vector_field= 'complex'
            self.torch_transform = lambda x:torch.rfft(x,1)
            self.torch_inverse_t = lambda x:torch.irfft(x,1,signal_sizes=(curve_len,))
            self.numpy_transform = lambda x: np.fft.rfft(x,axis=-1,n=curve_len)
            self.numpy_inverse_t = lambda x:np.fft.irfft(x,axis=-1,n=curve_len).real

        if method == 'dct':
            self.curve_dtype = [np.ndarray]
            self.vector_dtype= [np.ndarray]
            self.curve_field = 'real'
            self.vector_field= 'real'
            self.numpy_transform = lambda x: fftpack.dct(x,axis=-1,n=curve_len,norm='ortho').real
            self.numpy_inverse_t = lambda x:fftpack.idct(x,axis=-1,n=curve_len,norm='ortho').real

    def reduce(self,vector,mc):
        '''
        for 1D data_only
        for shift freq data: i.e. the zero frequency at the mid point
        '''
        TensorQ   = isinstance(vector,torch.Tensor)
        nArrayQ   = isinstance(vector,np.ndarray)
        if self.method == 'fft':
            if mc is None:mc = self.curve_len
            if mc%2 != self.curve_len%2:mc = mc +1 # make sure the reduced vector has same odd-even sige
            mid_index  = self.curve_len//2
            pick_len = mc//2
            left_index = int(np.ceil(mid_index-pick_len))
            right_index= int(np.floor(mid_index+pick_len + mc %2))

            if   TensorQ:return vector[...,left_index:right_index,:]
            elif nArrayQ:return vector[...,left_index:right_index]
        else:
            if   TensorQ:return vector[...,:mc,:]
            elif nArrayQ:return vector[...,:mc]

    def output_dim(self,mc):
        if mc is None:mc = self.curve_len
        if self.method == 'fft':
            if mc%2 != self.curve_len%2:mc = mc +1 # make sure the reduced vector has same odd-even sige
        else:
            if mc is not None:mc=mc
        return mc

    def curve2vector(self,curve,num=None):
        curve = self.check_curve_data(curve)
        TensorQ   = isinstance(curve,torch.Tensor)
        nArrayQ   = isinstance(curve,np.ndarray)
        transformer = self.numpy_transform if nArrayQ else self.torch_transform
        vector    = transformer(curve)
        if num is not None:
            vector = self.reduce(vector,num)
        return vector

    def vector2curve(self,vector,length=None,keep=True):
        origin_type = type(vector)
        vector    = self.check_vector_data(vector)
        TensorQ   = isinstance(vector,torch.Tensor)
        nArrayQ   = isinstance(vector,np.ndarray)
        inverseform = self.numpy_inverse_t if nArrayQ else self.torch_inverse_t
        if length is None: length = self.curve_len
        if self.method == 'fft' and nArrayQ:
            shape_l   = len(vector.shape)
            chanlen   = vector.shape[-1]
            pad_num_l = int(np.ceil((length-chanlen)/2))
            pad_num_r = int(np.floor((length-chanlen)/2))
            vector    = np.pad(vector,[[0,0]]*(shape_l-1)+[[pad_num_l,pad_num_r]])
        elif self.method == 'fft' and TensorQ:
            shape_l   = len(vector.shape)
            chanlen   = vector.shape[-2]
            pad_num_l = int(np.ceil((length-chanlen)/2))
            pad_num_r = int(np.floor((length-chanlen)/2))
            vector    = F.pad(vector,(0,0,pad_num_l,pad_num_r))
        elif self.method == 'rfft' and TensorQ:
            length    = length//2+1
            shape_l   = len(vector.shape)
            chanlen   = vector.shape[-2]
            pad_num_r = int(np.floor(length-chanlen))
            vector    = F.pad(vector,(0,0,0,pad_num_r))
        else:
            pass
        curve=inverseform(vector)
        if keep:
            curve= self.change_tensor_type(curve,origin_type)
        return curve

class CurveSample(CurveProcesser):
    '''
    'unisample','cplxsample'
    '''
    def __init__(self,curve_len=1001,method='unisample',feature='full',vector_type="curve",filter_type='none',sample_num=100,**kargs):
        super().__init__(curve_len,**kargs)
        self.method      = method
        self.class_name  = "Sample"
        self.feature     = feature
        self.vector_type = vector_type
        self.filter_type = filter_type
        method_allow     = ['unisample','cplxsample']
        if method not in method_allow:raise NotImplementedError
        self.sample_num  = sample_num
        self.save_name   = method+str(sample_num)+('' if feature == 'full' else '.'+feature)+("" if vector_type == 'curve' else '.'+vector_type)
        self.method_name = method+str(sample_num)\
                                 +('' if feature == 'full' else '.'+feature)\
                                 +("" if vector_type == 'curve' else '.'+vector_type)\
                                 +('' if filter_type == 'none' else '.'+filter_type)
        if method == 'unisample':
            self.chosen = None
            self.curve_dtype = [np.ndarray]
            self.vector_dtype= [np.ndarray]
            self.curve_field = 'real'
            self.vector_field= self.curve_field
        elif method == 'cplxsample':
            self.chosen = None
            self.curve_dtype = [np.ndarray]
            self.vector_dtype= [np.ndarray]
            self.curve_field = 'complex'
            self.vector_field= self.curve_field
        else:
            raise NotImplementedError

    def output_dim(self,num):
        return self.sample_num

    def reduce(self,vector,mc):
        return vector

    def curve2vector(self,curve,num=None):
        if self.sample_num == self.curve_len:return curve
        curve = self.check_curve_data(curve)
        if num is None:num=self.sample_num
        curve = self.check_curve_data(curve)
        if (self.method == "unisample") or (self.method =="cplxsample"):
            if self.chosen is None:
                length = curve.shape[-1]
                self.chosen = np.linspace(0,length-1,num).astype('int').tolist()
            return curve[...,self.chosen]


    def vector2curve(self,vector,num=None,keep=True):
        if self.sample_num == self.curve_len:return vector
        origin_type = type(vector)
        vector=self.check_vector_data(vector)
        if num is None:num = self.curve_len
        if self.method == "unisample" or self.method =="cplxsample":
            length = vector.shape[-1]
            o_shape= list(vector.shape[:-1])+[-1]
            vector = vector.reshape(-1,length)
            x_index= np.linspace(0,num-1,length).astype('int')
            y_out  = interp1d(x_index,vector,kind='cubic',axis=1)(np.arange(num))
            y_out  = y_out.reshape(o_shape)
        else:
            raise NotImplementedError
        if keep:
            y_out= self.change_tensor_type(y_out,origin_type)
        return y_out



import pywt
class CurveWavelet(CurveProcesser):
    def __init__(self,curve_len=1001,method='dwt',level=6,out_num=4,**kargs):
        super().__init__(curve_len,**kargs)
        method_allow   = ['dwt','cplxdwt']
        if method not in method_allow:raise NotImplementedError
        self.method    = method
        self.class_name= "Wavelet"
        self.declvl    = level
        self.out_num   = out_num
        self.method_name='{}.{}of{}'.format(method,out_num,level)

        if 'dwt' in method:
            channel = []
            for i in range(level):
                now_l = int(np.ceil(curve_len/np.power(2,i+1)))
                channel.append(now_l)
            channel.append(channel[-1])
            channel.reverse()
            self.channel = channel
            self.out_dim = sum(channel[:self.out_num])

            self.curve_dtype = [np.ndarray]
            self.vector_dtype= [np.ndarray]
            self.curve_field = 'real' if method == 'dwt' else 'complex'
            self.vector_field= self.curve_field

    def check_vector_data(self,vector):
        VectorComplexQ = (self.vector_field == 'complex')
        vector = self.check_data(vector,VectorComplexQ)
        if type(vector) not in self.vector_dtype:
            vector = self.change_tensor_type(vector,self.vector_dtype[0])
        assert vector.shape[-1] == self.out_dim
        other_part_shape = list(vector.shape)[:-1]
        out = np.split(vector,np.cumsum(self.channel[:self.out_num]),-1)[:-1]
        for c in self.channel[self.out_num:]:
            out.append(np.zeros(other_part_shape+[c]))
        return out

    def output_dim(self,num):
        return self.out_dim

    def reduce(self,vector,mc):
        return vector

    def curve2vector(self,curve,num=None):
        curve = self.check_curve_data(curve)
        num   = self.out_num
        if 'dwt' in self.method:
            out   = pywt.wavedec(curve, 'db1', level=6,axis=-1,mode='smooth')
            out   = np.concatenate(out[:num],-1)
            return out
        else:
            raise NotImplementedError

    def vector2curve(self,vector,keep=True):
        origin_type = type(vector)
        vector = self.check_vector_data(vector)
        if   self.method == 'dwt':
            curve    = pywt.waverec(vector,'db1',axis=-1,mode='smooth')[...,:self.curve_len]
        elif self.method == 'cplxdwt':
            vector_real = [v.real for v in vector]
            vector_imag = [v.imag for v in vector]
            curve_real  = pywt.waverec(vector_real,'db1',axis=-1,mode='smooth')[...,:self.curve_len]
            curve_imag  = pywt.waverec(vector_imag,'db1',axis=-1,mode='smooth')[...,:self.curve_len]
            curve       = curve_real+1j*curve_imag
        else:
            raise NotImplementedError
        if keep:curve= self.change_tensor_type(curve,origin_type)
        return curve

if __name__ == '__main__':
    feq  = np.range(1000)
    s1   = np.array([(x*np.sin(cos(x))+x) for x in range(1000)])
    seq        =s1
    max_leng   =len(seq)
    fft_result =np.fft.fft(seq)
    fft_store  =np.zeros_like(fft_result)
    #fft_store  =np.zeros(max_leng,dtype='complex128')
    mc         =100 #max choose number
    right      =fft_result[:mc+1]
    left       =fft_result[-mc:]
    freq       =np.concatenate([left,right])
    fft_store[:mc+1]=right
    fft_store[-mc:] =left
    ifft_result=fft.ifft(fft_store)
    filted_seq =ifft_result
    plt.plot(feq,seq.imag,feq,filted_seq.imag)
