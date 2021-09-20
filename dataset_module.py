import os,re,json
import torch
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from mltool import tableprint as tp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .normlization import normlizationer,norm_dict
from .utils import *
from .Curve2vector import IdentyProcess,CurveFourier,CurveWavelet
from . import criterion

logging_info = cPrint(True)
def parse_datalist(curve_path_list,image_path_list):
    if curve_path_list is None:return None,None
    if isinstance(curve_path_list,np.ndarray) and isinstance(image_path_list,np.ndarray):
        return curve_path_list,image_path_list
    if not isinstance(curve_path_list,list) and image_path_list is None:
    	# this code reveal the curve file and image file from a high level path ../Data##
    	curve_path_list,image_path_list = convertlist(curve_path_list)
    if isinstance(curve_path_list,str) and isinstance(image_path_list,str):
    	# this code for single image and curve file
    	if os.path.isfile(curve_path_list) and os.path.isfile(image_path_list):
    		curve_path_list=[curve_path_list]
    		image_path_list=[image_path_list]
    if not os.path.isfile(curve_path_list[0]) and image_path_list is None:
        curve_path_list_new=[]
        image_path_list_new=[]
        for path_list in curve_path_list:
            curve_path_l,image_path_l = convertlist(path_list)
            curve_path_list_new+=curve_path_l
            image_path_list_new+=image_path_l
        curve_path_list = curve_path_list_new
        image_path_list = image_path_list_new
    return curve_path_list,image_path_list

def load_data_numpy(curve_path_list,image_path_list):
    if isinstance(curve_path_list,np.ndarray) and isinstance(image_path_list,np.ndarray):
        return curve_path_list,image_path_list
    imagedata  = []
    curvedata  = []
    for curve_path,image_path in zip(curve_path_list,image_path_list):
    	imagedata.append(np.load(image_path).astype('float'))
    	curvedata.append(np.load(curve_path))
    imagedata = np.concatenate(imagedata)
    curvedata = np.concatenate(curvedata)
    return imagedata,curvedata
curve_branch_flag = {"T":"1","1":"1","R":"2","2":"2","P":"3","3":"3"}

def get_dataset_name(curve_branch= None,curve_flag=None,enhance_p=None,FeatureNum  = None,
                     val_filter  = None,volume=None,range_clip=None,**kargs):
    assert curve_branch is not None
    assert curve_flag is not None
    assert enhance_p is not None
    assert FeatureNum is not None
    # curve_branch= 'T',curve_flag='N',enhance_p='E',FeatureNum  = 1001,
    DataSetType = f'B{curve_branch_flag[str(curve_branch)]}{curve_flag}{enhance_p}S{FeatureNum}'
    if range_clip is not None:
        DataSetType+=f".{range_clip[0]}to{range_clip[1]}"
    if (val_filter is not None) and ('max' in val_filter):
        DataSetType+=f".{val_filter}"
    return DataSetType

class BaseDataSet(Dataset):
    DataSetType=""
    _collate_fn=None
    def _offline(self):
        # for the base class no need
        raise NotImplementedError

    def set_length(self,num):
        if num is not None:
            self.length = num
        #self.curvedata = self.curvedata[:num]
        #self.imagedata = self.imagedata[:num]
        #self.vector    = self.vector[:num] if self.vector is not None else None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        curve_data = self.curvedata[index]#(2,1001)
        image_data = self.imagedata[index]#(1,16,16)
        if self.vector is None:
            vector_data = self.transformer.curve2vector(curve_data)
            vector_data = self.transformer.reduce(vector_data,self.vector_dim)
            vector_data = np2torch(vector_data)
            vector_data = self.forf(vector_data)
        else:
            vector_data = self.vector[index]#(2,num)
        if not self.case_type == "train":
            return vector_data,image_data,curve_data


        return vector_data,image_data

    def recover(self,tensor):
        #transformer = CurveSample(sample_num=self.FeatureNum)
        return self.transformer.vector2curve(self.invf(tensor))

    def structure_normer(self):
        if   isinstance(self.normf,str) or self.normf is None:
            string = self.normf if self.normf is not None else "Identy"
            if self.verbose:print("this dataset use 【{}】 norm".format(string))
            self.normlizationer = normlizationer(self.vector,self.normf,eps=0.1)
            self.forf   =  self.normlizationer.forf
            self.invf   =  self.normlizationer.invf
        elif isinstance(self.normf,list):
            self.forf,self.invf = self.normf
            string = "Inherit"
        else:
            raise
        return self.forf,self.invf

    # def computer_accurancy(self,feaRs,feaPs,reals=None,train_model=False,inter_process=False):
    #     # usually, feaR, feaP,real is a torch GPU Tensor
    #     # predict a curve repr by (...,1001) vector
    #     if train_model:
    #         loss   = torch.nn.MSELoss()(feaRs,feaPs)
    #         accu   = loss
    #         return loss,accu
    #     curve_loss = self.transformer.curve_loss
    #     y_feat_p   = feaPs.detach().cpu() # should be a cpu torch tensor
    #     y_feat_r   = feaRs.detach().cpu() # should be a cpu torch tensor
    #     loss_feat  = curve_loss(y_feat_r,y_feat_p,-1)
    #     y_test_p   = self.recover(y_feat_p)
    #     y_test_c   = self.recover(y_feat_r)
    #     y_test_r   = reals.detach().cpu()# should be a cpu torch tensor
    #     loss_real  = curve_loss(y_test_r,y_test_p,-1)
    #     if inter_process:
    #         return loss_feat,loss_real,(y_test_p,y_test_c,y_test_r)
    #     else:
    #         loss=loss_feat.mean()
    #         accu=loss_real.mean()
    #     return loss,accu
########################################################################################
############## Insider code is more general for all machine learning project ###########
class SMSDataset(BaseDataSet):
    '''
    This is the Metasurface dataset import module.This module accept `offline` operation.
        -> curve_path_list: -str- or -numpy.ndarray-
                if it is string(dir), the code will search this dir and load all Data_xxx/Integrate_curve_xxx.npy file as the curvedata.
                if it is numpy.ndarray, the code will load the array as its curvedata, please care we wont offline when arry input.
                          if the option [DATAROOT] is None and option [offline_data_location] is None, the default offline path is the parent directory of [curve_path_list]
        -> image_path_list: -str- or -numpy.ndarray- or None
                if it is string(dir), the code will search this dir and load all Data_xxx/Integrate_image_xxx.npy file as the imagedata.
                if it is numpy.ndarray, the code will load the array as its imagedata, please care the dimention.
                if it is None, the [image_path_list] will replace the `curve` character in [curve_path_list:string] to image.
        -> curve_branch: 'T';-str-['T','R','P'] or -int- [1,2,3]
                The curve banch assigned for the task. 'T' or 1 point to the first branch (Transmittion).
                'R' or 2 point to the second branch (Reflection). And 'P' or 3 point to the absorption.
        -> curve_flag: 'N';-str-['N','C']
                assign use complex curve('C') or the norm value('N').
                use complex curve will limit the option in [type_predicted]
        -> FeatureNum: 1001; -int-
                the downsampling number.
        -> type_predicted: -str-['curve','tandem','inverse','demtan','GAN','multihot','combination','onehot','number']
                The output vector type.
                'curve','tandem','inverse','demtan' return a curve vector 1xL in [0,1].
                'multihot' return the 1xL Z_2 vector.
                'onehot'   return the index of the onehot vector.
                'combination' <-- deprecate.
                'number'   return 1 scaler.
        -> target_predicted:-str-
                The task flag for each output vector type.
                'curve': 1. 'simple': return the curve vector
                         2. 'whatevermaxnormed': return the curve vector normalized by its max value.
                'multihot': 1. 'location_of_peaks'
                            2. 'plat(#)': divide the frequency band into # piece and check has peak or not.
                'combination' <-- deprecate.
                'onehot': 1. 'location_of_max_peak' :  return the index of max_peak
                          2. 'left_or_right': return:  judge the max peak at left or right of the center point
                          3. 'balance_leftorright'  :  judge the max peak at left or right of the middle point
                'number': 1. 'maxvalue': return the max value for each curve.
        -> normf: -str- or -list-: default:'none'.
                the post data normalization. Notice this normalization must apply both train and test. And the some
                share parameter should collect from trainset. if the normf is list=[forf,invf], both forf and invf
                should Inherit from another dataset.forf,dataset.invf
        -> transformer: <-- deprecate. -str- or -class-
                Now we only use unisample transformer. TODO: add FFT, CFT, Wavelet option.
        -> enhance_p: 'E'. ['E','D','N']
                precision flag. 'E'-> np.round(data,3); 'D' -> np.round(data,6); 'N' origin precision;
        -> offline: True. bool. General offline switch.
        -> offline_data_location: default:None. -str-
                assign the offline data location both for save and load. If active, block [DATAROOT]
        -> DATAROOT:  default:None. -str-
                assignthe parent dir of offline path, and offline path is $DATAROOT/$DataSetType
        -> case_type: default: 'train'. 'train' or 'test'
                offline npy file's flag.
        -> dataset_quantity: default:None. ['latest',-int-]
                The offline data distinguish flag.
                if dataset_quantity is None, the offline file will use the suffix <- [# data file] automatively.
                If you have generate multiple offline data (usually happened when you add new data),
                you need assign this value to "latest" to force regenerate
                                     or assign the old one by right suffix.
        -> val_filter:None
                this flag for you dont want to filte some flat signal.
                for example, if we want to use data has at least 0.1 maximum value. set max10
        -> volume:None. 'int'
                this flag help you use dynamic dataset volume. incresing dataset volume.
        -> range_clip:None
                this flag for you dont want to use full frequency band.
                for example, usually we will use last 250 point of the PLG dataset and set range_clip=[750,1000]
    '''
    #allowed_normf=["heq","mean","heq-h","heq-e","mean2zero","gauss","resp_gauss"]

    allowed_curve_branch   = ["T",1,"R",2,"P",3]
    allowed_normf          = list(norm_dict.keys())+['none']


    def __init__(self,curve_path_list,image_path_list,FeatureNum=1001,curve_branch='T',curve_flag='N',
                      type_predicted=None,target_predicted=None,normf='none',enhance_p='E',
                      offline=True,offline_data_location=None,DATAROOT=None,dataset_quantity=None,case_type='train',
                      partIdx=None,
                      val_filter=None,volume=None,range_clip=None,verbose=True,
                      image_transfermer=None,**kargs):
        logging_info = cPrint(verbose)
        assert curve_branch in self.allowed_curve_branch

        if isinstance(normf,str) or normf is None:
            if normf in self.allowed_normf:
                logging_info('we will use 【{}】 norm on vector data'.format(normf))
                self.normf      = normf
            else:
                logging_info('please note this dataset force use none normf, your configuration {} is block'.format(normf))
                self.normf      = 'none'
        else:
            logging_info('Inherit norm method from last dataset')
            self.normf      = normf


        self.verbose = verbose
        self.vectorDim       = self.FeatureNum = FeatureNum
        self.type_predicted  = type_predicted
        self.target_predicted= target_predicted
        self.case_type       = case_type
        self.partIdx         = partIdx[self.case_type] if partIdx is not None else None
        ##############################################################
        ####################### Name Task ############################
        ##############################################################
        ### get the unique offline dataset name from the configuration.

        self.DataSetType = get_dataset_name(curve_branch=curve_branch,curve_flag=curve_flag,enhance_p=enhance_p,
                                                 FeatureNum=FeatureNum,
                                                 val_filter=val_filter,volume=volume,range_clip=range_clip)

        if target_predicted in ['dct','rfft','fft']:
            self.FeatureNum  = data_origin_len
            self.transformer =  CurveFourier(self.FeatureNum,target_predicted)
        elif target_predicted in ['dwt','cplxdwt']:
            self.FeatureNum  = data_origin_len
            self.transformer = CurveWavelet(data_origin_len,target_predicted)
        else:
            self.transformer= IdentyProcess()


        ##############################################################
        ####################### Offline Processing ##################
        ##############################################################
        ### get the unique offline dataset name from the configuration.
        curve_path_list_numpy,image_path_list_numpy = parse_datalist(curve_path_list,image_path_list)
        if (isinstance(curve_path_list,np.ndarray) and (not offline_data_location) and (not DATAROOT)) or (not offline):
            print("use array input, and not set the offline data save path. We will not offline generated data.")
            offline_data_location = "offline_data"
            do_processing_IC_data = True
            do_processing_feature = True
            offline = False
        else:
            if curve_path_list is None:dataset_quantity="select_from_offline"
            if DATAROOT is None:DATAROOT,_                      = os.path.split(curve_path_list)
            if not offline_data_location:offline_data_location  = os.path.join(DATAROOT,self.DataSetType)
            if not os.path.exists(offline_data_location):os.mkdir(offline_data_location)
            print(offline_data_location)
            if (dataset_quantity is None) or (dataset_quantity == "latest") or (len(os.listdir(offline_data_location))==0):
                tail_curve_path_name   = len(os.listdir(curve_path_list)) if isinstance(curve_path_list,str) else max(len(curve_path_list),len(curve_path_list_numpy))
            else:
                tail_curve_path_name   = str(dataset_quantity)
            offline_curvedata_name,do_processing_IC_data,tail_curve_path_name =self.check_offine_exist(offline_data_location,r"{}_curvedata_[\d]*.npy".format(case_type),
                                                                                       tail_curve_path_name,force=(dataset_quantity=="latest"))
            offline_imagedata_name = offline_curvedata_name.replace('curve','image')
            offline_featname_map  = {type_predicted:type_predicted}
            for t in ['curve','tandem','inverse','demtan','GAN']:offline_featname_map[t]= "curve"
            # the curve,tandam,inverse,demtan,GAN will share the same curve feat, so no need generate

            offline_featdata_name,do_processing_feature,tail_curve_path_name =self.check_offine_exist(offline_data_location,
                                         "{}.feat.{}.{}_[\d]*.npy".format(case_type,offline_featname_map[type_predicted],target_predicted),
                                                                                  tail_curve_path_name,force=(dataset_quantity=="latest"))
            offline_curvedata      = os.path.join(offline_data_location,offline_curvedata_name)
            offline_imagedata      = os.path.join(offline_data_location,offline_imagedata_name)
            offline_featdata       = os.path.join(offline_data_location,offline_featdata_name)

        if do_processing_IC_data or (offline == "force-curve"):
            self.imagedata,self.curvedata = load_data_numpy(curve_path_list_numpy,image_path_list_numpy)
            # 'enhance' and x=1-x now is a default option
            # for sure the curve is noice-off and sensitive for 1 rather than 0
            # it will transform the origin complex curve to its norm
            assert self.curvedata.dtype == 'complex128'
            if 'B1N' in self.DataSetType:
                self.curvedata  = np.abs(self.curvedata[...,0:1,:])
                self.curvedata  = 1-self.curvedata
            elif 'B2N' in self.DataSetType:
                self.curvedata  = np.abs(self.curvedata[...,1:2,:])
                self.curvedata  = self.curvedata
            elif 'B3N' in self.DataSetType:
                self.curvedata  = np.abs(self.curvedata[...,1:2,:])**2+np.abs(self.curvedata[...,0:1,:])**2
                self.curvedata  = 1-self.curvedata
            elif 'B1C' in self.DataSetType:
                self.curvedata  = self.curvedata[...,0:1,:]
            elif 'B1C' in self.DataSetType:
                self.curvedata  = self.curvedata[...,1:2,:]
            elif 'B3C' in self.DataSetType:
                self.curvedata  = self.curvedata
            ## --> processing E
            ## --> will default reverse the ratio 1 and 0
            if   enhance_p=='E':self.curvedata  = np.round(self.curvedata,3)
            elif enhance_p=='D':self.curvedata  = np.round(self.curvedata,6)
            elif enhance_p=='N':self.curvedata  = self.curvedata
            else:raise NotImplementedError

            if range_clip is not None:
                self.curvedata=self.curvedata[...,int(range_clip[0]):int(range_clip[1])]

            ## --> processing sampling
            data_origin_len = self.curvedata.shape[-1]
            assert self.FeatureNum<=data_origin_len
            # the self.FeatureNum can big than data_origin_len but it just copy value to extand
            if self.FeatureNum != data_origin_len and self.FeatureNum and isinstance(self.transformer,IdentyProcess):
                sample_index    = np.linspace(0,data_origin_len-1,self.FeatureNum).astype('int').tolist()
                self.curvedata  = self.curvedata[...,sample_index]
            if (val_filter is not None) and ('max' in val_filter):
                assert curve_flag =='N'
                value =  float(val_filter.replace('max',""))/100
                choose_index = np.where(self.curvedata.max(-1)>value)[0]
                self.curvedata = self.curvedata[choose_index]
                self.imagedata = self.imagedata[choose_index]

            ## --> offline save, accelerate next load
            if offline:
                logging_info('first generate curve/image data, we will save them in \n  --->|{}|<-----'.format(offline_data_location))
                np.save(offline_curvedata,self.curvedata)
                np.save(offline_imagedata,self.imagedata)
        else: # load from offline data
            if verbose:
                print('find offline curve data at --->|{}|<-----'.format(offline_curvedata))
                print('find offline image data at --->|{}|<-----'.format(offline_imagedata))
            self.imagedata = np.load(offline_imagedata).astype('float')
            self.curvedata = np.load(offline_curvedata)


        if do_processing_feature or (offline == "force-feat"):
            #####################################################################################
            # In the newest data manager, we use a dist control
            # there is a control parameter named type_predicted
            # For this dataset we cancel 'transformer' design since it is the Sample Transformer
            #####################################################################################
            if   type_predicted in ['curve','tandem','inverse','demtan','GAN'] :
                #we want to predict a curve or compressed curve 1001x[0,1]
                #we can compress curve via transformer
                #we can normlize curve using max norm and other norm
                #we can filte the curve like only use the 'smoothpeak' data
                self.vector = self.curvedata
                if   target_predicted =='simple':
                    pass
                elif target_predicted =='whatevermaxnormed':
                    assert curve_flag =='N'
                    self.vector_max   = self.curvedata.max(-1,keepdims=True)
                    self.vector       = self.curvedata/(self.vector_max+1e-6)
                elif target_predicted in ['dct','rfft','dwt']:
                    assert curve_flag =='N'
                    self.vector       = self.transformer.curve2vector(self.curvedata,self.vectorDim )
                elif target_predicted in ['fft','cplxdwt']:
                    assert curve_flag =='C'
                    self.vector       = self.transformer.curve2vector(self.curvedata,self.vectorDim )
            elif type_predicted == 'multihot':
                assert curve_flag =='N'
                if target_predicted   =='location_of_peaks':
                    #so it is a multi-one-hot-vector like [...,0,1,0,0,0,1,0,1,...]
                    #assert enhance_p=='N'
                    self.vector = find_peak(self.curvedata,include_boundary=False,complete=True)
                elif target_predicted =='peak_and_peakwise':
                    # add a logical bit to judge is there a peck in such frequency
                    raise
                elif target_predicted =='plat8':
                    divide_num = 8
                    #now we split the curve into 8 small interval.
                    #if there is a peck in it, active a one-hot
                    #so the vector like [0,0,0,1,0,1,0,0,0,0]
                    peaks       = find_peak(self.curvedata)
                    interv      = np.array_split(peaks,divide_num,-1)
                    self.vector = [kk.sum((1,2))>0 for kk in interv]
                    self.vector = np.stack(self.vector,-1)
            elif type_predicted == 'combination':
                assert curve_flag =='N'
                if target_predicted =='peak_and_curve':
                    #now we predict both peak and curve
                    #emphysis peak location may increase our precision
                    peak        = find_peak(self.curvedata)
                    vector      = self.curvedata
                    self.vector = np.concatenate([vector,peak],-1)
                elif target_predicted =='maxnormed':
                    self.vector_max   = self.curvedata.max(-1,keepdims=True)
                    self.vector       = self.curvedata/self.vector_max
                    self.vector       = np.concatenate([self.vector_max,self.vector],-1)
                elif target_predicted =='peakparameters':
                    from tqdm import tqdm
                    print("it will take a bit to generate peak parameters")
                    self.vector=[]
                    for curve in tqdm(self.curvedata):
                        start  = int(range_clip[0])*10/1001+2  if range_clip is not None else 2
                        end    = int(range_clip[1])*10/1001+2  if range_clip is not None else 12
                        xdata  = np.linspace(start,end,self.curvedata.shape[-1])
                        pcurve = PeakWiseCurve(xdata,curve[0])
                        self.vector.append(pcurve.state_array())
                    self.vector = np.array(self.vector)
            elif type_predicted == 'onehot':
                assert curve_flag =='N'
                #we want to predict a int number,
                #it can be the 'localtion for max peak'
                label_max = 2
                if target_predicted =='location_of_max_peak':
                    # use the CrossEntropyLoss in pytorch,
                    # the desire target is Index.
                    # the prediction is a (Batch,vectorDim) vector.
                    #self.vector      = np.zeros((len(self.curvedata),self.FeatureNum)) #(...,128) onehot
                    labels           = np.argmax(self.curvedata,-1).flatten()
                    #self.vector[np.arange(len(self.curvedata)), labels] = 1
                    self.vector = torch.Tensor(labels).long()
                elif target_predicted =='left_or_right':
                    #self.vector      = np.zeros((len(self.curvedata),self.FeatureNum)) #(...,128) onehot
                    labels           = np.argmax(self.curvedata,-1).flatten()
                    mid              = self.curvedata.shape[-1]/2
                    labels           = (labels>mid)+0
                    #self.vector[np.arange(len(self.curvedata)), labels] = 1
                    self.vector      = torch.Tensor(labels).long()
                elif target_predicted =='balance_leftorright':
                    #self.vector      = np.zeros((len(self.curvedata),self.FeatureNum)) #(...,128) onehot
                    labels           = np.argmax(self.curvedata,-1).flatten()
                    mid              = np.median(labels)
                    labels           = (labels>=mid)+0
                    #self.vector[np.arange(len(self.curvedata)), labels] = 1
                    self.vector = torch.Tensor(labels).long()
            elif type_predicted == 'number':
                assert curve_flag =='N'
                #we want to predict a real number,
                #it can be the 'value of the max peak'
                #it can be the 'area of the curve'
                if  target_predicted=='maxvalue':
                    self.vector       = np.max(self.curvedata,-1,keepdims=True)
            else:
                raise NotImplementedError
            if offline:
                logging_info('first generate feature data, we will save them in --->|{}|<-----'.format(offline_featdata))
                np.save(offline_featdata,self.vector)
        else:
            logging_info('find offline feature data at --->|{}|<-----'.format(offline_featdata))
            self.vector = np.load(offline_featdata)


        self.imagedata = self.imagedata.reshape(-1,1,16,16)
        ## if image_transfermer is not None:
        #     if image_transfermer == "1to1":
        #         self.imagedata = (self.imagedata - 0.5)/0.5
        #     elif image_transfermer == "contour":
        #         logging_info('=== using image 【contour】 as input ====')
        #         offline_contour = os.path.join(offline_data_location,offline_imagedata_name.replace('image','contour'))
        #         offline_edgepix = os.path.join(offline_data_location,offline_imagedata_name.replace('image','edge'))
        #         self.real_imagedata= self.imagedata
        #         if not os.path.exists(offline_contour):
        #             self.edgepixel,self.imagedata=get_contour_data(self.real_imagedata[:,0,:,:],dim=50)
        #             if offline:
        #                 logging_info('first generate edge/contour data, we will save them in \n  --->|{}|<-----'.format(offline_data_location))
        #                 np.save(offline_edgepix,self.edgepixel)
        #                 np.save(offline_contour,self.imagedata)
        #         else:
        #             if verbose:
        #                 print('find offline contour data at --->|{}|<-----'.format(offline_contour))
        #             self.edgepixel = np.load(offline_edgepix)
        #             self.imagedata = np.load(offline_contour)
        #     else:
        #         raise NotImplementedError
        # from numpy to torch.Tensor
        self.curvedata = np2torch(self.curvedata)
        self.imagedata = np2torch(self.imagedata)
        self.vector    = np2torch(self.vector)

        # normlization part
        self.forf,self.invf = self.structure_normer()
        self.vector         = self.forf(self.vector)
        if volume is not None:
            self.curvedata = self.curvedata[-volume:]
            self.imagedata = self.imagedata[-volume:]
            self.vector    =    self.vector[-volume:]

        #self.real_vector = self.curvedata

        if target_predicted =='inverseone':
            self.imagedata  = self.imagedata[:,:,8,8]
        self.image_type_shape= self.imagedata.shape[1:]
        self.curve_type_shape= self.vector.shape[1:]
        if len(self.curve_type_shape)==0:self.curve_type_shape=(self.vectorDim,)
        if type_predicted == 'onehot':
            self.curve_type_shape=(int(self.vector.max().item()+1),)

        self.transformer.forf  = self.forf
        self.transformer.invf  = self.invf
        self.length            = len(self.curvedata)

        print(f">>>> {case_type} dataset size {self.length}")

        if verbose:
            headers = ['type','fea_type','curve data shape', 'image data shape', 'vector data shape']
            data = [[case_type,self.transformer.feature,
                     tuple2str(self.curvedata.shape),
                     tuple2str(self.imagedata.shape),
                     tuple2str(self.vector.shape) if self.vector is not None else "generated"]]
            tp.table(data, headers,width=17)
    def check_offine_exist(self,offline_location,matcher_string,flag_num,force=False):
        pattener     = re.compile(matcher_string)
        has_offline  = check_has_file(offline_location,pattener)
        if flag_num  == "select_from_offline":
            assert len(has_offline)>0
            offline_name = has_offline[0]
            #print("notice we will use ")
            return offline_name, False ,re.findall(r"([\d])",offline_name)[0]
        offline_name = matcher_string.replace("[\d]*","{}").format(flag_num)
        offlinedQ    = offline_name in has_offline
        if len(has_offline)==0:
            print("Warning: No offline data detected, we will generate one")
            return offline_name, True, flag_num
        if not offlinedQ:
            print(f"Warning: detect unmatch between your assign data {offline_name} and offline data {has_offline}")
            if force=="force":
                print(f"Warning: we will generate new")
                return offline_name, True, flag_num
            else:
                print(f"Warning: we will use old offline data {has_offline[0]}")
                return has_offline[0], False, flag_num
        else:
            return offline_name, False, flag_num

    def show_a_demo(self,num=None):
        if num is None:random_index = np.random.randint(0,self.length)
        else:random_index=num
        curve_data  = self.curvedata[random_index,0].numpy()
        image_data  = self.imagedata[random_index,0].numpy()
        vector_data = self.vector[random_index]
        reconstruct = self.recover(vector_data[None])
        vector_data = (self.invf(vector_data[None])).numpy()


        x=np.arange(self.FeatureNum)
        type_predicted  = self.type_predicted
        target_predicted= self.target_predicted

        if target_predicted in ['simple','whatevermaxnormed','maxnormed']+['dct','rfft','dwt','fft','cplxdwt']:
            vector_data = vector_data[0,0]
            reconstruct = reconstruct[0,0]
            graph_fig, graph_axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3.2),
                                         gridspec_kw={
                                               'width_ratios': [1,1.7]
                                        })
            ax = graph_axes[0]
            if curve_data.shape[-1]==2:

                _=ax.plot(np.arange(len(curve_data)),curve_data[...,0],'r*',label='curve.real')
                _=ax.plot(np.arange(len(curve_data)),curve_data[...,1],'g*',label='curve.imag')
                if target_predicted in ['dct','rfft','dwt','fft','cplxdwt']:
                    _=ax.plot(np.arange(len(reconstruct)),reconstruct[...,0],'r',label='reconstruct.real')
                    _=ax.plot(np.arange(len(reconstruct)),reconstruct[...,1],'g',label='reconstruct.imag')
            else:
                if target_predicted in ['dct','rfft','dwt','fft','cplxdwt']:
                    _=ax.plot(np.arange(len(reconstruct)),reconstruct,'r*',label='reconstruct')
                    _=ax.plot(np.arange(len(curve_data)),curve_data,'g',label='curve')


            ax.legend()
            ax = graph_axes[1]
            if vector_data.shape[-1]==2:
                _=ax.plot(np.arange(len(vector_data)),vector_data[...,0],'r',label='vector.real')
                _=ax.plot(np.arange(len(vector_data)),vector_data[...,1],'g',label='vector.imag')
            else:
                _=ax.plot(np.arange(len(vector_data)),vector_data,'r',label='vector')
            ax.legend()
        if target_predicted =='plat8':
            vector_data = vector_data[0]
            divide_num = 8
            curve      = curve_data
            xranges    = np.array_split(np.arange(self.FeatureNum),divide_num)
            peak       = np.zeros(self.FeatureNum)
            for pos,v in zip(xranges,vector_data):peak[pos]=v
            peak       = peak*curve.max()
            plt.plot(x,curve,'r',x,peak,'b')
        if target_predicted =='location_of_peaks':
            vector_data = vector_data[0,0]
            curve      = curve_data
            peak       = vector_data*curve
            plt.plot(x,curve,'r',x,peak,'b')
        if target_predicted =='peak_and_curve':
            peak = vector_data[-self.FeatureNum:]
            curve= vector_data[:self.FeatureNum]
            peak = peak*curve.max()
            plt.plot(x,curve,'r',x,peak,'b')
        if target_predicted =='location_of_max_peak':
            peak = vector_data
            curve= curve_data
            peak_data = peak*curve.max()
            plt.plot(x,curve_train,'r',x,peak_data,'b')
        if target_predicted =='maxvalue':
            curve     = curve_data
            peak_indx = np.argmax(curve)
            peak_data = 0*curve
            peak_data[peak_indx] = curve.max()
            plt.plot(x,curve_train,'r',x,peak_data,'b')


class SMSDatasetN(SMSDataset):
    data_field = 'real'
    def __init__(self,curve_path_list,image_path_list,**kargs):

        super().__init__(curve_path_list,image_path_list,curve_flag='N',**kargs)

    def get_default_accu_type(self):
        type_predicted  =self.type_predicted
        target_predicted=self.target_predicted
        if type_predicted   == 'curve':accu_list=['MAError']
        elif type_predicted == 'inverse':accu_list=['MSError','BCELoss','BinaryImgError']
        elif type_predicted == 'multihot':accu_list=['BCELoss','BinaryImgError']
        elif type_predicted == 'combination':raise NotImplementedError
        elif type_predicted == 'onehot':accu_list=['CELoss','OneHotError']
        elif type_predicted == 'number':accu_list=['MSError']
        elif type_predicted in ['tandem','demtan','GAN']:accu_list=['MSError','BinaryImgError']
        return accu_list
    def computer_accurancy(self,data_import,accu_list=None,inter_process=False):
        with torch.no_grad():
            # usually, feaR, feaP,real is a torch GPU Tensor
            # accu_list =['MAError','MSError','MsaError','MseaError']
            type_predicted  =self.type_predicted
            target_predicted=self.target_predicted
            if accu_list is None:accu_list=self.get_default_accu_type()
            if   type_predicted == 'curve':
                feaRs,feaPs,reals=data_import
                # the reals is the curve before normalization
                # the feaRs is the normalized curve
                # the feaPs is the predicted normalized curve
                # when train, inter_process is False, only compare the MSE between normlized curve
                # when inter_process is True, will compare the MSE between origin curve
                # then the self.recover(feaRs) become the compress-decompress curve
                if reals is None:
                    predict  = y_feat_p = self.invf(feaPs)
                    target   = y_feat_r = self.invf(feaRs)
                else:
                    predict  = self.recover(feaPs)
                    reconst  = self.recover(feaRs)
                    target   = reals

            elif type_predicted in ['multihot','onehot','number','inverse']:
                feaRs,feaPs,reals=data_import
                predict  = feaPs
                target   = feaRs
                #so it is a multi-one-hot-vector like [...,0,1,0,0,0,1,0,1,...]
                #so feaPs ---->[0.1,0.001,0.5,0.8,0.9,0.4,....]
                #so feaRs ---->[0,0,0,1,1,0,....]
                # feaRs,feaPs,reals=data_import
                # loss = torch.nn.BCELoss()(feaPs,feaRs)
                # real = feaRs
                # pred = torch.round(feaPs)
                # accu = 1-(real==pred).float().mean() #for the model autosave,the save accu is inversed
            elif type_predicted == 'combination':
                feaRs,feaPs,reals=data_import
                print('for combination part, you need check ')
                raise NotImplementedError

            if type_predicted in ['tandem' ,'demtan','GAN']:
                Feat_Real,Feat_Pred,Imag_Real,Imag_Pred=data_import
                if type_predicted == 'GAN':Feat_Real=Feat_Real[...,:self.vector.shape[-1]]
                loss_pool  = {}
                for accu_type in accu_list:
                    if 'Binary' in accu_type:
                        loss_pool[accu_type] = criterion.loss_functions[accu_type](Imag_Pred,Imag_Real)
                    else:
                        loss_pool[accu_type] = criterion.loss_functions[accu_type](Feat_Pred,Feat_Real)
                if not inter_process:
                    for accu_type in accu_list:loss_pool[accu_type] = loss_pool[accu_type].mean().item()
                    return loss_pool
                else:
                    return loss_pool,(Feat_Real,Feat_Pred,Imag_Real,Imag_Pred)
            else:
                loss_pool  = {}
                for accu_type in accu_list:
                    predict_now=predict
                    target_now = target
                    accu_type_real=accu_type
                    if '_for_' in accu_type:
                        accu_type_real,_,accu_part =  accu_type.split("_")
                        predict_now = predict[list(range(*self.partIdx[accu_part]))]
                        target_now  =  target[list(range(*self.partIdx[accu_part]))]
                    loss_pool[accu_type] = criterion.loss_functions[accu_type_real](predict_now,target_now)
                if not inter_process:
                    for accu_type in accu_list:
                        loss_pool[accu_type] = loss_pool[accu_type].mean().item()
                    return loss_pool
                else:
                    return loss_pool,(predict,reconst,target)
    def criterion(self,custom_type='default'):
        if custom_type == 'default':
            type_predicted = self.type_predicted
            if   type_predicted == 'curve':   criterion_class = torch.nn.MSELoss
            elif type_predicted == 'inverse': criterion_class = torch.nn.BCELoss
            elif type_predicted == 'multihot':criterion_class = torch.nn.BCELoss
            elif type_predicted == 'onehot':  criterion_class = criterion.CrossEntropyLoss
            elif type_predicted == 'number':  criterion_class = torch.nn.MSELoss
            elif type_predicted in ['tandem','demtan']:  criterion_class = criterion.TandemLoss
        elif custom_type == 'SelfAttLoss':criterion_class = criterion.SelfAttentionedLoss
        elif custom_type == 'SelfEhaAttLoss1':criterion_class = criterion.SelfEnhanceLoss1
        elif custom_type == 'SEALoss2':criterion_class = criterion.SelfEnhanceLoss2
        elif custom_type == 'SEALoss3':criterion_class = criterion.SelfEnhanceLoss3
        elif custom_type == 'SEALoss4':criterion_class = criterion.SelfEnhanceLoss4
        elif custom_type == 'SEALoss5':criterion_class = criterion.SelfEnhanceLoss5
        elif custom_type == 'SEALoss6':criterion_class = criterion.SelfEnhanceLoss6
        elif custom_type == 'BalancedBCE':
            weight_zero = self.curvedata.mean(0,keepdim=True)
            criterion_class = criterion.BalancedBCEWrapper(weight_zero)
        elif custom_type == 'SEALoss3T':
            weight = self.curvedata.mean(0)
            criterion_class = criterion.SEALoss3TWrapper(weight)
        elif custom_type == 'SEALoss4T':
            weight = self.curvedata.mean(0)
            criterion_class = criterion.SEALoss4TWrapper(weight)
        elif custom_type == 'TandemLossL1':criterion_class = criterion.TandemLossL1
        elif custom_type == 'TandemLossL2':criterion_class = criterion.TandemLossL2
        elif custom_type == 'TandemLossSSIM':criterion_class = criterion.TandemLossSSIM
        elif custom_type == 'FocalLoss':criterion_class = criterion.FocalLoss
        elif custom_type == 'BCELoss':criterion_class    = criterion.BCEWithLogitsLoss
        elif custom_type == 'CELoss':criterion_class     = criterion.CrossEntropyLoss
        elif custom_type == 'FocalLoss1':criterion_class = criterion.FocalLossWrapper(gamma=1.1, alpha=0.5)
        else:
            print(f"we dont allow this crition {custom_type} now")
            raise NotImplementedError
        return criterion_class
    def use_classifier_loss(self,loss_type):
        # for classification problem, the target vector is always onehot
        # for CrossEntropyLoss:
        #     the target should be the index number #
        #     the predict should be the onehot vector [], not after sigmoid/softmax
        # for BCELossLogits Loss:
        #     the target should be the mult-hot/onehot vector[]
        #     the predict should be the  mult-hot/onehot vector[], not after sigmoid/softmax
        # for Focal Loss:
        #     the target should be the onehot vector[]
        #     the predict should be the  onehot vector[], not after sigmoid/softmax
        assert len(self.vector.shape)<=2
        if len(self.vector.shape)==1:
            # this mean it is index input,
            if not hasattr(self,"hot_vector"):
                self.hot_vector = torch.zeros(self.vector.shape[0], 2).scatter_(1, self.vector.unsqueeze(1).long(), 1)
            if not hasattr(self,"hot_index"):
                self.hot_index  = self.vector.reshape(-1,1)
        elif self.vector.shape[-1]==1:
            # this mean it is index input,
            if not hasattr(self,"hot_vector"):
                self.hot_vector = torch.zeros(self.vector.shape[0], 2).scatter_(1, self.vector.long(), 1)
            if not hasattr(self,"hot_index"):
                self.hot_index  = self.vector
        else:
            # this mean it is mult-hot/onehot vector
            if not hasattr(self,"hot_vector"):
                self.hot_vector = self.vector
            if not hasattr(self,"hot_index"):
                self.hot_index  = self.vector.max(-1)[1]
        if (loss_type == "BCELossLogits") or (loss_type == "BCELoss") :
            if self.hot_index.max().item()>1:# this mean it is large class_num classifier
                self.vector=self.hot_vector
            else:# this mean it is 2-class classifier
                self.vector=self.hot_index
        elif loss_type == "CELoss":self.vector=self.hot_index.flatten().long()
        elif loss_type == "FocalLoss1":self.vector=self.CEvector
        else:
            print("===== Not an aviliable Loss Function. We do nothing ========")
    @staticmethod
    def get_dataset_name(curve_branch= 'T',enhance_p='E',
                              FeatureNum  = 1001,
                              val_filter  = None,volume=None,range_clip=None,**kargs):
        return get_dataset_name(curve_branch=curve_branch,curve_flag='N',enhance_p=enhance_p,
                                     FeatureNum=FeatureNum,
                                     val_filter=val_filter,volume=volume,range_clip=range_clip)
class SMSDatasetC(SMSDataset):
    data_field = 'complex'
    def __init__(self,curve_path_list,image_path_list,**kargs):
        super().__init__(curve_path_list,image_path_list,curve_flag='C',**kargs)
    def get_default_accu_type(self):
        type_predicted  =self.type_predicted
        target_predicted=self.target_predicted
        if type_predicted   == 'curve':accu_list=['MAError']
        else:raise NotImplementedError
        return accu_list
    def computer_accurancy(self,data_import,accu_list=None,inter_process=False):
        with torch.no_grad():
            # usually, feaR, feaP,real is a torch GPU Tensor
            # accu_list =['MAError','MSError','MsaError','MseaError']
            type_predicted  =self.type_predicted
            target_predicted=self.target_predicted
            if accu_list is None:accu_list=self.get_default_accu_type()
            if   type_predicted == 'curve':
                feaRs,feaPs,reals=data_import
                if reals is None:
                    predict  = y_feat_p = self.invf(feaPs)
                    target   = y_feat_r = self.invf(feaRs)
                else:
                    feaPs   = self.recover(feaPs)
                    feaRs   = self.recover(feaRs)
                    predict  = feaPs
                    target   = reals
            else:raise NotImplementedError

            loss_pool  = {}
            for accu_type in accu_list:
                loss_pool[accu_type] = criterion.loss_functions[accu_type](predict,target)
            if not inter_process:
                for accu_type in accu_list:
                    loss_pool[accu_type] = loss_pool[accu_type].mean().item()
                return loss_pool
            else:
                return loss_pool,(feaPs,feaRs,reals)
    def criterion(self,custom_type='default'):
        if custom_type == 'default':
            type_predicted = self.type_predicted
            if   type_predicted == 'curve':   criterion_class = torch.nn.MSELoss
            else:raise NotImplementedError
        elif custom_type == 'ComplexMSE':criterion_class = criterion.ComplexMSELoss
        else:
            raise NotImplementedError
        return criterion_class

    @staticmethod
    def get_dataset_name(curve_branch= 'T',enhance_p='E',
                              FeatureNum  = 1001,
                              val_filter  = None,volume=None,range_clip=None,**kargs):
        return get_dataset_name(curve_branch=curve_branch,curve_flag='C',enhance_p=enhance_p,
                                     FeatureNum=FeatureNum,
                                     val_filter=val_filter,volume=volume,range_clip=range_clip)


class SMSDatasetB1NES32(SMSDatasetN):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        super().__init__(curve_path_list,image_path_list,FeatureNum=32,curve_branch='T',enhance_p='E',**kargs)
    @staticmethod
    def get_dataset_name(**kargs):
        return get_dataset_name(curve_flag='N',FeatureNum=32,curve_branch='T',enhance_p='E',**kargs)
class SMSDatasetB1NES128(SMSDatasetN):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        super().__init__(curve_path_list,image_path_list,FeatureNum=128,curve_branch='T',enhance_p='E',**kargs)
    @staticmethod
    def get_dataset_name(**kargs):
        return get_dataset_name(curve_flag='N',FeatureNum=128,curve_branch='T',enhance_p='E',**kargs)
class SMSDatasetB1NES1001(SMSDatasetN):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        super().__init__(curve_path_list,image_path_list,FeatureNum=1001,curve_branch='T',enhance_p='E',**kargs)
    @staticmethod
    def get_dataset_name(**kargs):
        return get_dataset_name(curve_flag='N',FeatureNum=1001,curve_branch='T',enhance_p='E',**kargs)

class SMSDatasetB1CES32(SMSDatasetC):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        super().__init__(curve_path_list,image_path_list,FeatureNum=32,curve_branch='T',enhance_p='E',**kargs)
    @staticmethod
    def get_dataset_name(**kargs):
        return get_dataset_name(curve_flag='C',FeatureNum=128,curve_branch='T',enhance_p='E',**kargs)
