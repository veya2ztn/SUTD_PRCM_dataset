from .dataset_module import *
class SMSDatasetGraph_Test(SMSDatasetN):

    def __init__(self,curve_path_list,image_path_list,transformer=None,type_predicted=None,target_predicted=None,
                      case_type='train',volume=None,normf='none',val_filter=None,offline=True,enhance_p='E',verbose=True,DATAROOT=None,
                      image_transfermer=None,**kargs):
        import dgl
        self.verbose = verbose
        self.data_field = 'real'
        self.forf = lambda x:x
        self.invf = lambda x:x
        self.transformer=None
        self.image_type_shape = (16,16)
        self.curve_type_shape = (1,)
        if (type_predicted is None) or (target_predicted is None):
            logging_info('type_predicted and target_predicted must be assignned ')
            raise
        if transformer is not None:
            logging_info('please note this dataset force use none transformer, your configuration {} is block'.format(transformer.name))
        self.type_predicted=type_predicted
        self.target_predicted=target_predicted


        DATAROOT = "/media/tianning/DATA/metasurface/Compressed/randomly_data_valid_3000/B1NES128"
        if not isinstance(curve_path_list,list) :
            tail_curve_path_name = len(os.listdir(curve_path_list))
        else:
            tail_curve_path_name = len(curve_path_list)

        offline_dglimgdata_name = f'{case_type}_imagedata_{tail_curve_path_name}.dgl'
        offline_dgleigdata_name = f'{case_type}_imagedata_{tail_curve_path_name}.dgl_node_eig.npy'
        offline_vectordata_name = f'{case_type}.feat.onehot.balance_leftorright_{tail_curve_path_name}.npy'
        # offline_dglimgdata_name = f'test_imagedata_3.dgl'
        # offline_dgleigdata_name = f'test_imagedata_3.dgl_node_eig.npy'
        # offline_vectordata_name = f'test.feat.onehot.balance_leftorright_3.npy'

        offline_dglimgdata_path = os.path.join(DATAROOT,offline_dglimgdata_name)
        offline_dgleigdata_path = os.path.join(DATAROOT,offline_dgleigdata_name)
        offline_vectordata_path = os.path.join(DATAROOT,offline_vectordata_name)

        graph_eigv  =         np.load(offline_dgleigdata_path)
        graph_labels=         np.load(offline_vectordata_path)
        graph_lists = dgl.load_graphs(offline_dglimgdata_path)[0]

        self.graph_lists = graph_lists
        self.vector      = torch.from_numpy(graph_labels).reshape(-1,1)
        self.n_samples   = len(self.graph_lists)
        self.pos_enc_dim =pos_enc_dim=1

        for g,vec in zip(self.graph_lists,graph_eigv):
            g.ndata['feat'] = g.ndata['feat'].long()
            g.ndata['eig']  = torch.from_numpy(vec).float()
        if pos_enc_dim > 0:self._add_positional_encodings(pos_enc_dim)


    def _add_positional_encodings(self, pos_enc_dim):
        for g in self.graph_lists:
            g.ndata['pos_enc'] = g.ndata['eig'][:,1:pos_enc_dim+1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.vector[idx]
    def _collate_fn(self,samples):
        graphs, labels = map(list, zip(*samples))
        labels         = torch.cat(labels).long().reshape(-1,1)
        tab_sizes_n    = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n    = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n        = torch.cat(tab_snorm_n).sqrt()

        tab_sizes_e    = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e    = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e        = torch.cat(tab_snorm_e).sqrt()
        #print(len(graphs), labels.shape, snorm_n.shape, snorm_e.shape)
        batched_graph  = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e


class SMSDatasetB1NES16(SMSDatasetB1NES):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        self.FeatureNum=16
        super().__init__(curve_path_list,image_path_list,**kargs)
class SMSDatasetB1NES32Full(SMSDatasetB1NES32):
    def __init__(self,curve_path_lists,image_path_lists,**kargs):
        assert image_path_lists is None
        curve_path_list_full=[]
        image_path_list_full=[]
        for curve_path_list in curve_path_lists:
            curve_path_list_now,image_path_list_now = convertlist(curve_path_list)
            curve_path_list_full.extend(curve_path_list_now)
            image_path_list_full.extend(image_path_list_now)
        super().__init__(curve_path_list_full,image_path_list_full,DATAROOT="/data/DATA/MetaSurface/all_data",**kargs)
class SMSDatasetB1NDS128(SMSDatasetB1NES):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        self.FeatureNum=128
        super().__init__(curve_path_list,image_path_list,enhance_p='D',**kargs)
class SMSDatasetB1NDS32(SMSDatasetB1NES):
    def __init__(self,curve_path_list,image_path_list,**kargs):
        self.FeatureNum=32
        super().__init__(curve_path_list,image_path_list,enhance_p='D',**kargs)

class MSBaseDataSet(BaseDataSet):
    DataSetType=""
    _collate_fn=None
    def __init__(self,vec_dim,normf,verbose,branch=None,volume=None,case_type='train',interprocess=False):
        '''
        curve_path is the Abs path of curve npy data (B,2,1001) if full curve set
                                                     (B,1,1001) if tiny curve set (for test)
                                                     (batch,branch,len,)
        image_path is the Abs path of image npy data (B,256)
        transformer is one curve-vector class

        If offline(default is True) is True,
            this load will search the same dir as curve_path
            load transformed data if there exists.
        Otherwise, it will scan,transform and save all the data first

        volume will change the metadata. Use set length change the data range is recommended
        '''

        #self.imagedata   = image_data
        #self.curvedata   = curve_data
        #self.vector      = vector
        #self.transformer = transformer
        self.FeatureNum    = vec_dim
        self.criterion  = torch.nn.MSELoss
        self.use_transform=True
        self.imagesize  = 16
        self.normf      = normf
        self.case_type  = case_type
        self.verbose    = verbose
        self.branch     = branch
        self.output_class='regression'
        # pickup curve data complex or real
        self.curve_field     = curve_field = self.transformer.curve_field
        self.curve_feature   = curve_feature = self.transformer.feature
        if curve_field == 'real' and not isinstance(self.curvedata,torch.Tensor):
            self.curvedata  = self.curvedata.real.astype('float')

        # pickup branch
        if self.branch is not None:
            self.curvedata  = self.curvedata[:,self.branch:branch+1,:] #(B,2,1001)
            if self.vector is not None:
                self.vector  = self.vector[:,self.branch:self.branch+1,:] #(B,2,1001)


        # pickup max volume size
        if volume is not None:
            self.curvedata = self.curvedata[:volume]
            self.imagedata = self.imagedata[:volume]
            self.vector    = self.vector[:volume] if self.vector is not None else None

        # conver to curve size
        self.curvedata = np2torch(self.curvedata)
        self.imagedata = np2torch(self.imagedata)
        self.imagedata = self.imagedata.reshape(-1,1,self.imagesize,self.imagesize)
        self.vector    = np2torch(self.vector) if self.vector is not None else None

        if self.transformer.vector_field == 'real' and not "norm" in self.transformer.feature:
            self.vector = self.vector[...,0:1]



        # normlization part
        if isinstance(self.normf,str) or self.normf is None:
            string = self.normf if self.normf is not None else "Identy"
            logging_info("this dataset use 【{}】 norm".format(string))
            self.normlizationer   = normlizationer(self.vector,self.normf)
            self.forf   =  self.normlizationer.forf
            self.invf   =  self.normlizationer.invf
        elif isinstance(self.normf,list):
            self.forf,self.invf = normf
            string = "Inherit"
        else:
            raise
        self.vector     = self.forf(self.vector) if self.vector is not None else None
        filter_type     = self.transformer.filter_type
        vector_type     = self.transformer.vector_type
        self.vector_type= vector_type
        self.vector_real= self.vector
        if not interprocess:
            if  ('maxp' in vector_type):
                assert self.transformer.class_name == "Sample" or "Identity"
                assert self.transformer.feature == 'norm'
                assert self.vector is not None
                assert string == "Identy" or "none" or Inherit
                print("now detect the max peak rather than whole curve")
                print("the predicted vector is digital vector")
                assert self.transformer.sample_num == 1001
                divide_num  = int(self.vector_type[4:])
                self.curvedata   = 1-self.curvedata
                self.vector_real = 1-self.vector_real
                self.vector      = 1-self.vector[...,200:].squeeze(1)
                labels           = torch.argmax(self.vector,1)
                one_hot          = torch.zeros_like(self.vector)
                one_hot[torch.arange(self.vector.shape[0]), labels] = 1
                peaks            = one_hot
                interv           = np.array_split(peaks,divide_num,-1)
                self.vector      = torch.stack([(kk.sum(1)>0) for kk in interv],-1)
                self.vector      = torch.argmax(self.vector,1)
                #self.vector      = self.vector.unsqueeze(1)
                self.output_class='digital'
            if  ('peak' in vector_type or "plat" in vector_type):
                assert self.transformer.class_name == "Sample" or "Identity"
                assert self.transformer.feature == 'norm'
                assert self.vector is not None
                assert string == "Identy" or "none" or Inherit
                print("now detect the peak rather than whole curve")
                print("the predicted vector is digital vector")
                if vector_type == "peak":
                    self.vector = find_peak(1-self.vector.squeeze(1),offsite=15)
                    self.vector = self.vector.unsqueeze(1)
                elif "plat" in vector_type:
                    assert self.transformer.sample_num == 1001
                    divide_num  = int(self.vector_type[4:])
                    self.curvedata   = 1-self.curvedata
                    self.vector_real = 1-self.vector_real
                    self.vector      = 1-self.vector[...,200:].squeeze(1)
                    peaks            = find_peak(self.vector)
                    interv           = np.array_split(peaks,divide_num,-1)
                    self.vector      = torch.stack([(kk.sum(1)>0) for kk in interv],-1)
                    self.vector      = self.vector.unsqueeze(1)
                elif vector_type == "peakandcurve":
                    vector    = 1-self.vector
                    self.curvedata = 1-self.curvedata
                    self.vector_real = vector
                    peak = find_peak(self.vector.squeeze(1),offsite=15).unsqueeze(1).float()
                    self.vector = torch.cat([vector,peak],-1)
                    self.forf = lambda x:x
                    self.invf = lambda x:x[...,:128]
            if  (filter_type=='smoothpeak'):
                assert self.transformer.class_name == "Sample" or "Identity"
                assert self.transformer.feature == 'norm'
                assert self.vector is not None
                assert self.curvedata.shape[1]==1
                print('use filted dataset,only smooth curve count')
                self.choose_index = curver_filte_smoothpeak(self.curvedata[:,0,:])
                self.curvedata    = self.curvedata[self.choose_index]
                self.imagedata    = self.imagedata[self.choose_index]
                self.vector       = self.vector[self.choose_index]
                self.vector_real  = self.vector_real[self.choose_index]
                #print(len(choose_index))
            if  (filter_type=='maxvalue'):
                assert self.curvedata.shape[1]==1
                print('use filted dataset,only smooth curve count')
                self.vector       = torch.max(self.vector,-1,keepdim=True)[0]
                self.output_class = 'number'
            if  (filter_type=='maxnormed'):
                assert self.curvedata.shape[1]==1
                print('use filted dataset,only smooth curve count')
                self.vector_max   = torch.max(self.vector,-1,keepdim=True)[0]
                self.vector       = self.vector/self.vector_max
                self.output_class = 'number'
        self.image_type_shape= self.imagedata.shape[1:]
        self.curve_type_shape= self.vector.shape[1:]
        if not interprocess and ('maxp' in vector_type):self.curve_type_shape = [divide_num]

        self.length    = len(self.curvedata)
        self.transformer.forf  = self.forf
        self.transformer.invf  = self.invf
        self.length = len(self.curvedata)
        if verbose:
            headers = ['type','fea_type','curve data shape', 'image data shape', 'vector data shape']
            data = [[case_type,self.transformer.feature,
                     tuple2str(self.curvedata.shape),
                     tuple2str(self.imagedata.shape),
                     tuple2str(self.vector.shape) if self.vector is not None else "generated"]]
            tp.table(data, headers,width=17)

            # print("the curve  data shape is {}".format(self.curvedata.shape))
            # print("the image  data shape is {}".format(self.imagedata.shape))
            # print("the vector data shape is {}".format(self.vector.shape)) if self.vector is not None else print("using generate data")
            # print("--------------------")
    def show_a_demo(self,num=None):
        if num is None:
            random_index = np.random.randint(0,self.length)
        else:
            random_index=num
        curve_data  = self.curvedata[random_index]
        image_data  = self.imagedata[random_index]
        curve_train = self.vector_real[random_index]
        curve_train = self.invf(curve_train)
        rec_train   = self.transformer.vector2curve(curve_train)
        curve_data  = self.transformer.torch2np(curve_data)[0]
        rec_train   = self.transformer.torch2np(rec_train)[0]
        x=np.arange(1001)
        if curve_train.shape[-1]==2:
            graph_fig, graph_axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
            graph_axes[0].plot(x,curve_data.real,'r',label='real')
            graph_axes[0].plot(x, rec_train.real,'b',label='rec')
            graph_axes[1].plot(x,curve_data.imag,'r',label='real')
            graph_axes[1].plot(x, rec_train.imag,'b',label='rec')
            graph_axes[0].legend(loc='upper right')
            graph_axes[1].legend(loc='upper right')
            plt.show()
        elif 'peak' in self.vector_type:
            vector = self.transformer.torch2np(self.vector_real[random_index])[0]
            if self.vector_type == "peak":
                peak = self.transformer.torch2np(self.vector[random_index])[0]
                vector = 1-vector
            elif  self.vector_type == "peakandcurve":
                peak = self.transformer.torch2np(self.vector[random_index][...,-128:])[0]
            graph_fig, graph_axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
            x=np.arange(len(curve_data))
            graph_axes[0].plot(x,curve_data,'r',label='real')
            graph_axes[0].plot(x, rec_train,'b',label='rec')
            x=np.arange(len(vector))
            graph_axes[1].plot(x,vector,'r',label='feat')
            graph_axes[1].plot(x,  peak,'b',label='peak')

            graph_axes[0].legend(loc='upper right')
            graph_axes[1].legend(loc='upper right')
        elif "plat" in self.vector_type:
            peak = self.transformer.torch2np(self.vector[random_index])[0]
            divide_num = int(self.vector_type[4:])
            xranges    = np.array_split(np.arange(801),divide_num)
            peak_real  = np.zeros(1001)
            for pos,v in zip(xranges,peak):peak_real[200+pos]=v
            peak_real=peak_real*curve_data.max()
            plt.plot(x,curve_data.real,'r',x,rec_train.real,'b',x,peak_real,'g')
        elif "maxp" in self.vector_type:
            peak       = self.vector[random_index][0].item()
            divide_num = int(self.vector_type[4:])
            xranges    = np.array_split(np.arange(801),divide_num)[peak]
            peak_real  = np.zeros(1001)
            peak_real[200+xranges]=1
            peak_real=peak_real*curve_data.max()
            plt.plot(x,curve_data.real,'r',x,rec_train.real,'b',x,peak_real,'g')
        else:
            plt.plot(x,curve_data.real,'r',x,rec_train.real,'b')


class MetaSurfaceSet(MSBaseDataSet):
    # deal with direct data set path
    def __init__(self,curve_path,image_path,transformer=None,
                 vec_dim=None,normf=None,offline=True,verbose=True,
                 **kargs):
        if transformer is None:
            print('you must assign a tranformer')
            raise
        self.image_path = image_path
        self.curve_path = curve_path
        self.transformer= transformer
        self.imagedata  = np.load(image_path).astype('float')
        self.curvedata  = np.load(curve_path)
        self.offline    = offline
        self.verbose    = verbose
        self.reoffline  = False
        curve_file_name = curve_path.replace('.npy','')#train.curve
        curve_feature   = self.transformer.feature
        vector_type     = self.transformer.vector_type
        choose_index    = None
        self.trans_path = curve_file_name + '.' + transformer.save_name +'.npy'#train.curve
        if curve_feature == "norm":
            self.curvedata  = np.abs(self.curvedata)
        if "enhance" in vector_type:
            self.curvedata  = np.round(self.curvedata,3)

            #print(self.curvedata.shape)
        if self.offline:
            self.vector = self._offline()
            self.vector = self.transformer.reduce(self.vector,vec_dim)
        else:
            self.vector = None


        super().__init__(vec_dim,normf,verbose,**kargs)

    def _offline(self):
        '''
        from the origin curve data      (B,2,1001)
        to   the compressed vector data (B,2,num)
        save in the self.trains_path  "xxxxxxx.npy"
        '''
        offline_verbose = True
        if os.path.exists(self.trans_path) and not self.reoffline:
            if offline_verbose:print("reload transformed data from  {}".format(self.trans_path),end='\r')
            vector = np.load(self.trans_path)
            return vector
        branches = self.curvedata.shape[1]
        vector=[]
        for branch in range(branches):
            branch_data = self.curvedata[:,branch,:]
            transf_data = self.transformer.curve2vector(branch_data)
            vector.append(transf_data)
        if offline_verbose:print("generate transformed data and save to   {}".format(self.trans_path),end='\r')
        vector=np.stack(vector,1)
        np.save(self.trans_path,vector)
        return vector
class MSDataSetList(MSBaseDataSet):
    def __init__(self,curve_path_list,image_path_list,transformer=None,
                 vec_dim=None,normf=None,verbose=True,offline=True,load_method='compact',**kargs):
        # we have three types dataset
        #  1. the origin curve data (...,1001). these data distributed in a list path like /train_list
        #  2. the origin image data (...,256).  these data distributed in a list path like /train_list
        #                                                                            .../train_list/Data01/
        #                                                                            .../train_list/Data02/
        #  3. the compress curve data in case the curve so large
        #         the compress method is recorded in [transformer]
        #  We default use 'offline' mode, so we will deposit compressed data.
        #  For 'distributed' method, all the compressed curve data will save in its father's file
        #                                                                           .../train_list/Data01/
        #  For 'compact' method,
        #      all the curve data (...,1001) image data (...,256) compress curve (...,fea_len) will
        #      concencate in three big matrix and save in a compact file named by data number.
        #                                                                       .../compact/train_number_97000
        #  So when call these data, next time,
        #     'distributed' method will collect all small file.
        #     'compact' method will call the big one directly.
        if transformer is None:
            print('you must assign a tranformer')
            raise
        self.curve_path = curve_path_list
        self.image_path = image_path_list

        self.imagedata  = []
        self.curvedata  = []
        self.vector     = []
        self.transformer= transformer
        if not isinstance(curve_path_list,list) and image_path_list is None:
            # this code reveal the curve file and image file from a high level path ../Data##
            curve_path_list,image_path_list = convertlist(curve_path_list)
        if isinstance(curve_path_list,str) and isinstance(image_path_list,str):
            # this code for single image and curve file
            if os.path.isfile(curve_path_list) and os.path.isfile(image_path_list):
                curve_path_list=[curve_path_list]
                image_path_list=[image_path_list]

        for curve_path,image_path in zip(curve_path_list,image_path_list):
            small_data_set = MetaSurfaceSet(curve_path,image_path,transformer,normf=None,offline=offline,verbose=False,interprocess=True,**kargs)
            self.imagedata.append(small_data_set.imagedata)
            self.curvedata.append(small_data_set.curvedata)
            self.vector.append(small_data_set.vector)
        self.imagedata = torch.cat(self.imagedata)
        self.curvedata = torch.cat(self.curvedata)
        if self.vector[0] is not None:
            self.vector = torch.cat(self.vector)
        else:
            self.vector = None
        #cat the curve data
        #cat the image data
        #cat the vector data if offline is True
        #inject into the dataset class
        super().__init__(vec_dim,normf,verbose,**kargs)
        #finish the final result like normlization, pickup branch, pickup max data point
class MSDataSetListLight(MSDataSetList):
    def __init__(self,curve_path,image_path,transformer,**krag):
        super().__init__(curve_path,image_path,transformer,branch=0,**krag)
class MetaSurfaceSetLight(MetaSurfaceSet):
    def __init__(self,curve_path,image_path,transformer,**krag):
        super().__init__(curve_path,image_path,transformer,branch=0,**krag)
class MetaSurfaceFast(Dataset):
    def __init__(self,curvedata,imagedata,vector,transform,forf,invf,case_type):
        super(MetaSurfaceFast).__init__()
        self.curvedata = torch.Tensor(curvedata)
        self.imagedata = torch.Tensor(imagedata)
        if vector is not None:
            self.vector    = torch.Tensor(vector)
        else:
            print('use generate vector')
            self.vector    =  forf(transform.curve2vector(self.curvedata))

        self.case_type = case_type
        self.invf      = invf
        self.forf      = forf
        self.transform = transform

    def __len__(self):
        return len(self.curvedata)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        curve_data  = self.curvedata[index]#(2,1001)
        image_data  = self.imagedata[index]#(1,16,16)
        vector_data = self.vector[index]#(2,num)
        if not self.case_type == "train":
            return vector_data,image_data,curve_data
        return vector_data,image_data
