from .dataset_module import *
from .utils import download_dropbox_url
#from .online_resource import online_path
with open(f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/.DATARoot.json",'r') as f:RootDict=json.load(f)
DATAROOT  = RootDict['DATAROOT']
SAVEROOT  = RootDict['SAVEROOT']


def get_FAST_B1NE_dataset_online(OfflinedataRoot=DATAROOT,curve_branch='T',dataset="PLG",FeatureNum=128,range_clip=None,
                          type_predicted="curve",target_predicted="simple",download=True,**kargs):
    DATASETROOT= os.path.join(OfflinedataRoot,f"{dataset}DATASET")
    if not os.path.exists(DATASETROOT):os.makedirs(DATASETROOT)
    download_url = online_path[f"{dataset}.B1NES{FeatureNum}"]
    file_name    = re.findall(r"(.*?)[\?]", download_url.split("/")[-1])[0]
    filepath     = os.path.join(DATASETROOT,file_name)
    if download:download_dropbox_url(download_url,filepath,download=="redownload")
    dataset_train= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                DATAROOT=DATASETROOT,
                                **kargs)
    dataset_valid= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch='T',enhance_p='E',case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],DATAROOT=DATASETROOT,
                                **kargs)
    dataset_train.accu_list = ['MSError']
    return dataset_train,dataset_valid

def get_FAST_B1NE_dataset(DATAROOT=DATAROOT,curve_branch='T',dataset="PLG",FeatureNum=128,range_clip=None,
                          type_predicted="curve",target_predicted="simple",download=True,normf='none',**kargs):
    DATASETROOT= os.path.join(DATAROOT,f"{dataset}DATASET")
    CURVETRAIN = f"{DATASETROOT}/train_data_list"
    IMAGETRAIN = None
    CURVE_TEST = f"{DATASETROOT}/valid_data_list"
    IMAGE_TEST = None
    dataset_train= SMSDatasetN(CURVETRAIN,IMAGETRAIN,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,normf=normf,
                                **kargs)
    dataset_valid= SMSDatasetN(CURVE_TEST,IMAGE_TEST,FeatureNum=FeatureNum,curve_branch='T',enhance_p='E',case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],
                                **kargs)
    dataset_train.accu_list = ['MSError']
    return dataset_train,dataset_valid


def get_balance_2_classifier_dataset(loseFunction="CELoss",
         CURVETRAIN = f"{DATAROOT}/randomly_data_valid_3000/train_data_list",
         IMAGETRAIN = None,
         CURVE_TEST = f"{DATAROOT}/randomly_data_valid_3000/valid_data_list",
         IMAGE_TEST = None,curve_branch='T',**kargs):
    type_predicted = "onehot"
    target_predicted="balance_leftorright"
    FeatureNum=128
    dataset_train= SMSDatasetN(CURVETRAIN,IMAGETRAIN,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p="E",case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                **kargs)
    dataset_valid= SMSDatasetN(CURVE_TEST,IMAGE_TEST,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p="E",case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],
                                **kargs)
    dataset_train.use_classifier_loss(loseFunction)
    dataset_valid.use_classifier_loss(loseFunction)
    dataset_train.accu_list = ["ClassifierA","ClassifierP","ClassifierN"]
    return dataset_train,dataset_valid
