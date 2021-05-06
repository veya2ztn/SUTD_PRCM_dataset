from .dataset_module import *
from .utils import download_dropbox_url
#from .online_resource import online_path



def get_FAST_B1NE_dataset(OfflinedataRoot,dataset="PLG",FeatureNum=128,range_clip=None,
                          type_predicted="curve",target_predicted="simple",download=True,**kargs):
    DATASETROOT= os.path.join(OfflinedataRoot,f"{dataset}DATASET")
    if not os.path.exists(DATASETROOT):os.makedirs(DATASETROOT)
    download_url = online_path[f"{dataset}.B1NES{FeatureNum}"]
    file_name    = re.findall(r"(.*?)[\?]", download_url.split("/")[-1])[0]
    filepath     = os.path.join(DATASETROOT,file_name)
    if download:download_dropbox_url(download_url,filepath,download=="redownload")
    dataset_train= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch='T',enhance_p='E',case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                DATAROOT=DATASETROOT,
                                **kargs)
    dataset_valid= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch='T',enhance_p='E',case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],DATAROOT=DATASETROOT,
                                **kargs)
    return dataset_train,dataset_valid

def get_balance_2_classifier_dataset(loseFunction="CELoss",
         CURVETRAIN = f"{DATAROOT}/randomly_data_valid_3000/train_data_list",
         IMAGETRAIN = None,
         CURVE_TEST = f"{DATAROOT}/randomly_data_valid_3000/valid_data_list",
         IMAGE_TEST = None):
    dataset_train= SMSDatasetB1NES128(CURVETRAIN,IMAGETRAIN,
                                type_predicted="onehot",target_predicted="balance_leftorright",
                                case_type='train',verbose=True)
    dataset_valid= SMSDatasetB1NES128(CURVE_TEST,IMAGE_TEST,
                                type_predicted="onehot",target_predicted="balance_leftorright",
                                normf=[dataset_train.forf,dataset_train.invf],
                                case_type='test')
    dataset_train.use_classifier_loss(loseFunction)
    dataset_valid.use_classifier_loss(loseFunction)
    return dataset_train,dataset_valid
