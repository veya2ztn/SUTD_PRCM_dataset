from .dataset_module import *
from .utils import download_dropbox_url
from .online_resource import online_path
with open(f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/.DATARoot.json",'r') as f:RootDict=json.load(f)
DATAROOT  = RootDict['DATAROOT']
SAVEROOT  = RootDict['SAVEROOT']

curve_branch_flag = {"T":"1","1":"1","R":"2","2":"2","P":"3","3":"3"}
def get_FAST_B1NE_dataset_online(OfflinedataRoot=DATAROOT,curve_branch='T',dataset="RDN",FeatureNum=128,range_clip=None,
                          type_predicted="curve",target_predicted="simple",download=True,normf="none",**kargs):
    DATASETROOT= os.path.join(OfflinedataRoot,f"{dataset}DATASET")
    #DATASETROOT=OfflinedataRoot
    DATAFLAG     = f'B{curve_branch_flag[curve_branch] }NES{FeatureNum}'
    DATASETPATH= os.path.join(DATASETROOT,DATAFLAG)
    if not os.path.exists(DATASETPATH):os.makedirs(DATASETPATH)
    download_url = online_path[f"{dataset}.{DATAFLAG}"]
    file_name    = re.findall(r"(.*?)[\?]", download_url.split("/")[-1])[0]
    filepath     = os.path.join(DATASETROOT,DATAFLAG,file_name)
    if download:download_dropbox_url(download_url,filepath,download==download)
    dataset_train= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,normf=normf,
                                DATAROOT=DATASETROOT,
                                **kargs)
    dataset_valid= SMSDatasetN(None,None,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],DATAROOT=DATASETROOT,
                                **kargs)
    dataset_train.accu_list = ['MSError']
    return dataset_train,dataset_valid

def get_FAST_B1NE_dataset(DATAROOT=DATAROOT,curve_branch='T',dataset="RDN",FeatureNum=128,
                          type_predicted="curve",target_predicted="simple",download=True,normf='none',**kargs):
    DATASETROOT= os.path.join(DATAROOT,f"{dataset}DATASET")
    CURVETRAIN = f"{DATASETROOT}/train_data_list"
    IMAGETRAIN = None
    CURVE_TEST = f"{DATASETROOT}/valid_data_list"
    IMAGE_TEST = None
    dataset_train= SMSDatasetN(CURVETRAIN,IMAGETRAIN,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='train',
                                type_predicted=type_predicted,target_predicted=target_predicted,normf=normf,
                                **kargs)
    dataset_valid= SMSDatasetN(CURVE_TEST,IMAGE_TEST,FeatureNum=FeatureNum,curve_branch=curve_branch,enhance_p='E',case_type='test',
                                type_predicted=type_predicted,target_predicted=target_predicted,
                                normf=[dataset_train.forf,dataset_train.invf],
                                **kargs)
    dataset_train.accu_list = ['MSError']
    return dataset_train,dataset_valid


def get_balance_2_classifier_dataset(loseFunction="CELoss",
         DATAROOT=DATAROOT,dataset="RDN",
         curve_branch='T',**kargs):
    DATASETROOT= os.path.join(DATAROOT,f"{dataset}DATASET")
    CURVETRAIN = f"{DATASETROOT}/train_data_list"
    IMAGETRAIN = IMAGE_TEST = None
    CURVE_TEST = f"{DATASETROOT}/valid_data_list"
    type_predicted  = "onehot"
    target_predicted= "balance_leftorright"
    FeatureNum      = 128
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
