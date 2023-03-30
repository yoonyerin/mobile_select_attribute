from os.path import join

"""CONSTANTS"""
#DATASET_DIR: str = './data/'
DATASET_DIR: str = '/Users/yerinyoon/Documents/cubig/mobile_attribute_select/data'
SERVICE_DIR: str= "/Users/yerinyoon/Documents/cubig/mobile_attribute_select/data/fake_dir"
ORIGINAL_DIR: str= "/Users/yerinyoon/Documents/cubig/mobile_attribute_select/data/real_dir"

IMG_DIR: str = join(DATASET_DIR, 'img_align_celeba')
SERVICE_IMG_DIR: str = join(SERVICE_DIR, 'fake_img_dir')
ORIGINAL_IMG_DIR: str = join(ORIGINAL_DIR, 'real_img_dir')
PARTITION_FILE: str = join(DATASET_DIR, 'list_eval_partition.csv')
ATTRIBUTE_FILE: str = join(DATASET_DIR, 'list_attr_celeba.csv')
TRAIN_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'train_attr_list.csv')
VAL_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'val_attr_list.csv')
TEST_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'test_attr_list.csv')
SERVICE_ATTRIBUTE_LIST: str = join(SERVICE_DIR, 'fake_label_file.csv')
ORIGINAL_LIST: str = join(ORIGINAL_DIR, 'real_label_file.csv')
CHECKPOINT_DIR: str = '/Users/yerinyoon/Documents/cubig/mobile_attribute_select/checkpoints'
BACKUP_DIR: str = './backups'
TESTSET_DIR: str = './data/testset'
INFERENCE_DIR: str = './inf'
    
"""HYPER PARAMETERS"""
# Miscs
manual_seed = 42 #1903
evaluate = False
gpu_id = '0'
disable_tqdm = True
auto_hibernate = True

# optimization
train_batch = 100 #256
dl_workers = 4
test_batch = 100 #128
epochs = 80 #60
lr = 0.01 #0.1, 0.01
lr_decay = 'step' #step, cos, linear, linear2exp, schedule
step = 30 # interval for learning rate decay in step mode
schedule = [30, 35, 40, 45, 50, 55, 56, 57, 58, 59, 60] # decrease learning rate at these epochs [150, 225]
turning_point = 100 # epoch number from linear to exponential decay mode
gamma = 0.1 #LR is multiplied by gamma on schedule 0.1
momentum = 0.9
weight_decay = 1e-4  #1e-4 
criterion = 'FocalLoss' #FocalLoss CE
optimizer = 'SGD' #SGD, Adam, AdamW
scheduler = 'ReduceLROnPlateau' #Manual ReduceLROnPlateau OneCycleLR CosineWarmupLR
patience = 5 # patience for ReduceLROnPlateau scheduler
no_bias_bn_decay = True # Turn off bias decay (default: True)
label_smoothing = 0 # 0 to turn off, 0.1 (default)
mixed_up = 0.2 # mixedup alpha value: 0 to turn off, 0.2 (default)

# Early Stopping
early_stopping = True
es_min = 30 # minimum patience
es_patience = 10 

# Checkpoints and loggers
ckp_resume = '' #path to latest checkpoint (default: none) #join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
ckp_logger_fname = join(CHECKPOINT_DIR, 'log.txt')
checkpoint_fname = join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
bestmodel_fname = join(CHECKPOINT_DIR, 'model_best.pth.tar')
tensorboard_dir = 'runs'
train_plotfig = join(CHECKPOINT_DIR, 'logs.eps')
train_saveplot = True
test_preds_fname = join(CHECKPOINT_DIR, 'test_preds.json')

# Architecture
arch = 'FaceAttrMobileNetV2' # #model architecture: FaceAttrResNet FaceAttrMobileNetV2 FaceAttrResNeXt
pt_layers = 50 # 18, 34, 50

fixed_attrs=["Wearing_Lipstick","Attractive", "High_Cheekbones", "High_Cheekbones", "Mouth_Slightly_Open", "Smiling"]
hair_priority={"Wearing_Lipstick":[0, 1, 4], "Attractive":[-1], "High_Cheekbones":[0, 3, 1, 4], "High_Cheekbones":[0, 4, 1], "Mouth_Slightly_Open":[0, 4, 3, 1],  "Smiling":[0, 1, 3, 4]}


## Selected attrs for fixed_attrs: "Wearing_Lipstick","Attractive", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Smiling"]
selected_1=["Narrow_Eyes", "Straight_Hair", "Pale_Skin",  "hair_color", "Receding_Hairline", "Bald", "Oval_Face", 
            "Double_Chin", "Bushy_Eyebrows", "5_o_Clock_Shadow", "Wavy_Hair", "Wearing_Earrings", "No_Beard", "Arched_Eyebrows", "Attractive", "Male",
            "Heavy_Makeup"]
selected_2=[ "Mouth_Slightly_Open", "Straight_Hair", "Bushy_Eyebrows", "Bangs", "5_o_Clock_Shadow", "Big_Lips", "Narrow_Eyes", 
             "Double_Chin", "Wavy_Hair", "Pointy_Nose", "Chubby", "Arched_Eyebrows", "Big_Nose", "Young", "Male", "Heavy_Makeup", "Wearing_Lipstick"]
selected_3=["hair_color", "Young", "Straight_Hair", "Receding_Hairline", "Chubby", "Big_Lips", "5_o_Clock_Shadow", "No_Beard", "Oval_Face", "Rosy_Cheeks", "Male",
            "Heavy_Makeup", "Wearing_Lipstick", "Mouth_Slightly_Open"]
selected_4=["Narrow_Eyes", "Straight_Hair", "Pale_Skin", "Mouth_Slightly_Open", "hair_color", "Receding_Hairline", "Oval_Face", "Goatee", "Big_Nose",
            "Attractive", "Arched_Eyebrows", "5_o_Clock_Shadow", "No_Beard", "Heavy_Makeup", "Wearing_Lipstick"]
selected_5=["Bald", "Pointy_Nose", "hair_color", "Young", "Straight_Hair", "Attractive", "Chubby", "Oval_Face", "Male", "Narrow_Eyes", "Rosy_Cheeks",
            "High_Cheekbones"]
selected_6=["hair_color", "Straight_Hair", "Big_Lips","Young",
            "Bags_Under_Eyes", "No_Beard", "Male", "Attractive", "Oval_Face", "Rosy_Cheeks", "Mouth_Slightly_Open", "High_Cheekbones"]

attibutes_by_fixed=[selected_1, selected_2, selected_3, selected_4, selected_5, selected_6]