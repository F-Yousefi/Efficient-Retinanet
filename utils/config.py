class Config(object):
    dir_to_dataset = "../dataset"
    device = "cuda"
    gpu_under_control = True
    high_temp = 78
    low_temp = 55
    delay_time = 0.5
    batch_size = 4
    num_workers = 2
    dir_to_pretrained_model = "./pre_trained_models/ER_max_mAP_efficientnet-b4.ckpt"
    backbone_model_name = "efficientnet-b4"
    dir_to_save_model = "./pre_trained_models/"
    dir_to_log = "./pre_trained_models/"
    epochs = 40


config = Config()
