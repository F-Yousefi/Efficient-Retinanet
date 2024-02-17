class Config(object):
    dir_to_dataset = "../dataset"
    dir_to_save_model = "./pre_trained_models/"
    dir_to_log = "./pre_trained_models/"
    gpu_under_control = False
    device = "cuda"
    high_temp = 78
    low_temp = 55
    delay_time = 0
    batch_size = 4
    num_workers = 2
    dir_to_pretrained_model = None
    backbone_model_name = "efficientnet-b0"
    epochs = 40


config = Config()
