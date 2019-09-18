import tensorflow as tf
import os, logging
import argparse
import yaml
from dataloader import landmarks_loader
from network import V_UNet

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))

def init_logging(out_base_dir,exp_name):
    os.makedirs(out_base_dir, exist_ok = True)
    out_dir = os.path.join(out_base_dir, exp_name)
    os.makedirs(out_dir, exist_ok = False)
    # init logging
    logging.basicConfig(filename = os.path.join(out_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return out_dir, logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True, help = "config path")
    parser.add_argument("--mode", default = "train")
    parser.add_argument("--log_dir", default = os.path.join(os.getcwd(), "log"), help = "login path")
    parser.add_argument("--checkpoint", help = "checkpoint path")
    parser.add_argument("--retrain", dest = "retrain", action = "store_true", help = "reset global_step to zero")
    parser.set_defaults(retrain = False)

    opt = parser.parse_args()
    config = get_config(opt.config)

    out_dir, logger = init_logging(opt.log_dir,exp_name = config['exp_name'])
    logger.info(opt)
    logger.info(yaml.dump(config))

    if opt.mode == "train":

        batch_size = config["batch_size"]
        img_shape = 2*[config["spatial_size"]] + [3]
        data_shape = [batch_size] + img_shape
        init_shape = [config["init_batches"] * batch_size] + img_shape
        sigma = config['sigma']
        data_dir = config['data_dir']
        img_list = config['img_list']

        loader = landmarks_loader(data_shape, train = True, sigma = sigma, data_dir = data_dir, img_list = img_list)

        init_loader = landmarks_loader(init_shape, train = True, sigma = sigma, data_dir = data_dir, img_list = img_list)

        valid_loader = landmarks_loader(data_shape, train = False, sigma = sigma, data_dir = data_dir, img_list = img_list, shuffle = False)

        model = V_UNet(config, out_dir, logger)
        if opt.checkpoint is not None:
            model.restore_graph(opt.checkpoint)
        else:
            model.init_graph(next(init_loader))
        if opt.retrain:
            model.reset_global_step()
        model.fit(loader, valid_loader)
    else:
        raise NotImplemented()
