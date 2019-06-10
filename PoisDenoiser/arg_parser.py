import argparse

def parse_args_joint_train():
    parser = argparse.ArgumentParser(description='Joint training of PoisNet (or UDNet)')

    parser.add_argument('--stages', type=int, default=2, help='# of stages.')
    parser.add_argument('--channels', type=int, default=8, help='Num of conv channels per each stage.')
    parser.add_argument('--train_batchsize', type=int, default=50, help='training batch size.')
    parser.add_argument('--val_batchsize', type=int, default=50, help='testing batch size.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train for.')
    parser.add_argument('--lr0', type=float, default=1e-2, help='learning rate. Default=1e-2.')
    parser.add_argument('--gpu_id', type=int, default=1, help='which gpu to use?')
    parser.add_argument('--num_images', type=int, default=None, help='# of images to train on')
    parser.add_argument('--start_epoch', type=int, default=None, help='index of epoch where to resume training from')
    parser.add_argument('--save_epoch', type=int, default=50, help='With what period (in epochs) to save models')
    parser.add_argument('--save_path_app', type=str, default='', \
                            help='model short type name, additional folder division. Recommended: <type>_<#imgs>. Ex: "pois_100" or "l2_100"')
    parser.add_argument('--model_type', type=str, default='', help='Type of model. Possible values: "pois" or "l2".')
    parser.add_argument('--milestones', type=str, default='', help='List of milestone points, Ex: "[1,2,3,4,5]". ')
    parser.add_argument('--loss', type=str, default='MSE', help='Type of loss. Possible values: "MSE", "L1", "perceptual_MSE", "pois".')
    parser.add_argument('--prox_param', action='store_true', help='A flag for creating a parameter for handling different noise levels. False by default.')
    parser.add_argument('--no_sharing_weights', action='store_true', help='A flag for no_sharing_weights or do sharing. False by default.')
    parser.add_argument('--do_VST', action='store_true', help='do VST or not. Default False.')
    parser.add_argument('--exp', type=str, default='confocal', help='Only for FMD dataset. Possible values: confocal and twophoton')
    opt = parser.parse_args()

    return opt