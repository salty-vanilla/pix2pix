import argparse
import os
import sys
sys.path.append(os.getcwd())
from pix2pix import Pix2Pix
from generator import UNet
from discriminator import ResidualDiscriminator
from image_sampler import ImageSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x_dir', type=str)
    parser.add_argument('y_dir', type=str)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--height', '-ht', type=int, default=256)
    parser.add_argument('--width', '-wd', type=int, default=256)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=10)
    parser.add_argument('--gp_weight', '-gp', type=float, default=10.)
    parser.add_argument('--l1_weight', '-l1', type=float, default=1.)
    parser.add_argument('--initial_steps', '-is', type=int, default=20)
    parser.add_argument('--initial_critics', '-ic', type=int, default=20)
    parser.add_argument('--normal_critics', '-nc', type=int, default=5)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="uniform")
    parser.add_argument('--upsampling', '-up', type=str, default="deconv")
    parser.add_argument('--dis_norm', '-dn', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 color_mode_x='rgb',
                                 color_mode_y='rgb',
                                 normalization_x='tanh',
                                 normalization_y='tanh',
                                 is_flip=False)

    generator = UNet((args.height, args.width, 3),
                     color_mode='rgb',
                     upsampling=args.upsampling,
                     is_training=True)

    discriminator = ResidualDiscriminator((args.height, args.width, 6),
                                          normalization=args.dis_norm,
                                          is_training=True)

    pix2pix = Pix2Pix(generator,
                      discriminator,
                      l1_weight=args.l1_weight,
                      gradient_penalty_weight=args.gp_weight,
                      is_training=True)
    exit()
    pix2pix.fit(image_sampler.flow_from_directory(args.x_dir,
                                                  args.y_dir,
                                                  batch_size=args.batch_size),
                result_dir=args.result_dir,
                model_dir=args.model_dir,
                save_steps=args.save_steps,
                visualize_steps=args.visualize_steps,
                nb_epoch=args.nb_epoch,
                initial_steps=args.initial_steps,
                initial_critics=args.initial_critics,
                normal_critics=args.normal_critics)


if __name__ == '__main__':
    main()
