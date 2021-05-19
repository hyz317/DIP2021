import argparse
from dataset.dataset import DataSet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_set_dir', type=str, default="data/training")
    parser.add_argument('--base_dir', type=str, default="data/base")
    args = parser.parse_args()

    dataset = DataSet(
        training_img_dir=args.train_set_dir,
        base_info_dir=args.base_dir
    )
    dataset.shuffle()
