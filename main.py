import argparse

from trainer import Trainer


def main(args):
    model_trainer = Trainer(args)
    if args.test:
        model_trainer.test()
    else:    
        model_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, 
        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, 
        help="Batch size.")
    parser.add_argument("--lr", type=float, default=8e-4, 
        help="Learning rate.")
    parser.add_argument("--lr_min", type=float, default=1e-5, 
        help="Minimum learning rate.")
    parser.add_argument("--mixup_alpha", type=float, default=1.2, 
        help="Alpha value used for sampling from beta distribution for mixup.")
    parser.add_argument("--save_dir", default="models/", 
        help="Directory to save the models.")
    parser.add_argument("--save_iter", type=int, default=5, 
        help="How often to save the model checkpoint.")
    parser.add_argument("--load", default="", 
        help="Load the model checkpoint.")
    parser.add_argument("--test", action="store_true", 
        help="Get results over the testing set.")

    main(parser.parse_args())