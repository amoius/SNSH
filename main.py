from snsh import SignedGraphConvolutionalNetwork
from param_parser import parameter_parser
from utils import tab_printer


def main():
    args = parameter_parser()
    tab_printer(args)
    trainer = SignedGraphConvolutionalNetwork(args)
    trainer.start()


if __name__ == "__main__":
    main()
