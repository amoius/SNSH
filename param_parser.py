import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Bitcoin OTC dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run SGCN.")

    parser.add_argument("--edge_list_url",
                        nargs="?",
                        default="./input/guinea/reledgelist.txt",
                        help="Edge list.")
    parser.add_argument("--output_url",
                        nargs="?",
                        default="./output/guinea/",
                        help="Edge list with relation.")
    parser.add_argument("--good_window_url",
                        nargs="?",
                        default="./input/guinea/good.tsv",
                        help="Good window.")

    parser.add_argument("--bad_window_url",
                        nargs="?",
                        default="./input/guinea/bad.tsv",
                        help="Bad window.")


    parser.add_argument("--num_ns",
                        type=int,
                        default=2,
                        help="Negative sampling.")

    parser.add_argument("--min_range",
                        type=int,
                        default=2,
                        help="Negative sampling.")

    parser.add_argument("--BATCH_SIZE",
                        type=int,
                        default=10,
                        help="Number of BATCH_SIZE. Default is 1000.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Embedding epochs. Default is 100.")

    parser.add_argument("--embedding_dim",
                        type=int,
                        default=2,
                        help="Embedding dimension. Default is 2.")
    return parser.parse_args()
