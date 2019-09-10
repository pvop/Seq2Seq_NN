import argparse
def Hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_vocab_size",default=30000,type=int,
                       help="source vocabulary size")
    parser.add_argument("--target_vocab_size", default=30000, type=int,
                       help="target vocabulary size")
    parser.add_argument("--dropout",default=0.3,type=float,
                        help="model dropout rate")
    parser.add_argument("--learning_rate",default=0.0001,type=float,
                        help="model learning rate")
    parser.add_argument("--max_gradient_norm",default=5.0,type=float,
                        help='max gradient norm to avoid gradient explosion')
    parser.add_argument("--embedding_size",default=512,type=int,
                        help="word embedding size")
    parser.add_argument("--rnn_size", default=512, type=int,
                        help="rnn size")
    parser.add_argument("--lstm_layers",default=4,type=int,
                        help="lstm layers")
    parser.add_argument("--beam_search",default=5,type=int,
                        help="beam search")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size in training and testing")
    parser.add_argument("--max_source_length", default=100, type=int,
                        help="max source length in training")
    parser.add_argument("--max_target_length", default=100, type=int,
                        help="max target length in training")


    return parser.parse_args()