import argparse
import preprocess
from models import hmm, structured_perceptron

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", 
        required=True, 
        help="Possible parts: 2, 3, 4, 5, 5-laplace, 5-good_turing, 5-structured_perceptron")
    parser.add_argument("--datasets", 
        required=True, 
        help="Input datasets to be used, separated by commas. Datasets should be stored in data/")
    parser.add_argument("--epochs", 
        default=8,
        help="Needed only when running 5-structured_perceptron or 5. Defaults to 8.")
    args = parser.parse_args()

    dataset_ls = args.datasets.split(",")
    datasets = [f"data/{ds}" for ds in dataset_ls]
    
    for ds in datasets:
        train_data = preprocess.read_paired_file("{}/train".format(ds))
        dev_data = preprocess.read_observation_file("{}/dev.in".format(ds))
        if args.part == "2":
            model = hmm.HMM()
            model.train(train_data)
            model.predict_part2(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "3":
            model = hmm.HMM()
            model.train(train_data)
            model.predict_part3(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "4":
            model = hmm.HMM()
            model.train(train_data)
            model.predict_part4(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "5-laplace":
            model = hmm.HMM()
            model.train(train_data)
            model.predict_part5_laplace_smoothing(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "5-good_turing":
            model = hmm.HMM()
            model.train_good_turing(train_data)
            model.predict_part5_good_turing(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "5-structured_perceptron":
            model = structured_perceptron.StructuredPerceptron()
            model.train(train_data, args.epochs)
            model.predict(dev_data) \
                .write_preds("{}/dev.p{}.out".format(ds, args.part))
        elif args.part == "5":
            test_data = preprocess.read_observation_file("{}/test.in".format(ds))
            model = structured_perceptron.StructuredPerceptron()
            model.train(train_data, args.epochs)
            model.predict(test_data) \
                .write_preds("{}/test.p{}.out".format(ds, args.part))
        else: 
            raise argparse.ArgumentError(message="Please input a valid argument for --part.")