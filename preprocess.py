from model import hmm

def read_paired_file(filename):
    with open(filename, encoding = 'utf-8') as f:
        file_content = f.read()
    # sort file into comments, with each comment being an element of the list
    l = file_content.strip().split('\n\n')
    
    # each word-line is then made into an element within each comment element
    l2 = [i.split('\n') for i in l]
    
    # each word in a word line is made the zeroth element of the 2nd nested list, and its respective entity/sentiment label the first element
    for idx, line in enumerate(l2):
        line = [i.split(" ") for i in line]
        l2[idx] = line
    return l2

# TODO: Implement reader for test data

def read_observation_file(filename):
    with open(filename) as f:
        file_content = f.read()
    # sort file into comments, with each comment being an element of the list
    l = file_content.strip().split('\n\n')
    
    # each word-line is then made into an element within each comment element
    l2 = [i.split('\n') for i in l]

    return l2

if __name__ == "__main__":
    datasets = ["data/SG", "data/EN", "data/CN"]
    for ds in datasets:
        train_data = read_paired_file("{}/train".format(ds))
        model = hmm.HMM()
        model.train(train_data)
        test_data = read_observation_file("{}/dev.in".format(ds))
        model.predict_part2(test_data) \
             .write_preds("{}/part2_dev.prediction".format(ds))