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

def read_observation_file(filename):
    with open(filename) as f:
        file_content = f.read()
    # sort file into comments, with each comment being an element of the list
    l = file_content.strip().split('\n\n')
    
    # each word-line is then made into an element within each comment element
    l2 = [i.split('\n') for i in l]

    return l2

if __name__ == "__main__":
    # train_data = read_paired_file("{}/train".format("data/MOCK"))
    # model = hmm.HMM()
    # model.train(train_data)
    # print(model.count_emissions)
    # print(model.count_states)
    # print(model.count_transitions)
    # print(model._calculate_emission_prob_with_unk("Why", "O"))
    datasets = ["data/SG", "data/EN", "data/CN"]
    # datasets = ["data/MOCK"]
    for ds in datasets:
        train_data = read_paired_file("{}/train".format(ds))
        model = hmm.HMM()
        model.train(train_data)
        test_data = read_observation_file("{}/dev.in".format(ds))
        model.predict_part2(test_data) \
             .write_preds("{}/dev.p2.out".format(ds))