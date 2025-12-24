import pickle
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-VocabDict.p', 'rb') as f_vocab:
    vocab_dict = pickle.load(f_vocab)

vocab_back_dict = {val: key for key, val in vocab_dict.items()}

with open('NEWDatasets/ccbench-dataset-preprocessed/6col-VocabBackDict.p', 'wb') as f_vocab_back:
    pickle.dump(vocab_back_dict, f_vocab_back, protocol=pickle.HIGHEST_PROTOCOL)
    