from fairseq.models.bart import BARTModel

model = "/home/gayane/BartLM/checkpoints"

bart = BARTModel.from_pretrained(
    model,
    checkpoint_file=model +'/home/gayane/BartLM/checkpoints/freesolv0.0005/checkpoint9.pt',
    data_name_or_path='/home/gayane/BartLM/freesolv-bin/processed'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('/home/gayane/BartLM/moleculenet/datasets/train_bbbp.csv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split()
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))