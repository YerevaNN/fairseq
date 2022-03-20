import torch
roberta = torch.hub.load('/home/gayane/BartLM/', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/home/gayane/BartLM/roberta.large', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)


tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
roberta.decode(tokens)  # 'Hello world!'

# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)


# # Download RoBERTa already finetuned for MNLI
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# roberta.eval()  # disable dropout for evaluation

# # Encode a pair of sentences and make a prediction
# tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
# roberta.predict('mnli', tokens).argmax()  # 0: contradiction

# # Encode another pair of sentences
# tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
# roberta.predict('mnli', tokens).argmax()  # 2: entailment

roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)


import torch
from fairseq.data.data_utils import collate_tokens

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()

batch_of_pairs = [
    ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    ['potatoes are awesome.', 'I like to run.'],
    ['Mars is very far from earth.', 'Mars is very close.'],
]

batch = collate_tokens(
    [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = roberta.predict('mnli', batch)
print(logprobs.argmax(dim=1))
# tensor([0, 2, 1, 0])


roberta.cuda()
roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], device='cuda:0', grad_fn=<LogSoftmaxBackward>)