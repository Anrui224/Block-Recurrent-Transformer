from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from transformers import AutoTokenizer


def train_pg19_tokenizer():
    paths = [str(x) for x in Path('/home/archen/pg19').glob("**/*.txt")]

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(paths,trainer)
    tokenizer.save('/home/archen/block_recurrent_transformer/data/pg19_tokenizer')


# tokenizer = Tokenizer(BPE()).from_file('/home/archen/block_recurrent_transformer/data/pg19_tokenizer')
tokenizer = AutoTokenizer.from_pretrained('t5-base')

paths = [str(x) for x in Path('/home/archen/pg19/test').glob("**/*.txt")]
count = 1
for path in paths:
    print(count)
    count += 1
    filename = Path(path).name
    fileroute = Path(r'/home/archen/pg19/test_processed').joinpath(filename)
    # print(str(filename))
    if fileroute.exists():
        continue
    with open(path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    while '' in data:
        data.remove('')
    data = '\n'.join(data)
    encoded = tokenizer.encode(data)
    
    with open(fileroute, 'w') as f:
        f.write(str(encoded))



