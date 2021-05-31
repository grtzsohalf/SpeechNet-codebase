import torch
from functools import partial
from src.text import load_text_encoder
from src.audio import create_transform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torchaudio

HALF_BATCHSIZE_AUDIO_LEN = 400 # Batch size will be halfed if the longest wavefile surpasses threshold
QUAT_BATCHSIZE_AUDIO_LEN = 800 # Batch size will be quatered if the longest wavefile surpasses threshold
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_AUDIO_LEN )
HALF_BATCHSIZE_TEXT_LEN = 150

def collect_audio_batch(batch, mode, audio_config):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>)
       e.g. [(file1,txt1),(file2,txt2),...] '''

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]
    files, text, waveforms, wave_lens, sample_rate = [], [], [], [], 0
    for b in batch:
        files.append(str(b[0]).split('/')[-1].split('.')[0])
        text.append(torch.LongTensor(b[1]))
        waveform, sample_rate = torchaudio.load(str(b[0]))
        waveforms.append(waveform)
        wave_lens.append(len(waveform[0]))
        # print(waveform.shape, sample_rate, len(waveform[0]))
    if max(wave_lens) > HALF_BATCHSIZE_AUDIO_LEN*(audio_config['frame_shift']*sample_rate/1000) and mode=='train':
        #print('HALF!')
        files = files[:len(batch)//2]
        text = text[:len(batch)//2]
        waveforms = waveforms[:len(batch)//2]
        wave_lens = wave_lens[:len(batch)//2]
    '''
    if max(wave_lens) > QUAT_BATCHSIZE_AUDIO_LEN*(audio_config['frame_shift']*sample_rate/1000) and mode=='train':
        #print('QUAT!')
        files = files[:len(batch)//4]
        text = text[:len(batch)//4]
        waveforms = waveforms[:len(batch)//4]
        wave_lens = wave_lens[:len(batch)//4]
    '''
    return files, text, waveforms, wave_lens, sample_rate


def collect_text_batch(batch, mode):
    '''Collects a batch of text, should be list of list of int token
       e.g. [txt1 <list>,txt2 <list>,...] '''

    # Bucketed batch should be [[txt1, txt2,...]]
    if type(batch[0][0]) is list:
        batch = batch[0]
    # Half batch size if input to long
    if len(batch[0])>HALF_BATCHSIZE_TEXT_LEN and mode=='train':
        batch = batch[:len(batch)//2]
    # Read batch
    text = [torch.LongTensor(b) for b in batch]
    # Zero-padding
    text = pad_sequence(text, batch_first=True)

    return text


def create_dataset(tokenizer, ascending, name, path, bucketing, batch_size,
                   train_split=None, dev_split=None, test_split=None):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    if train_split is not None:
        # Training mode
        mode = 'train'
        tr_loader_bs = 1 if bucketing and (not ascending) else batch_size
        bucket_size = batch_size if bucketing and (not ascending) else 1 # Ascending without bucketing
        dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
        tr_set = Dataset(path,train_split,tokenizer, bucket_size, ascending=ascending)
        # Messages to show
        msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                             dev_split.__str__(),len(dv_set),batch_size,bucketing)

        return tr_set, dv_set, tr_loader_bs, batch_size//4, mode, msg_list
    else:
        # Testing model
        mode = 'test'
        dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
        tt_set = Dataset(path,test_split,tokenizer, 1) # Do not use bucketing for test set
        # Messages to show
        msg_list = _data_msg(name,path,dev_split.__str__(),len(dv_set),
                             test_split.__str__(),len(tt_set),batch_size//4,False)
        msg_list = [m.replace('Dev','Test').replace('Train','Dev') for m in msg_list]
        return dv_set, tt_set, batch_size//4, batch_size//4, mode, msg_list


def create_textset(tokenizer, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of text dataset'''
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriTextDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
    tr_set = Dataset(path,train_split,tokenizer, bucket_size)

    # Messages to show
    msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                         dev_split.__str__(),len(dv_set),batch_size,bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size//4, msg_list

def load_dataset(gpu_rank, world_size, rank, n_jobs, use_gpu, pin_memory, ascending, asr_corpus, lm_corpus, audio, text, **kwargs):
    ''' Prepare dataloader for training/validation'''

    # Audio feature extractor
    # audio_transform, feat_dim = create_transform(audio.copy())
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset (in testing mode, tr_set=dv_set, dv_set=tt_set)
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(tokenizer,ascending,**asr_corpus)
    # Collect function
    collect_tr = partial(collect_audio_batch, mode=mode, audio_config=audio)
    # audio transform [:-1] --> remove specaug
    #collect_dv = partial(collect_audio_batch, audio_transform=audio_transform[:-1], mode='test')
    collect_dv = partial(collect_audio_batch, mode='test', audio_config=audio)
    # Shuffle/drop applied to training set only
    shuffle = (mode=='train' and not ascending)
    drop_last = shuffle
    # Create data loader
    if use_gpu and rank is not None:
        tr_sampler = DistributedSampler(tr_set, num_replicas=world_size, rank=rank)
        dv_sampler = DistributedSampler(dv_set, num_replicas=world_size, rank=rank)
        tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=False, sampler=tr_sampler, drop_last=drop_last, 
                            collate_fn=collect_tr, num_workers=n_jobs, pin_memory=use_gpu)
        dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, sampler=dv_sampler, drop_last=drop_last, 
                            collate_fn=collect_dv, num_workers=n_jobs, pin_memory=use_gpu)
    else:
        tr_sampler = None
        tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_tr,
                            num_workers=n_jobs, pin_memory=use_gpu)
        dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                            num_workers=n_jobs, pin_memory=pin_memory)
    # Messages to show
    feat_dim = audio['feat_dim'] * (audio['delta_order'] + 1)
    data_msg.append('I/O spec.  | Audio feature = {}\t| feature dim = {}\t| Token type = {}\t| Vocab size = {}'\
                    .format(audio['feat_type'],feat_dim,tokenizer.token_type,tokenizer.vocab_size))

    return tr_sampler, tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, data_msg


def load_textset(n_jobs, use_gpu, pin_memory, asr_corpus, lm_corpus, audio, text):

    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_textset(tokenizer,**lm_corpus)
    collect_tr = partial(collect_text_batch,mode='train')
    collect_dv = partial(collect_text_batch,mode='dev')
    # Dataloader (Text data stored in RAM, no need num_workers)
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_tr,
                        num_workers=0, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=0, pin_memory=pin_memory)

    # Messages to show
    data_msg.append('I/O spec.  | Token type = {}\t| Vocab size = {}'\
                    .format(tokenizer.token_type,tokenizer.vocab_size))

    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg


def _data_msg(name,path,train_split,tr_set,dev_split,dv_set,batch_size,bucketing):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name,path))
    msg_list.append('           | Train sets = {}\t| Number of utts = {}'.format(train_split,tr_set))
    msg_list.append('           | Dev sets = {}\t| Number of utts = {}'.format(dev_split,dv_set))
    msg_list.append('           | Batch size = {}\t\t| Bucketing = {}'.format(batch_size,bucketing))
    return msg_list
