import os
from data.dataset import TrainDataset, TrainDataset_Unsupervised
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)
    print(f'root_dir = {root_dir}')
    
    dataset_class = TrainDataset
    if args.unsupervised :
        dataset_class = TrainDataset_Unsupervised
    #if args.trigger_word == 'brain' :
    #    from data.dataset_brain import TrainDataset_Brain
    #    dataset_class = TrainDataset_Brain
    
    dataset = dataset_class(root_dir=root_dir,
                            anomaly_source_path=args.anomal_source_path,
                            anomal_position_source_path=args.anomal_position_source_path,
                            resize_shape=[512, 512],
                            tokenizer=tokenizer,
                            caption=args.trigger_word,
                            latent_res=args.latent_res,)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

