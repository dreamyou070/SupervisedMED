import os
from data.dataset import TrainDataset
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'{args.obj_name}/train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)

    dataset = TrainDataset(root_dir=root_dir,
                         anomaly_source_path=args.anomal_source_path,
                         resize_shape=[512, 512],
                         tokenizer=tokenizer,
                         caption=args.trigger_word,
                         use_perlin=True,
                         anomal_only_on_object=args.anomal_only_on_object,
                         anomal_training=True,
                         latent_res=args.latent_res,
                         do_anomal_sample=args.do_anomal_sample,
                         use_object_mask=args.do_object_detection, )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

