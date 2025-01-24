import os
import json
import random
import argparse
from tqdm import tqdm
from datasets import load_from_disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mme_path', required=True, help='config_path')
    parser.add_argument('--num_samples',
                        required=True,
                        type=int,
                        help='output samples number')
    parser.add_argument('--ordered',
                        action="store_true",
                        default=False,
                        help='use ordered sample or random sample')
    args = parser.parse_args()
    if not os.path.exists(args.mme_path):
        raise RuntimeError(f"{args.mme_path} is not exist !!!")
    mme_data = load_from_disk(args.mme_path)
    images_dir = os.path.join(args.mme_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    num_datas = len(mme_data)
    assert (num_datas >= args.num_samples)
    select_idxs = []
    if not args.ordered:
        select_idxs = random.sample(range(0, num_datas), args.num_samples)
    else:
        select_idxs = range(0, args.num_samples)
    data = []
    with tqdm(total=args.num_samples) as pbar:
        for idx in select_idxs:
            image = mme_data['image'][idx]
            image_path = os.path.join(images_dir, f"{idx}.png")
            image.save(image_path)
            data.append({
                "image": f"images/{idx}.png",
                "question": mme_data['question'][idx],
                "answer": mme_data['answer'][idx]
            })
            pbar.update(1)
    json_file = os.path.join(args.mme_path, "samples.json")
    with open(json_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
