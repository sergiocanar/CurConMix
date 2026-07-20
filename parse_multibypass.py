"""
Converts MultiBypassT40's COCO-style JSON annotations (label_files_challenge/*.json)
into a CholecT45.csv-shaped dataframe, so the rest of the pipeline (preprocess.get_folds,
dataset.py, train.py, ...) can read it unmodified.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def convert(data_root, frames_dir, out_csv):
    label_dir = os.path.join(data_root, "label_files_challenge")
    frames_root = os.path.join(data_root, frames_dir)
    video_jsons = sorted(f for f in os.listdir(label_dir) if f.endswith(".json"))

    n_triplet = n_instrument = n_verb = n_target = None
    video_dfs = []

    for video_json in video_jsons:
        video = video_json[:-len(".json")]
        with open(os.path.join(label_dir, video_json)) as f:
            data = json.load(f)

        cats = data["categories"]
        counts = (len(cats["triplet"]), len(cats["instrument"]), len(cats["verb"]), len(cats["target"]))
        if n_triplet is None:
            n_triplet, n_instrument, n_verb, n_target = counts
        else:
            assert counts == (n_triplet, n_instrument, n_verb, n_target), (
                f"{video}: category counts {counts} don't match {(n_triplet, n_instrument, n_verb, n_target)} "
                "seen in earlier videos"
            )

        images = data["images"]
        n_images = len(images)
        ids = [img["id"] for img in images]
        assert ids == list(range(n_images)), f"{video}: image ids are not contiguous 0..{n_images - 1}"

        for img_id in ids:
            frame_path = os.path.join(frames_root, video, f"{img_id:06d}.jpg")
            assert os.path.exists(frame_path), f"Missing frame: {frame_path}"

        triplet = np.zeros((n_images, n_triplet), dtype=int)
        instrument = np.zeros((n_images, n_instrument), dtype=int)
        verb = np.zeros((n_images, n_verb), dtype=int)
        target = np.zeros((n_images, n_target), dtype=int)

        for ann in data["annotations"]:
            idx = ann["image_id"]
            triplet[idx, ann["category_id"]] = 1
            instrument[idx, ann["instrument_id"]] = 1
            verb[idx, ann["verb_id"]] = 1
            target[idx, ann["target_id"]] = 1

        metadata = pd.DataFrame({
            "folder": ids,
            "frame": ids,
            "video": video,
            "image_path": [f"{video}/{img_id:06d}.jpg" for img_id in ids],
            "image_id": [f"{video}_{img_id}" for img_id in ids],
        })

        video_df = pd.concat(
            [
                metadata,
                pd.DataFrame(triplet).add_prefix("tri"),
                pd.DataFrame(instrument).add_prefix("inst"),
                pd.DataFrame(verb).add_prefix("v"),
                pd.DataFrame(target).add_prefix("t"),
            ],
            axis=1,
        )
        video_dfs.append(video_df)
        print(f"{video}: {n_images} frames, {len(data['annotations'])} annotations")

    final_df = pd.concat(video_dfs, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(final_df)} rows ({n_triplet} triplets, {n_instrument} instruments, "
          f"{n_verb} verbs, {n_target} targets) to {out_csv}")
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/multibypasst40_challenge_trainval")
    parser.add_argument("--frames-dir", default="videos_cutmargin512")
    parser.add_argument("--out-csv", default="data/multibypasst40_challenge_trainval/dataframes/MultiBypassT40.csv")
    args = parser.parse_args()
    convert(args.data_root, args.frames_dir, args.out_csv)
