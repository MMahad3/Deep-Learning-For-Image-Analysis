import os

# ✅ Use raw string or escaped path
main_dir = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008\ImageSets\Main"

def extract_ids(split="train"):
    image_ids = set()
    for file in os.listdir(main_dir):
        if file.endswith(f"_{split}.txt"):
            with open(os.path.join(main_dir, file)) as f:
                for line in f:
                    img_id, label = line.strip().split()
                    if int(label) == 1:
                        image_ids.add(img_id)
    return sorted(image_ids)

train_ids = extract_ids("train")
with open(os.path.join(main_dir, "train.txt"), "w") as f:
    for img_id in train_ids:
        f.write(img_id + "\n")

val_ids = extract_ids("val")
with open(os.path.join(main_dir, "val.txt"), "w") as f:
    for img_id in val_ids:
        f.write(img_id + "\n")

print(f"Saved train.txt ({len(train_ids)} images) and val.txt ({len(val_ids)} images) to: {main_dir}")

