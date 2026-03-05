import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_txt', type=str, required=True)
args = parser.parse_args()

classes = sorted(os.listdir(args.data_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

print("Clases encontradas:")
for cls, idx in class_to_idx.items():
    print(f"  {idx}: {cls}")

with open(args.output_txt, 'w') as f:
    for cls in classes:
        cls_dir = os.path.join(args.data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img in sorted(os.listdir(cls_dir)):
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                f.write(f'{cls}/{img} {class_to_idx[cls]}\n')

print(f"\nArchivo generado: {args.output_txt}")