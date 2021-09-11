import math
import os
import random

eval_percent = 0.2

image_names = {}

for shape in os.listdir('images/to_sort'):
    image_names[shape] = os.listdir(f"images/to_sort/{shape}")

min_count = min([len(name) for name in image_names.values()])

for shape in image_names.keys():
    to_remove = random.sample(image_names[shape], len(image_names[shape]) - min_count)
    if not os.path.isdir(f"images/unused/{shape}"):
        os.mkdir(f"images/unused/{shape}")
    for name in to_remove:
        os.replace(f"images/to_sort/{shape}/{name}", f"images/unused/{shape}/{name}")
        image_names[shape].remove(name)

eval_count = math.ceil(min_count * eval_percent)
for shape in image_names.keys():
    eval_images = random.sample(image_names[shape], eval_count)
    if not os.path.isdir(f"images/evaluation/{shape}"):
        os.mkdir(f"images/evaluation/{shape}")
    if not os.path.isdir(f"images/training/{shape}"):
        os.mkdir(f"images/training/{shape}")

    for name in image_names[shape]:
        if name in eval_images:
            os.replace(f"images/to_sort/{shape}/{name}", f"images/evaluation/{shape}/{name}")
        else:
            os.replace(f"images/to_sort/{shape}/{name}", f"images/training/{shape}/{name}")
