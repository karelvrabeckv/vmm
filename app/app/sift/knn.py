import numpy as np

# Find matches in the whole visual dictionary
def get_matches(img1_vecs_index, img1_vecs, img2_vecs_index, img2_vecs):
    matches = []

    # Iterate both descriptors simultaneously
    for img1_part_vecs_index, img2_part_vecs_index in zip(img1_vecs_index, img2_vecs_index):
        if len(img1_part_vecs_index) == 0:
            continue

        for img1_vec_index in img1_part_vecs_index:
#             distances = [np.linalg.norm(np.array(img1_vecs[img1_vec_index]) - np.array(img2_vecs[img2_vec_index])) for img2_vec_index in img2_part_vecs_index]
            distances = np.linalg.norm(img1_vecs[img1_vec_index] - np.take(img2_vecs, img2_part_vecs_index, axis=0), axis = 1)

            if len(distances) > 0:
              matches.append((img1_vec_index, img2_part_vecs_index[np.argmin(distances)]))

    return matches
