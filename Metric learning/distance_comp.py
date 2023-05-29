import json
from itertools import islice
import Siamese_comp_contrastive
import torch
import numpy as np
import dataset
def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def distance_estimator(query_set, gallery_set, model, device, N):
    model.eval()
    result = dict()
    for img1, img1_name in query_set:
        tmp = dict()
        for img2, img2_name in gallery_set:
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dissim = torch.nn.functional.pairwise_distance(out1, out2)
            tmp[img2_name] = np.round(dissim.item(), 6)
        tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=False)}  # sort tmp by values
        print("--------------------------------------------------------------------")
        print(img1_name)
        print(tmp)
        result[img1_name] = take(N, tmp.keys())
    return result
def main():
    pass
    # img1 = load_image(img1).to(device)
    # img2 = load_image(img2).to(device)
    # img3 = load_image(img3).to(device)
    # out1, out2 = model(img1, img2)
    # dissim = torch.nn.functional.pairwise_distance(out1, out2)
    # print(f"output of same {np.round(dissim.item(), 6)}")
    #
    # out1, out2 = model(img2, img3)
    # dissim = torch.nn.functional.pairwise_distance(out1, out2)
    # print(f"output of different {np.round(dissim.item(), 6)}")

    # --------------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Siamese_comp_contrastive.SiameseNetwork()

    model_path = '/Users/munkhdelger/PycharmProjects/ML_competition/checkpoints/Siamese_best__0.8434__2023-05-29__18:30:55.pth'
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    model = model.to(device)

    query_set, gallery_set = dataset.get_query_and_gallery()
    result = distance_estimator(query_set, gallery_set, model, device, N=10)
    # print(result)
    print("###############################################################")
    for i, j in result.items():
        print(i, j)

    query_random_guess = dict()
    query_random_guess['groupname'] = "Capybara"
    query_random_guess["images"] = result
    with open('/Users/munkhdelger/PycharmProjects/ML_competition/Metric learning/data.json', 'w') as f:
        json.dump(query_random_guess, f)


if __name__ == '__main__':
    main()
