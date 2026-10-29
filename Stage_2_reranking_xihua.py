import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import time 
import torch
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from sample4geo.dataset.visloc import get_transforms
from hcnet.network import HCNet
from hcnet.utils.utils import *


def projected_point_geo_error(px, py, sat_name, query_gps, sat_gps_dict, img_size, meter_per_pixel):
    # GPS
    sat_lat, sat_lon = sat_gps_dict[sat_name]

    cx = (img_size - 1) / 2
    cy = (img_size - 1) / 2

    dx = (px - cx) * meter_per_pixel      
    dy = (py - cy) * meter_per_pixel       

    pred_lat = sat_lat - dy / 111320.0
    pred_lon = sat_lon + dx / (111320.0 * np.cos(np.radians(sat_lat)))

    return haversine_distance(query_gps[0], query_gps[1], pred_lat, pred_lon)

def normalize_name(x):
    return str(x).replace("\\", "/").split("/")[-1].strip()

def load_image(dir):
    image = cv2.imread(dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = val_transforms(image=image)['image'].to(device).unsqueeze(0) 

    return image

def eval_query(new_satellite_index_names, pos_sample, topk=10):

    ranked_names = [normalize_name(x) for x in new_satellite_index_names[:topk]]
    positive_names = set([normalize_name(x) for x in pos_sample])

    cmc = np.zeros(topk, dtype=np.int32)

    if len(positive_names) == 0:
        cmc[0] = -1
        return 0.0, cmc, None

    hit_positions = [
        idx for idx, name in enumerate(ranked_names)
        if name in positive_names
    ]

    if len(hit_positions) == 0:
        return 0.0, cmc, None

    first_rank = hit_positions[0]
    cmc[first_rank:] = 1

    ap = 0.0
    ngood = len(positive_names)

    for i, pos in enumerate(hit_positions):
        d_recall = 1.0 / ngood
        precision = (i + 1) / (pos + 1)

        if pos != 0:
            old_precision = i / pos
        else:
            old_precision = 1.0

        ap += d_recall * (old_precision + precision) / 2.0

    return ap, cmc, first_rank


def project_center_and_distance(H, src_shape, dst_shape, use_inverse=False):
    """
    H: [B, 3, 3]
    src_shape: [B, C, H, W]
    dst_shape: [B, C, H, W]
    """

    if H.dim() == 2:
        H = H.unsqueeze(0)

    if use_inverse:
        H = torch.linalg.inv(H)

    B, C, H_src, W_src = src_shape
    _, _, H_dst, W_dst = dst_shape

    cx_src = (W_src-1) / 2.0
    cy_src = (H_src-1) / 2.0

    center = torch.tensor(
        [cx_src, cy_src, 1.0],
        dtype=H.dtype,
        device=H.device
    ).view(1, 3, 1).repeat(B, 1, 1)

    proj = H @ center

    z = proj[:, 2, 0]

    px = proj[:, 0, 0] / (z + 1e-8)
    py = proj[:, 1, 0] / (z + 1e-8)

    cx_dst = (W_dst-1) / 2.0
    cy_dst = (H_dst-1) / 2.0

    dist = torch.sqrt((px - cx_dst) ** 2 + (py - cy_dst) ** 2)

    valid_z = torch.abs(z) > 1e-6
    valid_inside = (
        (px >= 0) & (px < W_dst) &
        (py >= 0) & (py < H_dst)
    )
    valid_finite = torch.isfinite(dist) & torch.isfinite(px) & torch.isfinite(py)

    valid = valid_z & valid_inside & valid_finite

    dist = torch.where(valid, dist, torch.full_like(dist, 1e6)) 

    return dist, px, py, valid

def haversine_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [float(lat1), float(lon1), float(lat2), float(lon2)]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6378137
    return R * c

def get_top1_geo_error(top1_sat_name, query_gps, sat_gps_dict):

    if top1_sat_name not in sat_gps_dict:
        return None

    query_lat, query_lon = float(query_gps[0]), float(query_gps[1])
    sat_lat, sat_lon = sat_gps_dict[top1_sat_name]

    return haversine_distance(query_lat, query_lon, sat_lat, sat_lon)

def get_sat_geo_error(sat_name1, sat_name2, sat_gps_dict):

    sat_lat1, sat_lon1 = sat_gps_dict[sat_name1]
    sat_lat2, sat_lon2 = sat_gps_dict[sat_name2]

    return haversine_distance(sat_lat1, sat_lon1, sat_lat2, sat_lon2)

def compute_flow_convergence(flow_predictions, gamma=1.1, last_n=8):

    errors = []

    for i in range(1, len(flow_predictions)):
        delta = flow_predictions[i] - flow_predictions[i-1]
        error = torch.norm(delta.reshape(delta.shape[0], -1), dim=1)
        errors.append(error)

    errors = torch.stack(errors, dim=0)
    errors = errors[-last_n:]

    weights = gamma ** torch.arange(errors.shape[0],device=errors.device,dtype=errors.dtype)
    weights = weights / weights.sum()
    mean_errors = (errors * weights[:, None]).sum(dim=0)

    return mean_errors

if __name__ == '__main__':

    # test | UAV-Visloc | weinan | xihua | adm
    data_name  = 'xihua' 
    DATA_LIST = [7, 8]  

    # 'BEMN-Tiny' | 'BEMN-Base'
    model_name = 'BEMN-Tiny'

    hcnet_config = './hcnet/config/visloc.json'
    drone_resolution = '_392' 
    img_size = 392
    iters_lev0 = 16
    ori_refine = True

    rerank_k    = 5      # Top-K re-ranking
    candidate_k = 100 
    topk_eval   = 100

    # "./checkpoints/XIHUA/GenHNR/GenHNR-T"
    # "./checkpoints/XIHUA/GenHNR/GenHNR-B"
    checkpoint_start = "./checkpoints/XIHUA/GenHNR/GenHNR-T"

    gpu_ids: tuple = (0,)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if data_name == 'xihua':
        orginal_resolution = [1959, 1663, 1663, 1959, 1663, 1803, 1959, 1959] 
        data_root = './data/Xihua_visloc/XIHUA-Visloc'
        satellite_resolution = '' 
        mode = '_test'

    val_transforms, _, _ = get_transforms((img_size, img_size))

    # Load model
    hcnet_config = json.load(open(hcnet_config,'r'))
    hcnet_config = EasyDict(hcnet_config)
    hcnet = HCNet(hcnet_config, model_name, data_root)
    hcnet = hcnet.to(device)

    hcnet_config.image_size = img_size
    hcnet_config.sat_size   = img_size

    # Load pretrained Checkpoint
    if checkpoint_start is not None:
        print("Start from:", checkpoint_start)
        model_state_dict = torch.load(checkpoint_start)
        hcnet.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(gpu_ids) > 1:
        hcnet = torch.nn.DataParallel(hcnet, device_ids=gpu_ids)

    hcnet.eval()

    FIRST_CMCS = []
    FIRST_APS = []
    FIRST_VALID_NUMS = []
    FIRST_GEO_MEANS = []
    FIRST_GEO_NUMS = []

    SECOND_CMCS = []
    SECOND_APS = []
    SECOND_VALID_NUMS = []
    SECOND_GEO_MEANS = []
    SECOND_GEO_NUMS = []

    ALL_FIRST_GEO_ERRORS = []
    ALL_SECOND_GEO_ERRORS = []

    for k in range(len(DATA_LIST)):

        print("\n{}Test {}-th scenario{}".format(20 * "-", DATA_LIST[k], 20 * "-"))

        meter_per_pixel = orginal_resolution[DATA_LIST[k] - 1] / 392 * 0.247

        query_folder = data_root + f'/{DATA_LIST[k]:02}/drone' + drone_resolution
        query_paths = [os.path.join(query_folder, f)
                for f in sorted(os.listdir(query_folder))
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    
        query_info_path = data_root + f'/{DATA_LIST[k]:02}/{DATA_LIST[k]:02}_c.csv'
        if not Path(query_info_path).exists():
            query_info_path = data_root + f'/{DATA_LIST[k]:02}/{DATA_LIST[k]:02}.csv'

        query_info_df = pd.read_csv(query_info_path)

        if data_name == 'xihua':
            pos_pair_path = f"./retrieval_results/Top_retrieval_{DATA_LIST[k]}.csv"

        print(f'Read pos_pair from {pos_pair_path}')
            
        pos_pairs_df = pd.read_csv(pos_pair_path, header=None)

        pos_sample_path = data_root + f'/{DATA_LIST[k]:02}/satellite' + mode + satellite_resolution + f'/pairs.csv' 
        pos_sample_df = pd.read_csv(pos_sample_path, header=None)

        sat_info_path = data_root + f'/{DATA_LIST[k]:02}/satellite' + mode + satellite_resolution  + f'/{DATA_LIST[k]:02}_coordinates.csv' 
        sat_info_df = pd.read_csv(sat_info_path, header=None)

        sat_info_df["sat_name"] = sat_info_df.iloc[:, 0]

        sat_gps_dict = {row["sat_name"]: (float(row.iloc[1]), float(row.iloc[2]))for _, row in sat_info_df.iterrows()}

        times = []
        first_CMC = np.zeros(topk_eval, dtype=np.float32)
        first_APs = []
        first_valid_num = 0
        second_CMC = np.zeros(topk_eval, dtype=np.float32)
        second_APs = []
        second_valid_num = 0
        first_top1_geo_errors = []
        second_top1_geo_errors = []

        with torch.no_grad():
            start_id = 0
            pbar = tqdm(query_paths[start_id:], desc="检索：")
            for i, query_path in enumerate(pbar, start=start_id):

                if 'adm' in data_name:
                    query_gps = None
                elif 'UAV' not in data_name:
                    query_gps = query_info_df[query_info_df.iloc[:, 1] == normalize_name(query_path)].values[0, 2:4]           
                else:
                    query_gps = query_info_df[query_info_df.iloc[:, 1] == normalize_name(query_path)].values[0, 3:5]

                # Stage 1 retrieval results
                satellite_index_names = pos_pairs_df.iloc[i, 1:candidate_k+1] # 
                rerank_candidates     = satellite_index_names[:rerank_k]
                remaining_candidates  = satellite_index_names[rerank_k:]

                pos_sample = pos_sample_df[pos_sample_df.iloc[:, 0] == normalize_name(query_path)].values[0, 1:]
                pos_sample = [str(x).strip() for x in pos_sample if pd.notna(x) and str(x).strip() not in ["-1", ""]]

                query_image_single = load_image(query_path)
                pos_image = load_image(data_root + f'/{DATA_LIST[k]:02}/satellite' + mode + satellite_resolution + '/' + pos_sample[0])

                satellite_images = [pos_image]     
                valid_sat_names  = [pos_sample[0]] 
                start_id = 1

                query_name = normalize_name(query_path)

                for satellite_index_name in rerank_candidates:
                    satellite_path = data_root + f'/{DATA_LIST[k]:02}/satellite' + mode + satellite_resolution + '/' + satellite_index_name
                    satellite_image = load_image(satellite_path)
                    satellite_images.append(satellite_image)
                    valid_sat_names.append(satellite_index_name)

                if len(valid_sat_names) == 0:
                    reranked_top_names = []
                else:
                    # Homography estimation
                    satellite_images = torch.cat(satellite_images, dim=0)
                    query_images = query_image_single.repeat(len(valid_sat_names), 1, 1, 1)

                    four_pr, corr_map, delta_theta = hcnet(query_images, satellite_images, sat_gps=[], iters_lev0=iters_lev0, refine_mode=True, refine=ori_refine)

                    if isinstance(four_pr, (list, tuple)):
                        four_point = four_pr[-1]
                        mean_error = compute_flow_convergence(four_pr, gamma=1.1, last_n=8)

                        corr_map  = corr_map.squeeze(3)
                        B, H1, W1, H2, W2 = corr_map.shape
                        corr = corr_map.reshape(B, H1 * W1, H2 * W2)
                        diag_corr = torch.diagonal(corr, dim1=1, dim2=2)
                        score = diag_corr.sum(dim=1)
                    else:
                        four_point = four_pr

                    H = get_homograpy(four_point, query_images.shape)
                    dist, px, py, valid = project_center_and_distance(H, src_shape=query_images.shape, dst_shape=satellite_images.shape, use_inverse=False)

                    mean_error = mean_error[start_id:]
                    score = score[start_id:]
                    dist  = dist[start_id:]
                    
                    order = torch.argsort(dist).detach().cpu().numpy() 

                    # Dynamic re-ranking
                    selected_idx = None

                    best_idx = 0
                    Tcb = 25
                    Tcn = 20

                    for idx in order:

                        if idx == 0:
                            continue

                        # Sw
                        Sw = score[best_idx] / score[idx]  
                        Twb   = 1  + (0.5 - 1) * (min(dist[idx], 50)/50) 
                        Twn = 1  + (  2 - 1) * (min(dist[idx], 50)/50) 

                        candidate_reliable = mean_error[idx] < Tcn and Sw < Twn  

                        if not candidate_reliable:
                            continue
    
                        base_unreliable = mean_error[best_idx] > Tcb or Sw < Twb 

                        if base_unreliable:
                            selected_idx = idx
                            best_idx = idx
                        else:
                            better_error = mean_error[idx] < mean_error[best_idx] 
                            better_dist = dist[idx] < dist[best_idx]
                            if better_dist and better_error:  
                                selected_idx = idx
                                best_idx = idx

                    if selected_idx is not None:
                        remaining_order = [idx for idx in order if idx != selected_idx]
                        reranked_top_names = ([valid_sat_names[start_id:][selected_idx]] + [valid_sat_names[start_id:][idx] for idx in remaining_order])
                    else:
                        reranked_top_names = list(rerank_candidates)

                matched = pos_sample_df[pos_sample_df.iloc[:, 0] == query_name]

                if len(matched) == 0:
                    print("No positive sample found:", query_name)
                    continue

                pos_sample_raw = matched.values[0, 1:]

                pos_sample = [
                    str(x).strip()
                    for x in pos_sample_raw
                    if pd.notna(x) and str(x).strip() not in ["-1", ""]
                ]

                if len(pos_sample) == 0:
                    print("Positive samples are empty:", query_name)
                    continue

                # ----------------------------- Stage-1 results -----------------------------
                first_satellite_index_names = satellite_index_names 

                ap, cmc, rank = eval_query(
                    first_satellite_index_names,
                    pos_sample,
                    topk=topk_eval
                )

                if cmc[0] == -1:
                    continue

                first_CMC += cmc
                first_APs.append(ap)
                first_valid_num += 1

                first_top1_name = first_satellite_index_names.iloc[0]

                if query_gps is not None:
                    first_top1_error = get_top1_geo_error(
                        first_top1_name,
                        query_gps,
                        sat_gps_dict
                    )
                else:
                    first_top1_error = None

                if first_top1_error is not None:
                    first_top1_geo_errors.append(first_top1_error)
                
                # ----------------------------- Stage-2 results-----------------------------
                second_satellite_index_names = reranked_top_names + list(remaining_candidates)

                ap, cmc, rank = eval_query(
                    second_satellite_index_names,
                    pos_sample,
                    topk=topk_eval
                )

                if cmc[0] == -1:
                    continue

                second_CMC += cmc
                second_APs.append(ap)
                second_valid_num += 1

                second_top1_name = second_satellite_index_names[0]

                if selected_idx is not None:
                    b = selected_idx + start_id
                else:
                    b = start_id

                if valid[b] and query_gps is not None:
                    second_top1_error = projected_point_geo_error(
                        px[b].item(),
                        py[b].item(),
                        second_top1_name,
                        query_gps,
                        sat_gps_dict,
                        img_size=392,
                        meter_per_pixel=meter_per_pixel
                    )
                else:
                    second_top1_error = None

                if second_top1_error is not None:
                    second_top1_geo_errors.append(second_top1_error)

            first_CMC = first_CMC / first_valid_num
            first_mAP = np.mean(first_APs)
            first_top1_geo_mean = np.mean(first_top1_geo_errors) if len(first_top1_geo_errors) > 0 else np.nan
            first_top1_geo_median = np.median(first_top1_geo_errors) if len(first_top1_geo_errors) > 0 else np.nan

            second_CMC = second_CMC / second_valid_num
            second_mAP = np.mean(second_APs)
            second_top1_geo_mean = np.mean(second_top1_geo_errors) if len(second_top1_geo_errors) > 0 else np.nan
            second_top1_geo_median = np.median(second_top1_geo_errors) if len(second_top1_geo_errors) > 0 else np.nan

            # first stage
            FIRST_CMCS.append(first_CMC)
            FIRST_APS.append(first_mAP)
            FIRST_VALID_NUMS.append(first_valid_num)

            if len(first_top1_geo_errors) > 0:
                FIRST_GEO_MEANS.append(first_top1_geo_mean)
                FIRST_GEO_NUMS.append(len(first_top1_geo_errors))
                ALL_FIRST_GEO_ERRORS.extend(first_top1_geo_errors)


            # second stage
            SECOND_CMCS.append(second_CMC)
            SECOND_APS.append(second_mAP)
            SECOND_VALID_NUMS.append(second_valid_num)

            if len(second_top1_geo_errors) > 0:
                SECOND_GEO_MEANS.append(second_top1_geo_mean)
                SECOND_GEO_NUMS.append(len(second_top1_geo_errors))
                ALL_SECOND_GEO_ERRORS.extend(second_top1_geo_errors)

        print("The first  stage: Recall@1: {:.4f} - Recall@2: {:.4f} - Recall@5: {:.4f} - AP: {:.4f}".format(first_CMC[0] * 100, first_CMC[1] * 100, first_CMC[4] * 100, first_mAP * 100))
        print("The first  stage: Dis@1   : {:.4f} m".format(first_top1_geo_mean))
        print("The first  stage: Dis@1_median : {:.4f} m".format(first_top1_geo_median))

        print("The second stage: Recall@1: {:.4f} - Recall@2: {:.4f} - Recall@5: {:.4f} - AP: {:.4f}".format(second_CMC[0] * 100, second_CMC[1] * 100, second_CMC[4] * 100, second_mAP * 100))
        print("The second stage: Dis@1   : {:.4f} m".format(second_top1_geo_mean))
        print("The second stage: Dis@1_median : {:.4f} m".format(second_top1_geo_median))

    # Average results
    FIRST_CMCS = np.array(FIRST_CMCS)
    FIRST_APS = np.array(FIRST_APS)
    FIRST_VALID_NUMS = np.array(FIRST_VALID_NUMS)

    SECOND_CMCS = np.array(SECOND_CMCS)
    SECOND_APS = np.array(SECOND_APS)
    SECOND_VALID_NUMS = np.array(SECOND_VALID_NUMS)

    first_weights = FIRST_VALID_NUMS / np.sum(FIRST_VALID_NUMS)
    second_weights = SECOND_VALID_NUMS / np.sum(SECOND_VALID_NUMS)

    first_CMC = np.average(FIRST_CMCS, axis=0, weights=first_weights)

    first_mAP = np.average(FIRST_APS, weights=first_weights)

    second_CMC = np.average(SECOND_CMCS, axis=0, weights=second_weights)

    second_mAP = np.average(SECOND_APS, weights=second_weights)

    if len(FIRST_GEO_MEANS) > 0:
        first_top1_geo_mean = np.average(FIRST_GEO_MEANS, weights=FIRST_GEO_NUMS)
    else:
        first_top1_geo_mean = np.nan

    if len(SECOND_GEO_MEANS) > 0:
        second_top1_geo_mean = np.average(SECOND_GEO_MEANS, weights=SECOND_GEO_NUMS)
    else:
        second_top1_geo_mean = np.nan

    first_top1_geo_median = (np.median(ALL_FIRST_GEO_ERRORS)if len(ALL_FIRST_GEO_ERRORS) > 0 else np.nan)

    second_top1_geo_median = (np.median(ALL_SECOND_GEO_ERRORS)if len(ALL_SECOND_GEO_ERRORS) > 0 else np.nan)

    print("\n" + "=" * 100)

    print("The first  stage average: Recall@1: {:.4f} - Recall@2: {:.4f} - Recall@5: {:.4f} - AP: {:.4f}".format(first_CMC[0] * 100, first_CMC[1] * 100, first_CMC[4] * 100, first_mAP * 100))

    print("The first  stage average: Dis@1        : {:.4f} m".format(first_top1_geo_mean))

    print("The first  stage average: Dis@1_median : {:.4f} m".format(first_top1_geo_median))


    print("The second stage average: Recall@1: {:.4f} - Recall@2: {:.4f} - Recall@5: {:.4f} - AP: {:.4f}".format(second_CMC[0] * 100,second_CMC[1] * 100,second_CMC[4] * 100,second_mAP * 100))

    print("The second stage average: Dis@1        : {:.4f} m".format(second_top1_geo_mean))

    print("The second stage average: Dis@1_median : {:.4f} m".format(second_top1_geo_median))
