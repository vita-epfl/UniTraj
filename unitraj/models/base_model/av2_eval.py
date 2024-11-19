from av2.datasets.motion_forecasting.eval.metrics import (
    compute_ade,
    compute_fde,
    compute_brier_fde,
    compute_brier_ade,
    compute_is_missed_prediction,
    compute_world_ade,
    compute_world_fde,
    compute_world_brier_fde,
    compute_world_misses,
    compute_world_collisions
)

from typing import Final
import numpy as np


def transform_preds_to_argoverse_format(pred_dicts, top_k_for_eval=-1, eval_second=6):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    
    scene2preds = {}
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)
    
    num_frame_to_eval = 60
    
    batch_pred_trajs = np.zeros((num_scenario, topK, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, topK))
    gt_trajs = np.zeros((num_scenario, num_frame_to_eval, 7))
    gt_is_valid = np.zeros((num_scenario, num_frame_to_eval), dtype=int)
    object_type = np.zeros((num_scenario), dtype=object)
    object_id = np.zeros((num_scenario), dtype=int)
    scenario_id = np.zeros((num_scenario), dtype=object)
    
    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        cur_pred = preds_per_scene[0]
        sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
        cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
        cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]

        cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()

        batch_pred_trajs[scene_idx] = cur_pred['pred_trajs'][:topK, :, :]
        batch_pred_scores[scene_idx] = cur_pred['pred_scores'][:topK]
        gt_trajs[scene_idx] = cur_pred['gt_trajs'][-num_frame_to_eval:, [0, 1, 3, 4, 6, 7,
                                                                                8]]
        gt_is_valid[scene_idx] = cur_pred['gt_trajs'][-num_frame_to_eval:, -1]
        object_type[scene_idx] = cur_pred['object_type']
        object_id[scene_idx] = cur_pred['object_id']
            
        gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos


def argoverse2_evaluation(pred_dicts, top_k_for_eval=-1, eval_second=6, num_modes_for_eval=6):
    batch_pred_scores, batch_pred_trajs, gt_infos = transform_preds_to_argoverse_format(
        pred_dicts, top_k_for_eval=top_k_for_eval, eval_second=eval_second,
    )
    num_scenario = batch_pred_trajs.shape[0]

    ade_results = np.zeros((num_scenario, num_modes_for_eval))
    fde_results = np.zeros((num_scenario, num_modes_for_eval))
    miss_rate_results = np.zeros((num_scenario, num_modes_for_eval))
    brier_fde_results = np.zeros((num_scenario, num_modes_for_eval))
    brier_ade_results = np.zeros((num_scenario, num_modes_for_eval))
    
    gt_trajs = gt_infos['gt_trajectory']
    gt_is_valid = gt_infos['gt_is_valid']

    for scene_idx, pred_traj in enumerate(batch_pred_trajs):
        gt_traj_all = gt_trajs[scene_idx][..., :2]
        valid_mask = gt_is_valid[scene_idx]
        gt_traj = gt_traj_all[valid_mask == 1]
        pred_traj = pred_traj[:, valid_mask == 1]
        pred_score = batch_pred_scores[scene_idx]
        
        ade = compute_ade(pred_traj, gt_traj)
        fde = compute_fde(pred_traj, gt_traj)
        is_missed = compute_is_missed_prediction(pred_traj, gt_traj, miss_threshold_m=2.0)
        brier_fde = compute_brier_fde(pred_traj, gt_traj, pred_score)
        brier_ade = compute_brier_ade(pred_traj, gt_traj, pred_score)

        ade_results[scene_idx] = ade
        fde_results[scene_idx] = fde
        miss_rate_results[scene_idx] = is_missed
        brier_fde_results[scene_idx] = brier_fde
        brier_ade_results[scene_idx] = brier_ade

    rows = np.arange(num_scenario)
    min_ade_idx = np.argmin(ade_results, axis=-1)
    min_fde_idx = np.argmin(fde_results, axis=-1)
    min_ade = np.mean(ade_results[rows, min_ade_idx])
    min_fde = np.mean(fde_results[rows, min_fde_idx])
    brier_min_ade = np.mean(brier_ade_results[rows, min_ade_idx])
    brier_min_fde = np.mean(brier_fde_results[rows, min_fde_idx])
    miss_rate = np.mean(miss_rate_results[rows, min_fde_idx])

    result_dict = {
        "min_ADE": min_ade,
        "min_FDE": min_fde,
        "brier_min_ADE": brier_min_ade,
        "brier_min_FDE": brier_min_fde,
        "miss_rate": miss_rate,
    }
    
    
    return result_dict
