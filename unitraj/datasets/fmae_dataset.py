import numpy as np
from .base_dataset import BaseDataset
import torch


class FMAEDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)

    def convert_to_fmae_format(self, input):
        B = 32 #batch size
        N_agent = self.config["max_num_agents"]#10 #maximum number of agents per scenari --> get from config file
        N_lane = self.config["max_num_roads"] #80 #maximum number of lane segments per scenario --> get from config file
        h_steps = self.config["past_len"]#50 #num historical timesteps for agent tracks --> get from config file
        f_steps = self.config["future_len"] #60 #num future timesteps for agent tracks --> get from config file
        lane_sampling_pts = self.config["num_points_each_polyline"] #TODO:not sure if this is the right value #20 #sampling points per lane segment --> get from config file
        data = {
            "x": torch.rand(B, N_agent, h_steps, 2), #agent tracks as local differences from origin being the last position of the past trajectory
            "x_attr": torch.zeros((B, N_agent, 3), dtype=torch.uint8), #categorical agent attributes
            "x_positions": torch.rand(B, N_agent, h_steps, 2), #agent tracks in scene coordinates
            "x_centers": torch.rand(B, N_agent, 2), #center of agent track
            "x_angles": torch.rand(B, N_agent, h_steps+f_steps), #agent headings
            "x_velocity": torch.rand(B, N_agent, h_steps+f_steps), #velocity of agents as absolute values
            "x_velocity_diff": torch.rand(B, N_agent, h_steps), #velocity changes of agents
            "lane_positions": torch.rand(B, N_lane, lane_sampling_pts, 2), #lane segments in scene coordinates
            "lane_centers": torch.rand(B, N_lane, 2), #center of lane segments
            "lane_angles": torch.rand(B, N_lane), #orientation of lane segments
            "lane_attr": torch.rand(B, N_lane, 3), #categorial lane attributes
            "is_intersections": torch.rand(B, N_lane), # categorical lane attribute
            "y": torch.rand(B, N_agent, f_steps, 2), #agent future tracks as x,y positions -> ground truth in unitraj format
            "x_padding_mask": torch.zeros((B, N_agent, h_steps+f_steps), dtype=torch.bool), #padding mask for agent tracks
            "lane_padding_mask": torch.zeros((B, N_lane, lane_sampling_pts), dtype=torch.bool), #padding mask for lane segment points
            "x_key_padding_mask": torch.zeros((B, N_agent), dtype=torch.bool), #batch padding mask for agent tracks
            "lane_key_padding_mask": torch.zeros((B, N_lane), dtype=torch.bool), #batch padding mask for lane segments
            "num_actors": torch.full((B,), fill_value=N_agent, dtype=torch.int64),
            "num_lanes": torch.full((B,), fill_value=N_lane, dtype=torch.int64),
            "scenario_id": [] * B,
            "track_id": [] * B,
            "origin": torch.rand(B, 2), #scene to global coordinates position
            "theta": torch.rand(B), #scene to global coordinates orientation
        }

        data["origin"] = input["center_objects_world"][0:2]#input["obj_trajs_pos"][..., h_steps-1, 0:2]
        data["theta"] = np.asin(input["center_objects_world"][35])
        rotate_mat = torch.tensor(
            [
                [torch.cos(data["theta"]), -torch.sin(data["theta"])],
                [torch.sin(data["theta"]), torch.cos(data["theta"])],
            ],
        )
        data["x"] = input["obj_trajs_pos"][..., 0:2]#torch.matmul(input["obj_trajs_pos"][..., 0:2] - data["origin"], rotate_mat)
        #data["x_attr"]
        data["x_positions"] = input["obj_trajs"][..., 0:2]
        #data["x_centers"]
        data["x_velocity"] = np.linalg.norm(input["obj_trajs"][..., 37:39])
        #data["x_velocity_diff"] look at implementation in model
        #data["lane_positions"]
        #data["lane_centers"]
        #data["lane_angles"]
        #data["lane_attr"]
        #data["is_intersections"]
        #data["y"]
        data["x_padding_mask"] = input["obj_trajs_mask"]
        data["lane_padding_mask"] = input["map_polylines_mask"]
        #data["x_key_padding_mask"]
        #data["lane_key_padding_mask"]
        #data["num_actors"] -> already set
        #data["num_lanes"] -> already set
        data["scenario_id"] = input["scenario_id"]
        #data["track_id"] information not present in input



        return data
