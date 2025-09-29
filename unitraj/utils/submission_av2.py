import time
from pathlib import Path

import torch
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
from torch import Tensor


class SubmissionAv2:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.submission_file = Path(save_dir) / f"single_agent_{stamp}.parquet"
        self.challenge_submission = ChallengeSubmission(predictions={})

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        """
        trajectory: (B, M, 60, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        """
        scenario_ids = data["scenario_id"]
        track_ids = data["track_id"]
        batch = len(track_ids)

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        if inference:
            return global_trajectory, probability

        for i, (scene_id, track_id) in enumerate(zip(scenario_ids, track_ids)):
            self.challenge_submission.predictions[scene_id] = {
                track_id: (global_trajectory[i], probability[i])
            }

    def generate_submission_file(self):
        print("generating submission file for argoverse 2 motion forecasting challenge")
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.submission_file}")
