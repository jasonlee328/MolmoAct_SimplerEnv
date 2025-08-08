from typing import Optional, Sequence

import os
import re
import copy
import warnings
import ast
from datetime import datetime
import logging
import json
import random
import math
from collections import defaultdict
from io import BytesIO
import base64
import faulthandler
import xml.etree.ElementTree as ET


import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw, ImageFile, ImageOps
import requests
from tqdm import tqdm
from transformers import Qwen2Tokenizer, AutoProcessor
from transforms3d.euler import euler2axangle
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from olmo.vllm.molmo2_open.molmo2 import Molmo2ForConditionalGeneration
from olmo.util import prepare_cli_environment


from simpler_env.policies.molmoact.action_tokenize import *





def apply_chat_template(processor: AutoProcessor, text: str):
    messages = [
        {
            "role": "user",
            "content": [dict(type="text", text=text)]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt


class MolmoActInference:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "google_robot",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [256, 256],
        action_scale: float = 1.0,
        initial_confidence_threshold: float = 0.95,  # initial threshold for token confidence
        threshold_adjustment_factor: float = 0.1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        device = "cuda:0"
        self.processor = AutoProcessor.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            padding_side="left",
        )

        self.sampling_params = SamplingParams(
            max_tokens=448,
            temperature=0
        )

        self.model = LLM(
            model=saved_model_path,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )


        self.image_size = [256, 256]
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0


        self.step_to_point = defaultdict(list)
        self.timestep = 0
        stats_path = './simpler_env/policies/llava/dataset_statistics.json'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
        with open(stats_path, 'r') as f:
            self.dataset_stats = json.load(f)
        self.token_confidence_threshold = initial_confidence_threshold
        self.threshold_adjustment_factor = threshold_adjustment_factor

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.step_to_point = defaultdict(list)
        self.timestep = 0



    def scale_pt(pt, w, h):
        """
        Convert a point whose coordinates are in 0–255 space
        to image-pixel space (0‥w-1, 0‥h-1).
        """
        x, y = pt
        return (int(round(x / 255.0 * (w - 1))),
                int(round(y / 255.0 * (h - 1))))
                    
                    
                    
    def unnormalize_action_tokenized(self, generated_text):
        match = re.search(r"the action that the robot should take is\s*(\[[^\]]+\])", generated_text, re.IGNORECASE)
        if match:
            action_list_str = match.group(1)
        else:
            match = re.search(r"\[[^\]]+\]", generated_text)
            if match:
                action_list_str = match.group(0)
            else:
                raise ValueError("No action list found in the generated text.")

        token_list = action_list_str.strip("[]").split(",")
        token_list = [token.strip().strip('"').strip("'") for token in token_list]
        base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        token_ids = base_tokenizer.convert_tokens_to_ids(token_list)
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
        stats = self.dataset_stats[self.unnorm_key]["action"]
        action_low = np.array(stats["q01"])
        action_high = np.array(stats["q99"])
        mask = np.array(stats.get("mask", [True] * len(action_low)))
        unnormalized_action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions
        )
        return unnormalized_action

    
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
    
        assert image.dtype == np.uint8
        orig_h, orig_w = image.shape[:2]
        image = self._resize_image(image)
        new_h, new_w = image.shape[:2]
        img = Image.fromarray(image)
        language_instruction = self.task_description
        system_prompt = f"The task is {language_instruction}. What is the action that the robot should take. To figure out the action that the robot should take to {language_instruction}, let's think through it step by step. First, what is the depth map for this image? Second, what is the trajectory of the end effector? Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"

        inputs = [
            {
                "prompt": apply_chat_template(self.processor, system_prompt),
                "multi_modal_data": {
                    "image": [image]
                },
            },
        ]

        output = self.model.generate(inputs, sampling_params=self.sampling_params)

        generated_text = output[0].outputs[0].text
        annotated_image = image.copy()

        trajectory = None
        unnormalized_action = self.unnormalize_action_tokenized(generated_text)
        if "The trajectory of the end effector is" in generated_text:
            try:
                traj_part = generated_text.split("The trajectory of the end effector is")[-1]
                traj_part = traj_part.split("Based on")[0].strip()
                traj_str = traj_part.rstrip('.').strip()
                trajectory = ast.literal_eval(traj_str)
                traj_digits = []
                for num in trajectory:
                    for digit in str(num):
                        traj_digits.append(digit)

                for i in range(len(trajectory) - 1):
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i + 1]))
                    pt1 = scale_pt(pt1, new_w, new_h)
                    pt2 = scale_pt(pt2, new_w, new_h)
                    cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv.LINE_AA)
 
            except Exception as e:
                print("Failed to parse trajectory:", e)
        else:
            print("No trajectory found in generated text.")
            
            
        (h, w) = annotated_image.shape[:2]
        raw_action = {
            "world_vector": unnormalized_action[:3],
            "rotation_delta": unnormalized_action[3:6],
            "open_gripper": unnormalized_action[6:7],  # assuming the last value is gripper action
        }
        annotated_image = cv.resize(annotated_image, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
    
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action
    
            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
    
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
    
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
    
            action["gripper"] = relative_gripper_action
    
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
    
        action["terminate_episode"] = np.array([0.0])
    
        return raw_action, action, annotated_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
    
   
        
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        # images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
    
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
    
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
    
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
    
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)












