# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from sparseml.pytorch.optim import (
    MagnitudePruningParamsScorer,
    MovementPruningParamsScorer,
    create_pruning_param_scorer,
)


def _make_fake_params(n, shape):
    return [torch.nn.Parameter(torch.randn(*shape)) for _ in range(n)]


def _fake_params_random_update(params):
    for param in params:
        shape = param.data.shape
        param.data = torch.randn(*shape)
        param.grad = torch.randn(*shape)


@pytest.mark.parametrize(
    "score_type,n_updates",
    [
        ("magnitude", 0),
        ("magnitude", 1),
        ("movement", 5),
    ],
)
def test_pruning_scorer(score_type, n_updates):
    params = _make_fake_params(8, (24, 24))
    scorer = create_pruning_param_scorer(params, score_type)

    for i in range(n_updates):
        _fake_params_random_update(params)
        fake_masks = [(param != 0).type(param.dtype) for param in params]
        scorer.pre_optim_step_update(fake_masks)
    scores = scorer.score_parameters()
    assert len(scores) == len(params)

    # simulate mask update
    fake_masks = [torch.ones_like(p) for p in params]
    fake_mask_diffs = [-1.0 * torch.ones_like(p) for p in params]
    scorer.mask_update(fake_masks, fake_mask_diffs)

    for param, score in zip(params, scores):
        assert param is not None
        assert score is not None

        assert param.dtype == score.dtype
        assert param.data.shape == score.shape


@pytest.mark.parametrize(
    "expected_class,score_type",
    [
        (MagnitudePruningParamsScorer, "magnitude"),
        (MovementPruningParamsScorer, "movement"),
    ],
)
def test_create_pruning_param_scorer(expected_class, score_type):
    fake_params = _make_fake_params(5, (25, 25))
    scorer = create_pruning_param_scorer(fake_params, score_type)
    assert isinstance(scorer, expected_class)


def test_score_parameters_handles_list_scores(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "new_group", lambda backend=None, timeout=None: None)

    def fake_gather(tensor, gather_list=None, group=None, dst=0):
        if gather_list is not None:
            gather_list[0] = tensor

    monkeypatch.setattr(torch.distributed, "gather", fake_gather)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda val, src=0, group=None: None)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group=None: None)

    params = _make_fake_params(2, (2, 2))
    scorer = MovementPruningParamsScorer(params)
    scorer._movement_scores = [[0.0 for _ in range(p.numel())] for p in params]

    scores = scorer.score_parameters()
    assert len(scores) == len(params)
    assert all(isinstance(s, torch.Tensor) for s in scores)
