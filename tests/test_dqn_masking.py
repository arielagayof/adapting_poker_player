import torch

from poker_adapt.agents.dqn.dqn_agent import legal_action_mask, masked_argmax, masked_max


def test_legal_action_mask_marks_only_legal_indices():
    mask = legal_action_mask(action_dim=6, legal_action_ids=[0, 2, 5, 99, -1])
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_masked_argmax_ignores_illegal_actions():
    q_values = torch.tensor([[1.0, 9.0, 3.0, 8.0]], dtype=torch.float32)
    legal_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    best = masked_argmax(q_values, legal_mask)
    assert best.tolist() == [2]


def test_masked_max_returns_zero_when_no_legal_actions():
    q_values = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    legal_mask = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    value = masked_max(q_values, legal_mask)
    assert value.tolist() == [0.0]

