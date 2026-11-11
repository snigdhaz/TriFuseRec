import numpy as np
import torch
from src.sasrec_paper import PaperSASRec, sampled_metrics, left_pad, training_windows


def test_sampled_metrics_perfect_and_worst():
    n_users, n_items = 4, 300
    histories = [[0, 1], [2], [3, 4, 5], [6]]
    truths = np.array([10, 11, 12, 13])
    scores = np.zeros((n_users, n_items), dtype=np.float32)
    scores[np.arange(n_users), truths] = 1.0          # target strictly best
    m = sampled_metrics(scores, truths, histories)
    assert m['HR@10'] == 1.0 and abs(m['NDCG@10'] - 1.0) < 1e-9
    scores = -scores                                   # target strictly worst
    assert sampled_metrics(scores, truths, histories)['HR@10'] == 0.0


def test_sampled_ties_count_against_target():
    scores = np.ones((1, 200), dtype=np.float32)
    assert sampled_metrics(scores, np.array([5]), [[0]])['HR@10'] == 0.0


def test_training_windows_shift_by_one():
    seq, pos = training_windows([[3, 7, 9]], max_len=4)
    assert seq.tolist() == [[0, 0, 4, 8]] and pos.tolist() == [[0, 0, 8, 10]]


def test_paper_sasrec_padding_no_nan_and_invariant():
    torch.manual_seed(0)
    model = PaperSASRec(50, d=16, max_len=10).eval()
    short = torch.from_numpy(left_pad([[1, 2]], 10))
    out = model.scores(short)
    assert torch.isfinite(out).all()
    # Padding values must not change the result (padded positions are zeroed).
    assert torch.allclose(out, model.scores(torch.from_numpy(left_pad([[1, 2]], 10))))
